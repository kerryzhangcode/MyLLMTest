"""
HuggingFace Transformers 封装的 GPT 语言模型（与 mingpt/model.py 平行）。

为什么用这一份：
  - GPT2LMHeadModel：工业级实现，CUDA/CPU 优化、梯度检查点等可选
  - model.generate(use_cache=True)：内置 KV Cache，自回归生成时避免重复计算历史 K/V
  - GPT2Config：标准超参配置，和 minGPT 的 gpt-nano / micro 等预设一一对应

与 minGPT 的接口对齐（Trainer / train.py 无需改逻辑）：
  - forward(idx, targets) -> logits, loss
  - configure_optimizers(train_config)
  - generate(idx, max_new_tokens, ...)
  - block_size 属性

训练时：整段前向，不用 KV cache（并行算全序列）。
推理时：generate() 自动启用 KV cache。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from mingpt.utils import CfgNode as CN, normalize_model_config

# 与 minGPT model_type 对齐的规模预设
MODEL_PRESETS = {
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
}


class HFGPT(nn.Module):
    """
    薄封装层：内部是 transformers.GPT2LMHeadModel，对外保持 minGPT 同款 API。
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = "gpt-nano"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # 生成专用（训练不用 KV cache）
        C.temperature = 0.8
        C.top_k = 40
        C.do_sample = True
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.block_size = config.block_size
        self.config = config

        normalize_model_config(config)
        type_given = config.model_type is not None
        params_given = all(
            [config.n_layer is not None, config.n_head is not None, config.n_embd is not None]
        )
        assert type_given ^ params_given

        if type_given:
            if config.model_type not in MODEL_PRESETS:
                raise ValueError(
                    f"未知 model_type={config.model_type!r}，可选: {list(MODEL_PRESETS)}"
                )
            config.merge_from_dict(MODEL_PRESETS[config.model_type])

        n_embd = config.n_embd
        n_head = config.n_head
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"

        hf_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.block_size,
            n_embd=n_embd,
            n_layer=config.n_layer,
            n_head=n_head,
            n_inner=4 * n_embd,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            # 字符级小词表，不需要 GPT-2 的 layer norm epsilon 特殊处理
            use_cache=False,
        )

        self.hf_model = GPT2LMHeadModel(hf_config)
        # 与 minGPT 一致：tie 词嵌入与 lm_head 权重（可选，HF 默认已 tie）
        n_params = sum(p.numel() for p in self.hf_model.parameters())
        print("[HFGPT] backend=transformers.GPT2LMHeadModel, params=%.2fM" % (n_params / 1e6,))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        idx:    (B, T) 输入 token
        targets: (B, T) 下一 token 标签（与 minGPT Dataset 的 y 一致）

        手动算 CE，避免 HF labels 内部再 shift 与我们的 (x,y) 不一致。
        """
        b, t = idx.size()
        assert t <= self.block_size

        outputs = self.hf_model(input_ids=idx, use_cache=False)
        logits = outputs.logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def configure_optimizers(self, train_config):
        """与 minGPT 相同的 AdamW 分组：Linear 权重 decay，bias/LayerNorm/Embedding 不 decay。"""
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        union = decay | no_decay
        assert len(inter) == 0
        assert len(param_dict.keys() - union) == 0

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": train_config.weight_decay,
            },
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        使用 HuggingFace generate()，内部自动 KV cache（use_cache=True）。

        比 minGPT 手写循环更高效，尤其是生成长文本时。
        """
        temperature = temperature if temperature is not None else self.config.temperature
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        top_k = top_k if top_k is not None else self.config.top_k

        self.hf_model.eval()
        out = self.hf_model.generate(
            input_ids=idx,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_k=top_k if top_k and top_k > 0 else None,
            use_cache=True,
            pad_token_id=0,
        )
        self.hf_model.train()
        return out
