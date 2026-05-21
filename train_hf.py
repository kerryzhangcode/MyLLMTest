"""
TinyStories 训练入口 —— HuggingFace Transformers 后端（与 train.py 平行）。

数据流水线与 minGPT 版完全相同，仅替换模型为 models.hf_gpt.HFGPT：
  - 训练：GPT2LMHeadModel 全序列前向
  - 采样：model.generate(use_cache=True) 使用 KV cache

推荐流程：
  python -m data.download --max_stories 50000
  python -m data.prepare
  python train_hf.py

对比学习：
  python train.py      # 手写 minGPT
  python train_hf.py   # 成熟 HF 封装
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

from config.train_config_hf import get_config
from data.dataset import TinyStoriesDataset
from mingpt.trainer import Trainer
from mingpt.utils import config_to_checkpoint_dict, set_seed, setup_logging
from models.hf_gpt import HFGPT


def estimate_loss(model, dataset, device, max_batches: int = 20):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    losses = []
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            if b + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / len(losses)


def sample_story(model, dataset, device, max_new_tokens: int, prompt: str = "Once upon a time"):
    model.eval()
    ctx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ctx, max_new_tokens)
    text = dataset.decode(out[0].tolist())
    model.train()
    return text


def main():
    config = get_config()
    if len(sys.argv) > 1:
        config.merge_from_args(sys.argv[1:])

    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    train_dataset = TinyStoriesDataset(config.data)
    val_cfg = TinyStoriesDataset.get_default_config()
    val_cfg.data_dir = config.data.data_dir
    val_cfg.block_size = config.data.block_size
    val_cfg.split = "val"
    val_dataset = TinyStoriesDataset(val_cfg)

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = HFGPT(config.model)

    trainer = Trainer(config.trainer, model, train_dataset)
    best_val_loss = float("inf")

    def on_batch_end(trainer: Trainer):
        nonlocal best_val_loss

        if trainer.iter_num % config.log.print_every == 0:
            print(
                f"[HF] iter {trainer.iter_num:5d} | "
                f"loss {trainer.loss.item():.4f} | "
                f"{trainer.iter_dt * 1000:.1f} ms/iter"
            )

        if trainer.iter_num > 0 and trainer.iter_num % config.log.eval_every == 0:
            val_loss = estimate_loss(model, val_dataset, trainer.device)
            print(f"  [eval] val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = os.path.join(config.system.work_dir, "model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config_to_checkpoint_dict(config),
                        "iter_num": trainer.iter_num,
                        "val_loss": val_loss,
                        "backend": "hf_gpt",
                    },
                    ckpt,
                )
                print(f"  [save] {ckpt}")

        if trainer.iter_num > 0 and trainer.iter_num % config.log.sample_every == 0:
            story = sample_story(
                model,
                train_dataset,
                trainer.device,
                config.log.sample_max_new_tokens,
            )
            print("  [sample] KV cache 生成:\n" + "-" * 40)
            print(story[:800])
            print("-" * 40)

    trainer.set_callback("on_batch_end", on_batch_end)
    trainer.run()
    print("训练结束（HF 后端）。")


if __name__ == "__main__":
    main()
