"""
集中管理训练超参数。

模块划分：
  system  - 随机种子、输出目录
  data    - 数据路径、序列长度
  model   - GPT 规模（gpt-nano / micro / mini ...）
  trainer - 学习率、batch、迭代次数等
"""

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN

from data.dataset import TinyStoriesDataset


def get_config():
    C = CN()

    # ---------- 系统 ----------
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/tinystories"

    # ---------- 数据 ----------
    C.data = TinyStoriesDataset.get_default_config()
    C.data.data_dir = "./data_cache"
    C.data.block_size = 256  # 上下文长度；越大显存越高

    # ---------- 模型 ----------
    # gpt-nano ≈ 0.3M 参数，适合 CPU/入门 GPU 快速实验
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt-mini"
    C.model.embd_pdrop = 0.1
    C.model.resid_pdrop = 0.1
    C.model.attn_pdrop = 0.1

    # ---------- 训练器 ----------
    C.trainer = Trainer.get_default_config()
    C.trainer.max_iters = 20000
    C.trainer.batch_size = 32
    C.trainer.learning_rate = 3e-4
    C.trainer.weight_decay = 0.1
    C.trainer.grad_norm_clip = 1.0
    C.trainer.num_workers = 0  # macOS 上多进程 DataLoader 有时需设为 0

    # ---------- 日志与评估 ----------
    C.log = CN()
    C.log.print_every = 10
    C.log.eval_every = 200
    C.log.sample_every = 500
    C.log.sample_max_new_tokens = 200

    return C
