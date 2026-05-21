"""
HuggingFace 后端的训练配置（与 train_config.py 平行）。

用法：python train_hf.py
"""

from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN

from data.dataset import TinyStoriesDataset
from models.hf_gpt import HFGPT


def get_config():
    C = CN()

    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/tinystories_hf"

    C.data = TinyStoriesDataset.get_default_config()
    C.data.data_dir = "./data_cache"
    C.data.block_size = 256

    C.model = HFGPT.get_default_config()
    C.model.model_type = "gpt2"
    C.model.embd_pdrop = 0.1
    C.model.resid_pdrop = 0.1
    C.model.attn_pdrop = 0.1

    C.trainer = Trainer.get_default_config()
    C.trainer.max_iters = 2000
    C.trainer.batch_size = 32
    C.trainer.learning_rate = 3e-4
    C.trainer.weight_decay = 0.1
    C.trainer.grad_norm_clip = 1.0
    C.trainer.num_workers = 0

    C.log = CN()
    C.log.print_every = 10
    C.log.eval_every = 200
    C.log.sample_every = 500
    C.log.sample_max_new_tokens = 200

    return C
