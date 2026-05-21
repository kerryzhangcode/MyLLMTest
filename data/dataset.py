"""
语言模型训练用的 PyTorch Dataset。

每个样本是一段长度为 block_size 的输入 x，以及“向右平移一位”的目标 y：
  x: [t0, t1, ..., t_{B-1}]
  y: [t1, t2, ..., t_B]   （预测下一个字符）

这样模型在每个位置都在做 next-token prediction。
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from mingpt.utils import CfgNode as CN


class TinyStoriesDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.data_dir = "./data_cache"
        C.split = "train"  # train | val
        C.block_size = 256
        return C

    def __init__(self, config):
        self.config = config
        split = config.split
        assert split in {"train", "val"}

        bin_path = os.path.join(config.data_dir, f"{split}.bin")
        meta_path = os.path.join(config.data_dir, "meta.pkl")

        if not os.path.isfile(bin_path):
            raise FileNotFoundError(
                f"找不到 {bin_path}，请先运行 data.download 与 data.prepare"
            )

        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab_size = meta["vocab_size"]
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]

        # 每个样本需要 block_size+1 个连续 token（才能构造 x 和 y）
        self.length = len(self.data) - config.block_size - 1

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_block_size(self) -> int:
        return self.config.block_size

    def decode(self, ids: list[int] | np.ndarray) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        bs = self.config.block_size
        # 从随机起点 idx 截取一段连续 token
        chunk = self.data[idx : idx + bs + 1].astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
