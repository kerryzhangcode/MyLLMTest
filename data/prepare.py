"""
将原始文本转为模型可用的二进制 token 文件。

输出（nanoGPT / minGPT 常用格式）：
  data_cache/
    train.bin   # uint16 一维 token 序列（训练集 90%）
    val.bin     # 验证集 10%
    meta.pkl    # vocab_size, stoi, itos

TinyStories 使用**字符级分词**（character-level）：
  - 每个字符对应一个 token id
  - 词表很小（通常几十个 ASCII 字符），便于小模型学习
  - minGPT 同样支持任意 vocab_size，与 GPT-2 BPE（50257）只是分词方式不同
"""

import argparse
import os
import pickle

import numpy as np


def build_char_vocab(text: str):
    """从语料中统计所有出现过的字符，构建 stoi / itos。"""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]


def prepare(
    raw_txt_path: str,
    out_dir: str,
    train_ratio: float = 0.9,
) -> dict:
    """
    读取 raw txt → 划分 train/val → 写入 .bin 与 meta.pkl。

    Returns:
        meta 字典（含 vocab_size 等）
    """
    print(f"读取语料: {raw_txt_path}")
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        data = f.read()

    stoi, itos, vocab_size = build_char_vocab(data)
    print(f"词表大小 vocab_size = {vocab_size}")
    print(f"字符表: {''.join(itos[i] for i in range(vocab_size))!r}")

    n = len(data)
    split_at = int(n * train_ratio)
    train_text = data[:split_at]
    val_text = data[split_at:]

    train_ids = np.array(encode(train_text, stoi), dtype=np.uint16)
    val_ids = np.array(encode(val_text, stoi), dtype=np.uint16)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.pkl")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"train tokens: {len(train_ids):,} → {train_path}")
    print(f"val tokens:   {len(val_ids):,} → {val_path}")
    print(f"meta saved → {meta_path}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理 TinyStories 为 token 二进制文件")
    parser.add_argument("--raw_txt", type=str, default="./data_cache/tinystories_raw.txt")
    parser.add_argument("--out_dir", type=str, default="./data_cache")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    if not os.path.isfile(args.raw_txt):
        raise FileNotFoundError(
            f"找不到 {args.raw_txt}，请先运行: python -m data.download"
        )
    prepare(args.raw_txt, args.out_dir, args.train_ratio)
