"""
加载 checkpoint，从给定 prompt 续写 TinyStories 风格短文。

用法：
  python sample.py
  python sample.py --prompt="Lily was a little cat."
  python sample.py --ckpt=./out/tinystories/model.pt --max_new_tokens=300
"""

import argparse
import os
import pickle

import torch

from data.dataset import TinyStoriesDataset
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN, normalize_model_config


def load_model(ckpt_path: str, data_dir: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 从 checkpoint 恢复模型配置，若不存在则用 gpt-nano 默认
    if "config" in ckpt and "model" in ckpt["config"]:
        mcfg = CN(**ckpt["config"]["model"])
        normalize_model_config(mcfg)
    else:
        mcfg = GPT.get_default_config()
        mcfg.model_type = "gpt-nano"

    ds_cfg = TinyStoriesDataset.get_default_config()
    ds_cfg.data_dir = data_dir
    ds_cfg.split = "train"
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    ds_cfg.block_size = ckpt.get("config", {}).get("data", {}).get("block_size", 256)

    dataset = TinyStoriesDataset(ds_cfg)
    mcfg.vocab_size = meta["vocab_size"]
    mcfg.block_size = dataset.get_block_size()

    model = GPT(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./out/tinystories/model.pt")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"找不到权重 {args.ckpt}，请先完成训练")

    model, dataset = load_model(args.ckpt, args.data_dir, device)

    ctx = torch.tensor([dataset.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            ctx,
            args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=args.top_k,
        )
    print(dataset.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
