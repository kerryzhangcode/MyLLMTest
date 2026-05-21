"""
HuggingFace 后端采样：加载 model.pt，使用 generate() + KV cache。

用法：
  python sample_hf.py
  python sample_hf.py --ckpt=./out/tinystories_hf/model.pt --prompt="Lily was a cat."
"""

import argparse
import os
import pickle

import torch

from data.dataset import TinyStoriesDataset
from mingpt.utils import CfgNode as CN, normalize_model_config
from models.hf_gpt import HFGPT


def load_model(ckpt_path: str, data_dir: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "config" in ckpt and "model" in ckpt["config"]:
        mcfg = CN(**ckpt["config"]["model"])
        normalize_model_config(mcfg)
    else:
        mcfg = HFGPT.get_default_config()
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

    model = HFGPT(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./out/tinystories_hf/model.pt")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"找不到权重 {args.ckpt}，请先运行 train_hf.py")

    model, dataset = load_model(args.ckpt, args.data_dir, device)

    ctx = torch.tensor([dataset.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            ctx,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(dataset.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
