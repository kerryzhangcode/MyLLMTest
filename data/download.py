"""
从 HuggingFace 下载 TinyStories 子集并保存为纯文本。

数据集：karpathy/tinystories-gpt4-clean
  - GPT-4 生成的儿童向英文小故事
  - 字符集接近 ASCII，适合字符级（character-level）语言模型

首次运行需要网络；可用 --max_stories 限制条数以便快速试跑。
"""

import argparse
import os
from typing import Optional


def download_tinystories(
    out_dir: str,
    max_stories: Optional[int] = None,
    split: str = "train",
) -> str:
    """
    下载故事并写入单个 .txt 文件（故事之间用双换行分隔）。

    Returns:
        输出文件路径
    """
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tinystories_raw.txt")

    if os.path.isfile(out_path):
        print(f"已存在缓存文件，跳过下载: {out_path}")
        return out_path

    print("正在从 HuggingFace 加载 karpathy/tinystories-gpt4-clean ...")
    # streaming=True 可在不全量下载到内存的情况下迭代（适合大集）
    ds = load_dataset("karpathy/tinystories-gpt4-clean", split=split, streaming=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            count += 1
            if count % 10000 == 0:
                print(f"  已写入 {count} 条故事 ...")
            if max_stories is not None and count >= max_stories:
                break

    print(f"完成：共 {count} 条故事 → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 TinyStories 原始文本")
    parser.add_argument("--out_dir", type=str, default="./data_cache")
    parser.add_argument(
        "--max_stories",
        type=int,
        default=50000,
        help="最多下载多少条；设为 -1 表示不限制（完整集约 273 万条，耗时长）",
    )
    args = parser.parse_args()
    limit = None if args.max_stories < 0 else args.max_stories
    download_tinystories(args.out_dir, max_stories=limit)
