#!/usr/bin/env bash
# 一键跑通：下载 → 预处理 → 训练
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/3 下载 TinyStories 子集（5 万条，可改 --max_stories）==="
python -m data.download --max_stories 50000

echo "=== 2/3 字符级分词并写入 train.bin / val.bin ==="
python -m data.prepare

echo "=== 3/3 训练 minGPT ==="
python train.py
