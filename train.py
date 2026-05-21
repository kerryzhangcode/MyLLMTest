"""
TinyStories + minGPT 标准训练入口。

推荐流程：
  1. python -m data.download --max_stories 50000
  2. python -m data.prepare
  3. python train.py

命令行覆盖示例：
  python train.py --trainer.max_iters=5000 --model.model_type=gpt-micro
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

from config.train_config import get_config
from data.dataset import TinyStoriesDataset
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import config_to_checkpoint_dict, set_seed, setup_logging


def estimate_loss(model, dataset, device, max_batches: int = 20):
    """在验证集上估算平均交叉熵（越低越好）。"""
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
    """
    用训练好的模型续写故事。
    prompt 会被编码成 token 作为上下文。
    """
    model.eval()
    ctx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ctx, max_new_tokens, temperature=0.8, do_sample=True, top_k=40)
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

    # ----- 数据集 -----
    train_dataset = TinyStoriesDataset(config.data)
    val_cfg = TinyStoriesDataset.get_default_config()
    val_cfg.data_dir = config.data.data_dir
    val_cfg.block_size = config.data.block_size
    val_cfg.split = "val"
    val_dataset = TinyStoriesDataset(val_cfg)

    # ----- 模型（词表与 block_size 由数据决定）-----
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # ----- 训练器 -----
    trainer = Trainer(config.trainer, model, train_dataset)

    best_val_loss = float("inf")

    def on_batch_end(trainer: Trainer):
        nonlocal best_val_loss

        if trainer.iter_num % config.log.print_every == 0:
            print(
                f"iter {trainer.iter_num:5d} | "
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
                    },
                    ckpt,
                )
                print(f"  [save] 新最佳模型 → {ckpt}")

        if trainer.iter_num > 0 and trainer.iter_num % config.log.sample_every == 0:
            story = sample_story(
                model,
                train_dataset,
                trainer.device,
                config.log.sample_max_new_tokens,
            )
            print("  [sample] 生成片段:\n" + "-" * 40)
            print(story[:800])
            print("-" * 40)

    trainer.set_callback("on_batch_end", on_batch_end)
    trainer.run()

    print("训练结束。")


if __name__ == "__main__":
    main()
