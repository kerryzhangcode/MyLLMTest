"""
通用工具：随机种子、日志目录、轻量配置类 CfgNode。

CfgNode 模仿 yacs 的用法，支持嵌套属性和命令行覆盖，例如：
  python train.py --trainer.batch_size=32 --model.model_type=gpt-micro
"""

import json
import os
import random
import sys
from ast import literal_eval

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 的随机性，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_model_config(config) -> None:
    """
    GPT / HFGPT 要求 model_type 与 (n_layer, n_head, n_embd) 二选一。

    训练时 GPT.__init__ 会把 preset 展开写回 config，导致 checkpoint 里两套字段都有。
    加载或保存前调用本函数：若已有完整 n_*，则清掉 model_type；否则保留 model_type。
    """
    params_given = all(
        [
            getattr(config, "n_layer", None) is not None,
            getattr(config, "n_head", None) is not None,
            getattr(config, "n_embd", None) is not None,
        ]
    )
    if params_given:
        config.model_type = None
    elif getattr(config, "model_type", None) is not None:
        config.n_layer = None
        config.n_head = None
        config.n_embd = None


def config_to_checkpoint_dict(config) -> dict:
    """保存 checkpoint 用的 config 副本，model 段已规范化。"""
    d = config.to_dict()
    mcfg = CfgNode(**d["model"])
    normalize_model_config(mcfg)
    d["model"] = mcfg.to_dict()
    return d


def setup_logging(config) -> None:
    """
    在工作目录下保存本次运行的参数与完整配置 JSON，方便事后对照。
    config 需包含 config.system.work_dir。
    """
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "args.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv))
    with open(os.path.join(work_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """
    轻量配置节点：用属性访问代替字典，支持 merge_from_dict / merge_from_args。
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent: int) -> str:
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d: dict) -> None:
        self.__dict__.update(d)

    def merge_from_args(self, args: list) -> None:
        """
        解析命令行形如 --trainer.max_iters=1000 的覆盖项。
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval
            try:
                val = literal_eval(val)
            except ValueError:
                pass
            assert key[:2] == "--"
            key = key[2:]
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
