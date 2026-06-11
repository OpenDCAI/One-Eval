#!/usr/bin/env python3
"""
doctor.py — 环境自检（安装后、评测前跑一次，确认"装好就能跑"）。

为什么需要：本 skill 不自包含，脚本靠 import one_eval + dataflow。若主仓库依赖
没装全（典型：手工凑的残缺虚拟环境），评测会跑到一半才报神秘 ImportError。
本脚本提前体检：缺什么、怎么修，一次说清，不让用户撞墙。

检查项分两级：
  - 必需（缺则确定性评测无法运行）：python>=3.10 / one_eval 可导入 / dataflow /
    datasets / numpy / pandas / requests / yaml
  - 可选（缺只影响部分指标，确定性评测不受影响）：
    langchain_openai(LLM-judge 型 metric) / rouge_score(rouge_l) /
    sacrebleu(bleu/chrf) / matplotlib(出图)

用法：
  python scripts/doctor.py
退出码：0 = 必需项齐全（可选项缺失只告警）；非 0 = 有必需项缺失。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402  （把仓库根加进 sys.path）

# (import 名, 友好名, 缺失时的一句话影响)
REQUIRED = [
    ("one_eval", "one_eval（主仓库包）", "评测内核无法导入，整套不可用"),
    ("dataflow", "dataflow", "评测引擎缺失，run_eval 无法运行"),
    ("datasets", "datasets (HuggingFace)", "无法下载/加载 benchmark 数据"),
    ("numpy", "numpy", "指标计算依赖"),
    ("pandas", "pandas", "数据处理依赖"),
    ("requests", "requests", "API 连通探测依赖"),
    ("yaml", "pyyaml", "无法解析 evalspec.yaml"),
]
OPTIONAL = [
    ("langchain_openai", "langchain-openai", "LLM-judge 型 metric 跳过（确定性评测不受影响）"),
    ("rouge_score", "rouge-score", "rouge_l 指标不可用"),
    ("sacrebleu", "sacrebleu", "bleu / chrf 指标不可用"),
    ("matplotlib", "matplotlib", "make_plots 出图不可用"),
]

INSTALL_HINT = (
    "修复：在仓库根用主环境执行  pip install -e .  （或 uv pip install -e .）。"
    "详见 README 3.1 安装环境。"
)


def _has(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except Exception:
        return False


def main() -> int:
    print("One-Eval 环境自检\n" + "=" * 40)

    # Python 版本
    py_ok = sys.version_info >= (3, 10)
    pv = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[{'✓' if py_ok else '✗'}] Python {pv}（需 ≥ 3.10）")
    print(f"    解释器: {sys.executable}")

    missing_required = [] if py_ok else ["python>=3.10"]

    print("\n必需依赖：")
    for mod, friendly, impact in REQUIRED:
        ok = _has(mod)
        print(f"  [{'✓' if ok else '✗'}] {friendly}" + ("" if ok else f"  — {impact}"))
        if not ok:
            missing_required.append(friendly)

    print("\n可选依赖（缺失不影响确定性评测）：")
    for mod, friendly, impact in OPTIONAL:
        ok = _has(mod)
        print(f"  [{'✓' if ok else '○'}] {friendly}" + ("" if ok else f"  — {impact}"))

    print("\n" + "=" * 40)
    if missing_required:
        print(f"✗ 缺少必需项：{', '.join(missing_required)}")
        print(INSTALL_HINT)
        return 1
    print("✓ 必需项齐全，可以开始评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
