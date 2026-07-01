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
    ("langchain_openai", "langchain-openai", "LLM-judge 型 metric"),
    ("rouge_score", "rouge-score", "rouge_l 指标"),
    ("sacrebleu", "sacrebleu", "bleu / chrf 指标"),
    ("matplotlib", "matplotlib", "make_plots 出 PNG 图（HTML 报告不需要）"),
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


def _check_env_isolation() -> None:
    """检查当前 python 是否跑在隔离环境里（venv），避免污染系统/用户环境。

    只告警不阻断：One-Eval 依赖较重，强烈建议跑在独立 venv，
    别装进系统自带 python 或用户全局 site-packages。
    """
    import os

    exe = sys.executable or ""
    in_venv = (hasattr(sys, "real_prefix")
               or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix))
    venv_env = os.environ.get("VIRTUAL_ENV")

    print("\n运行环境隔离：")
    if venv_env or in_venv:
        print(f"  [✓] 虚拟环境(venv): {venv_env or sys.prefix}")
    else:
        # 没有任何隔离迹象，且像系统自带 python → 重点提示
        looks_system = exe.startswith("/usr/bin") or exe.startswith("/usr/local/bin/python") \
            or exe.startswith("/System/")
        mark = "✗" if looks_system else "!"
        print(f"  [{mark}] 未检测到隔离环境（venv）")
        print(f"      当前解释器: {exe}")
        print(f"      强烈建议独立环境：uv venv .venv（或 python -m venv .venv），"
              f"再 uv pip install -e .（或 pip install -e .）")
        print(f"      之后所有脚本都用该环境的 python 绝对路径调用，勿动用系统/全局环境。")


def _check_reusable_envs() -> None:
    """探测可直接复用的 Python 环境，优先复用、避免重复装 GB 级重包(torch 等)。

    很多机器已有现成环境（仓库 .venv/、已激活的 venv）。命中就提示直接用，
    不必新建——新建从零装 torch/open-dataflow 等最耗时，复用是最快路径。
    """
    import os

    print("\n可复用的 Python 环境探测：")
    found = []
    # 1) 仓库根的 .venv
    venv_py = common.REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        found.append(f"仓库内 .venv → 直接用：{venv_py}")
    # 2) 当前已激活的 venv
    if os.environ.get("VIRTUAL_ENV"):
        found.append(f"已激活 venv：{os.environ['VIRTUAL_ENV']}")

    if found:
        for f in found:
            print(f"  [✓] {f}")
        print("  ↳ 已有可用环境就直接用它跑脚本，**无需新建、无需重装**（重装 torch 等重包最慢）。")
    else:
        # 没有现成环境：推荐 uv（优先）或 python -m venv（兜底）
        has_uv = _which("uv")
        if has_uv:
            print("  [○] 未发现现成可复用环境。推荐：uv venv && uv pip install -e .")
        else:
            print("  [○] 未发现现成可复用环境。推荐：python -m venv .venv && pip install -e .")
        print("      装在独立环境里即可（别污染系统/全局 python）。")


def _which(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


def _check_skill_registration() -> None:
    """检查 skill 是否能被 Claude Code 自动发现（项目级 .claude/skills/ 软链）。

    Claude Code 只扫描 ~/.claude/skills/ 与 <项目>/.claude/skills/ 下的 <name>/SKILL.md。
    本仓库在 <repo>/.claude/skills/one-eval 放了指向 one-eval-skill 的相对软链，
    clone 下来即自动可见；缺失时只告警并给出补建命令（不影响"读文件照流程跑"的用法）。
    """
    link = common.REPO_ROOT / ".claude" / "skills" / "one-eval"
    print("\nClaude Code skill 注册：")
    if link.exists() and (link / "SKILL.md").exists():
        print(f"  [✓] 已注册为项目级 skill：{link}")
        print(f"      重启 Claude Code 后，/skills 列表里会出现 one-eval（在本仓库内）。")
    else:
        print(f"  [○] 未检测到项目级 skill 软链（不影响把 SKILL.md 当文档读的用法）")
        print(f"      如想让 /skills 自动发现，在仓库根执行：")
        print(f"      mkdir -p .claude/skills && ln -s ../../one-eval-skill .claude/skills/one-eval")


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

    print("\n可选依赖（非必需，纯 API 评测无需安装，缺失不影响主流程）：")
    for mod, friendly, impact in OPTIONAL:
        ok = _has(mod)
        tail = "" if ok else f"  — 缺失：仅当用到 {impact} 才需装"
        print(f"  [{'✓' if ok else '○'}] {friendly}" + tail)
    print("  ↳ 这些标 [○] 是正常的，按需再装即可——不要逐个征询用户是否安装。")

    _check_env_isolation()
    _check_reusable_envs()
    _check_skill_registration()

    print("\n" + "=" * 40)
    if missing_required:
        print(f"✗ 缺少必需项：{', '.join(missing_required)}")
        print(INSTALL_HINT)
        return 1
    print("✓ 必需项齐全，可以开始评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
