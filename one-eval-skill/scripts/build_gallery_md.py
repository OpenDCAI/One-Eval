#!/usr/bin/env python3
"""
build_gallery_md.py — 从 bench_gallery.json 生成 references/bench_gallery.md 的候选区。

为什么需要：
  96 个 bench 来自主仓库的 gallery 表，但**本版默认它们都还没测通**，
  因此全部进「候选区（未验证）」。READY 区初始为空，随着 bench 被 smoke 测通
  （run_eval.py 写入 .local_state.json）后，由人工/后续脚本补充进 READY 区。

  候选区只给「接入所需的原始信息」（eval_type 猜测、source_url、原始字段 bench_keys、分类），
  agent 真正接入某个候选 bench 时，仍需按 eval_types.md 确认 key_mapping、走 smoke 验证。

用法：
  python build_gallery_md.py                # 用仓库默认 json，写到 references/bench_gallery.md
  python build_gallery_md.py --json <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

DEFAULT_JSON = common.REPO_ROOT / "one_eval" / "utils" / "bench_table" / "bench_gallery.json"
DEFAULT_OUT = common.SKILL_DIR / "references" / "bench_gallery.md"

HEADER = """# Bench Gallery — benchmark 清单（READY 区 + 候选区）

> 本文件由 `scripts/build_gallery_md.py` 从主仓库 `bench_gallery.json` 生成候选区。
> **不要手改候选区**（会被覆盖）；READY 区可手工维护。

## 接入约定（务必先读 `eval_types.md`）
- **READY 区**：已 smoke 测通、key_mapping 已确认、本地数据就绪的 bench，可直接复用（免重测）。
- **候选区**：来自主仓库 gallery 的 {n} 个 bench，**本版默认都未验证**。接入某个候选 bench 时：
  1. `eval_type` 列只是依据原始字段做的**初步归类**，需按 `eval_types.md` 复核。
  2. `原始字段` 是 HF 上的列名，**不等于** key_mapping —— 嵌套字段要先拍平。
  3. 用 `prepare_bench.py` 下载预览结构 → 填 key_mapping → `run_eval.py --smoke` 验证。
  4. 测通后该 bench 进入 READY（`.local_state.json`），可手工登记到下方 READY 区。
"""

READY_SECTION = """
---

## READY 区（已测通，可直接复用）

> 初始为空。每测通一个 bench，在此登记：bench_name ｜ eval_type ｜ 本地数据路径 ｜ key_mapping。
> 运行时由 `run_eval.py` 通过 `.local_state.json` 自动识别 READY，无需在此手填即可复用；
> 这里的清单仅供人查阅「哪些已稳定可用」。

_（暂无）_
"""


def _guess_keys_hint(bench: dict) -> str:
    keys = bench.get("bench_keys") or []
    return ", ".join(str(k) for k in keys) if keys else "—"


def build_md(data: dict) -> str:
    benches = data.get("benches", []) or []
    by_cat = defaultdict(list)
    for b in benches:
        cat = (b.get("meta") or {}).get("category") or "Uncategorized"
        by_cat[cat].append(b)

    parts = [HEADER.format(n=len(benches)), READY_SECTION]
    parts.append("\n---\n\n## 候选区（未验证，按分类）\n")

    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda x: x.get("bench_name", ""))
        parts.append(f"\n### {cat}（{len(items)}）\n")
        parts.append("| bench_name | eval_type(初判) | source_url | 原始字段 |")
        parts.append("|---|---|---|---|")
        for b in items:
            name = b.get("bench_name", "?")
            etype = b.get("bench_dataflow_eval_type", "?")
            url = b.get("bench_source_url") or "—"
            keys = _guess_keys_hint(b)
            parts.append(f"| {name} | `{etype}` | {url} | {keys} |")
        parts.append("")

    return "\n".join(parts) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="生成 bench_gallery.md 候选区")
    p.add_argument("--json", default=str(DEFAULT_JSON), help="bench_gallery.json 路径")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出 md 路径")
    args = p.parse_args(argv or sys.argv[1:])

    src = Path(args.json)
    if not src.exists():
        print(f"✗ 找不到 gallery json: {src}", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    md = build_md(data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    n = len(data.get("benches", []) or [])
    print(f"✓ 已生成 {out}（候选 {n} 个，READY 区初始为空）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
