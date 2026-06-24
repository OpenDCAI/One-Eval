#!/usr/bin/env python3
"""
make_plots.py — 把评测结果渲染成分析图，供最终报告（图文并茂）嵌入。

定位：
  报告环节需要「图文并茂、有总有详」。本脚本只负责确定性地产出 PNG 图表，
  图怎么解读、怎么串成叙事，由调用方 agent 按 report_template 写文字。

输入：
  - eval_results.json（run_eval.py 产出）：每个 bench 的 DataFlow 诊断分数
  - metric_results.json（run_metrics.py 产出，可选）：primary metric + 多维度 metric 打分

产出（默认写到结果同目录的 plots/ 下）：
  - bench_scores.png        各 bench 主分数对比（条形图）
  - metric_heatmap.png      bench × metric 分数热力图（有多维 metric 时）
  - sample_validity.png     各 bench 有效样本占比（valid/total）

用法：
  python make_plots.py --results eval_outputs/eval_results.json \
      --metrics eval_outputs/metric_results.json --out eval_outputs/plots

退出码：0 = 至少产出一张图；非 0 = 无可用数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 非交互后端：无显示环境也能存图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文标签可读：优先用系统可用的中文字体，找不到则退回默认（英文标签仍正常）
for _f in ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_result_index(metrics: dict) -> dict:
    return {
        r.get("bench_name"): r
        for r in (metrics or {}).get("metric_results", []) or []
        if r.get("bench_name")
    }


def _primary_score_from_metric_row(row: dict):
    primary = row.get("primary_metric_result")
    if isinstance(primary, dict) and primary.get("score") is not None:
        return primary.get("score")
    for val in (row.get("metrics") or {}).values():
        if isinstance(val, dict) and val.get("priority") == "primary" and val.get("score") is not None:
            return val.get("score")
    return None


def plot_bench_scores(results: dict, metrics: dict, out_dir: Path):
    """各 bench 主分数对比条形图。"""
    metric_rows = _metric_result_index(metrics)
    rows = []
    for r in results.get("results", []):
        if not r.get("ok"):
            continue
        score = None
        if isinstance(r.get("primary_metric_result"), dict):
            score = r["primary_metric_result"].get("score")
        if score is None:
            score = _primary_score_from_metric_row(metric_rows.get(r.get("bench_name"), {}))
        if score is None:
            score = (r.get("dataflow_score") or {}).get("score")
        if score is not None:
            rows.append((r["bench_name"], score))
    if not rows:
        return None
    names = [r[0] for r in rows]
    scores = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.5))
    bars = ax.bar(names, scores, color="#4C78A8")
    ax.set_ylabel("Score")
    ax.set_title(f"Benchmark 主分数对比 — {results.get('model', '')}")
    ax.set_ylim(0, 1.0 if all(s <= 1 for s in scores) else max(scores) * 1.15)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{s:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = out_dir / "bench_scores.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_sample_validity(results: dict, out_dir: Path):
    """各 bench 有效样本占比（valid/total）堆叠条形。"""
    rows = []
    for r in results.get("results", []):
        s = r.get("dataflow_score") or {}
        total, valid = s.get("total_samples"), s.get("valid_samples")
        if total:
            rows.append((r["bench_name"], valid or 0, total))
    if not rows:
        return None
    names = [r[0] for r in rows]
    valid = [r[1] for r in rows]
    invalid = [r[2] - r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.5))
    ax.bar(names, valid, label="valid", color="#54A24B")
    ax.bar(names, invalid, bottom=valid, label="invalid", color="#E45756")
    ax.set_ylabel("Samples")
    ax.set_title("各 Benchmark 有效样本占比")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = out_dir / "sample_validity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_metric_heatmap(metrics: dict, out_dir: Path):
    """bench × metric 分数热力图（多维 metric 时才有意义）。"""
    mrs = metrics.get("metric_results", []) or []
    bench_names, metric_names, grid = [], [], []
    # 收集 metric 名全集
    all_metrics = []
    for mr in mrs:
        for mn in (mr.get("metrics") or {}):
            if mn not in all_metrics:
                all_metrics.append(mn)
    if not all_metrics:
        return None

    for mr in mrs:
        if not mr.get("metrics"):
            continue
        bench_names.append(mr["bench_name"])
        row = []
        for mn in all_metrics:
            cell = (mr["metrics"].get(mn) or {})
            row.append(cell.get("score") if not cell.get("error") else None)
        grid.append(row)
    if not bench_names:
        return None
    metric_names = all_metrics

    import numpy as np
    arr = np.array([[v if v is not None else np.nan for v in row] for row in grid],
                   dtype=float)

    fig, ax = plt.subplots(figsize=(max(5, len(metric_names) * 1.3),
                                    max(3.5, len(bench_names) * 0.7)))
    im = ax.imshow(arr, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_yticks(range(len(bench_names)))
    ax.set_yticklabels(bench_names)
    ax.set_title("Bench × Metric 分数热力图")
    for i in range(len(bench_names)):
        for j in range(len(metric_names)):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#222")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    p = out_dir / "metric_heatmap.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description="评测结果图表生成")
    p.add_argument("--results", required=True, help="eval_results.json 路径")
    p.add_argument("--metrics", help="metric_results.json 路径（可选）")
    p.add_argument("--out", help="图表输出目录（默认结果同目录 plots/）")
    args = p.parse_args(argv or sys.argv[1:])

    if not Path(args.results).exists():
        print(f"✗ 结果文件不存在: {args.results}", file=sys.stderr)
        return 2
    results = _load(args.results)

    out_dir = Path(args.out) if args.out else Path(args.results).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []
    metrics = _load(args.metrics) if args.metrics and Path(args.metrics).exists() else {}
    p1 = plot_bench_scores(results, metrics, out_dir)
    if p1:
        made.append(p1)
    p2 = plot_sample_validity(results, out_dir)
    if p2:
        made.append(p2)
    if metrics:
        p3 = plot_metric_heatmap(metrics, out_dir)
        if p3:
            made.append(p3)

    if not made:
        print("✗ 无可用数据生成图表", file=sys.stderr)
        return 1
    print("已生成图表：")
    for m in made:
        print(f"  • {Path(m).absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
