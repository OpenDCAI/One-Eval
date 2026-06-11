#!/usr/bin/env python3
"""
run_metrics.py — 在 dataflow 主评测之外，按用户选择的 metric 注册表补充多维度打分。

定位：
  run_eval.py 给出的是 dataflow 内核的「主分数」（每个 eval_type 的默认指标）。
  本脚本读取每个 bench 的明细输出（detail_path / records），把用户挑选的额外 metric
  （如 bleu / rouge / exact_match / 自定义 LLM 裁判等）逐一算出来，形成多维度评分。

数据来源：
  run_eval.py 落盘的 eval_results.json，每个 bench 含 detail_path（dataflow 写出的明细），
  其中每条记录带 prediction 与 reference，交给 MetricRunner 复用注册表里的指标实现。

用法：
  # 列出注册表里所有可用 metric（供 agent 给用户挑选）
  python run_metrics.py --list

  # 对某次评测结果补充 metric（metrics 用逗号分隔的注册名）
  python run_metrics.py --results eval_outputs/eval_results.json --metrics bleu,rouge_l

  # 指定每个 metric 的优先级（primary 进主表，secondary 进附表）
  python run_metrics.py --results <path> --metrics exact_match:primary,bleu:secondary

退出码：0 = 成功；非 0 = 结果文件缺失或全部 metric 失败。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402


def _list_metrics() -> int:
    """打印注册表里所有 metric 的元数据（按维度分组），供 agent / 用户选择。"""
    try:
        custom = common.ensure_metrics_loaded()
        from one_eval.core.metric_registry import get_registered_metrics_meta
    except Exception as e:
        print(f"✗ 无法导入 metric 注册表（评测引擎未装好）: {e}", file=sys.stderr)
        return 1

    metas = get_registered_metrics_meta()
    if not metas:
        print("（注册表为空）")
        return 0
    if custom:
        print(f"（已加载自定义 metric 模块: {', '.join(custom)}）\n")

    by_dim: dict = {}
    for m in metas:
        by_dim.setdefault(getattr(m, "dimension", "correctness"), []).append(m)

    print(f"可用 metric（共 {len(metas)} 个，按维度分组）：")
    for dim in sorted(by_dim):
        print(f"\n=== 维度: {dim} ===")
        for m in by_dim[dim]:
            aliases = f"  别名: {', '.join(m.aliases)}" if m.aliases else ""
            cats = f"  类别: {', '.join(m.categories)}" if m.categories else ""
            print(f"  • {m.name}{aliases}{cats}")
            if m.desc:
                print(f"      {m.desc}")
            if m.usage:
                print(f"      适用: {m.usage}")
    return 0


def _parse_metrics_arg(s: str) -> list:
    """解析 'name:priority,name2' → [{name, priority}]，priority 默认 secondary。"""
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, prio = item.split(":", 1)
            out.append({"name": name.strip(), "priority": prio.strip()})
        else:
            out.append({"name": item, "priority": "secondary"})
    return out


def _build_bench_for_metrics(bench_result: dict):
    """从单个 bench 的评测结果构造 BenchInfo，把明细路径塞进 meta.artifact_paths。

    MetricRunner._resolve_inputs 会优先读 meta.artifact_paths.records。
    """
    from one_eval.core.state import BenchInfo

    detail = bench_result.get("detail_path")
    bench = BenchInfo(
        bench_name=bench_result.get("bench_name"),
        bench_dataflow_eval_type=bench_result.get("bench_dataflow_eval_type"),
        dataset_cache=detail,
    )
    if detail:
        bench.meta["artifact_paths"] = {"records": detail}
    # key_mapping 里的 target key 作为 ref 提示（若有）
    km = bench_result.get("key_mapping") or {}
    ref_key = (km.get("input_target_key") or km.get("input_label_key")
               or km.get("input_better_key"))
    if ref_key:
        bench.meta["ref_key"] = ref_key
    return bench


def run_metrics(results_path: str, metrics_cfg: list) -> dict:
    """对 eval_results.json 里每个 bench 补充 metric 打分，返回汇总 dict。"""
    common.ensure_metrics_loaded()
    from one_eval.metrics.runner import MetricRunner

    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    benches = data.get("results", []) or []
    runner = MetricRunner()

    out = {"model": data.get("model"), "metric_results": []}
    for br in benches:
        name = br.get("bench_name")
        if not br.get("ok") or not br.get("detail_path"):
            out["metric_results"].append(
                {"bench_name": name, "skipped": True,
                 "reason": "主评测未成功或无明细输出"})
            continue
        bench = _build_bench_for_metrics(br)
        res = runner.run_bench(bench, metrics_cfg)
        out["metric_results"].append({"bench_name": name, **res})
    return out


def _print_summary(summary: dict) -> None:
    print(f"\n模型: {summary.get('model')}")
    for mr in summary["metric_results"]:
        name = mr.get("bench_name")
        if mr.get("skipped"):
            print(f"\n  [{name}] 跳过：{mr.get('reason')}")
            continue
        if mr.get("error"):
            print(f"\n  [{name}] 错误：{mr.get('error')}")
            continue
        print(f"\n  [{name}] 样本数={mr.get('num_samples')}")
        for mname, mres in (mr.get("metrics") or {}).items():
            if mres.get("error"):
                print(f"    ✗ {mname}: {mres['error']}")
            else:
                prio = mres.get("priority", "secondary")
                print(f"    • {mname} [{prio}]: {mres.get('score')}")


def main(argv=None):
    p = argparse.ArgumentParser(description="按注册表 metric 补充多维度打分")
    p.add_argument("--list", action="store_true", help="列出所有可用 metric")
    p.add_argument("--results", help="run_eval.py 输出的 eval_results.json 路径")
    p.add_argument("--metrics", help="逗号分隔的 metric 名（可加 :priority）")
    p.add_argument("--output", help="metric 结果输出 json（默认写到结果同目录）")
    args = p.parse_args(argv or sys.argv[1:])

    if args.list:
        return _list_metrics()

    if not args.results or not args.metrics:
        print("错误：需要 --results 与 --metrics（或用 --list 查看可用 metric）",
              file=sys.stderr)
        return 2

    results_path = args.results
    if not Path(results_path).exists():
        print(f"✗ 结果文件不存在: {results_path}", file=sys.stderr)
        return 2

    metrics_cfg = _parse_metrics_arg(args.metrics)
    if not metrics_cfg:
        print("✗ 未解析出任何 metric", file=sys.stderr)
        return 2

    summary = run_metrics(results_path, metrics_cfg)
    _print_summary(summary)

    out_path = Path(args.output) if args.output else \
        Path(results_path).parent / "metric_results.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nmetric 结果已写入: {out_path.absolute()}")

    # 全部 bench 都失败/跳过才算非 0
    any_ok = any(
        (not mr.get("skipped") and not mr.get("error") and mr.get("metrics"))
        for mr in summary["metric_results"]
    )
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
