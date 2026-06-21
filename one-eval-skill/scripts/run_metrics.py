#!/usr/bin/env python3
"""
run_metrics.py — 按 gallery evaluation contract 计算 benchmark 主分，并可补充诊断 metric。

定位：
  run_eval.py 负责 DataFlow 生成、明细和诊断分；benchmark 主分由本脚本读取
  meta.evaluation.primary_metric 后计算。用户显式传入的 --metrics 作为额外诊断维度。

数据来源：
  run_eval.py 落盘的 eval_results.json，每个 bench 含 detail_path（dataflow 写出的明细），
  其中每条记录带 prediction 与 reference，交给 MetricRunner 复用注册表里的指标实现。

用法：
  # 列出注册表里所有可用 metric（供 agent 给用户挑选）
  python run_metrics.py --list

  # 按 gallery contract 自动计算主分
  python run_metrics.py --results eval_outputs/eval_results.json

  # 对某次评测结果补充 diagnostic metric（metrics 用逗号分隔的注册名）
  python run_metrics.py --results eval_outputs/eval_results.json --metrics bleu,rouge_l

  # 指定每个 metric 的优先级（primary 进主表，secondary 进附表）
  python run_metrics.py --results <path> --metrics exact_match:primary,bleu:secondary

退出码：0 = 成功；非 0 = 结果文件缺失或全部 metric 失败。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

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


def _default_evaluation_for_eval_type(eval_type: str) -> dict:
    if eval_type == "key3_q_choices_a":
        return {
            "score_source": "metric_stage",
            "official_metric": "accuracy",
            "primary_metric": "choice_accuracy",
            "denominator": "total",
            "prediction_mode": "generation",
            "parser": {"type": "choice_letter", "choices": "A-D"},
            "failure_policy": {
                "parse_failed": "score_zero",
                "empty_output": "score_zero",
                "invalid_reference": "exclude",
            },
            "official_compatibility": {
                "equivalent": False,
                "reason": "Fallback generation+parse contract; gallery meta.evaluation is missing.",
            },
        }
    return {}


def _resolve_evaluation_contract(bench_result: dict) -> dict:
    evaluation = bench_result.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("primary_metric"):
        return evaluation

    gallery = common.get_gallery_bench(bench_result.get("bench_name")) or {}
    gallery_eval = ((gallery.get("meta") or {}).get("evaluation") or {})
    if isinstance(gallery_eval, dict) and gallery_eval.get("primary_metric"):
        return gallery_eval

    return _default_evaluation_for_eval_type(bench_result.get("bench_dataflow_eval_type"))


def _build_bench_for_metrics(bench_result: dict):
    """从单个 bench 的评测结果构造 BenchInfo，把明细路径塞进 meta.artifact_paths。

    MetricRunner._resolve_inputs 会优先读 meta.artifact_paths.records。
    """
    from one_eval.core.state import BenchInfo

    detail = bench_result.get("detail_path")
    gallery = common.get_gallery_bench(bench_result.get("bench_name")) or {}
    meta = dict(gallery.get("meta") or {})
    bench = BenchInfo(
        bench_name=bench_result.get("bench_name"),
        bench_dataflow_eval_type=bench_result.get("bench_dataflow_eval_type"),
        dataset_cache=detail,
        bench_prompt_template=bench_result.get("bench_prompt_template") or gallery.get("bench_prompt_template"),
    )
    meta.update({k: v for k, v in (bench_result.get("meta") or {}).items()})
    bench.meta.update(meta)
    if detail:
        bench.meta["artifact_paths"] = {"records": detail}
    # key_mapping 里的 target key 作为 ref 提示（若有）
    km = bench_result.get("key_mapping") or {}
    eval_type = bench_result.get("bench_dataflow_eval_type")
    pred_key = "generated_ans"
    if eval_type in ("key3_q_choices_a", "key3_q_choices_as"):
        pred_key = "generated_ans"
    ref_key = (
        km.get("input_target_key")
        or km.get("input_targets_key")
        or km.get("input_label_key")
        or km.get("input_labels_key")
        or km.get("input_better_key")
    )
    bench.meta["pred_key"] = pred_key
    if ref_key:
        bench.meta["ref_key"] = ref_key
    if km:
        bench.meta["key_mapping"] = km
    return bench


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "bench")).strip("_") or "bench"


def _primary_step3_path(results_path: str, bench_result: dict) -> Path:
    detail = Path(bench_result.get("detail_path") or "")
    if detail.name == "step_step2.jsonl" and detail.parent.exists():
        return detail.parent / "step_step3_primary.jsonl"
    return Path(results_path).parent / f"{_safe_name(bench_result.get('bench_name'))}_step3_primary.jsonl"


def _detail_score(detail: Any) -> Any:
    if isinstance(detail, dict) and "score" in detail:
        return detail.get("score")
    if isinstance(detail, bool):
        return 1.0 if detail else 0.0
    if isinstance(detail, (int, float)) or detail is None:
        return detail
    return None


def _drop_dataflow_diagnostics(record: dict) -> dict:
    legacy = {
        "eval_valid",
        "eval_score",
        "eval_error",
        "eval_pred",
        "eval_pred_choice",
        "eval_ref_choice",
        "eval_parse_strategy",
        "judge_response",
    }
    return {k: v for k, v in record.items() if k not in legacy}


def _primary_answer(
    idx: int,
    detail: Any,
    artifacts: dict,
    parse_result: dict | None = None,
) -> Any:
    pred_parse = (parse_result or {}).get("pred") or {}
    if pred_parse.get("normalized") is not None:
        return pred_parse.get("normalized")

    pred_choices = artifacts.get("pred_choices") or []
    if idx < len(pred_choices) and pred_choices[idx] is not None:
        return pred_choices[idx]

    for key in ("pred_vals", "extracted_values"):
        values = artifacts.get(key) or []
        if idx < len(values) and values[idx] is not None:
            return values[idx]

    if isinstance(detail, dict):
        for key in ("extracted", "answer", "pred", "prediction"):
            if detail.get(key) is not None:
                return detail.get(key)
    return None


def _write_primary_step3(results_path: str, bench_result: dict, bench, runner, primary_row: dict) -> str | None:
    """Write raw sample fields + generated answer + compact primary metric result.

    This becomes the report dashboard source of truth. DataFlow step2 fields remain
    diagnostic/legacy and are intentionally not copied into step3.
    """
    inputs = runner._resolve_inputs(bench)
    if not inputs:
        return None
    preds, refs, records, _align = runner._load_pred_ref_records(inputs, bench)
    details = primary_row.get("primary_metric_details") or []
    artifacts = primary_row.get("primary_metric_artifacts") or {}
    parse_results = artifacts.get("parse_results") or []

    out_path = _primary_step3_path(results_path, bench_result)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            detail = details[idx] if idx < len(details) else None
            parse_result = parse_results[idx] if idx < len(parse_results) else None
            score = _detail_score(detail)
            row = _drop_dataflow_diagnostics(dict(record))
            if "generated_ans" not in row:
                row["generated_ans"] = preds[idx] if idx < len(preds) else None
            row["primary_answer"] = _primary_answer(idx, detail, artifacts, parse_result)
            row["primary_score"] = score
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(out_path.absolute())


def run_metrics(results_path: str, metrics_cfg: list | None = None) -> dict:
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
        row = {"bench_name": name}

        evaluation = _resolve_evaluation_contract(br)
        if evaluation:
            bench.meta["evaluation"] = evaluation
            primary = runner.run_bench_with_contract(bench, evaluation)
            if primary.get("primary_metric_result"):
                step3_path = _write_primary_step3(results_path, br, bench, runner, primary)
                if step3_path:
                    primary["primary_detail_path"] = step3_path
                    br["primary_detail_path"] = step3_path
                    br.setdefault("artifact_paths", {})["primary_samples"] = step3_path
            primary.pop("primary_metric_details", None)
            row.update(primary)
        else:
            row["primary_metric_warning"] = "missing evaluation contract; DataFlow score remains diagnostic only"

        if metrics_cfg:
            diagnostic = runner.run_bench(bench, metrics_cfg)
            if diagnostic.get("metrics"):
                row["metrics"] = diagnostic.get("metrics")
                row.setdefault("num_samples", diagnostic.get("num_samples"))
                row.setdefault("alignment", diagnostic.get("alignment"))
            elif diagnostic.get("error"):
                row["diagnostic_error"] = diagnostic.get("error")
        out["metric_results"].append(row)
    Path(results_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
        primary = mr.get("primary_metric_result") or {}
        if primary:
            print(f"    ★ primary {primary.get('metric')}: {primary.get('score')} "
                  f"(source={primary.get('score_source')}, denominator={primary.get('denominator')})")
        elif mr.get("primary_metric_warning"):
            print(f"    ! {mr.get('primary_metric_warning')}")
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

    if not args.results:
        print("错误：需要 --results（或用 --list 查看可用 metric）", file=sys.stderr)
        return 2

    results_path = args.results
    if not Path(results_path).exists():
        print(f"✗ 结果文件不存在: {results_path}", file=sys.stderr)
        return 2

    metrics_cfg = _parse_metrics_arg(args.metrics) if args.metrics else []

    summary = run_metrics(results_path, metrics_cfg)
    _print_summary(summary)

    out_path = Path(args.output) if args.output else \
        Path(results_path).parent / "metric_results.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nmetric 结果已写入: {out_path.absolute()}")

    # 全部 bench 都失败/跳过才算非 0
    any_ok = any(
        (
            not mr.get("skipped")
            and not mr.get("error")
            and (mr.get("primary_metric_result") or mr.get("metrics"))
        )
        for mr in summary["metric_results"]
    )
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
