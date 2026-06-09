#!/usr/bin/env python3
"""
run_eval.py — One-Eval Skill 核心执行器（单次评测的确定性执行内核）。

职责（不做任何 LLM 编排，编排由调用方 agent 完成）：
  1. 解析 evalspec.yaml
  2. 对每个 benchmark：
     - 若已 READY（测通过）→ 复用本地数据，默认跳过 smoke
     - 否则下载（HFDownloadTool）→ smoke 子集（默认 3 条）先验证
     - 调 DataFlowEvalTool.run_eval 跑 dataflow 评测
     - 提取 dataflow 分数；smoke 通过后标记 READY
  3. 把每个 bench 的结果落盘为 JSON，供 run_metrics.py / 报告环节使用

用法：
  # smoke 阶段（默认每 bench 抽 3 条）
  python run_eval.py evalspec.yaml --smoke

  # 全量（max_samples 由 evalspec.runtime 决定，null=全量）
  python run_eval.py evalspec.yaml

退出码：0 = 全部 bench 成功；非 0 = 有 bench 失败。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

SMOKE_SAMPLES = 3  # 每个未 ready 的 bench 正式评测前抽样验证的条数


def _count_jsonl(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _truncate_jsonl(src: str, dst: str, n: int) -> str:
    """截取前 n 条到新文件，用于 smoke 子集测试。"""
    written = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            written += 1
            if written >= n:
                break
    return dst


def _ensure_dataset(bench_dict: dict, cache_dir: Path) -> str:
    """确保 benchmark 数据在本地，返回 jsonl 路径。

    优先复用已 READY 的本地数据；否则用 HFDownloadTool 下载。
    """
    bench_name = bench_dict["bench_name"]

    ready = common.get_ready_bench(bench_name)
    if ready and Path(ready["dataset_path"]).exists():
        return ready["dataset_path"]

    dl = bench_dict.get("download_config", {}) or {}
    repo_id = (bench_dict.get("bench_source_url", "") or "").replace(
        "https://huggingface.co/datasets/", "").strip("/")
    config_name = dl.get("config")
    split = dl.get("split", "test")
    if not repo_id:
        raise ValueError(f"bench {bench_name} 缺少可下载的 bench_source_url")

    safe = f"{repo_id.replace('/', '__')}__{config_name}__{split}.jsonl"
    out_path = cache_dir / safe
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    from one_eval.toolkits.hf_download_tool import HFDownloadTool
    tool = HFDownloadTool(cache_dir=str(cache_dir))
    res = tool.download_and_convert(
        repo_id=repo_id, config_name=config_name, split=split, output_path=out_path,
    )
    if not res.get("ok"):
        raise RuntimeError(f"下载失败: {res.get('error')}")
    return str(out_path)


def _extract_score(stats: dict) -> dict:
    """从 dataflow stats 提取核心分数（统一字段）。"""
    return {
        "accuracy": stats.get("accuracy", stats.get("score")),
        "score": stats.get("score", stats.get("accuracy")),
        "total_samples": stats.get("total_samples"),
        "valid_samples": stats.get("valid_samples"),
        "metric": stats.get("metric"),
    }


def run_one_bench(bench_dict: dict, model_dict: dict, cache_dir: Path,
                  output_dir: Path, smoke: bool, max_samples) -> dict:
    """评测单个 benchmark，返回结果 dict。"""
    from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool

    bench_name = bench_dict["bench_name"]
    eval_type = bench_dict.get("bench_dataflow_eval_type")
    is_ready = common.get_ready_bench(bench_name) is not None

    # 1. 准备数据
    dataset_path = _ensure_dataset(bench_dict, cache_dir)

    # 2. smoke 子集：未 ready 的 bench 正式评测前先抽样验证；已 ready 跳过
    run_path = dataset_path
    effective_smoke = smoke and not is_ready
    if effective_smoke:
        total = _count_jsonl(dataset_path)
        n = min(SMOKE_SAMPLES, total)
        smoke_path = cache_dir / f"{bench_name.replace('/', '__')}__smoke{n}.jsonl"
        run_path = _truncate_jsonl(dataset_path, str(smoke_path), n)
    elif max_samples:
        total = _count_jsonl(dataset_path)
        if total > max_samples:
            cut = cache_dir / f"{bench_name.replace('/', '__')}__n{max_samples}.jsonl"
            run_path = _truncate_jsonl(dataset_path, str(cut), max_samples)

    # 3. 构造 BenchInfo + ModelConfig，调 dataflow 评测
    bench = common.build_bench_info(bench_dict, dataset_cache=run_path)
    model_config = common.build_model_config(model_dict)

    tool = DataFlowEvalTool(output_root=str(output_dir / "_dataflow"))
    t0 = time.time()
    df_result = tool.run_eval(bench=bench, model_config=model_config)
    elapsed = round(time.time() - t0, 2)

    stats = df_result.get("stats", {}) or {}
    score = _extract_score(stats)

    result = {
        "bench_name": bench_name,
        "bench_dataflow_eval_type": eval_type,
        "mode": "smoke" if effective_smoke else "full",
        "reused_ready": is_ready,
        "dataset_path": dataset_path,
        "run_path": run_path,
        "elapsed_sec": elapsed,
        "dataflow_score": score,
        "detail_path": df_result.get("detail_path"),
        "key_mapping": df_result.get("key_mapping", bench_dict.get("key_mapping")),
        "ok": score.get("score") is not None,
    }

    # 4. smoke 通过 → 标记 READY（下次免重测）
    if effective_smoke and result["ok"]:
        common.mark_bench_ready(
            bench_name, dataset_path, eval_type,
            df_result.get("key_mapping", bench_dict.get("key_mapping", {})),
        )
        result["marked_ready"] = True

    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="One-Eval 核心执行器")
    p.add_argument("spec", help="evalspec.yaml 路径")
    p.add_argument("--smoke", action="store_true", help="只跑 smoke 子集（每 bench 3 条）")
    p.add_argument("--output-dir", help="覆盖 evalspec.runtime.output_dir")
    args = p.parse_args(argv or sys.argv[1:])

    spec = common.load_evalspec(args.spec)
    model_dict = spec.get("model", {})
    benches = spec.get("benchmarks", []) or []
    runtime = spec.get("runtime", {}) or {}

    if not benches:
        print("错误：evalspec.benchmarks 为空", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir or runtime.get("output_dir") or common.DEFAULT_OUTPUT_DIR)
    cache_dir = Path(runtime.get("cache_dir") or common.DEFAULT_CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_samples = runtime.get("max_samples")

    all_results = []
    n_fail = 0
    for i, bench_dict in enumerate(benches, 1):
        name = bench_dict.get("bench_name", f"bench_{i}")
        print(f"[{i}/{len(benches)}] {name} ...", flush=True)
        try:
            res = run_one_bench(bench_dict, model_dict, cache_dir, output_dir,
                                smoke=args.smoke, max_samples=max_samples)
            all_results.append(res)
            s = res["dataflow_score"]
            flag = "✓" if res["ok"] else "✗"
            print(f"  {flag} {res['mode']} | score={s.get('score')} "
                  f"| valid={s.get('valid_samples')}/{s.get('total_samples')} "
                  f"| {res['elapsed_sec']}s", flush=True)
            if not res["ok"]:
                n_fail += 1
        except Exception as e:
            import traceback
            print(f"  ✗ 失败: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            all_results.append({"bench_name": name, "ok": False, "error": str(e)})
            n_fail += 1

    # 落盘结果，供 run_metrics / 报告环节读取
    out_file = output_dir / "eval_results.json"
    out_file.write_text(
        json.dumps({"results": all_results, "model": model_dict.get("model_name_or_path")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {out_file}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

