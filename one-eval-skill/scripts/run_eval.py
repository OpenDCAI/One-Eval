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
RESULTS_FILENAME = "eval_results.json"


def _write_results(out_file: Path, all_results: list, model_dict: dict,
                   runtime: dict, run_id: str, smoke: bool, partial: bool) -> None:
    """把当前结果落盘（每完成一个 bench 调一次，实现增量保存 / 断点续跑）。

    顶层除 results 外带上：被测模型快照(脱敏)、run_id、生成时间、采样配置，
    使报告自包含、可复现；partial 标记本次是否还在进行中。
    """
    import datetime as _dt
    payload = {
        "run_id": run_id,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "partial": partial,                     # True=评测进行中/中断；False=本批全部跑完
        "model": model_dict.get("model_name_or_path"),   # 向后兼容：保留纯字符串字段
        "model_config": common.sanitize_model_config(model_dict),  # 脱敏完整快照
        "runtime": {"smoke": smoke, "max_samples": runtime.get("max_samples")},
        "results": all_results,
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def _load_done_benches(out_file: Path) -> dict:
    """续跑：从已有 eval_results.json 读出「已成功」的 bench 结果，按名字索引。

    只复用 ok=True 的；失败/未跑的 bench 会重跑。external_repo_pending 视为已完成。
    """
    if not out_file.exists():
        return {}
    try:
        data = json.loads(out_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    done = {}
    for r in data.get("results", []) or []:
        name = r.get("bench_name")
        if name and (r.get("ok") or r.get("mode") == "external_repo_pending"):
            done[name] = r
    return done


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
    bench_dict = common.normalize_benchmark_entry(bench_dict)
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
    """从 dataflow stats 提取诊断分数（统一字段）。"""
    return {
        "accuracy": stats.get("accuracy", stats.get("score")),
        "score": stats.get("score", stats.get("accuracy")),
        "total_samples": stats.get("total_samples"),
        "valid_samples": stats.get("valid_samples"),
        "metric": stats.get("metric"),
        "role": stats.get("role", "diagnostic"),
        "display_as_primary": bool(stats.get("display_as_primary", False)),
        "note": stats.get("note", "DataFlow score is diagnostic only; primary score is computed by metric stage."),
    }


def _external_repo_result(bench_dict: dict) -> dict:
    """external_repo bench 的优雅短路。

    不下载、不调内核、不报错：把 meta.repo_eval 原样带出，交给调用方 agent 按
    references/external_bench.md 在外部执行评测、再回填分数。当前版本不内置执行器。
    """
    bench_dict = common.normalize_benchmark_entry(bench_dict)
    meta = bench_dict.get("meta") or {}
    return {
        "bench_name": bench_dict.get("bench_name"),
        "bench_kind": common.BENCH_KIND_EXTERNAL,
        "bench_dataflow_eval_type": bench_dict.get("bench_dataflow_eval_type"),
        "mode": "external_repo_pending",
        "dataflow_score": {"score": None, "total_samples": None, "valid_samples": None,
                           "metric": None, "role": "diagnostic",
                           "display_as_primary": False},
        "repo_eval": meta.get("repo_eval", {}),
        "evaluation": meta.get("evaluation"),
        "prompt": meta.get("prompt"),
        "readiness": meta.get("readiness"),
        "note": "external_repo bench：需在外部仓库/沙箱执行评测后回填分数，详见 references/external_bench.md",
        "ok": True,  # 不算失败：只是待外部执行，不应让整批退出码非 0
    }


def run_one_bench(bench_dict: dict, model_dict: dict, cache_dir: Path,
                  output_dir: Path, smoke: bool, max_samples) -> dict:
    """评测单个 benchmark，返回结果 dict。"""
    from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool

    bench_dict = common.normalize_benchmark_entry(bench_dict)

    # external_repo：不走确定性内核，优雅短路（不下载/不调内核/不报错）。
    if common.get_bench_kind(bench_dict) == common.BENCH_KIND_EXTERNAL:
        return _external_repo_result(bench_dict)

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
        "evaluation": bench.meta.get("evaluation"),
        "prompt": df_result.get("prompt", bench.meta.get("prompt")),
        "readiness": bench.meta.get("readiness"),
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
    p.add_argument("--output-dir", help="覆盖 evalspec.runtime.output_dir（产物根目录）")
    p.add_argument("--resume", metavar="RUN_DIR",
                   help="从已有 run 目录续跑：复用其中已成功的 bench，只补跑失败/未跑的")
    args = p.parse_args(argv or sys.argv[1:])

    spec = common.load_evalspec(args.spec)
    model_dict = spec.get("model", {})
    benches = spec.get("benchmarks", []) or []
    runtime = spec.get("runtime", {}) or {}

    if not benches:
        print("错误：evalspec.benchmarks 为空", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir or runtime.get("output_dir") or common.DEFAULT_OUTPUT_DIR)
    cache_dir = Path(runtime.get("cache_dir") or common.DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_samples = runtime.get("max_samples")

    # --- run 目录：续跑复用旧目录，否则新建带时间戳的隔离目录（避免互相覆盖）---
    if args.resume:
        run_dir = Path(args.resume).resolve()
        if not run_dir.exists():
            print(f"错误：--resume 目录不存在: {run_dir}", file=sys.stderr)
            return 2
        run_id = run_dir.name
        # 同步 latest 指针
        try:
            (output_root / common.LATEST_RUN_FILE).write_text(str(run_dir), encoding="utf-8")
        except Exception:
            pass
    else:
        run_id = common.new_run_id()
        run_dir = common.make_run_dir(output_root, run_id)

    out_file = run_dir / RESULTS_FILENAME
    done = _load_done_benches(out_file) if args.resume else {}

    # --- 首行显式打印被测模型，杜绝「测了什么模型」的歧义（问题1）---
    mtype = "API" if model_dict.get("is_api") else "本地 vLLM"
    model_name = model_dict.get("model_name_or_path", "（未指定！）")
    print(f"被测模型: {model_name}  [{mtype}]", flush=True)
    print(f"run_id: {run_id}  |  产物目录: {run_dir}", flush=True)
    if args.resume and done:
        print(f"续跑：已复用 {len(done)} 个成功 bench：{', '.join(done)}", flush=True)
    print("", flush=True)

    all_results = []
    n_fail = 0
    for i, bench_dict in enumerate(benches, 1):
        bench_dict = common.enrich_external_bench(bench_dict)  # external_repo 按名从 gallery 回填 repo_eval
        name = bench_dict.get("bench_name", f"bench_{i}")
        if name in done:
            all_results.append(done[name])
            print(f"[{i}/{len(benches)}] {name} … ⏭ 跳过（已成功，续跑复用）", flush=True)
            continue
        print(f"[{i}/{len(benches)}] {name} ...", flush=True)
        try:
            res = run_one_bench(bench_dict, model_dict, cache_dir, run_dir,
                                smoke=args.smoke, max_samples=max_samples)
            all_results.append(res)
            if res.get("mode") == "external_repo_pending":
                print(f"  ⊙ external_repo | 待外部执行回填（见 references/external_bench.md）",
                      flush=True)
            else:
                s = res["dataflow_score"]
                flag = "✓" if res["ok"] else "✗"
                print(f"  {flag} {res['mode']} | score={s.get('score')} "
                      f"| valid={s.get('valid_samples')}/{s.get('total_samples')} "
                      f"| {res.get('elapsed_sec')}s", flush=True)
            if not res["ok"]:
                n_fail += 1
        except Exception as e:
            import traceback
            print(f"  ✗ 失败: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            all_results.append({"bench_name": name, "ok": False, "error": str(e)})
            n_fail += 1
        # 每完成一个 bench 立即增量落盘 → 中途崩溃也不丢已跑结果（问题5）
        _write_results(out_file, all_results, model_dict, runtime, run_id,
                       args.smoke, partial=True)

    # 全部跑完，标记 partial=False 终态落盘
    _write_results(out_file, all_results, model_dict, runtime, run_id,
                   args.smoke, partial=False)
    print(f"\n结果已写入: {out_file.resolve()}")
    print(f"（最新 run 指针: {(output_root / common.LATEST_RUN_FILE).resolve()}）")
    if n_fail:
        print(f"有 {n_fail} 个 bench 未成功；修好后可续跑："
              f"python scripts/run_eval.py {args.spec} --resume {run_dir}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
