#!/usr/bin/env python
"""
LiveCodeBench evaluation bridge script for One-Eval.
Bridges One-Eval's env-var conventions to LiveCodeBench's native runner.

Flow:
  1. Read model config from One-Eval env vars
  2. Register a custom model entry in LiveCodeBench's LanguageModelStore
  3. Invoke lcb_runner's generation + evaluation pipeline
  4. Collect pass@1 (and pass@5) into a standardized scores JSON

Model config is read from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES

LiveCodeBench internally uses OPENAI_KEY (not OPENAI_API_KEY), this script
handles the mapping transparently.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("livecodebench_oneeval")


def parse_args():
    parser = argparse.ArgumentParser(description="LiveCodeBench evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default=os.environ.get("ONEEVAL_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--api_base", type=str,
                        default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--max_samples", type=int,
                        default=int(os.environ.get("ONEEVAL_MAX_SAMPLES", "-1")))
    parser.add_argument("--release_version", type=str,
                        default=os.environ.get("ONEEVAL_LCB_RELEASE", "release_latest"))
    parser.add_argument("--n", type=int, default=10,
                        help="Number of generations per problem (for pass@k)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_workers", type=int, default=-1,
                        help="Number of parallel workers (-1 = auto-detect CPU count)")
    parser.add_argument("--num_process_evaluate", type=int, default=1,
                        help="Number of evaluation processes (default: 1 for memory safety)")
    parser.add_argument("--timeout", type=int, default=6,
                        help="Code execution timeout in seconds")
    parser.add_argument("--scenario", type=str, default="codegeneration",
                        choices=["codegeneration", "selfrepair",
                                 "testoutputprediction", "codeexecution"])
    return parser.parse_args()


def setup_env(args):
    """Map One-Eval env conventions to LiveCodeBench's expected env vars."""
    if args.api_key:
        os.environ["OPENAI_KEY"] = args.api_key

    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base


def register_custom_model(model_name: str):
    """
    Register a custom model in LiveCodeBench's LanguageModelStore
    so we can evaluate any OpenAI-compatible endpoint.
    """
    from lcb_runner.lm_styles import (
        LanguageModel,
        LanguageModelStore,
        LMStyle,
    )

    if model_name not in LanguageModelStore:
        custom_model = LanguageModel(
            model_name=model_name,
            model_repr=model_name,
            model_style=LMStyle.OpenAIChat,
            release_date=datetime(2025, 1, 1),
            link=None,
        )
        LanguageModelStore[model_name] = custom_model
        log.info(f"Registered custom model: {model_name} (OpenAIChat style)")
    else:
        log.info(f"Model already registered: {model_name}")


def run_lcb_pipeline(args):
    """
    Run LiveCodeBench generation + evaluation using internal APIs.
    This gives us control over max_samples truncation.
    """
    # Remove stale output to prevent caching issues when max_samples changes
    import shutil
    stale_output = Path("output") / args.model_name
    if stale_output.exists():
        shutil.rmtree(stale_output)
        log.info(f"Removed stale output directory: {stale_output}")

    lcb_argv = [
        "--model", args.model_name,
        "--scenario", args.scenario,
        "--n", str(args.n),
        "--temperature", str(args.temperature),
        "--evaluate",
        "--multiprocess", str(args.max_workers),
        "--num_process_evaluate", str(args.num_process_evaluate),
    ]

    if args.release_version:
        lcb_argv.extend(["--release_version", args.release_version])

    if args.timeout:
        lcb_argv.extend(["--timeout", str(args.timeout)])

    log.info(f"LCB args: {lcb_argv}")

    old_argv = sys.argv
    sys.argv = ["lcb_runner"] + lcb_argv
    try:
        from lcb_runner.runner.parser import get_args as lcb_get_args
        from lcb_runner.lm_styles import LanguageModelStore as LMStore
        from lcb_runner.runner.runner_utils import build_runner
        from lcb_runner.utils.path_utils import get_output_path
        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.runner.scenario_router import (
            build_prompt_benchmark,
            combine_results,
            sort_and_extract_save_results,
            get_metrics,
        )

        lcb_args = lcb_get_args()
        model = LMStore[lcb_args.model]
        benchmark, format_prompt = build_prompt_benchmark(lcb_args)

        if args.max_samples > 0 and args.max_samples < len(benchmark):
            log.info(f"Truncating benchmark from {len(benchmark)} to {args.max_samples} samples")
            benchmark = benchmark[:args.max_samples]

        log.info(f"Running generation on {len(benchmark)} problems, n={args.n}")

        output_path = get_output_path(model.model_repr, lcb_args)
        runner = build_runner(lcb_args, model)
        results = runner.run_main(benchmark, format_prompt)

        combined_results = combine_results(
            lcb_args.scenario, results, model, lcb_args.cot_code_execution
        )

        save_results = [
            instance.insert_output(outputs_list, extracted_list)
            for instance, (outputs_list, extracted_list) in zip(
                benchmark, combined_results
            )
        ]
        save_results, combined_results = sort_and_extract_save_results(
            lcb_args.scenario, save_results
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=4)
        log.info(f"Generation saved to: {output_path}")

        log.info("Running evaluation (pass@k)...")
        metrics = get_metrics(lcb_args.scenario, lcb_args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])

        eval_file = output_path.replace(".json", "_eval.json")
        eval_all_file = output_path.replace(".json", "_eval_all.json")

        from lcb_runner.utils.scenarios import Scenario
        if lcb_args.scenario == Scenario.codegeneration:
            metadatas = metrics[2] if metrics else [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)
        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)

        log.info(f"Evaluation saved to: {eval_file}")
        return metrics[0] if metrics else {}

    finally:
        sys.argv = old_argv


def collect_results(args) -> dict:
    """
    Collect evaluation results from LiveCodeBench's output directory.
    LiveCodeBench saves results under ./output/{model_name}/
    """
    output_base = Path("output") / args.model_name
    scores = {}

    if not output_base.exists():
        log.warning(f"Output directory not found: {output_base}")
        log.info("Searching for output files in alternative locations...")
        for candidate in Path("output").rglob("*.json"):
            if "eval" in candidate.name.lower() or "metric" in candidate.name.lower():
                log.info(f"Found candidate: {candidate}")
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and ("pass@1" in data or "pass@1" in str(data)):
                        scores = data
                        break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    if not scores:
        for json_file in sorted(output_base.rglob("*.json")):
            if "eval" in json_file.name.lower() or "metric" in json_file.name.lower():
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        scores.update(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    return scores


def extract_pass_at_k(raw_scores: dict) -> dict:
    """Extract pass@1 and pass@5 from raw scores dict."""
    result = {}

    for key, value in raw_scores.items():
        key_lower = key.lower().replace(" ", "").replace("_", "")
        if "pass@1" in key_lower or key == "pass@1":
            result["pass@1"] = float(value)
        elif "pass@5" in key_lower or key == "pass@5":
            result["pass@5"] = float(value)

    if not result and raw_scores:
        for key, value in raw_scores.items():
            try:
                result[key] = float(value)
            except (ValueError, TypeError):
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        try:
                            result[f"{key}.{sub_key}"] = float(sub_val)
                        except (ValueError, TypeError):
                            continue

    return result


def write_oneeval_scores(args, scores: dict):
    """Write results in One-Eval's expected format and per-sample detail JSONL."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Locate eval_all.json for per-sample detail
    eval_all_path = None
    output_base = Path("output") / args.model_name
    if output_base.exists():
        candidates = sorted(output_base.rglob("*_eval_all.json"))
        if candidates:
            eval_all_path = candidates[-1]

    # Build enriched per-sample JSONL from eval_all data
    total_samples = 0
    detail_name = None
    if eval_all_path and eval_all_path.exists():
        try:
            eval_all = json.loads(eval_all_path.read_text(encoding="utf-8"))
            if isinstance(eval_all, list) and eval_all:
                total_samples = len(eval_all)
                detail_name = f"samples_{timestamp}.jsonl"
                detail_dest = output_dir / detail_name
                with open(detail_dest, "w", encoding="utf-8") as fout:
                    for item in eval_all:
                        # Pick the best solution (first passing one, or first)
                        code_list = item.get("code_list") or []
                        graded_list = item.get("graded_list") or []
                        solution = ""
                        if code_list:
                            for code, passed in zip(code_list, graded_list):
                                if passed:
                                    solution = code
                                    break
                            if not solution:
                                solution = code_list[0]

                        pass_at_1 = item.get("pass@1", 0.0)
                        record = {
                            "task_id": item.get("question_id", ""),
                            "prompt": item.get("question_content", ""),
                            "solution": solution,
                            "question_title": item.get("question_title", ""),
                            "difficulty": item.get("difficulty", ""),
                            "platform": item.get("platform", ""),
                            "eval_score": float(pass_at_1),
                            "eval_valid": True,
                            "pass@1": float(pass_at_1),
                            "n_passed": sum(1 for g in graded_list if g),
                            "n_total": len(graded_list),
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.info(f"Per-sample detail written to: {detail_dest} ({total_samples} problems)")
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Could not load eval_all for detail: {e}")

    result = {
        "bench_name": "livecodebench",
        "model_name": args.model_name,
        "scenario": args.scenario,
        "release_version": args.release_version,
        "n": args.n,
        "total_samples": total_samples,
        "temperature": args.temperature,
        "timestamp": timestamp,
        **scores,
    }
    if detail_name:
        result["detail_path"] = detail_name

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Scores written to: {score_file}")
    log.info(f"Results: {json.dumps(scores, indent=2)}")

    return result


def main():
    args = parse_args()
    log.info(f"LiveCodeBench One-Eval bridge starting")
    log.info(f"  Model: {args.model_name}")
    log.info(f"  Scenario: {args.scenario}")
    log.info(f"  Release: {args.release_version}")
    log.info(f"  max_samples={args.max_samples}, n={args.n}, temp={args.temperature}")

    setup_env(args)
    register_custom_model(args.model_name)

    log.info("=" * 60)
    log.info("Running generation + evaluation")
    log.info("=" * 60)
    raw_scores = run_lcb_pipeline(args)

    if not raw_scores:
        raw_scores = collect_results(args)

    if not raw_scores:
        log.error("No evaluation results found!")
        sys.exit(1)

    scores = extract_pass_at_k(raw_scores)
    if not scores:
        log.warning("Could not extract pass@k metrics, using raw scores")
        scores = raw_scores

    write_oneeval_scores(args, scores)
    log.info("Done!")


if __name__ == "__main__":
    main()
