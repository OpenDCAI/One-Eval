#!/usr/bin/env python
"""
HumanEval+ evaluation bridge script for One-Eval.
Bridges One-Eval's env-var conventions to evalplus's native runner.

Flow:
  1. Read model config from One-Eval env vars
  2. Run evalplus.codegen to generate code samples via OpenAI-compatible API
  3. Run evalplus.evaluate to execute tests and compute pass@k
  4. Collect pass@1 (and pass@10) into a standardized scores JSON

Model config is read from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("humanevalplus_oneeval")

CODEGEN_TIMEOUT = 3600
EVALUATE_TIMEOUT = 1800


def parse_args():
    parser = argparse.ArgumentParser(description="HumanEval+ evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of generations per problem (for pass@k)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"],
                        help="Dataset to evaluate on (humaneval for HumanEval+, mbpp for MBPP+)")
    return parser.parse_args()


def run_codegen(args) -> Path:
    """Run evalplus.codegen to generate code samples."""
    env = os.environ.copy()

    cmd = [
        sys.executable, "-m", "evalplus.codegen",
        "--model", args.model_name,
        "--dataset", args.dataset,
        "--backend", "openai",
    ]

    api_base = os.environ.get("OPENAI_API_BASE", "")
    if api_base:
        cmd.extend(["--base-url", api_base])

    if args.temperature == 0.0:
        # evalplus requires --greedy for temperature=0 (OpenAI provider asserts
        # temperature > 0 when do_sample=True). --greedy forces n_samples=1 internally.
        if args.n_samples > 1:
            log.warning(f"temperature=0 with n_samples={args.n_samples} is contradictory "
                        f"(identical outputs). Falling back to greedy (n_samples=1).")
        cmd.append("--greedy")
    else:
        cmd.extend(["--n-samples", str(args.n_samples)])
        cmd.extend(["--temperature", str(args.temperature)])

    if args.max_samples > 0:
        log.info(f"NOTE: max_samples={args.max_samples} set, but evalplus requires full dataset "
                 f"for evaluation. Codegen will be limited to {args.max_samples} problems, "
                 f"evaluate will only score the generated subset.")
        # Determine the id-range that covers exactly max_samples problems.
        # Datasets may have gaps (e.g. MBPP+ has no id=5), so we sort all real
        # ids and pick the range [first_id, nth_id + 1) to guarantee coverage.
        start_id = 0
        end_id = args.max_samples
        try:
            if args.dataset == "mbpp":
                from evalplus.data import get_mbpp_plus
                problems = get_mbpp_plus()
            else:
                from evalplus.data import get_human_eval_plus
                problems = get_human_eval_plus()
            ids = sorted(
                int(key.split("/")[1])
                for key in problems.keys()
                if "/" in key and key.split("/")[1].isdigit()
            )
            if ids:
                start_id = ids[0]
                # Take the first max_samples ids and set end to cover them all
                take = ids[:args.max_samples]
                end_id = take[-1] + 1
                log.info(f"Dataset '{args.dataset}': using id-range [{start_id},{end_id}) "
                         f"to cover {len(take)} problems (ids may have gaps)")
        except Exception as e:
            log.warning(f"Could not determine id range, using [{start_id},{end_id}): {e}")
            end_id = start_id + args.max_samples
        cmd.extend(["--id-range", f"[{start_id},{end_id}]"])

    log.info(f"Running codegen: {' '.join(cmd)}")
    log.info(f"Timeout: {CODEGEN_TIMEOUT}s")

    try:
        result = subprocess.run(
            cmd, env=env, timeout=CODEGEN_TIMEOUT,
            stdout=sys.stdout, stderr=sys.stderr
        )
    except subprocess.TimeoutExpired:
        log.warning(f"codegen timed out after {CODEGEN_TIMEOUT}s, checking for partial results...")
        pass
    else:
        if result.returncode != 0:
            log.warning(f"codegen exited with code {result.returncode}, checking for partial results...")

    results_dir = Path("evalplus_results") / args.dataset
    if not results_dir.exists():
        log.error(f"Expected results directory not found: {results_dir}")
        sys.exit(1)

    samples_files = sorted(results_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    samples_files = [f for f in samples_files if not f.name.endswith(".raw.jsonl")]
    if not samples_files:
        log.error(f"No JSONL sample files found in {results_dir}")
        sys.exit(1)

    samples_file = samples_files[-1]
    line_count = sum(1 for _ in open(samples_file, encoding="utf-8"))
    log.info(f"Generated samples: {samples_file} ({line_count} lines)")
    return samples_file


def run_evaluate(args, samples_file: Path) -> tuple:
    """Run evalplus.evaluate on generated samples.

    When max_samples is set (partial generation), we patch evalplus's assertion
    that requires all problems to be present, since we only generated a subset.

    Returns:
        (scores_dict, eval_results_path or None)
    """
    env = os.environ.copy()

    # Remove stale eval_results to force fresh evaluation of all samples
    # evalplus names it {stem}_eval_results.json (underscore, not dot)
    for pattern in [
        samples_file.parent / (samples_file.stem + "_eval_results.json"),
        samples_file.with_suffix("").with_suffix(".eval_results.json"),
    ]:
        if pattern.exists():
            pattern.unlink()
            log.info(f"Removed stale eval_results: {pattern}")

    # If partial generation, patch the assertion in evaluate.py
    if args.max_samples > 0:
        # In Docker: the repo is mounted at /workspace/repo with evalplus/ as local package
        # Python resolves local evalplus/ before site-packages, so patch the repo copy first
        eval_path = Path("evalplus/evaluate.py")  # relative to cwd = /workspace/repo
        if not eval_path.exists():
            eval_path = Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "evalplus" / "evaluate.py"
        if eval_path.exists():
            content = eval_path.read_text(encoding="utf-8")
            patched = content.replace(
                'assert len(completion_id) == len(problems), "Missing problems in samples"',
                '# assert len(completion_id) == len(problems), "Missing problems in samples"  # patched by oneeval'
            )
            if patched != content:
                eval_path.write_text(patched, encoding="utf-8")
                log.info(f"Patched evaluate.py to allow partial evaluation ({eval_path})")

    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", args.dataset,
        "--samples", str(samples_file),
    ]

    log.info(f"Running evaluate: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, env=env, timeout=EVALUATE_TIMEOUT,
            capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        log.error(f"evaluate timed out after {EVALUATE_TIMEOUT}s")
        sys.exit(1)

    output = result.stdout + "\n" + result.stderr
    log.info(f"evaluate output:\n{output[-3000:]}")

    if result.returncode != 0:
        log.error(f"evaluate failed with code {result.returncode}")
        sys.exit(1)

    scores = parse_evalplus_output(output)
    if not scores:
        scores = parse_eval_results_file(samples_file)

    # Locate eval_results JSON (evalplus writes it next to samples file)
    eval_results_path = None
    candidates = [
        samples_file.with_suffix("").with_suffix(".eval_results.json"),
        samples_file.parent / (samples_file.stem + "_eval_results.json"),
    ]
    # Also search for any eval_results file in the directory
    candidates.extend(sorted(samples_file.parent.glob("*eval_results*.json")))
    for candidate in candidates:
        if candidate.exists():
            eval_results_path = candidate
            log.info(f"Found eval_results: {eval_results_path}")
            break

    return scores, eval_results_path


def parse_evalplus_output(output: str) -> dict:
    """Parse pass@k scores from evalplus output.

    evalplus typically outputs lines like:
      Base
      pass@1: 0.XXXX
      HumanEval+
      pass@1: 0.XXXX
    """
    scores = {}
    current_section = ""

    for line in output.splitlines():
        stripped = line.strip()

        if stripped.lower() in ("base", "humaneval", "mbpp"):
            current_section = "base"
            continue
        elif stripped.lower() in ("humaneval+", "mbpp+", "plus"):
            current_section = "plus"
            continue

        match = re.match(r"(pass@\d+):\s*([\d.]+)", stripped)
        if match:
            key = match.group(1)
            val = float(match.group(2))
            if current_section == "plus":
                scores[f"{key} (plus)"] = val
            elif current_section == "base":
                scores[f"{key} (base)"] = val
            else:
                scores[key] = val

    # Ensure we have a top-level pass@1 (prefer plus score)
    if "pass@1" not in scores:
        if "pass@1 (plus)" in scores:
            scores["pass@1"] = scores["pass@1 (plus)"]
        elif "pass@1 (base)" in scores:
            scores["pass@1"] = scores["pass@1 (base)"]

    return scores


def parse_eval_results_file(samples_file: Path) -> dict:
    """Try to find and parse evalplus's eval_results.json."""
    scores = {}
    results_dir = samples_file.parent

    eval_file = samples_file.with_suffix("").with_suffix(".eval_results.json")
    if not eval_file.exists():
        candidates = list(results_dir.rglob("*eval_results*.json")) + list(results_dir.rglob("*eval*.json"))
        if candidates:
            eval_file = candidates[0]
        else:
            return scores

    try:
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, value in data.items():
                if "pass@" in key.lower():
                    try:
                        scores[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            if not scores and "eval" in data:
                eval_data = data["eval"]
                if isinstance(eval_data, dict):
                    for key, value in eval_data.items():
                        if "pass@" in key.lower():
                            try:
                                scores[key] = float(value)
                            except (ValueError, TypeError):
                                pass
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    return scores


def write_oneeval_scores(args, scores: dict, samples_file: Path, eval_results_path: Path = None):
    """Write results in One-Eval's expected format and copy per-sample detail."""
    import shutil

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Count actual generated samples from the JSONL
    total_samples = 0
    if samples_file.exists():
        total_samples = sum(1 for _ in open(samples_file, encoding="utf-8"))

    result = {
        "bench_name": f"{args.dataset}plus",
        "model_name": args.model_name,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "total_samples": total_samples,
        "temperature": args.temperature,
        "timestamp": timestamp,
        **scores,
    }

    # Load eval_results (per-problem pass/fail) for enriching sample detail
    eval_data = {}
    if eval_results_path and eval_results_path.exists():
        try:
            eval_data = json.loads(eval_results_path.read_text(encoding="utf-8"))
            eval_dest = output_dir / f"eval_results_detail_{timestamp}.json"
            shutil.copy2(eval_results_path, eval_dest)
            result["eval_results_file"] = eval_dest.name
            log.info(f"Eval results detail copied to: {eval_dest}")
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Could not load eval_results: {e}")

    # Build enriched per-sample JSONL: merge codegen output with prompt + pass/fail
    detail_jsonl = None
    if samples_file.exists():
        dest_name = f"samples_{timestamp}.jsonl"
        dest = output_dir / dest_name
        eval_map = eval_data.get("eval", {}) if isinstance(eval_data, dict) else {}

        # Load problem prompts from evalplus dataset
        problems = {}
        try:
            if args.dataset == "humaneval":
                from evalplus.data import get_human_eval_plus
                problems = get_human_eval_plus()
            elif args.dataset == "mbpp":
                from evalplus.data import get_mbpp_plus
                problems = get_mbpp_plus()
        except Exception as e:
            log.warning(f"Could not load problem prompts: {e}")

        with open(samples_file, encoding="utf-8") as fin, \
             open(dest, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    fout.write(line + "\n")
                    continue
                task_id = record.get("task_id", "")
                # Enrich with problem prompt
                if task_id and task_id in problems:
                    prob = problems[task_id]
                    record["prompt"] = prob.get("prompt", "")
                    record["entry_point"] = prob.get("entry_point", "")
                # Enrich with pass/fail from eval_results if available
                if task_id and task_id in eval_map:
                    task_eval = eval_map[task_id]
                    if isinstance(task_eval, list) and task_eval:
                        first = task_eval[0]
                        base_pass = False
                        plus_pass = False
                        if isinstance(first, dict):
                            record["base_status"] = first.get("base_status", "")
                            record["plus_status"] = first.get("plus_status", "")
                            base_pass = first.get("base_status") == "pass"
                            plus_pass = first.get("plus_status") == "pass"
                        elif isinstance(first, list) and len(first) >= 2:
                            record["base_status"] = "pass" if first[0] == 1 else "fail"
                            record["plus_status"] = "pass" if first[1] == 1 else "fail"
                            base_pass = first[0] == 1
                            plus_pass = first[1] == 1
                        # eval_score: 1.0 if passes HumanEval+ (stricter), else 0.0
                        record["eval_score"] = 1.0 if plus_pass else 0.0
                        record["eval_valid"] = True
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        result["detail_path"] = dest_name
        detail_jsonl = dest
        log.info(f"Per-sample detail written to: {dest}")

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Scores written to: {score_file}")
    log.info(f"Results: {json.dumps(scores, indent=2)}")

    return result


def main():
    args = parse_args()
    log.info("HumanEval+ One-Eval bridge starting")
    log.info(f"  Model: {args.model_name}")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  n_samples={args.n_samples}, temperature={args.temperature}")

    samples_file = run_codegen(args)
    scores, eval_results_path = run_evaluate(args, samples_file)

    if not scores:
        log.error("No evaluation results found!")
        sys.exit(1)

    write_oneeval_scores(args, scores, samples_file, eval_results_path)
    log.info("Done!")


if __name__ == "__main__":
    main()
