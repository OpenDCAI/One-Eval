#!/usr/bin/env python
"""
OmniaBench evaluation bridge script for One-Eval.
Bridges One-Eval's env-var conventions to OmniaBench's orchestrate_eval.py.

OmniaBench evaluates general-purpose AI agents across 644 curated tasks
spanning 4 routes: DAG (multi-turn), DAG-S (single-turn), Solver, Program.

Official protocol (paper):
  - Agent temperature: 0
  - User simulator: DeepSeek-V4-Pro (thinking disabled)
  - Rubric judge: DeepSeek-V4-Pro (thinking disabled)
  - Max steps: 200 per route
  - Metric: pass@1 (combined_score == 1.0 for all rubric items)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omniabench_oneeval")

REPO_ROOT = Path(__file__).resolve().parent
EVAL_DIR = REPO_ROOT / "evaluation"
CONFIGS_DIR = EVAL_DIR / "configs"
SCRIPTS_DIR = EVAL_DIR / "scripts"
DATA_DIR = EVAL_DIR / "data" / "routes"

ALL_ROUTES = ["route1", "route2", "route3", "route4"]
PROFILE_NAME = "oneeval_runtime"

ROUTE_TASK_COUNTS = {
    "route1": 354,
    "route2": 60,
    "route3": 30,
    "route4": 200,
}
TOTAL_TASKS = sum(ROUTE_TASK_COUNTS.values())  # 644


def parse_args():
    parser = argparse.ArgumentParser(description="OmniaBench evaluation for One-Eval")

    # --- One-Eval standard arguments ---
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for evaluation results")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Agent model name (e.g. gpt-4o, claude-sonnet-4-6)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max total tasks to evaluate (-1 = all 644)")

    # --- OmniaBench-specific arguments ---
    parser.add_argument("--routes", type=str, default="route1,route2,route3,route4",
                        help="Comma-separated routes to evaluate (route1/route2/route3/route4)")
    parser.add_argument("--pass_k", type=int, default=1,
                        help="pass@k: number of attempts per task")
    parser.add_argument("--max_task_workers", type=int, default=8,
                        help="Max concurrent task workers")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Agent inference temperature (paper uses 0.0)")
    parser.add_argument("--judge_model", type=str, default="",
                        help="Rubric judge model (defaults to --model_name; paper: deepseek-v4-pro)")
    parser.add_argument("--user_model", type=str, default="",
                        help="User simulator model (defaults to --model_name; paper: deepseek-v4-pro)")
    parser.add_argument("--infer_mode", type=str, default="fc",
                        choices=["fc", "prompt"],
                        help="Inference mode: fc (function-calling) or prompt")
    parser.add_argument("--reasoning_effort", type=str, default="high",
                        help="Agent reasoning effort (high/medium/low)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sampling: read real global_ids from route data files
# ---------------------------------------------------------------------------

def load_route_global_ids(route_id: str) -> list[int]:
    """Load actual global_ids from a route's data file."""
    data_file = DATA_DIR / f"{route_id}.json"
    if not data_file.exists():
        log.warning(f"Route data file not found: {data_file}")
        return []

    data = json.loads(data_file.read_text(encoding="utf-8"))
    ids = []
    if isinstance(data, list):
        for item in data:
            gid = item.get("global_id")
            if gid is not None:
                ids.append(int(gid))
    elif isinstance(data, dict):
        for item in data.values():
            if isinstance(item, dict):
                gid = item.get("global_id")
                if gid is not None:
                    ids.append(int(gid))
    return sorted(ids)


HF_DATASET_REPO_ID = "scuuy666/OmniaBench"


def ensure_route_data(routes: list[str]):
    """Download route data from HuggingFace if any route file is missing."""
    missing = [r for r in routes if not (DATA_DIR / f"{r}.json").exists()]
    if not missing:
        return

    log.info(f"Route data missing for {missing}, downloading from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning("huggingface_hub not installed, cannot auto-download route data")
        return

    data_dir = EVAL_DIR / "data"
    try:
        snapshot_download(repo_id=HF_DATASET_REPO_ID, repo_type="dataset", local_dir=str(data_dir))
        log.info("Route data download complete.")
    except Exception as e:
        log.warning(f"Failed to download route data: {e}")


def sample_global_ids(routes: list[str], max_samples: int) -> list[int]:
    """Select the first max_samples global_ids sequentially across routes in order."""
    route_ids = {}
    total_available = 0
    for route in routes:
        ids = load_route_global_ids(route)
        route_ids[route] = ids
        total_available += len(ids) if ids else ROUTE_TASK_COUNTS.get(route, 0)

    if max_samples <= 0 or max_samples >= total_available:
        return []  # no sampling needed

    sampled = []
    remaining = max_samples

    for route in routes:
        if remaining <= 0:
            break
        ids = route_ids[route]
        if not ids:
            log.warning(f"No IDs loaded for {route}, cannot sample")
            continue
        take = ids[:remaining]
        sampled.extend(take)
        remaining -= len(take)

    return sampled


# ---------------------------------------------------------------------------
# Profile generation and command building
# ---------------------------------------------------------------------------

def write_profiles_json(args):
    """Generate profiles.json for OmniaBench from One-Eval arguments."""
    judge_model = args.judge_model or args.model_name
    user_model = args.user_model or args.model_name

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")

    profile = {
        "agent_model": args.model_name,
        "agent_provider": "openai",
        "agent_api_key": "OMNIABENCH_AGENT_API_KEY",
        "agent_base_url": "OMNIABENCH_AGENT_BASE_URL",
        "agent_reasoning_effort": args.reasoning_effort,
        "user_model": user_model,
        "user_provider": "openai",
        "user_api_key": "OMNIABENCH_USER_API_KEY",
        "user_base_url": "OMNIABENCH_USER_BASE_URL",
        "rubric_judge_model": judge_model,
        "rubric_judge_provider": "openai",
        "rubric_judge_api_key": "OMNIABENCH_JUDGE_API_KEY",
        "rubric_judge_base_url": "OMNIABENCH_JUDGE_BASE_URL",
        "infer_mode": args.infer_mode,
    }

    profiles = {
        "default_profile": PROFILE_NAME,
        "profiles": {PROFILE_NAME: profile},
    }

    profiles_path = CONFIGS_DIR / "profiles.json"
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Written profiles.json to {profiles_path}")

    # Set env vars that OmniaBench reads via env-var-name indirection
    os.environ["OMNIABENCH_AGENT_API_KEY"] = api_key
    os.environ["OMNIABENCH_AGENT_BASE_URL"] = api_base
    os.environ["OMNIABENCH_USER_API_KEY"] = api_key
    os.environ["OMNIABENCH_USER_BASE_URL"] = api_base
    os.environ["OMNIABENCH_JUDGE_API_KEY"] = api_key
    os.environ["OMNIABENCH_JUDGE_BASE_URL"] = api_base


def build_orchestrate_cmd(args, sampled_ids: list[int]) -> list[str]:
    """Build the orchestrate_eval.py command line."""
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "orchestrate_eval.py"),
        "--profile", PROFILE_NAME,
        "--routes", *routes,
        "--pass-k", str(args.pass_k),
        "--max-task-workers", str(args.max_task_workers),
    ]

    if sampled_ids:
        cmd.extend(["--global-id", *[str(gid) for gid in sampled_ids]])

    # Pass temperature to run_eval.py as extra arg (orchestrator forwards unknown args)
    cmd.extend(["--temperature", str(args.temperature)])

    return cmd


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def find_result_summary(results_dir: Path, profile: str) -> Path | None:
    """Locate the route_summary JSON produced by orchestrate_eval."""
    candidates = sorted(
        results_dir.glob(f"route_summary-{profile}-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    candidates = sorted(
        results_dir.glob("route_summary-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_combined_jsonl(results_dir: Path, profile: str) -> Path | None:
    """Locate the combined .runs.jsonl for per-sample detail."""
    candidates = sorted(
        results_dir.glob(f"**/combined-{profile}-*.runs.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    candidates = sorted(
        results_dir.glob("**/*.runs.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_summary(summary_path: Path) -> dict:
    """Parse official OmniaBench route_summary JSON.

    Official structure:
    {
      "profile": "...",
      "run_tag": "...",
      "routes": [
        {"route_id": "route1", "status": "ok", "task_count": 354,
         "scored_count": 350, "pass_at_1": 0.18, "score": 0.45, ...},
        ...
      ],
      "overall": {
        "task_count": 644,
        "scored_count": 630,
        "pass_at_1": 0.1724,
        "score": 0.42
      }
    }
    """
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    result = {
        "pass_at_1": 0.0,
        "avg_rubric_score": 0.0,
        "total_tasks": 0,
        "scored_tasks": 0,
        "by_route": {},
        "is_partial": False,
    }

    if not isinstance(data, dict):
        log.warning(f"Unexpected summary format: {type(data)}")
        return result

    # Parse overall
    overall = data.get("overall", {})
    if isinstance(overall, dict):
        result["pass_at_1"] = overall.get("pass_at_1", 0.0)
        result["avg_rubric_score"] = overall.get("score", 0.0)
        result["total_tasks"] = overall.get("task_count", 0)
        result["scored_tasks"] = overall.get("scored_count", 0)

    # Parse per-route
    routes_list = data.get("routes", [])
    if isinstance(routes_list, list):
        for row in routes_list:
            if not isinstance(row, dict):
                continue
            route_id = row.get("route_id", row.get("route", "unknown"))
            if row.get("status") != "ok":
                continue
            result["by_route"][route_id] = {
                "pass_at_1": row.get("pass_at_1", 0.0),
                "score": row.get("score", 0.0),
                "task_count": row.get("task_count", 0),
                "scored_count": row.get("scored_count", 0),
            }

    # Check if partial
    expected = TOTAL_TASKS
    actual = result["total_tasks"]
    if 0 < actual < expected:
        result["is_partial"] = True

    return result


def _normalize_sample_record(raw: dict) -> dict:
    """Transform a raw OmniaBench runs.jsonl record into a clean flat format for reporting."""
    result = raw.get("result", {}) if isinstance(raw.get("result"), dict) else {}
    task_info = result.get("task_info", {}) if isinstance(result.get("task_info"), dict) else {}
    rubric_eval = result.get("rubric_eval", {}) if isinstance(result.get("rubric_eval"), dict) else {}
    verifier_eval = result.get("verifier_eval", {}) if isinstance(result.get("verifier_eval"), dict) else {}

    total_reward = result.get("total_reward")
    eval_valid = total_reward is not None

    record = {
        "task_id": task_info.get("task_id", ""),
        "task_index": raw.get("task_index") or result.get("task_index"),
        "env_id": task_info.get("env_id", ""),
        "prompt": task_info.get("task", ""),
        "eval_score": total_reward if total_reward is not None else 0.0,
        "eval_valid": eval_valid,
        "rubric_reward": result.get("rubric_reward"),
        "rubric_earned": rubric_eval.get("earned_score"),
        "rubric_total": rubric_eval.get("total_score"),
        "rubric_summary": rubric_eval.get("summary", ""),
        "result_status": result.get("result_status", ""),
        "steps": result.get("steps"),
        "termination_reason": result.get("termination_reason", ""),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }

    if verifier_eval:
        record["verifier_score"] = verifier_eval.get("score")
        record["verifier_success"] = verifier_eval.get("verify_success")

    return record


def write_oneeval_results(args, parsed: dict, detail_jsonl: Path | None):
    """Write standardized scores_*.json and samples_*.jsonl for One-Eval."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Transform and write normalized detail JSONL for reporting
    detail_name = ""
    if detail_jsonl and detail_jsonl.exists():
        detail_name = f"samples_{timestamp}.jsonl"
        out_path = output_dir / detail_name
        count = 0
        with open(detail_jsonl, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    normalized = _normalize_sample_record(raw)
                    fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1
                except (json.JSONDecodeError, Exception) as e:
                    log.warning(f"Skipping malformed sample record: {e}")
        log.info(f"Wrote {count} normalized samples to {out_path}")

    scores = {
        "bench_name": "omniabench",
        "model_name": args.model_name,
        "timestamp": timestamp,
        "total_samples": parsed["total_tasks"],
        "scored_samples": parsed["scored_tasks"],
        "pass_at_1": parsed["pass_at_1"],
        "avg_rubric_score": parsed["avg_rubric_score"],
        "is_partial": parsed["is_partial"],
        "by_route": parsed["by_route"],
        "detail_path": detail_name,
        "config": {
            "routes": args.routes,
            "pass_k": args.pass_k,
            "max_task_workers": args.max_task_workers,
            "temperature": args.temperature,
            "infer_mode": args.infer_mode,
            "reasoning_effort": args.reasoning_effort,
            "judge_model": args.judge_model or args.model_name,
            "user_model": args.user_model or args.model_name,
            "lang_filter": "all",
        },
    }

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Scores written to {score_file}")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("OmniaBench evaluation via One-Eval bridge")
    log.info(f"  Agent model:       {args.model_name}")
    log.info(f"  Judge model:       {args.judge_model}")
    log.info(f"  User model:        {args.user_model}")
    log.info(f"  Routes:            {args.routes}")
    log.info(f"  pass@k:            {args.pass_k}")
    log.info(f"  Workers:           {args.max_task_workers}")
    log.info(f"  Max samples:       {args.max_samples}")
    log.info(f"  Temperature:       {args.temperature}")
    log.info(f"  Reasoning effort:  {args.reasoning_effort}")
    log.info(f"  Infer mode:        {args.infer_mode}")
    log.info(f"  Output dir:        {args.output_dir}")
    log.info("=" * 60)

    # Step 1: Generate profiles.json with model config
    write_profiles_json(args)

    # Step 2: Ensure route data exists before sampling
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    if args.max_samples > 0:
        ensure_route_data(routes)

    # Step 3: Compute sampling if max_samples set
    sampled_ids = []
    if args.max_samples > 0:
        log.info(f"Sampling first {args.max_samples} tasks sequentially across {len(routes)} routes...")
        sampled_ids = sample_global_ids(routes, args.max_samples)
        if sampled_ids:
            log.info(f"  Sampled {len(sampled_ids)} global_ids")
        else:
            log.info("  No sampling needed (max_samples >= total available)")

    # Step 3: Build and run orchestrate_eval.py
    cmd = build_orchestrate_cmd(args, sampled_ids)
    log.info(f"Running: {' '.join(cmd[:10])}... ({len(cmd)} args total)")

    env = os.environ.copy()
    result = subprocess.run(
        cmd,
        cwd=str(EVAL_DIR),
        env=env,
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        log.error(f"orchestrate_eval.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Step 4: Locate and parse results
    results_dir = EVAL_DIR / "results"
    summary_path = find_result_summary(results_dir, PROFILE_NAME)

    if not summary_path:
        log.error(f"No route_summary JSON found in {results_dir}")
        sys.exit(1)

    log.info(f"Found summary: {summary_path}")
    parsed = parse_summary(summary_path)

    # Step 5: Locate per-sample detail
    detail_jsonl = find_combined_jsonl(results_dir, PROFILE_NAME)

    # Step 6: Write One-Eval standardized output
    scores = write_oneeval_results(args, parsed, detail_jsonl)

    log.info("=" * 60)
    log.info("OmniaBench evaluation complete!")
    log.info(f"  pass@1 (leaderboard metric): {scores['pass_at_1']:.4f}")
    log.info(f"  avg rubric score:            {scores['avg_rubric_score']:.4f}")
    log.info(f"  Total tasks:                 {scores['total_samples']}")
    log.info(f"  Scored tasks:                {scores['scored_samples']}")
    if scores["is_partial"]:
        log.warning(f"  ⚠️  PARTIAL RUN ({scores['total_samples']}/{TOTAL_TASKS} tasks)")
    for route, info in scores.get("by_route", {}).items():
        log.info(f"  {route}: pass@1={info['pass_at_1']:.4f}, score={info['score']:.4f} (n={info['task_count']})")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())