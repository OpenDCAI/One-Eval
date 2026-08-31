#!/usr/bin/env python
"""
BFCL (Berkeley Function Calling Leaderboard) evaluation bridge for One-Eval.

Flow:
  1. Read model config from One-Eval env vars
  2. Import BFCL's internal Python API
  3. Load test entries, truncate to max_samples
  4. Run generation via BFCL's handler system
  5. Run evaluation via BFCL's eval_runner
  6. Collect per-category scores into a standardized scores JSON

Model config from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES

Test category selection via:
  --test_category or BFCL_TEST_CATEGORIES env var (comma-separated or preset name)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bfcl_oneeval")

CATEGORY_PRESETS = {
    "ast": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
    ],
    "live": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
    "non_live": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
    ],
    "single_turn": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "default": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
}
CATEGORY_PRESETS["all"] = (
    CATEGORY_PRESETS["ast"] + CATEGORY_PRESETS["live"] + CATEGORY_PRESETS["multi_turn"]
)


def parse_args():
    parser = argparse.ArgumentParser(description="BFCL evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--test_category",
        type=str,
        default="all",
        help="Comma-separated categories or a preset (ast/live/multi_turn/all/default)",
    )
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.001)
    return parser.parse_args()


def resolve_categories(category_str: str) -> list:
    """Resolve category string to a list of category names.

    Supports:
    - Our preset names (ast/live/multi_turn/all/default/non_live/single_turn)
    - BFCL native collection names (same as above, passed through as-is)
    - Comma-separated individual category names
    """
    category_str = category_str.strip()
    if category_str in CATEGORY_PRESETS:
        return CATEGORY_PRESETS[category_str]
    # Could be a comma-separated list of individual categories or BFCL collections
    return [c.strip() for c in category_str.split(",") if c.strip()]


def setup_env():
    """Ensure API credentials from environment are propagated to BFCL's expected vars."""
    api_base = os.environ.get("OPENAI_API_BASE", "")
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base


def register_model(model_name: str, temperature: float):
    """
    Patch or register a model in BFCL's MODEL_CONFIG_MAPPING to use
    OpenAICompletionsHandler (chat/completions API) instead of the default
    Responses API, while preserving official config values like is_fc_model.

    For known models (e.g. gpt-4o-2024-11-20), only swap the handler via
    dataclasses.replace() so is_fc_model, underscore_to_dot etc are preserved.
    For unknown models, create new ModelConfig and infer is_fc_model from name.
    """
    from dataclasses import replace as dc_replace
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
    except ImportError as e:
        log.warning(f"Could not import BFCL model config: {e}")
        return

    if model_name in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[model_name] = dc_replace(
            MODEL_CONFIG_MAPPING[model_name],
            model_handler=OpenAICompletionsHandler,
        )
        is_fc = MODEL_CONFIG_MAPPING[model_name].is_fc_model
        log.info(
            f"Patched '{model_name}' to use OpenAICompletionsHandler "
            f"(is_fc_model={is_fc})"
        )
    else:
        is_fc = "FC" in model_name
        MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
            model_name=model_name,
            display_name=model_name,
            url="",
            org="Custom",
            license="",
            model_handler=OpenAICompletionsHandler,
            input_price=None,
            output_price=None,
            is_fc_model=is_fc,
        )
        log.info(
            f"Registered custom model '{model_name}' with OpenAICompletionsHandler "
            f"(is_fc_model={is_fc})"
        )


def run_generation(args, categories: list):
    """
    Run BFCL generation using internal API with dataset truncation support.
    """
    try:
        from bfcl_eval._llm_response_generation import (
            generate_results,
            get_involved_test_entries,
            collect_test_cases,
        )

        gen_args = argparse.Namespace(
            model=[args.model_name],
            test_category=categories,
            temperature=args.temperature,
            include_input_log=False,
            exclude_state_log=False,
            num_threads=args.num_threads,
            num_gpus=1,
            backend="vllm",
            gpu_memory_utilization=0.9,
            result_dir=None,
            run_ids=False,
            allow_overwrite=True,
            skip_server_setup=True,
            local_model_path=None,
            lora_modules=None,
            enable_lora=False,
            max_lora_rank=None,
        )

        # Try to set result_dir from BFCL's config
        try:
            from bfcl_eval.constants.eval_config import RESULT_PATH
            gen_args.result_dir = RESULT_PATH
        except ImportError:
            gen_args.result_dir = Path("result")

        log.info(f"Loading test entries for categories: {categories}")
        all_test_file_paths, all_test_categories, all_test_entries_involved = get_involved_test_entries(
            gen_args.test_category, gen_args.run_ids
        )

        total_entries = len(all_test_entries_involved)
        if args.max_samples > 0 and args.max_samples < total_entries:
            log.info(f"Truncating from {total_entries} to {args.max_samples} samples")
            all_test_entries_involved = all_test_entries_involved[: args.max_samples]

        log.info(f"Running generation on {len(all_test_entries_involved)} entries")

        test_cases_total = collect_test_cases(
            gen_args,
            args.model_name,
            all_test_categories,
            all_test_file_paths,
            deepcopy(all_test_entries_involved),
        )

        if test_cases_total:
            generate_results(gen_args, args.model_name, test_cases_total)
            log.info(f"Generation complete: {len(test_cases_total)} test cases processed")
        else:
            log.info("No new test cases to generate (all already cached)")

        return True

    except ImportError as e:
        log.warning(f"Internal API import failed: {e}")
        log.info("Falling back to CLI-based generation")
        return _run_generation_cli(args, categories)


def _run_generation_cli(args, categories: list) -> bool:
    """Fallback: run generation via subprocess CLI (bfcl_eval module)."""
    success = True
    for category in categories:
        log.info(f"Generating for category: {category}")
        cmd = [sys.executable, "-m", "bfcl_eval", "generate",
               "--model", args.model_name,
               "--test-category", category,
               "--num-threads", str(args.num_threads)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            log.error(f"Generation failed for {category}: {result.stderr[:300]}")
            success = False
        else:
            log.info(f"Generation complete for {category}")
    return success


def run_evaluation(args, categories: list):
    """Run BFCL evaluation on generated results.

    Supports partial evaluation (max_samples < total): filters prompt and
    possible_answer to only include entries whose IDs appear in the result file,
    bypassing the assert in BFCL's runners that requires equal lengths.
    """
    try:
        from bfcl_eval.eval_checker.eval_runner import (
            get_handler,
            multi_turn_runner,
            ast_file_runner,
            relevance_file_runner,
        )
        from bfcl_eval.eval_checker.eval_runner_helper import (
            record_cost_latency,
            record_result,
            generate_leaderboard_csv,
            update_leaderboard_table_with_local_score_file,
        )
        from bfcl_eval.constants.eval_config import (
            POSSIBLE_ANSWER_PATH,
            PROMPT_PATH,
            RESULT_PATH,
            SCORE_PATH,
        )
        from bfcl_eval.utils import (
            load_file,
            find_file_with_suffix,
            is_multi_turn,
            is_relevance_or_irrelevance,
            is_java,
            is_js,
        )
    except ImportError as e:
        log.warning(f"Internal eval API import failed: {e}")
        log.info("Falling back to CLI-based evaluation")
        return _run_evaluation_cli(args, categories)

    log.info("Running evaluation via internal API (partial-aware)")

    model_name = args.model_name
    model_name_fs = model_name.replace("/", "_")

    result_dir = RESULT_PATH
    score_dir = SCORE_PATH

    state = {"leaderboard_table": {}}

    try:
        handler = get_handler(model_name)
    except Exception as e:
        log.warning(f"get_handler failed for '{model_name}': {e}")
        log.info("Falling back to CLI-based evaluation")
        return _run_evaluation_cli(args, categories)

    evaluated_count = 0
    for category in categories:
        log.info(f"Evaluating category: {category}")

        # Load model results — skip if no result file exists for this category
        result_model_dir = result_dir / model_name_fs
        if not result_model_dir.exists():
            result_model_dir = result_dir / model_name
        try:
            result_file = find_file_with_suffix(result_model_dir, category)
        except FileNotFoundError:
            log.info(f"No result file for {category}, skipping evaluation")
            continue
        model_result = load_file(result_file, sort_by_id=True)
        if not model_result:
            log.info(f"Empty result file for {category}, skipping evaluation")
            continue

        try:
            record_cost_latency(state["leaderboard_table"], model_name_fs, model_result)
        except Exception as e:
            log.warning(f"record_cost_latency failed for {category}: {e}")

        # Load full prompt
        prompt_file = find_file_with_suffix(PROMPT_PATH, category)
        prompt = load_file(prompt_file, sort_by_id=True)

        # Filter prompt to match result IDs (partial eval support)
        if len(model_result) < len(prompt):
            result_ids = {entry.get("id") for entry in model_result}
            prompt = [p for p in prompt if p.get("id") in result_ids]
            log.info(
                f"Partial eval for {category}: {len(model_result)} of "
                f"{len(load_file(prompt_file))} samples"
            )

        try:
            if is_relevance_or_irrelevance(category):
                accuracy, total_count = relevance_file_runner(
                    handler, model_result, prompt, model_name_fs, category, score_dir
                )
            else:
                # Load and filter possible_answer
                possible_answer_file = find_file_with_suffix(POSSIBLE_ANSWER_PATH, category)
                possible_answer = load_file(possible_answer_file, sort_by_id=True)

                if len(model_result) < len(possible_answer):
                    result_ids = {entry.get("id") for entry in model_result}
                    possible_answer = [a for a in possible_answer if a.get("id") in result_ids]

                if is_multi_turn(category):
                    accuracy, total_count = multi_turn_runner(
                        handler, model_result, prompt, possible_answer,
                        model_name_fs, category, score_dir,
                    )
                else:
                    language = "Java" if is_java(category) else ("JavaScript" if is_js(category) else "Python")
                    accuracy, total_count = ast_file_runner(
                        handler, model_result, prompt, possible_answer,
                        language, category, model_name_fs, score_dir,
                    )

            record_result(state["leaderboard_table"], model_name_fs, category, accuracy, total_count)
            log.info(f"Completed {category}: accuracy={accuracy:.4f} ({total_count} samples)")
            evaluated_count += 1
        except Exception as e:
            log.error(f"Evaluation failed for {category}: {e}")

    if evaluated_count == 0:
        log.warning("No categories were successfully evaluated via internal API")
        log.info("Falling back to CLI-based evaluation")
        return _run_evaluation_cli(args, categories)

    # Post-processing (leaderboard CSV) — non-fatal
    try:
        update_leaderboard_table_with_local_score_file(state["leaderboard_table"], score_dir)
        generate_leaderboard_csv(
            state["leaderboard_table"], score_dir, [model_name_fs], categories
        )
    except Exception as e:
        log.warning(f"Leaderboard CSV generation failed (non-fatal): {e}")

    return True


def _run_evaluation_cli(args, categories: list) -> bool:
    """Fallback: run evaluation via subprocess CLI (bfcl_eval module)."""
    success = True
    for category in categories:
        log.info(f"Evaluating category: {category}")
        cmd = [sys.executable, "-m", "bfcl_eval", "evaluate",
               "--model", args.model_name,
               "--test-category", category]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            log.error(f"Evaluation failed for {category}: {result.stderr[:300]}")
            success = False
        else:
            log.info(f"Evaluation complete for {category}")
    return success


def collect_scores(model_name: str) -> tuple:
    """
    Collect evaluation scores from BFCL's output directories.
    BFCL writes:
    - Per-category JSONs: {SCORE_PATH}/{model_name}/BFCL_v3_{category}_score.json
      Format: {"accuracy": 0.9, "correct_count": 9, "total_count": 10}
    - Summary CSVs: {SCORE_PATH}/data_overall.csv, data_non_live.csv, etc.
      Values use percentage format with % sign (e.g. "90.00%")
    Returns (scores, counts) where counts maps category -> total_count.
    """
    scores = {}
    counts = {}

    # Resolve BFCL's actual score directory
    try:
        from bfcl_eval.constants.eval_config import SCORE_PATH
        score_base = SCORE_PATH
    except ImportError:
        score_base = Path("score")

    log.info(f"Looking for scores in: {score_base}")

    # 1. Read per-category score JSONs (most reliable)
    score_model_dir = score_base / model_name
    if not score_model_dir.exists():
        score_model_dir = score_base / model_name.replace("/", "_")

    if score_model_dir.exists():
        for json_file in score_model_dir.glob("*_score.json"):
            try:
                first_line = json_file.read_text(encoding="utf-8").strip().split("\n")[0]
                data = json.loads(first_line)
                if isinstance(data, dict) and "accuracy" in data:
                    category = json_file.stem.replace("BFCL_v3_", "").replace("_score", "")
                    scores[category] = data["accuracy"]
                    counts[category] = data.get("total_count", 0)
                    log.info(f"  {category}: {data['accuracy']} ({data.get('correct_count', '?')}/{data.get('total_count', '?')})")
            except (json.JSONDecodeError, UnicodeDecodeError, IndexError):
                continue
    else:
        log.warning(f"Score model dir not found: {score_model_dir}")

    # 2. If no JSONs found, try CSV summaries
    if not scores:
        csv_files = list(score_base.glob("data_*.csv")) if score_base.exists() else []
        for csv_file in csv_files:
            try:
                import csv as csv_mod
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        row_model = row.get("Model", row.get("model", ""))
                        if model_name.lower() in row_model.lower():
                            for k, v in row.items():
                                if k.lower() in ("rank", "model", ""):
                                    continue
                                try:
                                    v_str = v.strip()
                                    if v_str.endswith("%"):
                                        scores[k] = float(v_str[:-1]) / 100.0
                                    elif v_str and v_str != "N/A":
                                        scores[k] = float(v_str)
                                except (ValueError, TypeError):
                                    pass
                            break
            except Exception as e:
                log.warning(f"Failed to parse {csv_file}: {e}")

    return scores, counts


def compute_overall(scores: dict, counts: dict) -> float:
    """Compute overall accuracy using the BFCL v3 official 3-group formula.

    Overall = unweighted_avg(non_live, live, multi_turn)
      non_live  = unweighted_avg(simple_ast, multiple, parallel, parallel_multiple, irrelevance)
                  where simple_ast = unweighted_avg(simple, java, javascript)
      live      = sample-count-weighted avg of 6 live categories
      multi_turn= unweighted_avg of 4 multi-turn categories
    """
    def _unw(*keys):
        vals = [scores[k] for k in keys if k in scores]
        return sum(vals) / len(vals) if vals else None

    def _wtd(*keys):
        present = [k for k in keys if k in scores and counts.get(k, 0) > 0]
        if not present:
            return None
        total_n = sum(counts[k] for k in present)
        return sum(scores[k] * counts[k] for k in present) / total_n

    # Non-Live: 2-level hierarchy matching eval_runner_helper.py
    simple_ast = _unw("simple", "java", "javascript")
    non_live_inputs = [
        simple_ast,
        scores.get("multiple"),
        scores.get("parallel"),
        scores.get("parallel_multiple"),
        scores.get("irrelevance"),
    ]
    non_live_inputs = [v for v in non_live_inputs if v is not None]
    non_live = sum(non_live_inputs) / len(non_live_inputs) if non_live_inputs else None

    # Live: weighted by sample count; fall back to unweighted if counts missing
    live = _wtd(
        "live_simple", "live_multiple", "live_parallel",
        "live_parallel_multiple", "live_irrelevance", "live_relevance",
    )
    if live is None:
        live = _unw(
            "live_simple", "live_multiple", "live_parallel",
            "live_parallel_multiple", "live_irrelevance", "live_relevance",
        )

    # Multi-Turn: unweighted
    multi_turn = _unw(
        "multi_turn_base", "multi_turn_miss_func",
        "multi_turn_miss_param", "multi_turn_long_context",
    )

    groups = [g for g in [non_live, live, multi_turn] if g is not None]
    return sum(groups) / len(groups) if groups else 0.0


def _extract_prompt_text(question, is_multi_turn: bool) -> str:
    """Extract human-readable prompt text from BFCL question structure."""
    if not question or not isinstance(question, list) or not question[0]:
        return ""
    if is_multi_turn:
        turn_texts = []
        for turn_idx, turn_msgs in enumerate(question, 1):
            if not isinstance(turn_msgs, list):
                continue
            for msg in turn_msgs:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    turn_texts.append(f"**Turn {turn_idx}:** {msg.get('content', '')}")
                    break
        return "\n\n".join(turn_texts)
    else:
        msgs = question[0] if isinstance(question[0], list) else question
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""


def _parse_error_fields(entry: dict) -> tuple:
    """Extract (error_message, error_type, error_details) from a score file entry."""
    raw_error = entry.get("error", [])
    error_message = ""
    error_type = entry.get("error_type", "")
    error_details = ""
    if isinstance(raw_error, dict):
        error_message = raw_error.get("error_message", "")
        if isinstance(error_message, list):
            error_message = "; ".join(str(m) for m in error_message)
        error_type = raw_error.get("error_type", error_type)
        details = raw_error.get("details")
        if details:
            error_details = json.dumps(details, ensure_ascii=False)
    elif isinstance(raw_error, list) and raw_error:
        error_message = "; ".join(str(m) for m in raw_error)
    return error_message, error_type, error_details


def write_oneeval_scores(args, scores: dict, counts: dict, categories: list):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = compute_overall(scores, counts)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Resolve BFCL paths
    try:
        from bfcl_eval.constants.eval_config import SCORE_PATH, RESULT_PATH
        score_base = SCORE_PATH
        result_base = RESULT_PATH
    except ImportError:
        score_base = Path("score")
        result_base = Path("result")

    score_model_dir = score_base / args.model_name
    if not score_model_dir.exists():
        score_model_dir = score_base / args.model_name.replace("/", "_")

    result_model_dir = result_base / args.model_name
    if not result_model_dir.exists():
        result_model_dir = result_base / args.model_name.replace("/", "_")

    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name
    total_samples = 0

    # Per-category accuracy and total_count from score file summaries
    category_accuracy = {}
    category_total_count = {}
    if score_model_dir.exists():
        for json_file in sorted(score_model_dir.glob("*_score.json")):
            try:
                lines = json_file.read_text(encoding="utf-8").strip().split("\n")
                summary = json.loads(lines[0])
                category = json_file.stem.replace("BFCL_v3_", "").replace("_score", "")
                category_accuracy[category] = summary.get("accuracy", 0.0)
                category_total_count[category] = summary.get("total_count", 0)
            except (json.JSONDecodeError, OSError, IndexError):
                continue

    # --- Build failed-entry lookup from score files ---
    # Key: (category, entry_id) → dict with error info and model_result
    # This handles duplicate IDs across categories correctly.
    failed_lookup = {}  # (category, entry_id) -> {error_message, error_type, error_details, model_result}
    if score_model_dir.exists():
        for json_file in sorted(score_model_dir.glob("*_score.json")):
            try:
                lines = json_file.read_text(encoding="utf-8").strip().split("\n")
                if len(lines) <= 1:
                    continue
                category = json_file.stem.replace("BFCL_v3_", "").replace("_score", "")
                for line in lines[1:]:
                    entry = json.loads(line)
                    entry_id = entry.get("id", "")
                    error_message, error_type, error_details = _parse_error_fields(entry)
                    model_result = entry.get("model_result", [])
                    failed_lookup[(category, entry_id)] = {
                        "error_message": error_message,
                        "error_type": error_type,
                        "error_details": error_details,
                        "model_result": model_result,
                    }
            except (json.JSONDecodeError, OSError):
                continue

    # --- Load test data for prompt extraction ---
    test_data = {}
    try:
        from pathlib import Path as P
        possible_data_dirs = [
            P(__file__).parent.parent / "bfcl_eval" / "data",
            P("bfcl_eval") / "data",
            P("/workspace/repo/berkeley-function-call-leaderboard/bfcl_eval/data"),
        ]
        bfcl_data_dir = None
        for candidate in possible_data_dirs:
            if candidate.exists():
                bfcl_data_dir = candidate
                break
        if bfcl_data_dir:
            for data_file in bfcl_data_dir.glob("BFCL_v3_*.json"):
                try:
                    with open(data_file, "r", encoding="utf-8") as f:
                        for line in f:
                            e = json.loads(line)
                            test_data[e.get("id")] = e
                except (json.JSONDecodeError, OSError):
                    continue
            log.info(f"Loaded {len(test_data)} test entries for prompt extraction")
    except Exception as e:
        log.warning(f"Could not load BFCL test data: {e}")

    # --- Write detail JSONL: result files as ground truth ---
    # Each line in a result file = one evaluated sample.
    # Score files only attach error info to failed entries.
    with open(detail_path, "w", encoding="utf-8") as fout:
        if not result_model_dir.exists():
            log.warning(f"Result model dir not found: {result_model_dir}")
        else:
            scored_categories = set(category_accuracy.keys())
            for result_file in sorted(result_model_dir.glob("*_result.json")):
                category = result_file.stem.replace("BFCL_v3_", "").replace("_result", "")

                # Only include categories that have been evaluated (have a score file)
                if scored_categories and category not in scored_categories:
                    log.warning(
                        f"Result file exists for '{category}' but no score file found, "
                        f"skipping (unevaluated category)"
                    )
                    continue

                try:
                    for line in result_file.read_text(encoding="utf-8").strip().split("\n"):
                        if not line:
                            continue
                        entry = json.loads(line)
                        entry_id = entry.get("id", "")

                        # Extract prompt from test data
                        prompt_text = f"[{category}]"
                        if entry_id in test_data:
                            test_entry = test_data[entry_id]
                            question = test_entry.get("question", [[]])
                            is_mt = (
                                question and isinstance(question, list)
                                and len(question) > 1 and isinstance(question[0], list)
                            )
                            extracted = _extract_prompt_text(question, is_mt)
                            if extracted:
                                prompt_text = extracted

                        # Check if this entry failed
                        fail_info = failed_lookup.get((category, entry_id))
                        if fail_info:
                            model_result = fail_info["model_result"]
                            solution = json.dumps(model_result, ensure_ascii=False) if model_result else ""
                            record = {
                                "task_id": entry_id,
                                "category": category,
                                "prompt": prompt_text,
                                "solution": solution,
                                "eval_score": 0.0,
                                "eval_valid": True,
                                "error_message": fail_info["error_message"],
                                "error_type": fail_info["error_type"],
                                "error_details": fail_info["error_details"],
                            }
                        else:
                            solution = entry.get("result", "")
                            if isinstance(solution, (list, dict)):
                                solution = json.dumps(solution, ensure_ascii=False)
                            record = {
                                "task_id": entry_id,
                                "category": category,
                                "prompt": prompt_text,
                                "solution": str(solution) if solution else "",
                                "eval_score": 1.0,
                                "eval_valid": True,
                                "error_message": "",
                                "error_type": "",
                                "error_details": "",
                            }

                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_samples += 1
                except (json.JSONDecodeError, OSError):
                    continue

    # Validate total_samples against score file summaries
    expected_total = sum(category_total_count.values()) if category_total_count else None
    if expected_total and total_samples != expected_total:
        log.warning(
            f"Detail row count ({total_samples}) differs from score summaries "
            f"total_count sum ({expected_total}). Using actual written count."
        )

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    # Determine if this is the full official BFCL evaluation
    ALL_OFFICIAL_CATEGORIES = set(CATEGORY_PRESETS["all"])
    evaluated_set = set(category_accuracy.keys())
    is_official_full = evaluated_set >= ALL_OFFICIAL_CATEGORIES

    score_key = "overall_accuracy" if is_official_full else "subset_accuracy"

    result = {
        "bench_name": "bfcl",
        "model_name": args.model_name,
        "test_categories": categories,
        "timestamp": timestamp,
        score_key: overall,
        "official_comparable": is_official_full,
        "total_samples": total_samples,
        "detail_path": detail_name,
        "category_scores": {k: v for k, v in scores.items()},
    }
    if not is_official_full:
        result["overall_accuracy"] = overall

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Scores written to: {score_file}")
    log.info(f"{'Overall' if is_official_full else 'Subset'} accuracy: {overall:.4f}")
    log.info(f"Official comparable: {is_official_full}")
    log.info(f"Per-category scores: {json.dumps(scores, indent=2)}")
    return result


def main():
    args = parse_args()
    categories = resolve_categories(args.test_category)

    log.info("BFCL One-Eval bridge starting")
    log.info(f"  Model: {args.model_name}")
    log.info(f"  Categories ({len(categories)}): {categories}")
    log.info(f"  max_samples={args.max_samples}, threads={args.num_threads}")

    setup_env()
    register_model(args.model_name, args.temperature)

    # Clean stale score and result files to prevent reading data from previous runs
    import shutil
    try:
        from bfcl_eval.constants.eval_config import SCORE_PATH, RESULT_PATH
        score_base = SCORE_PATH
        result_base = RESULT_PATH
    except ImportError:
        score_base = Path("score")
        result_base = Path("result")

    stale_score_dir = score_base / args.model_name
    if not stale_score_dir.exists():
        stale_score_dir = score_base / args.model_name.replace("/", "_")
    if stale_score_dir.exists():
        shutil.rmtree(stale_score_dir)
        log.info(f"Removed stale score directory: {stale_score_dir}")

    stale_result_dir = result_base / args.model_name
    if not stale_result_dir.exists():
        stale_result_dir = result_base / args.model_name.replace("/", "_")
    if stale_result_dir.exists():
        shutil.rmtree(stale_result_dir)
        log.info(f"Removed stale result directory: {stale_result_dir}")

    log.info("=" * 60)
    log.info("Phase 1: Generation")
    log.info("=" * 60)
    gen_ok = run_generation(args, categories)
    if not gen_ok:
        log.error("Generation phase failed")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase 2: Evaluation")
    log.info("=" * 60)
    eval_ok = run_evaluation(args, categories)
    if not eval_ok:
        log.error("Evaluation phase failed")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase 3: Collecting scores")
    log.info("=" * 60)
    scores, counts = collect_scores(args.model_name)

    if not scores:
        log.error("No evaluation results found!")
        log.error("Check BFCL generation and evaluation output above for errors.")
        sys.exit(1)

    # Verify all requested categories have scores
    missing_cats = [c for c in categories if c not in scores]
    if missing_cats:
        log.warning(
            f"Missing scores for categories: {missing_cats}. "
            f"Got scores for: {list(scores.keys())}"
        )

    write_oneeval_scores(args, scores, counts, categories)
    log.info("Done!")


if __name__ == "__main__":
    main()
