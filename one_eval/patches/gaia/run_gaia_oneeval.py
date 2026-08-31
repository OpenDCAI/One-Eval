#!/usr/bin/env python
"""
GAIA benchmark evaluation bridge for One-Eval.

Uses smolagents framework with the following capabilities:
- Dual-agent architecture (CodeAgent + ToolCallingAgent for web search)
- Full file attachment support (PDF/XLSX/PPTX/images/audio/ZIP)
- Answer reformulation step for precise output formatting
- Advanced web browsing tools (stateful browser with PageUp/Down/Find)
- Multi-modal support (Vision + audio transcription)

Architecture based on smolagents/examples/open_deep_research/run_gaia.py
"""

import argparse
import json
import logging
import os
import re
import string
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gaia_oneeval_aligned")

# Module-level lock for thread-safe JSONL writing (shared across all threads)
append_answer_lock = threading.Lock()


# ============================================================
# Official GAIA scorer imported from gaia_scorer.py
# ============================================================
# question_scorer is imported above from the official implementation

# ============================================================
# P0: File Attachment Support
# ============================================================

def download_gaia_dataset_with_files(split: str, data_dir: Path):
    """
    Download GAIA dataset including all attachment files.
    This is CRITICAL - without real files, ~40% of questions are impossible.
    """
    from huggingface_hub import snapshot_download

    if not data_dir.exists():
        log.info(f"Downloading GAIA dataset with attachments to {data_dir}...")
        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=str(data_dir),
            ignore_patterns=[".gitattributes", "README.md"],
        )
        log.info("✅ Dataset downloaded")
    else:
        log.info(f"Dataset already exists at {data_dir}")


def load_gaia_data_with_file_paths(split: str, max_samples: int, data_dir: Path):
    """
    Load GAIA dataset and construct real file paths for attachments.
    Loads from HuggingFace Hub directly (compatible with datasets>=3.0).
    """
    from datasets import load_dataset

    log.info(f"Loading GAIA dataset (split={split})...")
    ds = load_dataset(
        "gaia-benchmark/GAIA",
        name="2023_all",
        split=split,
        token=os.environ.get("HF_TOKEN"),
    )

    data = []
    for item in ds:
        file_path = ""
        file_name = item.get("file_name", "") or ""
        if file_name:
            file_path = str(data_dir / "2023" / split / file_name)
            if not Path(file_path).exists():
                log.warning(f"Attachment not found: {file_path}")

        data.append({
            "task_id": item["task_id"],
            "question": item["Question"],
            "level": item.get("Level", 1),
            "final_answer": item.get("Final answer", ""),
            "file_name": file_name,
            "file_path": file_path,
        })

    if max_samples > 0:
        data = data[:max_samples]
        log.info(f"Limited to {len(data)} samples")
    else:
        log.info(f"Loaded {len(data)} samples")

    return data


# ============================================================
# Import official tools from smolagents repo
# ============================================================

# This script will be copied to smolagents repo root, but runs from examples/open_deep_research/
# Python adds the script's directory (repo root) to sys.path, not CWD.
# We need CWD on the path so `from scripts.xxx` resolves to examples/open_deep_research/scripts/
sys.path.insert(0, os.getcwd())

try:
    # Import from scripts/ directory (relative to current working directory)
    from scripts.text_inspector_tool import TextInspectorTool
    from scripts.visual_qa import visualizer
    from scripts.text_web_browser import (
        SimpleTextBrowser, VisitTool,
        PageUpTool, PageDownTool, FinderTool, FindNextTool, ArchiveSearchTool
    )
    from scripts.reformulator import prepare_response
    from scripts.run_agents import get_single_file_description, get_zip_description

    # Import official GAIA scorer from scripts/ directory
    from scripts.gaia_scorer import question_scorer

    # Use DuckDuckGoSearchTool (free, no API key required)
    from smolagents import DuckDuckGoSearchTool

    TOOLS_AVAILABLE = True
except ImportError as e:
    log.error(f"Failed to import official smolagents tools: {e}")
    log.error("Make sure this script is running in the smolagents repo root")
    TOOLS_AVAILABLE = False


# ============================================================
# P1: Dual-Agent Architecture (EXACT copy of official implementation)
# ============================================================

def create_agent_team(model, token_counts):
    """
    Replicate the EXACT official agent architecture from HuggingFace.

    Returns the manager agent (CodeAgent with a managed search sub-agent).
    """
    from smolagents import CodeAgent, ToolCallingAgent, TokenUsage

    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    # Browser configuration - EXACT copy from official
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    BROWSER_CONFIG = {
        "viewport_size": 1024 * 5,
        "downloads_folder": "downloads_folder",
        "request_kwargs": {
            "headers": {"User-Agent": user_agent},
            "timeout": 300,
        },
        "serpapi_key": os.getenv("SERPAPI_API_KEY"),
    }
    os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        DuckDuckGoSearchTool(),  # Free, no API key required
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    def increment_web_agent_token_counts(final_answer, memory_step, agent):
        token_counts_web = agent.monitor.get_total_token_counts()
        # v1.25.0 returns TokenUsage object with .input_tokens / .output_tokens attributes
        token_counts["input"] += token_counts_web.input_tokens
        token_counts["output"] += token_counts_web.output_tokens
        return True

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=0,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        final_answer_checks=[increment_web_agent_token_counts],
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=0,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


# ============================================================
# P0 + P2: Prompt Engineering (EXACT copy from official)
# ============================================================

def build_augmented_question(example: dict, visual_inspection_tool, document_inspection_tool) -> str:
    """
    Build the augmented question with:
    - Official aggressive prompt
    - File descriptions (if attachments exist)

    This is the EXACT prompt that achieved 44.24% on GAIA validation.
    """
    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_path"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                example["file_path"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        augmented_question += prompt_use_files

    return augmented_question


# ============================================================
# P2: Per-question execution with independent agent
# ============================================================

def answer_single_question(example: dict, model_id: str, answers_file: str):
    """
    Run agent on a single question (independent agent per question for isolation).

    This follows the official implementation:
    1. Create fresh agent + model
    2. Build augmented prompt
    3. Run agent
    4. Reformulate answer
    5. Save result
    """
    from smolagents import LiteLLMModel
    from datetime import datetime

    # Official custom_role_conversions (for proper tool message handling)
    custom_role_conversions = {
        "tool-call": "assistant",
        "tool-response": "user",
    }

    # Model parameters
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
    }
    if api_key:
        model_params["api_key"] = api_key
    if api_base:
        model_params["api_base"] = api_base

    # Official uses different params for different models
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096

    model = LiteLLMModel(**model_params)

    # Tool instances for file description
    document_inspection_tool = TextInspectorTool(model, 100000)
    visual_inspection_tool = visualizer

    # Token tracking
    total_token_counts = {
        "input": 0,
        "output": 0,
    }

    # Create agent team (fresh for each question)
    agent = create_agent_team(model, total_token_counts)

    # P2: Build augmented question with file descriptions
    augmented_question = build_augmented_question(example, visual_inspection_tool, document_inspection_tool)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages()

        # P0: Answer reformulation (CRITICAL step)
        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)

        # Clean up memory for storage
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None

        # Serialize ChatMessage objects to dicts before JSON writing
        intermediate_steps = [
            msg.dict() if hasattr(msg, 'dict') else msg
            for msg in agent_memory
        ]

        # Check for errors
        parsing_error = True if any(["AgentParsingError" in str(step) for step in intermediate_steps]) else False
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False
        exception = None

    except Exception as e:
        log.error(f"Error on question {example['task_id']}: {e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Aggregate token counts
    token_counts_manager = agent.monitor.get_total_token_counts()
    # v1.25.0 returns TokenUsage object with .input_tokens / .output_tokens attributes
    total_token_counts["input"] += token_counts_manager.input_tokens
    total_token_counts["output"] += token_counts_manager.output_tokens

    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example.get("level", ""),
        "task_id": example["task_id"],
        "true_answer": example["final_answer"],
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
    }

    # Append to JSONL (thread-safe using module-level lock)
    with append_answer_lock:
        with open(answers_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(annotated_example, ensure_ascii=False) + "\n")

    return annotated_example


# ============================================================
# Main execution
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GAIA benchmark evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--data_dir", type=str, default="data/gaia",
                        help="Directory for GAIA dataset with attachments")
    return parser.parse_args()


def evaluate(data: list, predictions: dict) -> dict:
    """Score all predictions and compute metrics."""
    total = defaultdict(int)
    correct = defaultdict(int)

    for item in data:
        task_id = item["task_id"]
        level = item["level"]
        ground_truth = item["final_answer"]
        pred_entry = predictions.get(task_id, {})

        if isinstance(pred_entry, str):
            prediction = pred_entry
        else:
            prediction = pred_entry.get("prediction", "")

        level_key = f"level{level}"
        total["overall"] += 1
        total[level_key] += 1

        if question_scorer(prediction, ground_truth):
            correct["overall"] += 1
            correct[level_key] += 1

    results = {}
    for key in total:
        results[key] = {
            "total": total[key],
            "correct": correct[key],
            "accuracy": correct[key] / total[key] * 100 if total[key] > 0 else 0.0,
        }

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("GAIA Benchmark Evaluation - One-Eval Bridge")
    log.info("=" * 70)
    log.info(f"  Model: {args.model_name}")
    split = "validation"
    log.info(f"  Split: {split}")
    log.info(f"  Max samples: {args.max_samples}")
    log.info(f"  Concurrency: {args.concurrency}")
    log.info("=" * 70)

    # Check if tools are available
    if not TOOLS_AVAILABLE:
        log.error("❌ Required smolagents tools not available!")
        log.error("Please ensure smolagents_tools/ directory contains:")
        log.error("  - text_inspector_tool.py, visual_qa.py, text_web_browser.py")
        log.error("  - reformulator.py, run_agents.py, mdconvert.py, cookies.py")
        return 1

    # Download GAIA dataset with attachments
    data_dir = Path(args.data_dir)
    download_gaia_dataset_with_files(split, data_dir)

    # Load data with real file paths
    data = load_gaia_data_with_file_paths(split, args.max_samples, data_dir)

    # Check for existing results (resume support using task_id)
    answers_file = Path(args.output_dir) / "predictions.jsonl"
    done_task_ids = set()
    if answers_file.exists():
        import pandas as pd
        try:
            done_df = pd.read_json(answers_file, lines=True)
            done_task_ids = set(done_df["task_id"].tolist())
            log.info(f"Resuming from {len(done_task_ids)} cached results")
        except Exception as e:
            log.warning(f"Could not load previous results: {e}")

    # Filter to only unanswered questions (by task_id, more robust than question text)
    tasks_to_run = [item for item in data if item["task_id"] not in done_task_ids]
    log.info(f"Processing {len(tasks_to_run)} questions ({len(data) - len(tasks_to_run)} cached)")

    # Run with ThreadPoolExecutor (each question gets independent agent)
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                answer_single_question,
                example,
                args.model_name,
                str(answers_file)
            )
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing"):
            try:
                f.result()
            except Exception as e:
                log.error(f"Task failed: {e}")

    log.info("All tasks completed")

    # Load all predictions
    import pandas as pd
    predictions_df = pd.read_json(answers_file, lines=True)
    predictions = {row["task_id"]: row for _, row in predictions_df.iterrows()}

    # Evaluate
    log.info("Computing scores...")
    results = evaluate(data, predictions)

    # Build per-sample detail JSONL (One-Eval format)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = Path(args.output_dir) / detail_name

    with open(detail_path, "w", encoding="utf-8") as fout:
        for item in data:
            task_id = item["task_id"]
            pred_entry = predictions.get(task_id, {})

            if isinstance(pred_entry, str):
                prediction = pred_entry
                prompt = item["question"]
                agent_logs = []
            else:
                prediction = pred_entry.get("prediction", "") or ""
                prompt = pred_entry.get("augmented_question", item["question"]) or item["question"]
                agent_logs = pred_entry.get("intermediate_steps", []) or []

            is_correct = question_scorer(str(prediction), item["final_answer"])

            record = {
                "task_id": task_id,
                "level": f"Level {item['level']}",
                "prompt": prompt,
                "solution": str(prediction),
                "ground_truth": item["final_answer"],
                "eval_score": 1.0 if is_correct else 0.0,
                "eval_valid": True,
                "agent_logs": agent_logs if isinstance(agent_logs, list) else [],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write scores in One-Eval format
    score_output = {
        "average": {
            "accuracy": results["overall"]["accuracy"] / 100.0  # ratio scale (0-1)
        },
        "total_samples": len(data),
        "detail_path": detail_name,
        "by_level": {
            k: {
                "accuracy": v["accuracy"] / 100.0,  # ratio scale (0-1)
                "correct": v["correct"],
                "total": v["total"]
            }
            for k, v in results.items() if k != "overall"
        },
        "details": {
            "total": results["overall"]["total"],
            "correct": results["overall"]["correct"],
        },
        # Metadata for reproducibility and transparency
        "metadata": {
            "scaffold": "hf_open_deep_research",
            "scaffold_version": "v1.25.0",
            "scaffold_commit": "827964d923316b44c5492b26a920ade335ecd589",
            "main_model": args.model_name,
            "vision_model": "gpt-4o",  # Hardcoded in visual_qa.py line 167
            "vision_model_note": "Vision model is hardcoded to gpt-4o in smolagents visual_qa.py and cannot be changed",
            "dataset_repo": "gaia-benchmark/GAIA",
            "split": split,
            "search_provider": "duckduckgo",
            "manager_max_steps": 12,
            "search_max_steps": 20,
            "planning_interval": 4,
            "max_tokens": 4096,
            "concurrency": args.concurrency,
            "completed": len(predictions),
            "deviations_from_upstream": [
                "run all validation tasks (upstream v1.25.0 filters to file tasks only)",
                "task_id-based resume instead of question-text based",
                "module-level thread lock instead of per-function lock",
                "serialize ChatMessage to dict before JSON writing"
            ]
        }
    }

    score_file = Path(args.output_dir) / f"scores_{timestamp}.json"
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_output, f, indent=2, ensure_ascii=False)

    log.info("=" * 70)
    log.info("Evaluation complete!")
    log.info(f"  Overall: {score_output['average']['accuracy']:.4f} "
             f"({results['overall']['correct']}/{results['overall']['total']})")
    for key in sorted(results):
        if key.startswith("level"):
            r = results[key]
            log.info(f"  {key}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
    log.info(f"  Results saved to: {args.output_dir}")
    log.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

