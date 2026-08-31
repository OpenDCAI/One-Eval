#!/usr/bin/env python
"""
MMMU evaluation bridge script for One-Eval.
Integrates MMMU's parse_and_eval pipeline with One-Eval's external_repo system.

Flow:
  1. Load MMMU dataset from HuggingFace (30 subjects)
  2. Run inference via OpenAI-compatible API (multimodal, base64 images)
  3. Format outputs into MMMU's expected directory structure
  4. Call MMMU's main_parse_and_eval.py for answer extraction + scoring
  5. Aggregate results into scores_*.json

Model config is read from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mmmu_oneeval")

# 30 MMMU subjects
SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]


def parse_args():
    parser = argparse.ArgumentParser(description="MMMU evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    return parser.parse_args()


def encode_image(image) -> str:
    """Encode a PIL image to base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt_text(sample):
    """Build the full prompt text (with <image N> placeholders kept in place)."""
    question = sample["question"]
    options = sample.get("options")

    if sample["question_type"] == "multiple-choice" and options:
        if isinstance(options, str):
            options = eval(options)
        option_str = "\n".join(
            f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)
        )
        prompt = (
            f"{question}\n{option_str}\n"
            "Answer with the option's letter from the given choices directly."
        )
    else:
        prompt = (
            f"{question}\n"
            "Answer the question using a single word or phrase."
        )
    return prompt


def get_images(sample):
    """Extract all images from a sample (image_1 through image_7) as a dict."""
    images = {}
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            images[i] = img
    return images


def build_interleaved_content(prompt_text, images):
    """Split prompt by <image N> placeholders and interleave actual images."""
    parts = re.split(r"<image (\d+)>", prompt_text)
    content = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                content.append({"type": "text", "text": text})
        else:
            img_idx = int(part)
            if img_idx in images:
                b64 = encode_image(images[img_idx])
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
    if not content:
        content.append({"type": "text", "text": prompt_text})
    return content


def call_model(client, model_name, content, temperature, max_tokens,
               max_retries=3):
    """Call OpenAI-compatible API with interleaved content."""
    messages = [{"role": "user", "content": content}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"API error (attempt {attempt+1}): {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"API failed after {max_retries} attempts: {e}")
                return ""


def infer_subject(client, model_name, subject, split, max_samples,
                  temperature, max_tokens, output_dir):
    """Run inference for one subject, write output.json in MMMU format."""
    subject_dir = output_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_file = subject_dir / "output.json"

    # Create images directory for storing sample images
    images_dir = subject_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results for resumption
    existing = {}
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            existing = {item["id"]: item for item in data}
        except Exception:
            pass

    # Load dataset
    try:
        ds = load_dataset("MMMU/MMMU", subject, split=split)
    except Exception as e:
        log.error(f"Failed to load {subject}: {e}")
        return []

    samples = list(ds)
    if max_samples > 0:
        samples = samples[:max_samples]

    results = []
    for sample in samples:
        sample_id = sample["id"]

        # Save images to disk for all samples (needed for HTML report)
        images = get_images(sample)
        image_paths = []
        for i, img in images.items():
            img_filename = f"{sample_id}_{i}.png"
            img_path = images_dir / img_filename
            if not img_path.exists():
                try:
                    img.save(img_path, format="PNG")
                except Exception as e:
                    log.warning(f"Failed to save image {img_filename}: {e}")
            # Store relative path from subject_dir
            image_paths.append(f"images/{img_filename}")

        # Skip if already done
        if sample_id in existing and existing[sample_id].get("response"):
            existing_item = existing[sample_id]
            # Add image_paths to existing results if missing
            if "image_paths" not in existing_item:
                existing_item["image_paths"] = image_paths
            results.append(existing_item)
            continue

        prompt_text = build_prompt_text(sample)
        content = build_interleaved_content(prompt_text, images)

        response = call_model(
            client, model_name, content, temperature, max_tokens
        )

        # Build output in MMMU's expected format for main_parse_and_eval.py
        options = sample.get("options")
        if isinstance(options, str):
            options = eval(options)

        all_choices = [chr(65 + i) for i in range(len(options))] if options else []
        index2ans = {chr(65 + i): opt for i, opt in enumerate(options)} if options else {}

        item = {
            "id": sample_id,
            "question": sample.get("question", ""),  # Add question text for HTML report
            "question_type": sample["question_type"],
            "answer": sample.get("answer", ""),
            "all_choices": all_choices,
            "index2ans": index2ans,
            "response": response,
            "image_paths": image_paths,  # Add image paths for HTML report
        }
        results.append(item)

    # Save
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def run_mmmu_eval(mmmu_root, output_dir):
    """Call MMMU's main_parse_and_eval.py to parse and evaluate."""
    eval_script = mmmu_root / "mmmu" / "main_parse_and_eval.py"
    if not eval_script.exists():
        log.error(f"Eval script not found: {eval_script}")
        return None

    cmd = [
        sys.executable,
        str(eval_script),
        "--path", str(output_dir),
        "--subject", "ALL",
    ]

    log.info(f"Running MMMU eval: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(mmmu_root / "mmmu"),
    )

    if result.returncode != 0:
        log.error(f"MMMU eval failed:\n{result.stderr}")
        return None

    log.info(f"MMMU eval stdout:\n{result.stdout}")
    return result.stdout


def collect_results(output_dir):
    """Collect per-subject result.json files into aggregated scores."""
    scores = {}
    total_correct = 0
    total_count = 0

    for subject_dir in sorted(output_dir.iterdir()):
        result_file = subject_dir / "result.json"
        if not result_file.exists():
            continue

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            acc = data.get("acc", 0.0)
            num = data.get("num_example", 0)
            scores[subject_dir.name] = {"accuracy": acc, "count": num}
            total_correct += int(acc * num)
            total_count += num
        except Exception as e:
            log.warning(f"Failed to read {result_file}: {e}")

    overall_acc = (total_correct / total_count) if total_count > 0 else 0.0

    return {
        "average": {"accuracy": overall_acc},
        "total_samples": total_count,
        "by_subject": scores,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # MMMU repo root (this script is placed at mmmu/run_mmmu_oneeval.py)
    mmmu_root = Path(__file__).parent.parent

    client = OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
    )

    # Stage 1: Inference
    # When max_samples is set, treat it as TOTAL sample cap (aligned with dataflow):
    # iterate subjects in order, take samples sequentially until reaching the cap.
    # Only download data for subjects we actually need.
    log.info(f"=== Stage 1: Inference ({args.split}, max_samples={args.max_samples}) ===")

    all_infer_results = []
    remaining = args.max_samples if args.max_samples > 0 else float("inf")

    for subject in SUBJECTS:
        if remaining <= 0:
            break
        # Per-subject cap: take at most 'remaining' from this subject
        per_subject_cap = int(remaining) if remaining != float("inf") else -1
        results = infer_subject(
            client, args.model_name, subject, args.split,
            per_subject_cap, args.temperature, args.max_tokens, output_dir,
        )
        log.info(f"  {subject}: {len(results)} samples done")
        all_infer_results.extend(results)
        if args.max_samples > 0:
            remaining -= len(results)

    log.info(f"Total inference: {len(all_infer_results)} samples across subjects")

    # Stage 2: Parse and Eval (using MMMU's official script)
    log.info("=== Stage 2: Parse and Eval ===")
    run_mmmu_eval(mmmu_root, output_dir)

    # Stage 3: Aggregate scores and build per-sample detail
    log.info("=== Stage 3: Aggregate Scores ===")
    scores = collect_results(output_dir)

    # Import MMMU's official answer parsing for correct eval_score
    sys.path.insert(0, str(mmmu_root / "mmmu"))
    try:
        from utils.eval_utils import parse_multi_choice_response, parse_open_response
    except ImportError:
        parse_multi_choice_response = None
        parse_open_response = None
        log.warning("Could not import MMMU eval_utils, eval_score will use naive comparison")

    # Build enriched per-sample JSONL for HTML report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name
    total_samples = 0

    with open(detail_path, "w", encoding="utf-8") as fout:
        for subject_dir in sorted(output_dir.iterdir()):
            output_file = subject_dir / "output.json"
            if not subject_dir.is_dir() or not output_file.exists():
                continue
            subject = subject_dir.name
            try:
                items = json.loads(output_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for item in items:
                answer = item.get("answer", "")
                response = item.get("response", "")
                all_choices = item.get("all_choices", [])
                index2ans = item.get("index2ans", {})
                question_type = item.get("question_type", "")

                # Use MMMU's official parsing to determine correctness
                if question_type == "multiple-choice" and parse_multi_choice_response:
                    parsed = parse_multi_choice_response(response, all_choices, index2ans)
                    correct = (parsed == answer)
                elif question_type == "open" and parse_open_response:
                    parsed = parse_open_response(response)
                    correct = (str(parsed).strip().lower() == str(answer).strip().lower())
                else:
                    correct = response.strip().upper() == answer.strip().upper()

                # Build a readable prompt from available info
                question_text = item.get("question", "")
                if index2ans:
                    choices_str = " | ".join(f"{k}. {v}" for k, v in index2ans.items())
                    prompt_text = f"[{subject} / {question_type}]\n{question_text}\nChoices: {choices_str}\nCorrect Answer: {answer}"
                else:
                    prompt_text = f"[{subject} / {question_type}]\n{question_text}\nCorrect Answer: {answer}"

                # Build image paths relative to the samples JSONL location
                image_paths = []
                if "image_paths" in item:
                    for img_path in item["image_paths"]:
                        # Convert from "images/xxx.png" to "Subject/images/xxx.png"
                        full_path = f"{subject}/{img_path}"
                        image_paths.append(full_path)

                record = {
                    "task_id": item.get("id", ""),
                    "subject": subject,
                    "question_type": question_type,
                    "prompt": prompt_text,
                    "solution": response,
                    "ground_truth": answer,
                    "eval_score": 1.0 if correct else 0.0,
                    "eval_valid": True,
                }
                if image_paths:
                    record["image_paths"] = image_paths
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    # Update total_samples in scores (may differ from collect_results if result.json is incomplete)
    if total_samples > 0:
        scores["total_samples"] = total_samples
    scores["detail_path"] = detail_name

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"MMMU evaluation complete!")
    log.info(f"Overall Accuracy: {scores['average']['accuracy']:.4f}")
    log.info(f"Total Samples: {scores['total_samples']}")
    log.info(f"Results saved to: {score_file}")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
