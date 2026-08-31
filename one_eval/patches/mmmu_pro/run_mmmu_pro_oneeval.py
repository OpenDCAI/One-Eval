#!/usr/bin/env python
"""
MMMU-Pro evaluation bridge script for One-Eval.
Integrates MMMU-Pro's inference + scoring pipeline with One-Eval's external_repo system.

MMMU-Pro has 3 configs: "vision", "standard (10 options)", "standard (4 options)"
And 2 modes: "direct" (answer directly) / "cot" (chain-of-thought)

Flow:
  1. Load MMMU/MMMU_Pro dataset from HuggingFace
  2. Run inference via OpenAI-compatible API (multimodal, base64 images)
  3. Parse "Answer: X" from model responses
  4. Aggregate accuracy by subdomain → domain → overall
  5. Output scores_*.json

Model config is read from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES, ONEEVAL_SPLIT
"""

import argparse
import ast
import base64
import io
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mmmu_pro_oneeval")

# Prompt templates (from mmmu-pro/prompts.yaml)
PROMPTS = {
    "direct": {
        "vision": "Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
        "standard": "Answer with the option letter from the given choices directly.",
    },
    "cot": {
        "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
        "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    },
}

# Domain groupings (from MMMU-Pro evaluate.py)
DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science", "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture", "Architecture_and_Engineering", "Computer_Science",
        "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="MMMU-Pro evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--setting", type=str, default="vision",
                        choices=["vision", "standard (10 options)", "standard (4 options)"])
    parser.add_argument("--mode", type=str, default="direct",
                        choices=["direct", "cot"])
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    return parser.parse_args()


def encode_image(image) -> str:
    """Encode a PIL image to base64."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_options(options_str):
    """Parse options string into a list."""
    try:
        return ast.literal_eval(options_str)
    except Exception:
        return []



def build_messages(sample, setting, mode):
    """Build API messages for a single sample.

    Aligned with official infer_gpt.py message structure:
    - Vision: user[prompt_text, image]
    - Standard: user[clean_text, img1, img2, ...] (images appended in placeholder order)
    """
    prompt_key = "vision" if setting == "vision" else "standard"
    prompt_suffix = PROMPTS[mode][prompt_key]

    content = []

    if setting == "vision":
        # Vision mode: instruction text first, then the single composite image
        content.append({"type": "text", "text": prompt_suffix})
        img = sample.get("image")
        if img is not None:
            b64 = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
    else:
        # Standard mode: build full text (question + options + prompt),
        # extract <image N> order, replace placeholders with <image>,
        # then append images in that order after the text.
        # (Aligned with official replace_images_tokens + make_interleave_content)
        question = sample["question"]
        options = parse_options(sample.get("options", "[]"))
        option_str = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )
        full_text = f"{question}\n{option_str}\n{prompt_suffix}"

        # Extract image indices in the order they appear in full text
        image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", full_text)]
        # Replace placeholders with generic <image> marker
        clean_text = re.sub(r"<image\s+\d+>", "<image>", full_text)

        content.append({"type": "text", "text": clean_text})

        # Append images in placeholder occurrence order
        for idx in image_order:
            img = sample.get(f"image_{idx}")
            if img is not None:
                b64 = encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

    return [{"role": "user", "content": content}]


def call_model(client, model_name, messages, max_tokens, max_retries=3):
    """Call OpenAI-compatible API."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
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


def parse_answer(response, all_choices, index2ans):
    """Parse answer letter from response.

    Aligned with MMMU-Pro official evaluate.py parse_multi_choice_response.
    """
    if not response:
        return random.choice(all_choices) if all_choices else ""

    # Step 1: Find last "Answer:" pattern (official uses rfind)
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching_options = [opt for opt in all_choices if opt in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    # Step 2+: Standard fallback chain
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return random.choice(all_choices) if all_choices else ""
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        return candidates[np.argmax(start_indexes)]
    else:
        return candidates[0]


def process_sample(client, model_name, sample, setting, mode, max_tokens, images_dir=None):
    """Process a single sample: inference + parse answer."""
    messages = build_messages(sample, setting, mode)
    response = call_model(client, model_name, messages, max_tokens)

    options = parse_options(sample.get("options", "[]"))
    all_choices = [chr(65 + i) for i in range(len(options))]
    index2ans = {chr(65 + i): opt for i, opt in enumerate(options)}

    pred = parse_answer(response, all_choices, index2ans)
    is_correct = pred == sample.get("answer", "")

    # Save images to disk and record paths
    image_paths = []
    if images_dir:
        sample_id = sample["id"]
        if setting == "vision":
            # Vision mode: single 'image' field
            img = sample.get("image")
            if img is not None:
                img_filename = f"{sample_id}_1.png"
                img_path = images_dir / img_filename
                try:
                    img.save(img_path, format="PNG")
                    image_paths.append(f"images/{img_filename}")
                except Exception as e:
                    log.warning(f"Failed to save image {img_filename}: {e}")
        else:
            # Standard mode: image_1 through image_7
            for i in range(1, 8):
                img = sample.get(f"image_{i}")
                if img is not None:
                    img_filename = f"{sample_id}_{i}.png"
                    img_path = images_dir / img_filename
                    try:
                        img.save(img_path, format="PNG")
                        image_paths.append(f"images/{img_filename}")
                    except Exception as e:
                        log.warning(f"Failed to save image {img_filename}: {e}")

    return {
        "id": sample["id"],
        "question": sample.get("question", ""),  # Add question text for HTML report
        "options": sample.get("options", "[]"),  # Add options for HTML report
        "response": response,
        "pred": pred,
        "answer": sample.get("answer", ""),
        "is_correct": is_correct,
        "subject": sample.get("subject", "unknown"),
        "image_paths": image_paths,
    }


def aggregate_scores(results):
    """Aggregate results by subject → domain → overall (0-1 scale)."""
    # Per-subject
    subject_stats = {}
    for r in results:
        subj = r["subject"]
        if subj not in subject_stats:
            subject_stats[subj] = {"correct": 0, "total": 0}
        subject_stats[subj]["total"] += 1
        if r["is_correct"]:
            subject_stats[subj]["correct"] += 1

    by_subject = {}
    for subj, stats in subject_stats.items():
        by_subject[subj] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "count": stats["total"],
        }

    # Per-domain
    by_domain = {}
    for domain, subjects in DOMAIN_CAT2SUB_CAT.items():
        domain_correct = 0
        domain_total = 0
        for subj in subjects:
            if subj in subject_stats:
                domain_correct += subject_stats[subj]["correct"]
                domain_total += subject_stats[subj]["total"]
        if domain_total > 0:
            by_domain[domain] = {
                "accuracy": domain_correct / domain_total,
                "count": domain_total,
            }

    # Overall
    total_correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    overall_acc = total_correct / total if total > 0 else 0.0

    return {
        "average": {"accuracy": overall_acc},
        "total_samples": total,
        "by_domain": by_domain,
        "by_subject": by_subject,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory for storing sample images
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
    )

    # Load dataset
    log.info(f"Loading MMMU/MMMU_Pro, config='{args.setting}', split='test'")
    ds = load_dataset("MMMU/MMMU_Pro", args.setting, split="test")
    samples = list(ds)

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        log.info(f"Limited to {len(samples)} samples")
    else:
        log.info(f"Loaded {len(samples)} samples")

    # Load existing results for resumption
    setting_safe = args.setting.replace(" ", "_").replace("(", "").replace(")", "")
    jsonl_file = output_dir / f"{args.model_name}_{setting_safe}_{args.mode}.jsonl"
    existing = {}
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing[item["id"]] = item
        log.info(f"Loaded {len(existing)} existing results for resumption")

    # Run inference
    log.info(f"=== Inference: {args.model_name}, setting={args.setting}, mode={args.mode} ===")
    results = []

    def do_sample(sample):
        if sample["id"] in existing:
            existing_item = existing[sample["id"]]
            # Ensure image_paths exists for existing results
            if "image_paths" not in existing_item:
                sample_id = sample["id"]
                image_paths = []
                if args.setting == "vision":
                    img = sample.get("image")
                    if img is not None:
                        img_filename = f"{sample_id}_1.png"
                        img_path = images_dir / img_filename
                        if not img_path.exists():
                            try:
                                img.save(img_path, format="PNG")
                            except Exception as e:
                                log.warning(f"Failed to save image {img_filename}: {e}")
                        image_paths.append(f"images/{img_filename}")
                else:
                    for i in range(1, 8):
                        img = sample.get(f"image_{i}")
                        if img is not None:
                            img_filename = f"{sample_id}_{i}.png"
                            img_path = images_dir / img_filename
                            if not img_path.exists():
                                try:
                                    img.save(img_path, format="PNG")
                                except Exception as e:
                                    log.warning(f"Failed to save image {img_filename}: {e}")
                            image_paths.append(f"images/{img_filename}")
                existing_item["image_paths"] = image_paths
            return existing_item
        return process_sample(
            client, args.model_name, sample, args.setting, args.mode, args.max_tokens, images_dir
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(do_sample, s): s for s in samples}
        done_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done_count += 1
                if done_count % 50 == 0:
                    log.info(f"  Progress: {done_count}/{len(samples)}")
            except Exception as e:
                sample = futures[future]
                log.error(f"  Failed {sample.get('id', '?')}: {e}")
                results.append({
                    "id": sample.get("id", ""),
                    "response": "",
                    "pred": "",
                    "answer": sample.get("answer", ""),
                    "is_correct": False,
                    "subject": sample.get("subject", "unknown"),
                })

    # Save JSONL
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(results)} results to {jsonl_file}")

    # Score
    log.info("=== Scoring ===")
    scores = aggregate_scores(results)

    # Build per-sample detail JSONL for HTML report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name
    total_samples = 0

    with open(detail_path, "w", encoding="utf-8") as fout:
        for r in results:
            options = parse_options(r.get("options", "[]"))
            # Build a readable prompt with question and options
            subject = r.get("subject", "unknown")
            answer = r.get("answer", "")
            pred = r.get("pred", "")
            question = r.get("question", "")

            # Format options if available
            options_text = ""
            if options:
                options_text = "\nOptions:\n" + "\n".join(
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
                )

            prompt_text = f"[{subject} / {args.setting}]\n{question}{options_text}\nPredicted: {pred} | Correct Answer: {answer}"

            record = {
                "task_id": r.get("id", ""),
                "subject": subject,
                "prompt": prompt_text,
                "solution": r.get("response", ""),
                "ground_truth": answer,
                "predicted": pred,
                "eval_score": 1.0 if r.get("is_correct") else 0.0,
                "eval_valid": True,
            }
            # Add image_paths if present
            if "image_paths" in r and r["image_paths"]:
                record["image_paths"] = r["image_paths"]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_samples += 1

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    scores["total_samples"] = total_samples
    scores["detail_path"] = detail_name

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"MMMU-Pro evaluation complete!")
    log.info(f"Setting: {args.setting} | Mode: {args.mode}")
    log.info(f"Overall Accuracy: {scores['average']['accuracy']:.4f}")
    log.info(f"Total Samples: {scores['total_samples']}")
    log.info(f"Results saved to: {score_file}")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
