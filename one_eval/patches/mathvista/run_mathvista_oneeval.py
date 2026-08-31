#!/usr/bin/env python
"""
MathVista evaluation bridge script for One-Eval.
Integrates MathVista's 3-stage pipeline with One-Eval's external_repo system.

Model config is read from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add repo root to path for imports (script runs from evaluation/ dir)
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from openai import OpenAI
from rich.logging import RichHandler

from build_query import create_query_data
from extract_answer import extract_answer
from calculate_score import normalize_extracted_answer
from models.gpt import GPT_Model
from utilities import read_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)


def parse_args():
    parser = argparse.ArgumentParser(description="MathVista evaluation for One-Eval")

    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Model name (e.g., gpt-4o)')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples to evaluate (-1 for all)')

    # MathVista parameters
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_type', type=str, default='solution',
                        choices=['solution', 'code'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=1024)

    return parser.parse_args()


def load_mathvista_data(dataset_name, split, max_samples=-1):
    """Load MathVista dataset from HuggingFace"""
    logging.info(f"Loading dataset {dataset_name}, split {split}...")
    data_list = load_dataset(dataset_name, split=split)
    data = {item['pid']: item for item in data_list}

    if max_samples > 0:
        pids = list(data.keys())[:max_samples]
        data = {pid: data[pid] for pid in pids}
        logging.info(f"Limited to {len(data)} samples")
    else:
        logging.info(f"Loaded {len(data)} samples")

    return data


def stage1_generate_responses(model, data, query_data, output_file, output_dir):
    """Stage 1: Generate responses from model"""
    logging.info("Stage 1: Generating responses...")

    # Create images directory for storing sample images
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if os.path.exists(output_file):
        logging.info(f"Loading existing responses from {output_file}")
        results = read_json(output_file)

    total = len(data)
    for idx, (pid, problem) in enumerate(data.items(), 1):
        if pid in results and results[pid].get('response'):
            logging.info(f"[{idx}/{total}] Skipping {pid} (already exists)")
            continue

        logging.info(f"[{idx}/{total}] Processing {pid}")

        query_text = query_data.get(pid, problem.get('query', ''))
        image = problem.get('decoded_image', None)

        # Save image to disk for HTML report
        image_paths = []
        if image is not None:
            img_filename = f"{pid}.png"
            img_path = images_dir / img_filename
            try:
                image.save(img_path, format="PNG")
                image_paths.append(f"images/{img_filename}")
            except Exception as e:
                logging.warning(f"Failed to save image {img_filename}: {e}")

        try:
            response = model.get_response(user_prompt=query_text, decoded_image=image)

            results[pid] = {
                'pid': pid,
                'question': problem.get('question', ''),
                'query': query_text,
                'response': response,
                'prediction': response,
                'answer': problem.get('answer', ''),
                'question_type': problem.get('question_type', ''),
                'answer_type': problem.get('answer_type', ''),
                'precision': problem.get('precision', 0),
                'choices': problem.get('choices', []),
                'task': problem.get('task', ''),
                'skills': problem.get('skills', []),
                'image_paths': image_paths,
            }

            if idx % 10 == 0:
                save_json(results, output_file)
                logging.info(f"Saved {len(results)} responses")

        except Exception as e:
            logging.error(f"Error processing {pid}: {e}")
            results[pid] = {
                'pid': pid,
                'response': '',
                'prediction': '',
                'error': str(e)
            }

    save_json(results, output_file)
    logging.info(f"Stage 1 complete: {len(results)} responses saved")
    return results


def stage2_extract_answers(model, results, output_file):
    """Stage 2: Extract short answers from responses"""
    logging.info("Stage 2: Extracting answers...")

    extracted = {}
    for pid, item in results.items():
        response = item.get('prediction', '')

        problem = {
            'pid': pid,
            'question_type': item.get('question_type', ''),
            'answer_type': item.get('answer_type', ''),
            'choices': item.get('choices', []),
            'query': item.get('query', '')
        }

        extraction = extract_answer(model, response, problem, quick_extract=True)
        extracted[pid] = {**item, 'extraction': extraction}

    save_json(extracted, output_file)
    logging.info(f"Stage 2 complete: {len(extracted)} answers extracted")
    return extracted


def stage3_calculate_scores(extracted, data, output_dir):
    """Stage 3: Calculate accuracy scores (0-1 scale) and build per-sample detail."""
    logging.info("Stage 3: Calculating scores...")

    from collections import defaultdict

    tot = defaultdict(int)
    hit = defaultdict(int)
    per_sample = []

    for pid, item in extracted.items():
        extraction = item.get('extraction', '')
        problem = data[pid]

        answer = problem.get('answer', '')
        question_type = problem.get('question_type', '')
        answer_type = problem.get('answer_type', '')
        precision = problem.get('precision', 0)
        choices = problem.get('choices', [])
        meta = problem.get('metadata') or {}
        task = meta.get('task') or problem.get('task', 'Overall')
        skills = meta.get('skills') or problem.get('skills', [])

        try:
            normalized = normalize_extracted_answer(
                extraction, choices, question_type, answer_type, precision
            )
        except Exception:
            normalized = None

        is_correct = (normalized == answer) if normalized is not None else False

        tot['Overall'] += 1
        tot[task] += 1
        for skill in skills:
            tot[skill] += 1

        if is_correct:
            hit['Overall'] += 1
            hit[task] += 1
            for skill in skills:
                hit[skill] += 1

        per_sample.append({
            "task_id": pid,
            "task": task,
            "question_type": question_type,
            "answer_type": answer_type,
            "prompt": item.get('query', ''),
            "solution": item.get('response', ''),
            "extraction": str(extraction),
            "ground_truth": str(answer),
            "eval_score": 1.0 if is_correct else 0.0,
            "eval_valid": True,
            "image_paths": item.get('image_paths', []),
        })

    scores = {}
    for key in tot:
        scores[key] = {
            'total': tot[key],
            'correct': hit[key],
            'accuracy': hit[key] / tot[key] if tot[key] > 0 else 0.0
        }

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Write per-sample detail JSONL
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = os.path.join(output_dir, detail_name)
    with open(detail_path, "w", encoding="utf-8") as fout:
        for record in per_sample:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info(f"Per-sample detail written to: {detail_path} ({len(per_sample)} samples)")

    result = {
        'average': {'accuracy': scores['Overall']['accuracy']},
        'total_samples': scores['Overall']['total'],
        'detail_path': detail_name,
        'by_task': {k: v for k, v in scores.items() if k != 'Overall'}
    }

    score_file = os.path.join(output_dir, f"scores_{timestamp}.json")
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logging.info(f"Stage 3 complete: Overall accuracy = {result['average']['accuracy']:.4f}")
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    response_file = os.path.join(args.output_dir, 'responses.json')
    extracted_file = os.path.join(args.output_dir, 'extracted.json')

    # Load data
    data = load_mathvista_data('AI4Math/MathVista', 'testmini', args.max_samples)

    # Build query data using MathVista's original function
    logging.info("Creating query data...")
    from types import SimpleNamespace
    query_args = SimpleNamespace(
        shot_num=args.shot_num,
        shot_type=args.shot_type,
        use_caption=False,
        use_ocr=False,
    )
    query_data = create_query_data(data, {}, {}, query_args)

    # Create model client
    client = OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
    model = GPT_Model(
        client=client,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    # gpt-4o supports vision but the class only enables it for "vision" in name
    model.use_image = True

    # Stage 1: Generate responses
    results = stage1_generate_responses(model, data, query_data, response_file, args.output_dir)

    # Stage 2: Extract answers
    extracted = stage2_extract_answers(model, results, extracted_file)

    # Stage 3: Calculate scores + build per-sample detail
    scores = stage3_calculate_scores(extracted, data, args.output_dir)

    logging.info("=" * 60)
    logging.info(f"MathVista evaluation complete!")
    logging.info(f"Overall Accuracy: {scores['average']['accuracy']:.4f}")
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
