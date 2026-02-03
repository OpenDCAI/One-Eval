"""
Workflow for generating bench_gallery.json

This workflow iterates through all benchmarks in benchData.ts,
runs the metadata collection pipeline (without actual model evaluation),
and saves the results to bench_gallery.json format.

Features:
- Checkpoint/resume: Saves progress after each batch, can resume from interruption
- Parallel processing: Processes multiple benchmarks concurrently within batches
- Retry mechanism: Retries failed benchmarks at the end

Usage:
    python -m one_eval.graph.workflow_bench_gallery
    python -m one_eval.graph.workflow_bench_gallery --resume  # Resume from checkpoint
    python -m one_eval.graph.workflow_bench_gallery --parallel 3  # 3 concurrent tasks
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import copy

from one_eval.core.state import NodeState, BenchInfo
from one_eval.nodes.dataset_structure_node import DatasetStructureNode
from one_eval.nodes.bench_config_recommend_node import BenchConfigRecommendNode
from one_eval.nodes.download_node import DownloadNode
from one_eval.nodes.dataset_keys_node import DatasetKeysNode
from one_eval.nodes.bench_task_infer_node import BenchTaskInferNode
from one_eval.logger import get_logger

log = get_logger("WorkflowBenchGallery")


# ============================================================================
# Checkpoint Management
# ============================================================================

@dataclass
class CheckpointData:
    """Checkpoint data for resumable processing."""
    processed_bench_names: List[str] = field(default_factory=list)
    failed_bench_names: List[str] = field(default_factory=list)
    gallery_data: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = 0.0
    total_benchmarks: int = 0


def load_checkpoint(checkpoint_path: Path) -> Optional[CheckpointData]:
    """Load checkpoint from file if exists."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return CheckpointData(
            processed_bench_names=data.get("processed_bench_names", []),
            failed_bench_names=data.get("failed_bench_names", []),
            gallery_data=data.get("gallery_data", []),
            last_updated=data.get("last_updated", 0.0),
            total_benchmarks=data.get("total_benchmarks", 0),
        )
    except Exception as e:
        log.warning(f"Failed to load checkpoint: {e}")
        return None


def save_checkpoint(checkpoint_path: Path, checkpoint: CheckpointData) -> None:
    """Save checkpoint to file."""
    checkpoint.last_updated = time.time()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "processed_bench_names": checkpoint.processed_bench_names,
        "failed_bench_names": checkpoint.failed_bench_names,
        "gallery_data": checkpoint.gallery_data,
        "last_updated": checkpoint.last_updated,
        "total_benchmarks": checkpoint.total_benchmarks,
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"Checkpoint saved: {len(checkpoint.processed_bench_names)}/{checkpoint.total_benchmarks} processed")


# ============================================================================
# TypeScript Parser
# ============================================================================

def parse_bench_data_ts(ts_file_path: str) -> List[Dict[str, Any]]:
    """
    Parse benchData.ts and extract benchmark information.

    Uses a state-machine approach to handle nested objects properly.

    Args:
        ts_file_path: Path to benchData.ts file

    Returns:
        List of benchmark dictionaries
    """
    with open(ts_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the array content between BENCH_DATA: BenchItem[] = [ ... ];
    match = re.search(r'BENCH_DATA:\s*BenchItem\[\]\s*=\s*\[(.*)\];', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find BENCH_DATA array in file")

    array_content = match.group(1).strip()

    # Parse using brace counting to extract top-level objects
    benchmarks = []
    objects = extract_top_level_objects(array_content)

    for obj_str in objects:
        try:
            # Convert TS object to JSON-compatible string
            json_str = convert_ts_object_to_json(obj_str)
            bench = json.loads(json_str)
            benchmarks.append(bench)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse object: {e}\nObject: {obj_str[:100]}...")
            continue

    return benchmarks


def extract_top_level_objects(content: str) -> List[str]:
    """
    Extract top-level objects from an array content string.
    Handles nested braces properly.
    """
    objects = []
    depth = 0
    current_obj = []
    in_string = False
    escape_next = False

    for char in content:
        if escape_next:
            current_obj.append(char)
            escape_next = False
            continue

        if char == '\\' and in_string:
            current_obj.append(char)
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            current_obj.append(char)
            continue

        if in_string:
            current_obj.append(char)
            continue

        if char == '{':
            depth += 1
            current_obj.append(char)
        elif char == '}':
            depth -= 1
            current_obj.append(char)
            if depth == 0:
                obj_str = ''.join(current_obj).strip()
                if obj_str:
                    objects.append(obj_str)
                current_obj = []
        elif depth > 0:
            current_obj.append(char)

    return objects


def convert_ts_object_to_json(ts_obj: str) -> str:
    """
    Convert TypeScript object notation to valid JSON.
    Handles unquoted keys and trailing commas.
    """
    result = ts_obj

    # Replace unquoted keys with quoted keys
    # Match word characters followed by colon, but not inside strings
    # This regex looks for: start of line or comma/brace + whitespace + key + colon
    result = re.sub(r'([\{,]\s*)(\w+)(\s*):', r'\1"\2"\3:', result)

    # Remove trailing commas before } or ]
    result = re.sub(r',(\s*[}\]])', r'\1', result)

    return result


# ============================================================================
# Data Conversion
# ============================================================================

def convert_to_bench_info(bench_dict: Dict[str, Any]) -> BenchInfo:
    """
    Convert a parsed benchmark dictionary to BenchInfo dataclass.

    Args:
        bench_dict: Dictionary from benchData.ts

    Returns:
        BenchInfo instance
    """
    meta = bench_dict.get("meta", {})

    # Extract HuggingFace repo from datasetUrl
    dataset_url = meta.get("datasetUrl", "")
    hf_repo = ""
    if "huggingface.co/datasets/" in dataset_url:
        hf_repo = dataset_url.split("huggingface.co/datasets/")[-1]

    bench_info = BenchInfo(
        bench_name=bench_dict.get("id", ""),
        bench_table_exist=True,  # Since it's from benchData.ts
        bench_source_url=dataset_url,
        bench_dataflow_eval_type=None,  # Will be inferred
        bench_prompt_template=None,
        bench_keys=meta.get("datasetKeys", []),
        meta={
            "bench_name": bench_dict.get("id", ""),
            "source": "bench_table",
            "aliases": [bench_dict.get("id", ""), bench_dict.get("name", "")],
            "category": meta.get("category"),
            "tags": meta.get("tags", []),
            "description": meta.get("description", ""),
            "hf_meta": {
                "bench_name": bench_dict.get("id", ""),
                "hf_repo": hf_repo,
                "card_text": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "exists_on_hf": bool(hf_repo),
            }
        }
    )

    return bench_info


def bench_info_to_gallery_format(bench: BenchInfo) -> Dict[str, Any]:
    """
    Convert BenchInfo to bench_gallery.json format.

    Args:
        bench: BenchInfo instance

    Returns:
        Dictionary in bench_gallery.json format
    """
    return {
        "bench_name": bench.bench_name,
        "bench_table_exist": bench.bench_table_exist,
        "bench_source_url": bench.bench_source_url,
        "bench_dataflow_eval_type": bench.bench_dataflow_eval_type,
        "bench_prompt_template": bench.bench_prompt_template,
        "bench_keys": bench.bench_keys,
        "meta": bench.meta,
    }


# ============================================================================
# Pipeline Processing
# ============================================================================

async def process_single_bench_structure(
    bench: BenchInfo,
    node_structure: DatasetStructureNode,
) -> BenchInfo:
    """Process a single benchmark through DatasetStructureNode."""
    state = NodeState(user_query="bench_gallery_generation", benches=[bench])
    state = await node_structure.run(state)
    return state.benches[0]


async def process_single_bench_config(
    bench: BenchInfo,
    node_config: BenchConfigRecommendNode,
) -> BenchInfo:
    """Process a single benchmark through BenchConfigRecommendNode."""
    state = NodeState(user_query="bench_gallery_generation", benches=[bench])
    state = await node_config.run(state)
    return state.benches[0]


async def process_single_bench_download(
    bench: BenchInfo,
    node_download: DownloadNode,
) -> BenchInfo:
    """Process a single benchmark through DownloadNode."""
    state = NodeState(user_query="bench_gallery_generation", benches=[bench])
    state = await node_download.run(state)
    return state.benches[0]


async def process_single_bench_keys(
    bench: BenchInfo,
    node_keys: DatasetKeysNode,
) -> BenchInfo:
    """Process a single benchmark through DatasetKeysNode."""
    state = NodeState(user_query="bench_gallery_generation", benches=[bench])
    state = await node_keys.run(state)
    return state.benches[0]


async def process_single_bench_infer(
    bench: BenchInfo,
    node_infer: BenchTaskInferNode,
) -> BenchInfo:
    """Process a single benchmark through BenchTaskInferNode."""
    state = NodeState(user_query="bench_gallery_generation", benches=[bench])
    state = await node_infer.run(state)
    return state.benches[0]


async def process_bench_parallel(
    bench: BenchInfo,
    skip_download: bool,
    semaphore: asyncio.Semaphore,
) -> Tuple[BenchInfo, bool, Optional[str]]:
    """
    Process a single benchmark through the entire pipeline.

    Returns:
        Tuple of (processed_bench, success, error_message)
    """
    async with semaphore:
        # Initialize nodes (per-task to avoid state conflicts)
        node_structure = DatasetStructureNode()
        node_config = BenchConfigRecommendNode()
        node_download = DownloadNode()
        node_keys = DatasetKeysNode()
        node_infer = BenchTaskInferNode()

        try:
            log.info(f"[{bench.bench_name}] Starting pipeline...")

            # Step 1: Get dataset structure
            bench = await process_single_bench_structure(bench, node_structure)

            # Step 2: Recommend config
            bench = await process_single_bench_config(bench, node_config)

            if not skip_download:
                # Step 3: Download dataset
                bench = await process_single_bench_download(bench, node_download)

                # Step 4: Extract keys
                bench = await process_single_bench_keys(bench, node_keys)

            # Step 5: Infer task type
            bench = await process_single_bench_infer(bench, node_infer)

            success = bench.bench_dataflow_eval_type is not None
            if success:
                log.info(f"[{bench.bench_name}] Pipeline completed successfully")
            else:
                log.warning(f"[{bench.bench_name}] Pipeline completed but eval_type is None")

            return bench, success, None

        except Exception as e:
            error_msg = str(e)
            log.error(f"[{bench.bench_name}] Pipeline failed: {error_msg}")
            if not bench.meta:
                bench.meta = {}
            bench.meta["processing_error"] = error_msg
            return bench, False, error_msg


async def run_metadata_pipeline(
    benches: List[BenchInfo],
    skip_download: bool = False,
    batch_size: int = 5,
    parallel: int = 1,
    checkpoint_path: Optional[Path] = None,
    checkpoint: Optional[CheckpointData] = None,
) -> Tuple[List[BenchInfo], List[str]]:
    """
    Run the metadata collection pipeline on a list of benchmarks.

    Args:
        benches: List of BenchInfo to process
        skip_download: If True, skip download and use existing keys
        batch_size: Number of benchmarks to process in each batch
        parallel: Number of concurrent tasks within each batch
        checkpoint_path: Path to save checkpoints
        checkpoint: Existing checkpoint data to resume from

    Returns:
        Tuple of (processed_benches, failed_bench_names)
    """
    if checkpoint is None:
        checkpoint = CheckpointData(total_benchmarks=len(benches))

    # Filter out already processed benchmarks
    pending_benches = [
        b for b in benches
        if b.bench_name not in checkpoint.processed_bench_names
    ]

    if not pending_benches:
        log.info("All benchmarks already processed")
        # Reconstruct from checkpoint
        return [], checkpoint.failed_bench_names

    log.info(f"Processing {len(pending_benches)} pending benchmarks ({len(checkpoint.processed_bench_names)} already done)")

    # Semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(parallel)

    processed_benches = []
    failed_names = list(checkpoint.failed_bench_names)

    # Process in batches
    total = len(pending_benches)
    for i in range(0, total, batch_size):
        batch = pending_benches[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        log.info(f"Processing batch {batch_num}/{total_batches}, benchmarks {i + 1}-{min(i + batch_size, total)} of {total}")

        # Process batch in parallel
        tasks = [
            process_bench_parallel(bench, skip_download, semaphore)
            for bench in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for bench, result in zip(batch, results):
            if isinstance(result, Exception):
                log.error(f"[{bench.bench_name}] Exception: {result}")
                if not bench.meta:
                    bench.meta = {}
                bench.meta["processing_error"] = str(result)
                failed_names.append(bench.bench_name)
                processed_benches.append(bench)
            else:
                processed_bench, success, _error = result
                processed_benches.append(processed_bench)
                if not success:
                    if processed_bench.bench_name not in failed_names:
                        failed_names.append(processed_bench.bench_name)

            # Update checkpoint
            checkpoint.processed_bench_names.append(bench.bench_name)
            checkpoint.gallery_data.append(bench_info_to_gallery_format(
                processed_bench if not isinstance(result, Exception) else bench
            ))

        checkpoint.failed_bench_names = failed_names

        # Save checkpoint after each batch
        if checkpoint_path:
            save_checkpoint(checkpoint_path, checkpoint)

    return processed_benches, failed_names


async def retry_failed_benchmarks(
    failed_names: List[str],
    all_benches: List[BenchInfo],
    skip_download: bool,
    max_retries: int = 2,
    parallel: int = 1,
) -> List[BenchInfo]:
    """
    Retry processing for failed benchmarks.

    Args:
        failed_names: List of benchmark names that failed
        all_benches: All benchmark infos
        skip_download: Whether to skip download
        max_retries: Maximum retry attempts
        parallel: Number of concurrent tasks

    Returns:
        List of successfully retried benchmarks
    """
    if not failed_names:
        return []

    log.info(f"Retrying {len(failed_names)} failed benchmarks (max {max_retries} attempts)...")

    # Find the failed benchmarks
    failed_benches = [b for b in all_benches if b.bench_name in failed_names]

    semaphore = asyncio.Semaphore(parallel)
    recovered = []

    for attempt in range(max_retries):
        if not failed_benches:
            break

        log.info(f"Retry attempt {attempt + 1}/{max_retries} for {len(failed_benches)} benchmarks")

        tasks = [
            process_bench_parallel(copy.deepcopy(bench), skip_download, semaphore)
            for bench in failed_benches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        still_failed = []
        for bench, result in zip(failed_benches, results):
            if isinstance(result, Exception):
                still_failed.append(bench)
            else:
                processed_bench, success, _ = result
                if success:
                    recovered.append(processed_bench)
                    log.info(f"[{bench.bench_name}] Recovered on retry")
                else:
                    still_failed.append(bench)

        failed_benches = still_failed

        if failed_benches and attempt < max_retries - 1:
            await asyncio.sleep(2)  # Brief delay between retries

    if failed_benches:
        log.warning(f"{len(failed_benches)} benchmarks still failed after retries: {[b.bench_name for b in failed_benches]}")

    return recovered


# ============================================================================
# Main Entry Point
# ============================================================================

async def generate_bench_gallery(
    bench_data_ts_path: str,
    output_path: str,
    skip_download: bool = False,
    batch_size: int = 5,
    parallel: int = 1,
    limit: Optional[int] = None,
    resume: bool = False,
    retry: bool = True,
    max_retries: int = 2,
) -> None:
    """
    Main function to generate bench_gallery.json from benchData.ts.

    Args:
        bench_data_ts_path: Path to benchData.ts
        output_path: Path to output bench_gallery.json
        skip_download: If True, skip downloading datasets
        batch_size: Number of benchmarks to process in each batch
        parallel: Number of concurrent tasks within each batch
        limit: If set, only process first N benchmarks (for testing)
        resume: If True, resume from checkpoint
        retry: If True, retry failed benchmarks at the end
        max_retries: Maximum retry attempts for failed benchmarks
    """
    output_file = Path(output_path)
    checkpoint_path = output_file.parent / f".{output_file.stem}_checkpoint.json"

    log.info(f"Reading benchmarks from {bench_data_ts_path}")

    # Parse benchData.ts
    bench_dicts = parse_bench_data_ts(bench_data_ts_path)
    log.info(f"Found {len(bench_dicts)} benchmarks")

    if limit:
        bench_dicts = bench_dicts[:limit]
        log.info(f"Limited to first {limit} benchmarks")

    # Convert to BenchInfo
    benches = [convert_to_bench_info(b) for b in bench_dicts]

    # Load checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            log.info(f"Resuming from checkpoint: {len(checkpoint.processed_bench_names)}/{checkpoint.total_benchmarks} already processed")
        else:
            log.info("No checkpoint found, starting fresh")

    if checkpoint is None:
        checkpoint = CheckpointData(total_benchmarks=len(benches))

    # Run pipeline
    log.info(f"Starting metadata collection pipeline (parallel={parallel}, batch_size={batch_size})...")
    processed_benches, failed_names = await run_metadata_pipeline(
        benches,
        skip_download=skip_download,
        batch_size=batch_size,
        parallel=parallel,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )

    # Retry failed benchmarks
    if retry and failed_names:
        recovered = await retry_failed_benchmarks(
            failed_names,
            benches,
            skip_download,
            max_retries=max_retries,
            parallel=parallel,
        )

        # Update checkpoint with recovered benchmarks
        for bench in recovered:
            # Find and replace in gallery_data
            for i, item in enumerate(checkpoint.gallery_data):
                if item["bench_name"] == bench.bench_name:
                    checkpoint.gallery_data[i] = bench_info_to_gallery_format(bench)
                    break

            # Remove from failed list
            if bench.bench_name in checkpoint.failed_bench_names:
                checkpoint.failed_bench_names.remove(bench.bench_name)

    # Build final gallery from checkpoint data
    gallery = {"benches": checkpoint.gallery_data}

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gallery, f, ensure_ascii=False, indent=2)

    log.info(f"Saved bench_gallery.json to {output_path}")

    # Print summary
    success_count = sum(1 for item in checkpoint.gallery_data if item.get("bench_dataflow_eval_type"))
    total_count = len(checkpoint.gallery_data)
    failed_count = len(checkpoint.failed_bench_names)

    log.info(f"="*50)
    log.info(f"Summary:")
    log.info(f"  Total benchmarks: {total_count}")
    log.info(f"  Successfully processed: {success_count}")
    log.info(f"  Failed: {failed_count}")
    if checkpoint.failed_bench_names:
        log.info(f"  Failed benchmarks: {checkpoint.failed_bench_names}")
    log.info(f"="*50)

    # Clean up checkpoint on successful completion
    if failed_count == 0 and checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Checkpoint file cleaned up (all benchmarks successful)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate bench_gallery.json from benchData.ts")
    parser.add_argument(
        "--input",
        default="one_eval/utils/bench_table/benchData.ts",
        help="Path to benchData.ts"
    )
    parser.add_argument(
        "--output",
        default="one_eval/bench_gallery.json",
        help="Path to output bench_gallery.json"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets (use existing keys from benchData.ts)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of benchmarks to process in each batch"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent tasks within each batch (default: 1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N benchmarks (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic retry of failed benchmarks"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts for failed benchmarks (default: 2)"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / args.input
    output_path = project_root / args.output

    asyncio.run(generate_bench_gallery(
        bench_data_ts_path=str(input_path),
        output_path=str(output_path),
        skip_download=args.skip_download,
        batch_size=args.batch_size,
        parallel=args.parallel,
        limit=args.limit,
        resume=args.resume,
        retry=not args.no_retry,
        max_retries=args.max_retries,
    ))
