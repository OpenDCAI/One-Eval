"""Helpers for registering uploaded, local dataflow benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def collect_key_paths(value: Any, prefix: str = "") -> List[str]:
    """Return dotted paths for fields in a JSON-compatible object."""
    paths: List[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(collect_key_paths(child, path))
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        paths.extend(collect_key_paths(value[0], prefix))
    return paths


def inspect_jsonl(path: Path, *, preview_rows: int = 5) -> Dict[str, Any]:
    """Validate a JSONL file and return row count plus representative keys.

    The evaluation engine consumes one JSON object per non-empty line, so we
    validate the complete file before it becomes a registered benchmark.
    """
    row_count = 0
    keys: List[str] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} must contain a JSON object")

                row_count += 1
                if row_count <= preview_rows:
                    for key in collect_key_paths(row):
                        if key not in keys:
                            keys.append(key)
    except UnicodeDecodeError as exc:
        raise ValueError("file must be UTF-8 encoded JSONL") from exc

    if row_count == 0:
        raise ValueError("file does not contain any JSON objects")

    return {"num_rows": row_count, "keys": keys}


def build_local_bench_entry(
    *,
    bench_name: str,
    description: str,
    category: str,
    bench_type: str,
    dataset_path: Path,
    source_name: str,
    num_rows: int,
    keys: List[str],
    eval_type: Optional[str] = None,
    key_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the gallery schema for a local uploaded dataflow benchmark."""
    mapping = key_mapping if isinstance(key_mapping, dict) else {}
    relative_source = f"local://custom_benches/{source_name}"
    structure_features = {key: {"dtype": "unknown"} for key in keys}

    return {
        "bench_name": bench_name,
        "bench_kind": "dataflow",
        "bench_table_exist": True,
        "bench_source_url": relative_source,
        "bench_dataflow_eval_type": eval_type,
        "bench_prompt_template": None,
        "bench_keys": keys,
        "dataset_cache": str(dataset_path),
        "download_status": "success",
        "meta": {
            "bench_name": bench_name,
            "source": "user_upload",
            "aliases": [bench_name],
            "category": category,
            "tags": [bench_type, "local"],
            "description": description,
            "description_zh": description,
            "local_path": str(dataset_path),
            "key_mapping": mapping,
            "hf_meta": {
                "bench_name": bench_name,
                "hf_repo": None,
                "card_text": "",
                "tags": [bench_type, "local"],
                "exists_on_hf": False,
            },
            # A synthetic structure lets the normal preparation graph proceed
            # without probing HuggingFace for a local-only dataset.
            "structure": {
                "repo_id": bench_name,
                "revision": "local-upload",
                "subsets": [
                    {
                        "subset": "local",
                        "splits": [{"name": "uploaded", "num_examples": num_rows}],
                        "features": structure_features,
                    }
                ],
                "ok": True,
                "error": None,
            },
            "download_config": {
                "config": "local",
                "split": "uploaded",
                "reason": "local upload",
            },
        },
    }
