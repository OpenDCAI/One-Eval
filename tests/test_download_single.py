from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import snapshot_download


def safe_dirname(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").strip()


def download_one(
    hf_repo: str,
    cache_root: Path,
) -> Tuple[bool, str]:
    target_dir = cache_root / safe_dirname(hf_repo)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Trying to download dataset: {hf_repo} ===")
    print(f"Target dir: {target_dir}")
    print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT', '(not set)')}")
    print(f"HF_HUB_OFFLINE: {os.getenv('HF_HUB_OFFLINE', '(not set)')}")

    try:
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.h5", "*.ckpt"],
        )
        print(f"[OK] Download success: {hf_repo}")
        return True, "success"
    except Exception as e:
        print(f"[FAIL] Download failed: {hf_repo}")
        print(f"Error: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Test HuggingFace dataset download (same behavior as DownloadNode)."
    )
    parser.add_argument(
        "repos",
        nargs="*",
        help="HF repo ids to test, e.g. openai/gsm8k truthful_qa",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="./cache_test",
        help="Directory to store test downloads (default: ./cache_test)",
    )

    args = parser.parse_args()

    # 默认测试几个你当前报错的数据集
    default_repos = [
        "openai/gsm8k",
        "HuggingFaceH4/MATH-500",
        "truthful_qa",
        "hellaswag",
    ]
    repos: List[str] = args.repos or default_repos

    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Cache root: {cache_root}")
    print(f"Repos to test: {repos}")

    all_ok = True
    for repo in repos:
        ok, _ = download_one(repo, cache_root)
        all_ok = all_ok and ok

    if all_ok:
        print("\n=== All downloads succeeded ===")
    else:
        print("\n=== Some downloads failed ===")


if __name__ == "__main__":
    main()