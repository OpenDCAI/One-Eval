from __future__ import annotations

import asyncio
import os
import shutil
import inspect
import traceback
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger

log = get_logger("DownloadNode")


def _safe_dirname(name: str) -> str:
    # "openai/gsm8k" -> "openai__gsm8k"
    return name.replace("/", "__").replace("\\", "__").strip()


def _get_hf_repo(bench: BenchInfo) -> Optional[str]:
    meta = bench.meta or {}
    hf_meta = (meta.get("hf_meta") or {}) if isinstance(meta, dict) else {}
    repo = hf_meta.get("hf_repo")
    if isinstance(repo, str) and repo.strip():
        return repo.strip()
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dir_nonempty(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def _download_one_hf_repo(
    hf_repo: str,
    target_dir: Path,
) -> Tuple[bool, str]:
    """
    Returns: (ok, msg)
    """
    try:
        _ensure_dir(target_dir)

        hf_endpoint = os.getenv("HF_ENDPOINT", "(not set)")
        hf_offline = os.getenv("HF_HUB_OFFLINE", "(not set)")
        http_proxy = os.getenv("HTTP_PROXY", "(not set)")
        https_proxy = os.getenv("HTTPS_PROXY", "(not set)")
        all_proxy = os.getenv("ALL_PROXY", "(not set)")
        no_proxy = os.getenv("NO_PROXY", "(not set)")
        requests_ca_bundle = os.getenv("REQUESTS_CA_BUNDLE", "(not set)")
        ssl_cert_file = os.getenv("SSL_CERT_FILE", "(not set)")

        log.info(
            f"[DownloadNode] start snapshot_download repo={hf_repo} target_dir={target_dir} "
            f"HF_ENDPOINT={hf_endpoint} HF_HUB_OFFLINE={hf_offline} "
            f"HTTP_PROXY={http_proxy} HTTPS_PROXY={https_proxy} ALL_PROXY={all_proxy} NO_PROXY={no_proxy} "
            f"REQUESTS_CA_BUNDLE={requests_ca_bundle} SSL_CERT_FILE={ssl_cert_file}"
        )

        # 直接把 dataset repo 拉到目标目录（不写入全局 HF cache 的那套层级目录）
        kwargs = {
            "repo_id": hf_repo,
            "repo_type": "dataset",
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,  # 防止跨机器/跨盘符软链问题
            "ignore_patterns": ["*.h5", "*.ckpt"],  # 可按需删掉；避免误拉很大的无关文件
        }

        # 按需显式传入 endpoint（若 huggingface_hub 版本支持该参数）
        try:
            sig = inspect.signature(snapshot_download)
            if (
                "endpoint" in sig.parameters
                and isinstance(hf_endpoint, str)
                and hf_endpoint
                and hf_endpoint != "(not set)"
            ):
                kwargs["endpoint"] = hf_endpoint
        except Exception:
            pass

        snapshot_download(**kwargs)
        return True, "success"
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return False, err


class DownloadNode(BaseNode):
    """
    Step2-Node1: DownloadNode
    - 从 state.benches 读取 meta.hf_meta.hf_repo
    - 下载到 bench.dataset_cache（若为空则 ./cache）
    - 每个 bench 最多重试 3 次
    - 不 raise，失败写 bench.download_status 和 bench.meta['download_error']
    """

    def __init__(self, max_retries: int = 3):
        self.name = "DownloadNode"
        self.logger = log
        self.max_retries = max_retries

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DownloadNode] state.benches 为空，跳过下载。")
            return state

        # 默认 cache 根目录：当前工作目录下 ./cache
        default_cache_root = Path(os.getcwd()) / "cache"
        _ensure_dir(default_cache_root)

        downloaded_count = 0
        with tqdm(total=len(benches), desc="Downloading benches", unit="bench") as pbar:
            for bench in benches:
                try:
                    bench.download_status = "pending"

                    hf_repo = _get_hf_repo(bench)
                    if not hf_repo:
                        bench.download_status = "failed"
                        (bench.meta or {}).update({"download_error": "missing hf_meta.hf_repo"})
                        self.logger.error(
                            f"[DownloadNode] {bench.bench_name} 缺少 meta.hf_meta.hf_repo，无法下载。"
                        )
                        continue

                    cache_root = Path(bench.dataset_cache).expanduser().resolve() if bench.dataset_cache else default_cache_root
                    _ensure_dir(cache_root)

                    bench_dir = cache_root / _safe_dirname(hf_repo)
                    bench.dataset_cache = str(cache_root)  # 统一把根目录写回去（你也可以改成 bench_dir）

                    # 已经下载过就直接跳过
                    if _dir_nonempty(bench_dir):
                        bench.download_status = "success"
                        (bench.meta or {}).pop("download_error", None)
                        self.logger.info(f"[DownloadNode] {bench.bench_name} 已存在缓存：{bench_dir}")
                        downloaded_count += 1
                        pbar.n = downloaded_count
                        pbar.refresh()
                        continue

                    last_err = ""
                    for attempt in range(1, self.max_retries + 1):
                        ok, msg = _download_one_hf_repo(hf_repo, bench_dir)
                        if ok:
                            bench.download_status = "success"
                            (bench.meta or {}).pop("download_error", None)
                            self.logger.success(
                                f"[DownloadNode] 下载成功: {bench.bench_name} <- {hf_repo} -> {bench_dir}"
                            )
                            downloaded_count += 1
                            pbar.n = downloaded_count
                            pbar.refresh()
                            break

                        last_err = msg
                        self.logger.error(
                            f"[DownloadNode] 下载失败(第{attempt}/{self.max_retries}次): {bench.bench_name} <- {hf_repo} | err={msg}"
                        )

                        # 清理可能的半成品，避免下次误判 non-empty
                        try:
                            if bench_dir.exists():
                                shutil.rmtree(bench_dir, ignore_errors=True)
                        except Exception:
                            pass

                    if bench.download_status != "success":
                        bench.download_status = "failed"
                        (bench.meta or {}).update({"download_error": last_err})
                except Exception as e:
                    # 单个 bench 的异常不影响整体流程
                    bench.download_status = "failed"
                    (bench.meta or {}).update({"download_error": str(e)})
                    self.logger.error(f"[DownloadNode] 处理 bench={bench.bench_name} 时异常: {e}")

        return state
