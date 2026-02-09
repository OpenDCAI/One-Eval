from __future__ import annotations

import os
from pathlib import Path

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.logger import get_logger
from one_eval.toolkits.hf_download_tool import HFDownloadTool

log = get_logger("DownloadNode")


class DownloadNode(BaseNode):
    """
    Step2-Node3: DownloadNode
    - 读取 bench.meta["download_config"]
    - 调用 HFDownloadTool 下载并转换
    - 更新 bench.dataset_cache 指向生成的 jsonl 文件
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

        default_cache_root = Path(__file__).resolve().parents[2] / "cache"
        if not default_cache_root.exists():
            default_cache_root.mkdir(parents=True, exist_ok=True)

        tool = HFDownloadTool(cache_dir=str(default_cache_root))

        for bench in benches:
            try:
                # 0. 预检查：如果 bench 已经标记为下载成功且文件存在，直接跳过
                if bench.download_status == "success" and bench.dataset_cache:
                     p = Path(bench.dataset_cache)
                     if p.exists() and p.stat().st_size > 0:
                         self.logger.info(f"[{bench.bench_name}] 已标记为成功且文件存在: {p}")
                         continue

                bench.download_status = "pending"

                # 1. 获取推荐配置
                dl_config = bench.meta.get("download_config")
                hf_repo = None
                if bench.meta and "hf_meta" in bench.meta:
                    hf_repo = bench.meta["hf_meta"].get("hf_repo")

                if not hf_repo:
                    hf_repo = bench.bench_name

                if not dl_config:
                    self.logger.warning(
                        f"[{hf_repo}] 缺少 download_config，尝试使用默认配置 (default/test)"
                    )
                    dl_config = {"config": "default", "split": "test"}

                config_name = dl_config.get("config", "default")
                split_name = dl_config.get("split", "test")

                # === Config 自动修正逻辑 ===
                # 如果 config 是 default，但结构信息里没有 default，则尝试自动修正
                structure = bench.meta.get("structure", {})
                if structure and structure.get("ok"):
                    subsets = structure.get("subsets", [])
                    available_configs = [s.get("subset") for s in subsets if s.get("subset")]
                    
                    if available_configs and config_name not in available_configs:
                        self.logger.info(f"[{hf_repo}] Config '{config_name}' 不在可用列表中 {available_configs}，尝试自动修正...")
                        if "main" in available_configs:
                            config_name = "main"
                        else:
                            config_name = available_configs[0]
                        self.logger.info(f"[{hf_repo}] 修正后的 Config: {config_name}")

                # 2. 确定输出路径
                # 结构: cache/repo__config__split.jsonl
                safe_repo = hf_repo.replace("/", "__")
                filename = f"{safe_repo}__{config_name}__{split_name}.jsonl"
                output_path = default_cache_root / filename

                if output_path.exists() and output_path.stat().st_size > 0:
                    self.logger.info(f"[{hf_repo}] 文件已存在: {output_path}")
                    bench.dataset_cache = str(output_path)
                    bench.download_status = "success"
                    continue
                
                safe_bench = str(getattr(bench, "bench_name", "") or "").replace("/", "__")
                if safe_bench:
                    exact = list(default_cache_root.glob(f"*__{safe_bench}__{config_name}__{split_name}.jsonl"))
                    candidates = exact or list(default_cache_root.glob(f"*__{safe_bench}__{config_name}__*.jsonl")) or list(default_cache_root.glob(f"*__{safe_bench}__*.jsonl"))
                    candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
                    if candidates:
                        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
                        self.logger.info(f"[{hf_repo}] 复用已存在缓存文件: {chosen}")
                        bench.dataset_cache = str(chosen)
                        bench.download_status = "success"
                        if bench.meta and isinstance(bench.meta, dict):
                            bench.meta.pop("download_error", None)
                        continue

                # 3. 下载
                ok = False
                last_err = ""
                for i in range(self.max_retries):
                    res = tool.download_and_convert(
                        repo_id=hf_repo,
                        config_name=config_name,
                        split=split_name,
                        output_path=output_path,
                    )

                    if res.get("ok"):
                        ok = True
                        break
                    else:
                        last_err = res.get("error")
                        self.logger.warning(
                            f"[{hf_repo}] 下载重试 {i+1}/{self.max_retries}: {last_err}"
                        )

                if ok:
                    bench.dataset_cache = str(output_path)
                    bench.download_status = "success"
                    self.logger.info(f"[{hf_repo}] 下载成功 -> {output_path}")
                else:
                    bench.download_status = "failed"
                    if not bench.meta:
                        bench.meta = {}
                    bench.meta["download_error"] = last_err
                    self.logger.error(f"[{hf_repo}] 下载失败: {last_err}")

            except Exception as e:
                bench.download_status = "failed"
                if not bench.meta:
                    bench.meta = {}
                bench.meta["download_error"] = str(e)
                self.logger.error(f"[{bench.bench_name}] 异常: {e}")

        return state
