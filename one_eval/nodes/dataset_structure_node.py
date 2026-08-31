from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Any

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.logger import get_logger
from one_eval.toolkits.hf_dataset_structure_tool import HFDatasetStructureTool

log = get_logger("DatasetStructureNode")


class DatasetStructureNode(BaseNode):
    """
    Step2-Node1: DatasetStructureNode
    - 遍历 state.benches
    - 调用 HFDatasetStructureTool.probe 获取数据集结构
    - 将结构信息保存到 bench.meta["structure"]
    """

    def __init__(self):
        self.name = "DatasetStructureNode"
        self.logger = log

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DatasetStructureNode] state.benches 为空，跳过。")
            return state

        tool = HFDatasetStructureTool()

        self.logger.info(f"[DatasetStructureNode] 开始解析 {len(benches)} 个数据集的结构...")

        for bench in benches:
            # Local uploaded benches already have a real dataset_cache and a
            # synthetic structure registered by the upload endpoint. Never
            # probe their name as a HuggingFace repo.
            if (
                bench.dataset_cache
                and Path(bench.dataset_cache).exists()
                and isinstance(bench.meta, dict)
                and bench.meta.get("source") == "user_upload"
            ):
                bench.download_status = "success"
                self.logger.info(f"[DatasetStructureNode] 使用本地上传文件: {bench.dataset_cache}")
                continue

            # 检查是否已经存在 structure 信息
            if bench.meta and bench.meta.get("structure") and bench.meta["structure"].get("ok"):
                self.logger.info(f"[DatasetStructureNode] 跳过 {bench.bench_name}，已存在结构信息")
                continue

            repo_id = None
            if bench.meta and "hf_meta" in bench.meta:

                repo_id = bench.meta["hf_meta"].get("hf_repo")

            if not repo_id:
                repo_id = bench.bench_name  # Fallback

            if not repo_id:
                self.logger.warning(f"[DatasetStructureNode] 无法获取 repo_id: {bench}")
                continue

            try:
                self.logger.info(f"正在探测: {repo_id}")
                # 同步调用
                structure = tool.probe(repo_id=repo_id)

                if structure.get("ok"):
                    if not bench.meta:
                        bench.meta = {}
                    bench.meta["structure"] = structure
                    self.logger.info(f"解析成功: {repo_id}")
                else:
                    self.logger.error(f"解析失败: {repo_id} error={structure.get('error')}")
                    if not bench.meta:
                        bench.meta = {}
                    bench.meta["structure_error"] = structure.get("error")

            except Exception as e:
                self.logger.error(f"探测异常: {repo_id} error={e}")
                if not bench.meta:
                    bench.meta = {}
                bench.meta["structure_error"] = str(e)

        return state
