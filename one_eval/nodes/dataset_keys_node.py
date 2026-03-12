from __future__ import annotations

import json
from pathlib import Path

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("DatasetKeysNode")


class DatasetKeysNode(BaseNode):
    """
    Step3-Node1: DatasetKeysNode
    - 读取 bench.dataset_cache 指定的 JSONL 文件
    - 提取第一行数据的 keys
    - 保存到 bench.bench_keys
    """

    def __init__(self):
        self.name = "DatasetKeysNode"
        self.logger = log

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DatasetKeysNode] state.benches 为空，跳过。")
            return state

        for bench in benches:
            if not bench.dataset_cache:
                self.logger.warning(f"[{bench.bench_name}] dataset_cache 为空，跳过 key 提取")
                continue

            cache_path = Path(bench.dataset_cache)
            if not cache_path.exists():
                self.logger.warning(f"[{bench.bench_name}] 文件不存在: {cache_path}")
                continue

            try:
                # 读取前 5 行，提取 keys + 预览数据
                with cache_path.open("r", encoding="utf-8") as f:
                    preview_rows = []
                    keys = None
                    for _ in range(5):
                        line = f.readline()
                        if not line:
                            break
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            if keys is None:
                                keys = list(row.keys())
                            preview_rows.append(row)
                    if not preview_rows:
                        self.logger.warning(f"[{bench.bench_name}] 文件为空")
                        continue

                    bench.bench_keys = keys or []
                    if bench.meta is None:
                        bench.meta = {}
                    if isinstance(bench.meta, dict):
                        bench.meta["preview_data"] = preview_rows
                        bench.meta["local_path"] = str(cache_path)
                    self.logger.info(f"[{bench.bench_name}] 提取 keys: {bench.bench_keys}，预览行数: {len(preview_rows)}")

            except Exception as e:
                self.logger.error(f"[{bench.bench_name}] 读取异常: {e}")

        return state
