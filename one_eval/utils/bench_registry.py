from __future__ import annotations
import json
import os
from typing import List, Dict, Optional
from one_eval.logger import get_logger

log = get_logger("BenchRegistry")


class BenchRegistry:
    """
    本地 Benchmark 注册中心：
    - 支持用户指定 benchmark (specific_benches)
    - 支持根据 domain 与 bench.task_type 匹配的自动推荐
    """

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            log.error(f"Bench config file not found: {config_path}")
            self.data = {}
            return

        with open(config_path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Dict] = json.load(f)

        # 统一 bench_name 全部转小写，方便匹配
        self.lower_map = {name.lower(): name for name in self.data.keys()}

        log.info(f"[BenchRegistry] Loaded {len(self.data)} benches from {config_path}")

    # ---------------------------------------------------------
    # API
    # ---------------------------------------------------------

    def _match_bench_by_name_or_alias(self, query: str) -> Optional[str]:
        """支持大小写不敏感匹配 + alias 匹配"""
        query = query.lower()

        # bench_name 匹配
        if query in self.lower_map:
            return self.lower_map[query]

        # alias 匹配
        for bench_name, info in self.data.items():
            aliases = info.get("aliases", []) or []
            for alias in aliases:
                if query in alias.lower():
                    return bench_name

        return None

    def search(
        self,
        specific_benches: List[str],
        domain: List[str]
    ) -> List[Dict]:
        """
        先匹配用户指定的 benchmark，再进行 domain 匹配推荐。
        """
        results = []

        domain_set = {d.lower() for d in domain}

        # --------------------------
        # Step 1. 处理用户指定 benchmark
        # --------------------------
        for name in specific_benches:
            matched = self._match_bench_by_name_or_alias(name)
            if matched:
                bench = {
                    "bench_name": matched,
                    "source": "specified",
                    **self.data[matched]
                }
                results.append(bench)
            else:
                log.warning(f"[BenchRegistry] User specified bench '{name}' not found.")

        # --------------------------
        # Step 2. 根据 domain 匹配推荐 benchmark
        # --------------------------
        for bench_name, info in self.data.items():

            # 已经被用户指定过的避免重复
            if any(r["bench_name"] == bench_name for r in results):
                continue

            task_type = info.get("task_type")
            if not task_type:
                continue

            # 如果 bench.task_type 与 domain 有交集 → 命中
            if isinstance(task_type, str):
                task_type_list = [task_type]
            else:
                task_type_list = task_type

            task_set = {t.lower() for t in task_type_list}

            if domain_set & task_set:  # 有交集
                bench = {
                    "bench_name": bench_name,
                    "source": "local_recommend",
                    **info
                }
                results.append(bench)

        return results
