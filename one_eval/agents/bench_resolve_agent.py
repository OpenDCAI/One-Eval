from __future__ import annotations
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from huggingface_hub import DatasetCard, list_datasets

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger

log = get_logger("BenchResolveAgent")


class BenchResolveAgent(CustomAgent):

    _gallery_index: Optional[Dict[str, Dict]] = None

    @property
    def role_name(self) -> str:
        return "BenchResolveAgent"

    # 这里不需要 prompt，整个 Agent 不调用 LLM
    @property
    def system_prompt_template_name(self) -> str:
        return ""

    @property
    def task_prompt_template_name(self) -> str:
        return ""

    def _load_gallery_index(self) -> Dict[str, Dict]:
        """加载 bench_gallery.json，构建 bench_name -> 完整配置 的索引"""
        if BenchResolveAgent._gallery_index is not None:
            return BenchResolveAgent._gallery_index
        gallery_path = Path(__file__).parent.parent / "utils" / "bench_table" / "bench_gallery.json"
        if not gallery_path.exists():
            BenchResolveAgent._gallery_index = {}
            return {}
        with open(gallery_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        index: Dict[str, Dict] = {}
        for bench in data.get("benches", []):
            name = bench.get("bench_name", "")
            if name:
                index[name.lower()] = bench
                for alias in (bench.get("meta") or {}).get("aliases", []):
                    if isinstance(alias, str) and alias:
                        index[alias.lower()] = bench
        BenchResolveAgent._gallery_index = index
        return index

    def _lookup_gallery(self, bench_name: str) -> Optional[Dict]:
        """在 gallery 中查找 bench，返回完整配置或 None"""
        if not bench_name:
            return None
        index = self._load_gallery_index()
        return index.get(bench_name.lower())

    def _extract_query_info(self, state: NodeState) -> Dict[str, Any]:
        """只拿 domain / specific_benches，给后续逻辑用"""
        q = {}
        if isinstance(state.result, dict):
            q = state.result.get("QueryUnderstandAgent", {}) or {}

        return {
            "domain": q.get("domain") or [],
            "specific_benches": q.get("specific_benches") or [],
        }

    def _resolve_hf_bench(self, bench_name: str) -> Optional[Dict[str, Any]]:
        """
        尝试根据 bench_name 从 HuggingFace 拉取数据集信息：
        1) 直接当 repo_id 调用 DatasetCard.load
        2) 若失败，使用 list_datasets(search=bench_name) 搜索，优先匹配 id 末尾等于 bench_name 的
        找不到则返回 None
        """
        if not isinstance(bench_name, str):
            return None

        bench_name = bench_name.strip()
        if not bench_name:
            return None

        # 1) 直接当作 repo_id 尝试
        try:
            card = DatasetCard.load(bench_name)
            data = getattr(card, "data", {}) or {}
            return {
                "bench_name": bench_name,
                "hf_repo": bench_name,
                "card_text": card.text or "",
                "tags": data.get("tags", []),
                "exists_on_hf": True,
            }
        except Exception:
            pass

        # 2) 用搜索 + 后缀精确匹配
        try:
            candidates = list(list_datasets(search=bench_name, limit=10))
        except Exception as e:
            log.warning(f"list_datasets(search={bench_name}) 失败: {e}")
            return None

        bench_lower = bench_name.lower()
        chosen_id = None
        for d in candidates:
            ds_id = d.id
            short_id = ds_id.split("/")[-1].lower()
            if short_id == bench_lower:
                chosen_id = ds_id
                break

        if not chosen_id and candidates:
            chosen_id = candidates[0].id

        if not chosen_id:
            return None

        try:
            card = DatasetCard.load(chosen_id)
            data = getattr(card, "data", {}) or {}
            return {
                "bench_name": bench_name,          # 用户原始名称
                "hf_repo": chosen_id,              # 真正的 HF repo_id
                "card_text": card.text or "",
                "tags": data.get("tags", []),
                "exists_on_hf": True,
            }
        except Exception as e:
            log.warning(f"DatasetCard.load({chosen_id}) 失败: {e}")
            return {
                "bench_name": bench_name,
                "hf_repo": chosen_id,
                "card_text": "",
                "tags": [],
                "exists_on_hf": False,
            }

    def _search_hf_by_query(self, query: str, limit: int = 10, exclude_bench_names: set = None) -> List[Dict[str, Any]]:
        """
        根据查询关键词主动在 HF 上搜索 benchmark 数据集。
        exclude_bench_names: 精确的 bench_name 集合（本地已选中的），用于去重。
        """
        if not query or not query.strip():
            return []
        exclude_bench_names = exclude_bench_names or set()

        try:
            candidates = list(list_datasets(search=query, limit=limit))
        except Exception as e:
            log.warning(f"list_datasets(search={query}) 失败: {e}")
            return []

        results = []
        for d in candidates:
            ds_id = d.id
            # 排除精确匹配本地已选的 bench_name
            if ds_id.lower() in exclude_bench_names:
                continue
            short_name = ds_id.split("/")[-1].lower()
            if short_name in exclude_bench_names:
                continue
            # 排除 gallery 中映射到已选本地 bench 的结果
            gallery_entry = self._lookup_gallery(ds_id) or self._lookup_gallery(short_name)
            if gallery_entry:
                gallery_name = gallery_entry.get("bench_name", "").lower()
                if gallery_name in exclude_bench_names:
                    continue

            # 尝试获取 card 信息
            card_text = ""
            tags = []
            try:
                card = DatasetCard.load(ds_id)
                card_text = (card.text or "")[:500]
                data = getattr(card, "data", {}) or {}
                tags = data.get("tags", [])
            except Exception:
                pass

            results.append({
                "bench_name": ds_id,
                "hf_repo": ds_id,
                "card_text": card_text,
                "tags": tags,
                "exists_on_hf": True,
            })

        return results

    async def run(self, state: NodeState) -> NodeState:
        # 如果前一个 Agent 判定可以直接跳过，则不做任何事
        if state.temp_data.get("skip_resolve"):
            log.info("skip_resolve=True（hf_count=0），直接返回")
            return state

        info = self._extract_query_info(state)
        specific_benches: List[str] = info["specific_benches"] or []
        hf_count = int(getattr(state, 'hf_count', 2) or 2)

        # 第一个 Agent 推荐的 bench 名称列表
        bench_names: List[str] = state.temp_data.get("bench_names_suggested", []) or []
        bench_descs: Dict[str, str] = state.temp_data.get("bench_descs", {}) or {}
        hf_search_query: str = state.temp_data.get("hf_search_query", "") or ""

        # 已有的本地 bench_info（来自 BenchNameSuggestNode）
        bench_info: Dict[str, Dict[str, Any]] = getattr(state, "bench_info", {}) or {}
        existing_keys = set(bench_info.keys())

        existing_benches: List[BenchInfo] = getattr(state, "benches", []) or []

        # 去重集合：包含本地已选中的 bench_name 及其 gallery 映射名
        local_bench_names: List[str] = state.temp_data.get("local_bench_names", []) or []
        local_selected_set = {n.lower() for n in local_bench_names}
        # 也加入 gallery 中的 bench_name（处理 repo_id 和 gallery 短名不一致的情况）
        for n in list(local_selected_set):
            gallery_entry = self._lookup_gallery(n) or self._lookup_gallery(n.split("/")[-1])
            if gallery_entry:
                local_selected_set.add(gallery_entry.get("bench_name", "").lower())
        # 也加入 existing_benches 中的 bench_name
        for b in existing_benches:
            local_selected_set.add(b.bench_name.lower())

        # ================ Step 1: 解析用户指定的 specific_benches ================
        names_to_resolve: List[str] = []

        for name in bench_names:
            if not name:
                continue
            if name in existing_keys:
                if "desc" not in bench_info[name] and name in bench_descs:
                    bench_info[name]["desc"] = bench_descs[name]
                continue
            names_to_resolve.append(name)

        for name in specific_benches:
            if not name:
                continue
            if name in existing_keys:
                continue
            if name not in names_to_resolve:
                names_to_resolve.append(name)

        log.info(f"需要在 HF 上解析的名称: {names_to_resolve}")

        hf_resolved: List[Dict[str, Any]] = []

        for name in names_to_resolve:
            resolved = self._resolve_hf_bench(name)
            if not resolved:
                continue

            hf_resolved.append(resolved)

            repo_id = resolved.get("hf_repo") or name
            if repo_id not in bench_info:
                bench_info[repo_id] = {
                    "bench_name": repo_id,
                    "source": "hf_resolve",
                    "aliases": [name],
                    "hf_meta": resolved,
                    "desc": bench_descs.get(name, ""),
                }
                local_selected_set.add(repo_id.lower())

        # ================ Step 2: 主动搜索 HF 填充配额 ================
        hf_added_count = len(hf_resolved)
        remaining_quota = hf_count - hf_added_count

        if remaining_quota > 0 and hf_search_query:
            log.info(f"HF 配额剩余 {remaining_quota}，主动搜索: {hf_search_query}")
            search_results = self._search_hf_by_query(
                hf_search_query,
                limit=remaining_quota + 10,  # 多搜一些以防去重后不够
                exclude_bench_names=local_selected_set,
            )

            # 已添加的 HF bench_name 集合，防止本轮内重复
            hf_added_ids = set()

            for resolved in search_results:
                if hf_added_count >= hf_count:
                    break

                repo_id = resolved.get("hf_repo") or resolved.get("bench_name")
                if repo_id.lower() in hf_added_ids:
                    continue

                hf_resolved.append(resolved)
                bench_info[repo_id] = {
                    "bench_name": repo_id,
                    "source": "hf_search",
                    "aliases": [],
                    "hf_meta": resolved,
                    "desc": resolved.get("card_text", "")[:200],
                }
                hf_added_ids.add(repo_id.lower())
                hf_added_count += 1

            log.info(f"HF 主动搜索补充了 {len(hf_added_ids)} 个结果")

        # ================ Step 3: 构建 BenchInfo 写回 state ================
        # 对 HF 搜索结果，检查是否在 gallery 中有完整配置
        existing_bench_names = set()
        for b in existing_benches:
            existing_bench_names.add(b.bench_name.lower())
            # 也加入 repo_id 的 short_name，处理 "Salesforce/CRMArena" vs "crmarena" 的情况
            existing_bench_names.add(b.bench_name.split("/")[-1].lower())
            # 也加入 gallery 映射名
            ge = self._lookup_gallery(b.bench_name) or self._lookup_gallery(b.bench_name.split("/")[-1])
            if ge:
                existing_bench_names.add(ge.get("bench_name", "").lower())
        hf_benches: List[BenchInfo] = []
        hf_added_bench_names: set = set()  # 防止多个 HF 结果映射到同一个 gallery bench

        for repo_id, binfo in bench_info.items():
            if binfo.get("source") not in ("hf_resolve", "hf_search"):
                continue  # 本地的已在 existing_benches 里

            # 跳过与本地已有 bench 或已添加的 HF bench 重复的
            if repo_id.lower() in existing_bench_names or repo_id.lower() in hf_added_bench_names:
                continue
            if repo_id.split("/")[-1].lower() in existing_bench_names or repo_id.split("/")[-1].lower() in hf_added_bench_names:
                continue

            hf_meta = (binfo.get("hf_meta", {}) or {})
            hf_repo = hf_meta.get("hf_repo") or repo_id

            # 检查 gallery 中是否有这个 bench 的完整配置
            gallery_entry = self._lookup_gallery(repo_id) or self._lookup_gallery(
                repo_id.split("/")[-1]
            )

            if gallery_entry:
                # gallery 映射后的 bench_name 也需要去重检查
                gallery_bench_name = gallery_entry.get('bench_name', '')
                if gallery_bench_name.lower() in existing_bench_names or gallery_bench_name.lower() in hf_added_bench_names:
                    log.info(f"[去重] HF 结果 {repo_id} 映射到 gallery {gallery_bench_name}，与已有结果重复，跳过")
                    continue
                # HF 搜到的 bench 在 gallery 中有完整配置 → 使用 gallery 配置，标记为 hf_gallery
                bench = BenchInfo(
                    bench_name=gallery_entry['bench_name'],
                    bench_table_exist=gallery_entry.get('bench_table_exist', True),
                    bench_source_url=gallery_entry.get('bench_source_url'),
                    bench_dataflow_eval_type=gallery_entry.get('bench_dataflow_eval_type'),
                    bench_prompt_template=gallery_entry.get('bench_prompt_template'),
                    bench_keys=gallery_entry.get('bench_keys', []),
                    meta={
                        **gallery_entry.get('meta', {}),
                        'from_gallery': True,
                        'source': 'hf_gallery',  # HF 搜到 + gallery 有配置
                    },
                )
                log.info(f"[HF+Gallery] {repo_id} → gallery 有完整配置（eval_type={bench.bench_dataflow_eval_type}）")
            else:
                # 纯 HF 结果，没有 gallery 配置
                bench = BenchInfo(
                    bench_name=repo_id,
                    bench_table_exist=False,
                    bench_source_url=hf_repo,
                    meta={
                        **{k: v for k, v in binfo.items() if k != '_gallery_entry'},
                        "from_gallery": False,
                        "source": binfo.get("source", "hf_search"),
                    },
                )

            hf_benches.append(bench)
            # 记录已添加的 bench_name，防止后续 HF 结果映射到同一个 bench
            hf_added_bench_names.add(bench.bench_name.lower())
            hf_added_bench_names.add(repo_id.lower())

        state.benches = existing_benches + hf_benches
        state.bench_info = bench_info

        state.agent_results["BenchResolveAgent"] = {
            "bench_names_input": bench_names,
            "specific_benches": specific_benches,
            "hf_resolved": hf_resolved,
            "hf_count": hf_count,
            "hf_search_query": hf_search_query,
        }

        log.info(
            f"最终 bench 数量: {len(state.benches)}"
            f"（本地: {len(existing_benches)}, HF: {len(hf_benches)}）"
        )
        return state
