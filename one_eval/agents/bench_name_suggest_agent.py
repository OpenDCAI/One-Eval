from __future__ import annotations
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.utils.bench_registry import BenchRegistry
from one_eval.logger import get_logger

log = get_logger("BenchNameSuggestAgent")


class BenchNameSuggestAgent(CustomAgent):

    @property
    def role_name(self) -> str:
        return "BenchNameSuggestAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "bench_search.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "bench_search.hf_query"

    def _extract_query_info(self, state: NodeState) -> Dict[str, Any]:
        """
        从 QueryUnderstandAgent 的输出中抽取必要信息：
        state.result 约定为:
        {
          "QueryUnderstandAgent": {...}
        }
        """
        q = {}
        if isinstance(state.result, dict):
            q = state.result.get("QueryUnderstandAgent", {}) or {}

        return {
            "domain": q.get("domain") or [],
            "specific_benches": q.get("specific_benches") or [],
            "user_query": getattr(state, "user_query", ""),
        }

    async def run(self, state: NodeState) -> NodeState:
        # log.info("[BenchNameSuggestAgent] 执行开始")

        info = self._extract_query_info(state)
        domain: List[str] = info["domain"]
        specific_benches: List[str] = info["specific_benches"]
        user_query: str = info["user_query"]

        human_feedback: str = info.get("human_feedback", "")
        prev_benches: List[str] = info.get("benches", [])

        # ================ Step 1: 本地 BenchRegistry 搜索 ================
        from pathlib import Path
        config_path = Path(__file__).parents[1] / "utils" / "bench_table" / "bench_config.json"
        registry = BenchRegistry(str(config_path))
        local_matches = registry.search(
            specific_benches=specific_benches,
            domain=domain,
        )

        # log.warning(
        #     f"[BenchNameSuggestAgent] 检索关键词: specific_benches={specific_benches}, domain={domain}"
        # )
        log.info(f"本地匹配到 {len(local_matches)} 个 bench")

        bench_info: Dict[str, Dict[str, Any]] = {
            m["bench_name"]: m for m in local_matches
        }

        # 如果本地 >= 3，认为已经足够，不再调用 LLM 推荐
        if len(local_matches) >= 3:
            state.benches = [
                BenchInfo(
                    bench_name=name,
                    bench_table_exist=True,
                    bench_source_url=None,
                    meta=bench_info[name],
                )
                for name in bench_info.keys()
            ]
            state.bench_info = bench_info
            state.agent_results["BenchNameSuggestAgent"] = {
                "local_matches": local_matches,
                "bench_names": [],
                "skip_resolve": True,
            }
            # 标记后续 BenchResolveAgent 可以直接跳过
            state.temp_data["skip_resolve"] = True
            log.info("本地 bench >= 3，跳过推荐")
            return state

        # ================ Step 2: 通过 LLM 推荐 benchmark 名称列表 ================
        # 强制使用最新 Prompt，防止 Registry 缓存
        sys_prompt = """
你是 One-Eval 系统中的 BenchSearchAgent（具体由 BenchNameSuggestAgent 实现）。
你的工作是根据用户的任务需求，推荐合适的 benchmark 名称列表。

你需要遵守以下要求：
1. 你只负责“给出 benchmark 名称”，具体下载与评测由后续模块完成。
2. 你必须优先考虑在学术界 / 工业界广泛使用的、公开的评测基准。
3. 输出形式必须是严格的 JSON，能够被 Python 的 json.loads 正确解析。
4. 不要输出任何解释性文字、注释、Markdown，仅输出 JSON。
5. 注意你不一定是第一次被调用，可能用户后续的需求回溯到了这个节点，因此当你发现这一点时请注意调整你的输出内容，不要轻易删除已经给用户推荐过的bench除非用户特别说明，最终的bench主要看用户新的需求。
"""
        task_prompt_tmpl = """
下面是与当前评测任务相关的信息。请你根据这些信息，给出“推荐的 benchmark 名称列表”以及每个 benchmark 的简短介绍。

你需要返回一个 JSON，格式必须严格为：

{{
  "bench_list": [
    {{
        "name": "gsm8k",
        "desc": "Grade School Math, 小学数学应用题，考察多步推理能力"
    }},
    {{
        "name": "HuggingFaceH4/MATH-500",
        "desc": "竞赛级数学题，难度较高"
    }}
  ]
}}

要求：
1. "bench_list" 是一个对象数组，每个对象包含 "name" (benchmark名称) 和 "desc" (简短介绍)。
2. "desc" 字段限制在 20 字以内，简要说明该 benchmark 的主要内容或特点，方便用户快速了解。
3. 如果你知道 HuggingFace 上的完整仓库名（例如 "openai/gsm8k"、"HuggingFaceH4/MATH-500"），优先使用完整仓库名。
4. 如果你不确定仓库前缀，可以只给出常用简称（例如 "gsm8k"、"mmlu"），后续系统会尝试匹配。
5. 不要包含与评测无关的数据集（例如纯预训练语料、无标注文本、通用聊天日志等）。
6. 不要输出除上述 JSON 以外的任何内容。

----------------
用户原始需求:
{user_query}

用户新增的需求（没有则是第一次调用该节点）：
{human_feedback}

之前我们已经调用过 BenchSearchNode 节点，推荐了以下 benchmark:(没有则为空)
{prev_benches}

任务领域:
{domain}

本地已经找到的 benchmark:
{local_benches}
"""
        task_prompt = task_prompt_tmpl.format(
            user_query=user_query,
            human_feedback=human_feedback,
            prev_benches=",".join(prev_benches),
            domain=",".join(domain),
            local_benches=",".join(bench_info.keys()),
        )

        llm = self.create_llm(state)
        resp = await llm.call(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task_prompt),
            ],
            bind_post_tools=False,
        )

        parsed = self.parse_result(resp.content)
        
        bench_names: List[str] = []
        bench_descs: Dict[str, str] = {}

        if isinstance(parsed, dict):
            # 优先尝试 bench_list (新格式)
            raw_objs = parsed.get("bench_list")
            if raw_objs and isinstance(raw_objs, list):
                for item in raw_objs:
                    if isinstance(item, dict):
                        name = str(item.get("name", "")).strip()
                        desc = str(item.get("desc", "")).strip()
                        if name and name.lower() not in {"none", "null", "-"}:
                            bench_names.append(name)
                            if desc:
                                bench_descs[name] = desc
            else:
                # 回退到 bench_names (旧格式)
                raw_list = parsed.get("bench_names") or []
                for x in raw_list:
                    if isinstance(x, (str, int, float)):
                        name = str(x).strip()
                        if name and name.lower() not in {"none", "null", "-"}:
                            bench_names.append(name)

        # 去重保持顺序
        bench_names = list(dict.fromkeys(bench_names))

        log.info(f"LLM 推荐 bench_names: {bench_names}")
        log.info(f"LLM 推荐 bench_descs keys: {list(bench_descs.keys())}")

        # 把本地匹配 + 推荐名称写入中间状态，方便后续 BenchResolveAgent 使用
        state.bench_info = bench_info
        state.temp_data["bench_names_suggested"] = bench_names
        state.temp_data["bench_descs"] = bench_descs

        state.agent_results["BenchNameSuggestAgent"] = {
            "local_matches": local_matches,
            "bench_names": bench_names,
            "skip_resolve": False,
        }

        return state
