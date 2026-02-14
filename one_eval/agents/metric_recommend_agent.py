from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger
from one_eval.metrics.dispatcher import metric_dispatcher as metric_registry
from one_eval.core.metric_registry import get_registered_metrics_meta

log = get_logger("MetricRecommendAgent")

class MetricRecommendAgent(CustomAgent):
    """
    Step 3 Agent: Metric推荐
    双轨制策略：
    1. Analyst Track: 未知 Benchmark 基于 Info (name、prompt_template) 调用 LLM 分析。
    2. Registry Track: 已知 Benchmark 直接查表，用于兜底策略。
    """

    @property
    def role_name(self) -> str:
        return "MetricRecommendAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "metric_recommend.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "metric_recommend.task"

    def _check_registry(self, bench: BenchInfo) -> Optional[List[Dict[str, Any]]]:
        """
        检查注册表是否有预定义的 Metric。
        """
        return metric_registry.get_metrics(bench.bench_name)
    
    def _normalize_metric_format(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化指标格式，确保包含必需字段。
        """
        name = metric.get("name") or metric.get("metric_name")
        if not name:
            raise ValueError(f"Metric 缺少必需字段 'name': {metric}")

        priority = metric.get("priority")
        if priority not in ["primary", "secondary", "diagnostic"]:
            if priority is not None:
                log.warning(f"Metric {name} 的 priority '{priority}' 不在标准值中，使用默认规则")
            priority = metric_registry.get_default_priority(name)

        desc_map = getattr(self, "_metric_desc_map", {})
        desc = desc_map.get(name, "")

        normalized = {
            "name": name,
            "priority": priority,
            "desc": desc,
        }

        if "args" in metric:
            normalized["args"] = metric["args"]
        elif "params" in metric:
            normalized["args"] = metric["params"]
        elif "k" in metric:
            normalized["args"] = {"k": metric["k"]}

        return normalized
    
    def _validate_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证并规范化指标列表。
        """
        validated = []
        seen = set()
        if not metrics:
            return []
        for metric in metrics:
            try:
                normalized = self._normalize_metric_format(metric)
                name = normalized.get("name")
                if not name or name in seen:
                    continue
                validated.append(normalized)
                seen.add(name)
            except (ValueError, KeyError) as e:
                log.warning(f"跳过无效的指标配置: {metric}, 错误: {e}")
                continue
        return validated

    def _infer_eval_type(self, bench: BenchInfo) -> Optional[str]:
        return (
            bench.bench_dataflow_eval_type
            or bench.meta.get("bench_dataflow_eval_type")
            or bench.meta.get("eval_type")
        )



    def _ensure_primary(self, metrics: List[Dict[str, Any]]) -> None:
        if not metrics:
            return
        if any(m.get("priority") == "primary" for m in metrics):
            return
        metrics[0]["priority"] = "primary"

    def _read_preview_from_file(self, file_path: str, limit: int = 2) -> List[Any]:
        """
        从文件中读取预览数据 (支持 jsonl 和 json)
        """
        if not file_path:
            return []
            
        path = Path(file_path)
        preview = []
        if not path.exists():
            return preview
            
        try:
            if path.suffix.lower() == '.jsonl':
                with path.open('r', encoding='utf-8') as f:
                    for _ in range(limit):
                        line = f.readline()
                        if not line: break
                        try:
                            preview.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            elif path.suffix.lower() == '.json':
                with path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        preview = data[:limit]
                    elif isinstance(data, dict):
                        # 尝试常见的 key
                        for key in ['rows', 'records', 'data', 'examples', 'items']:
                            if key in data and isinstance(data[key], list):
                                preview = data[key][:limit]
                                break
        except Exception as e:
            log.warning(f"无法读取文件预览: {file_path}, 错误: {e}")
            
        return preview

    def _format_bench_context(self, benches: List[BenchInfo], task_domain: Optional[str] = None) -> str:
        """
        将 Benchmark 信息格式化为 LLM 可读的上下文。
        """
        context_parts = []

        for b in benches:
            task_type = b.meta.get("task_type", "unknown")
            if isinstance(task_type, list):
                task_type = ", ".join(task_type)
            
            # 1. 尝试从文件读取 (Strict Mode: Only use eval_detail_path)
            examples = []
            source_file = b.meta.get("eval_detail_path")
            
            if source_file:
                examples = self._read_preview_from_file(str(source_file))

            if isinstance(examples, list) and len(examples) > 0:
                display_examples = examples[:2]
                # 限制每个 sample 的长度，防止 token 爆炸
                examples_str = "\n".join([
                    f"  Sample {i+1}: {json.dumps(ex, ensure_ascii=False)[:1000]}"
                    for i, ex in enumerate(display_examples)
                ])
            else:
                examples_str = "  无样例数据 (无法读取源文件)"
            
            # 让 LLM 自己从 Sample 中看
            eval_type = self._infer_eval_type(b)
            prompt_template = b.bench_prompt_template
            if isinstance(prompt_template, str) and len(prompt_template) > 600:
                prompt_template = prompt_template[:300] + "\n...[SNIP]...\n" + prompt_template[-300:]

            part = (
                f"### Benchmark: {b.bench_name}\n"
                f"- state.task_domain: {task_domain or 'Unknown'}\n"
                f"- bench_dataflow_eval_type: {eval_type or 'Unknown'}\n"
                f"- 任务类型: {task_type}\n"
                f"- 领域标签: {b.meta.get('domain', 'Unknown')}\n"
                f"- 描述: {b.meta.get('description', 'No description provided')}\n"
                f"- 推理Prompt模板(截断): {prompt_template or 'None'}\n"
                f"- 样例数据 (Raw JSON):\n{examples_str}\n"
            )
            context_parts.append(part)
        return "\n".join(context_parts)

    async def run(self, state: NodeState) -> NodeState:
        """
        执行增强型指标推荐：
        
        - LLM 为主导 (Primary): 接收用户需求，参考所有已注册 Metric，进行一次性推荐。
        - Registry 为辅 (Reference): 提供标准建议作为上下文，但不直接采纳。
        """
        if not state.benches:
            log.warning("State 中没有发现 Benches 信息，跳过 Metric 推荐。")
            return state

        if not state.metric_plan:
            state.metric_plan = {}

        self._metric_desc_map = {m.name: m.desc for m in get_registered_metrics_meta()}

        target_benches: List[BenchInfo] = []
        registry_suggestions: Dict[str, List[Dict[str, Any]]] = {}

        # --- 预处理：收集 Registry 建议和处理 User Override ---
        for bench in state.benches:
            bench_name = bench.bench_name
            
            # 0. User Override (Meta中直接指定) - 最高优先级，永远生效
            if bench.meta.get("metrics"):
                user_metrics = bench.meta["metrics"]
                if isinstance(user_metrics, list):
                    validated = self._validate_metrics(user_metrics)
                    if validated:
                        self._ensure_primary(validated)
                        state.metric_plan[bench_name] = validated
                        log.info(f"[{bench_name}] ✓ 使用 Meta 指定的 Metrics ({len(validated)} 个)")
                        continue
            
            # 收集 Registry 建议 (作为参考或兜底)
            registry_metrics = self._check_registry(bench)
            if registry_metrics:
                registry_suggestions[bench_name] = self._validate_metrics(registry_metrics)

            # 始终将 Benchmark 加入 LLM 分析列表 (LLM First 策略)
            # Registry 建议将作为 Context 提供给 LLM，不再直接阻断 LLM 调用
            target_benches.append(bench)

        # --- LLM Analysis Phase ---
        if target_benches:
            log.info(f"[用户需求] 正在调用 LLM 分析以下 Benchmark: {[b.bench_name for b in target_benches]}")
            
            # 1. 准备上下文 
            # 将 registry_suggestions 始终作为参考提供
            bench_context_str = self._format_bench_context(
                target_benches, 
                task_domain=state.task_domain
            )
            
            # 2. 从 Registry 获取动态文档 (Menu)
            metric_library_doc = metric_registry.get_metric_library_doc()
            decision_logic_doc = metric_registry.get_decision_logic_doc()
            
            # 3. 构建 Prompt
            sys_prompt = self.get_prompt(
                self.system_prompt_template_name,
                metric_library_doc=metric_library_doc 
            )
            
            # 动态调整 Prompt 指令
            req_instruction = str(state.user_query)
            req_instruction += "\n\n(重要：请优先满足用户的具体需求。如果用户需求与标准指标冲突，以用户需求为准。)"

            task_prompt = self.get_prompt(
                self.task_prompt_template_name,
                bench_context=bench_context_str,
                user_requirement=req_instruction,
                decision_logic_doc=decision_logic_doc 
            )

            msgs = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task_prompt)
            ]

            llm = self.create_llm(state)
            resp = await llm.call(msgs, bind_post_tools=False)
            llm_content = resp.content if hasattr(resp, 'content') else str(resp)
            
            parsed_result = self.parse_result(llm_content)

            # 处理 LLM 返回结果
            if isinstance(parsed_result, dict):
                for b_name, metrics in parsed_result.items():
                    matched_bench = next((b for b in target_benches if b.bench_name == b_name), None)
                    if not matched_bench:
                        continue
                    
                    if isinstance(metrics, list) and len(metrics) > 0:
                        validated = self._validate_metrics(metrics)
                        if validated:
                            self._ensure_primary(validated)
                            state.metric_plan[b_name] = validated
                            log.info(f"[{b_name}] ✓ LLM 推荐 Metrics ({len(validated)} 个)")
                        else:
                            log.warning(f"[{b_name}] LLM 推荐 Metrics 验证失败")
                    else:
                        log.warning(f"LLM 返回的 {b_name} metric 格式不正确")
            else:
                log.warning(f"LLM 返回结果格式非 Dict: {parsed_result}")

        # --- Final Fallback ---
        # 针对 User Input Mode 下 LLM 可能遗漏的情况，或者 LLM 失败的情况
        for bench in state.benches:
            if bench.bench_name not in state.metric_plan:
                if bench.bench_name in registry_suggestions:
                    log.info(f"[{bench.bench_name}] LLM 未返回有效结果，回退使用 Registry 建议。")
                    fallback = registry_suggestions[bench.bench_name]
                else:
                    log.warning(f"[{bench.bench_name}] Registry和LLM均未命中，使用默认兜底指标。")
                    fallback = [
                        {"name": "exact_match", "priority": "primary"},
                        {"name": "extraction_rate", "priority": "diagnostic"}
                    ]
                validated = self._validate_metrics(fallback)
                self._ensure_primary(validated)
                state.metric_plan[bench.bench_name] = validated

        state.result[self.role_name] = {
            "metric_plan": state.metric_plan,
            "registry_hits": list(registry_suggestions.keys()), # 记录哪些有规则建议
            "llm_analyzed": [b.bench_name for b in target_benches]
        }
        
        log.info(f"Metric 推荐完成: 共 {len(state.metric_plan)} 个 Benchmark")
        return state
