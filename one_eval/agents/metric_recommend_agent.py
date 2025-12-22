from __future__ import annotations
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger
from one_eval.utils.metric_registry import metric_registry

log = get_logger("MetricRecommendAgent")

class MetricRecommendAgent(CustomAgent):
    """
    Step 3 Agent: Metric推荐
    双轨制策略：
    1. Registry Track: 已知 Benchmark 直接查表。
    2. Analyst Track: 未知 Benchmark 基于 Info 和 Examples 调用 LLM 分析。
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

    def _check_registry(self, bench_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        检查注册表是否有预定义的 Metric。
        """
        # 如果 Registry 返回 None，说明它不知道
        return metric_registry.get_metrics(bench_name)
    
    def _normalize_metric_format(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化指标格式，确保包含必需字段。
        将旧格式转换为统一格式：{name, priority, desc, args}
        """
        normalized = {
            "name": metric.get("name") or metric.get("metric_name"),
            "priority": metric.get("priority", "secondary"),
            "desc": metric.get("desc") or metric.get("description", ""),
        }
        
        # 处理参数：args, params, k 等
        if "args" in metric:
            normalized["args"] = metric["args"]
        elif "params" in metric:
            normalized["args"] = metric["params"]
        elif "k" in metric:
            normalized["args"] = {"k": metric["k"]}
        
        # 验证必需字段
        if not normalized["name"]:
            raise ValueError(f"Metric 缺少必需字段 'name': {metric}")
        
        # 验证 priority
        if normalized["priority"] not in ["primary", "secondary", "diagnostic"]:
            log.warning(f"Metric {normalized['name']} 的 priority '{normalized['priority']}' 不在标准值中，使用 'secondary'")
            normalized["priority"] = "secondary"
        
        return normalized
    
    def _validate_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证并规范化指标列表。
        返回规范化后的指标列表，过滤掉无效项。
        """
        validated = []
        for metric in metrics:
            try:
                normalized = self._normalize_metric_format(metric)
                validated.append(normalized)
            except (ValueError, KeyError) as e:
                log.warning(f"跳过无效的指标配置: {metric}, 错误: {e}")
                continue
        return validated

    def _format_bench_context(self, benches: List[BenchInfo]) -> str:
        """
        将 Benchmark 信息格式化为 LLM 可读的上下文。
        包含：名称、领域、任务类型、描述、样例数据等。
        """
        context_parts = []
        for b in benches:
            # 获取任务类型
            task_type = b.meta.get("task_type", "unknown")
            if isinstance(task_type, list):
                task_type = ", ".join(task_type)
            
            # 获取样例数据
            examples = b.meta.get("examples", []) or b.meta.get("few_shot", []) or b.meta.get("preview", [])
            if isinstance(examples, list) and len(examples) > 0:
                # 截取前2个样例，避免 token 过多
                display_examples = examples[:2]
                examples_str = "\n".join([
                    f"  样例 {i+1}: {str(ex)[:300]}"  # 限制每个样例长度
                    for i, ex in enumerate(display_examples)
                ])
            else:
                examples_str = "  无样例数据"
            
            # 获取输入输出字段信息
            input_col = b.meta.get("input_column", b.meta.get("question_column", "question"))
            output_col = b.meta.get("output_column", b.meta.get("answer_column", "answer"))
            
            part = (
                f"### Benchmark: {b.bench_name}\n"
                f"- 任务类型: {task_type}\n"
                f"- 领域标签: {b.meta.get('domain', 'Unknown')}\n"
                f"- 描述: {b.meta.get('description', 'No description provided')}\n"
                f"- 输入字段: {input_col}\n"
                f"- 输出字段: {output_col}\n"
                f"- 样例数据:\n{examples_str}\n"
            )
            context_parts.append(part)
        return "\n".join(context_parts)

    async def run(self, state: NodeState) -> NodeState:
        """
        执行双轨制指标推荐：
        1. Registry Track: 优先使用 metric_registry 的规则化注册表
        2. LLM Track: 未知 Benchmark 调用 LLM 分析推荐
        """
        if not state.benches:
            log.warning("State 中没有发现 Benches 信息，跳过 Metric 推荐。")
            return state

        # 初始化 metric_plan
        if not state.metric_plan:
            state.metric_plan = {}

        unknown_benches: List[BenchInfo] = []
        registry_hits: Dict[str, List[Dict[str, Any]]] = {}
        
        # --- Track 1: Registry Lookup (优先级从高到低) ---
        for bench in state.benches:
            bench_name = bench.bench_name
            
            # 1. 最高优先级：用户显式指定的 metrics（在 bench.meta 中）
            if bench.meta.get("metrics"):
                user_metrics = bench.meta["metrics"]
                if isinstance(user_metrics, list):
                    validated = self._validate_metrics(user_metrics)
                    if validated:
                        state.metric_plan[bench_name] = validated
                        log.info(f"[{bench_name}] ✓ 使用用户指定的 Metrics ({len(validated)} 个)")
                    else:
                        log.warning(f"[{bench_name}] 用户指定的 Metrics 验证失败，将使用注册表或 LLM")
                        unknown_benches.append(bench)
                else:
                    log.warning(f"[{bench_name}] bench.meta['metrics'] 格式不正确（应为列表），将使用注册表或 LLM")
                    unknown_benches.append(bench)
                continue

            # 2. 次优先级：检查全局 metric_registry
            registry_metrics = self._check_registry(bench_name)
            
            if registry_metrics:
                # 只要 Registry 返回了非空，就说明它是确定的（精确或模糊匹配成功）
                validated = self._validate_metrics(registry_metrics)
                state.metric_plan[bench_name] = validated
                registry_hits[bench_name] = validated
            else:
                # Registry 返回 None，说明不认识 -> 加入 LLM 待分析列表
                unknown_benches.append(bench)

        # --- Track 2: LLM Analysis (仅处理未知 Benchmark) ---
        if unknown_benches:
            log.info(f"正在调用 LLM 分析以下 Benchmark 的 Metrics: {[b.bench_name for b in unknown_benches]}")
            
            # 构造 Prompt 上下文
            bench_context_str = self._format_bench_context(unknown_benches)
            
            sys_prompt = self.get_prompt(self.system_prompt_template_name)
            task_prompt = self.get_prompt(
                self.task_prompt_template_name,
                bench_context=bench_context_str,
                user_requirement=state.user_query or "无特殊要求"
            )

            msgs = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task_prompt)
            ]

            llm = self.create_llm(state)
            # 调用 LLM：CustomLLMCaller 使用 call 方法（异步）
            resp = await llm.call(msgs, bind_post_tools=False)
            llm_content = resp.content if hasattr(resp, 'content') else str(resp)
            
            parsed_result = self.parse_result(llm_content)

            # 处理 LLM 返回结果：格式应为 {"benchmark_name": [metric_config, ...]}
            if isinstance(parsed_result, dict):
                for b_name, metrics in parsed_result.items():
                    # 匹配到对应的 bench
                    matched_bench = next((b for b in unknown_benches if b.bench_name == b_name), None)
                    if not matched_bench:
                        log.warning(f"LLM 返回的 benchmark '{b_name}' 不在待分析列表中，跳过")
                        continue
                    
                    if isinstance(metrics, list) and len(metrics) > 0:
                        validated = self._validate_metrics(metrics)
                        if validated:
                            state.metric_plan[b_name] = validated
                            log.info(f"[{b_name}] ✓ LLM 推荐 Metrics ({len(validated)} 个)")
                        else:
                            log.warning(f"[{b_name}] LLM 推荐的 Metrics 验证失败，使用默认指标")
                            # 使用注册表的默认回退指标
                            fallback_metrics = metric_registry.get_metrics(b_name)
                            state.metric_plan[b_name] = self._validate_metrics(fallback_metrics)
                    else:
                        log.warning(f"LLM 返回的 {b_name} metric 格式不正确（应为非空列表）: {metrics}")
                        # 使用默认回退
                        fallback_metrics = metric_registry.get_metrics(b_name)
                        state.metric_plan[b_name] = self._validate_metrics(fallback_metrics)
            else:
                log.warning(f"LLM 返回结果格式非 Dict: {parsed_result}")
                # 为所有未知 bench 使用默认回退
                for bench in unknown_benches:
                    if bench.bench_name not in state.metric_plan:
                        fallback_metrics = metric_registry.get_metrics(bench.bench_name)
                        state.metric_plan[bench.bench_name] = self._validate_metrics(fallback_metrics)

        # 最终验证：确保所有 bench 都有 metric_plan
        for bench in state.benches:
            if bench.bench_name not in state.metric_plan:
                log.warning(f"[{bench.bench_name}] Registry和LLM均未命中，使用默认文本生成指标。")
                state.metric_plan[bench.bench_name] = [
                    {"name": "exact_match", "priority": "primary"},
                    {"name": "extraction_rate", "priority": "diagnostic"}
                ]

        # 更新 State 结果区，方便调试查看
        state.result[self.role_name] = {
            "metric_plan": state.metric_plan,
            "registry_hits": list(registry_hits.keys()),
            "llm_analyzed": [b.bench_name for b in unknown_benches] if unknown_benches else []
        }
        
        log.info(f"Metric 推荐完成: 共 {len(state.metric_plan)} 个 Benchmark")
        return state
