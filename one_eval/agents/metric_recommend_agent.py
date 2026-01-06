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
    查表策略：
        - 查看 benchinfo 中是否指定了 metric
        - 基于 bench_dataflow_eval_type 进行第一次分流
        - 启发式 task_family (主要基于 prompt_template 分析)匹配， 进行第二次分流
        - (eval_type, task_family)   ---->  template 的映射
    2. Analyst Track: 未知 Benchmark 基于 Info (name、prompt_template) 调用 LLM 分析。
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

    def _check_registry(self, bench: BenchInfo, task_domain: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        检查注册表是否有预定义的 Metric。
        """
        eval_type = bench.bench_dataflow_eval_type
        if not eval_type and bench.meta:
            eval_type = bench.meta.get("bench_dataflow_eval_type") or bench.meta.get("eval_type")

        prompt_template = bench.bench_prompt_template

        try:
            return metric_registry.get_metrics(
                bench.bench_name,
                bench_meta=bench.meta,
                eval_type=eval_type,
                prompt_template=prompt_template,
                task_domain=task_domain,
            )
        except TypeError:
            return metric_registry.get_metrics(bench.bench_name)
    
    def _normalize_metric_format(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化指标格式，确保包含必需字段。
        """
        normalized = {
            "name": metric.get("name") or metric.get("metric_name"),
            "priority": metric.get("priority", "secondary"),
            "desc": metric.get("desc") or metric.get("description", ""),
        }
        
        if "args" in metric:
            normalized["args"] = metric["args"]
        elif "params" in metric:
            normalized["args"] = metric["params"]
        elif "k" in metric:
            normalized["args"] = {"k": metric["k"]}
        
        if not normalized["name"]:
            raise ValueError(f"Metric 缺少必需字段 'name': {metric}")
        
        if normalized["priority"] not in ["primary", "secondary", "diagnostic"]:
            log.warning(f"Metric {normalized['name']} 的 priority '{normalized['priority']}' 不在标准值中，使用 'secondary'")
            normalized["priority"] = "secondary"
        
        return normalized
    
    def _validate_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证并规范化指标列表。
        """
        validated = []
        if not metrics:
            return []
        for metric in metrics:
            try:
                normalized = self._normalize_metric_format(metric)
                validated.append(normalized)
            except (ValueError, KeyError) as e:
                log.warning(f"跳过无效的指标配置: {metric}, 错误: {e}")
                continue
        return validated

    def _format_bench_context(self, benches: List[BenchInfo], task_domain: Optional[str] = None) -> str:
        """
        将 Benchmark 信息格式化为 LLM 可读的上下文。
        """
        context_parts = []
        for b in benches:
            task_type = b.meta.get("task_type", "unknown")
            if isinstance(task_type, list):
                task_type = ", ".join(task_type)
            
            examples = b.meta.get("examples", []) or b.meta.get("few_shot", []) or b.meta.get("preview", [])
            if isinstance(examples, list) and len(examples) > 0:
                display_examples = examples[:2]
                examples_str = "\n".join([
                    f"  样例 {i+1}: {str(ex)[:300]}"
                    for i, ex in enumerate(display_examples)
                ])
            else:
                examples_str = "  无样例数据"
            
            input_col = b.meta.get("input_column", b.meta.get("question_column", "question"))
            output_col = b.meta.get("output_column", b.meta.get("answer_column", "answer"))
            
            eval_type = b.bench_dataflow_eval_type or b.meta.get("bench_dataflow_eval_type") or b.meta.get("eval_type")
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
                f"- 输入字段: {input_col}\n"
                f"- 输出字段: {output_col}\n"
                f"- 推理Prompt模板(截断): {prompt_template or 'None'}\n"
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

        if not state.metric_plan:
            state.metric_plan = {}

        unknown_benches: List[BenchInfo] = []
        registry_hits: Dict[str, List[Dict[str, Any]]] = {}
        
        # --- Track 1: Registry Lookup ---
        for bench in state.benches:
            bench_name = bench.bench_name
            
            # 1. 用户显式指定
            if bench.meta.get("metrics"):
                user_metrics = bench.meta["metrics"]
                if isinstance(user_metrics, list):
                    validated = self._validate_metrics(user_metrics)
                    if validated:
                        state.metric_plan[bench_name] = validated
                        log.info(f"[{bench_name}] ✓ 使用用户指定的 Metrics ({len(validated)} 个)")
                    else:
                        unknown_benches.append(bench)
                else:
                    unknown_benches.append(bench)
                continue

            # 2. 注册表查找
            registry_metrics = self._check_registry(bench, task_domain=state.task_domain)
            
            if registry_metrics:
                validated = self._validate_metrics(registry_metrics)
                state.metric_plan[bench_name] = validated
                registry_hits[bench_name] = validated
            else:
                unknown_benches.append(bench)

        # --- Track 2: LLM Analysis (仅处理未知 Benchmark) ---
        if unknown_benches:
            log.info(f"正在调用 LLM 分析以下 Benchmark 的 Metrics: {[b.bench_name for b in unknown_benches]}")
            
            # 1. 准备上下文
            bench_context_str = self._format_bench_context(unknown_benches, task_domain=state.task_domain)
            
            # 2. 从 Registry 获取动态文档 (关键修改点)
            # 这些文档将注入到 Prompt 模板中，替代硬编码
            metric_library_doc = metric_registry.get_metric_library_doc()
            decision_logic_doc = metric_registry.get_decision_logic_doc()
            
            # 3. 构建 Prompt
            # System Prompt: 注入指标库定义
            sys_prompt = self.get_prompt(
                self.system_prompt_template_name,
                metric_library_doc=metric_library_doc 
            )
            
            # Task Prompt: 注入决策逻辑
            task_prompt = self.get_prompt(
                self.task_prompt_template_name,
                bench_context=bench_context_str,
                user_requirement=state.user_query or "无特殊要求",
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
                    matched_bench = next((b for b in unknown_benches if b.bench_name == b_name), None)
                    if not matched_bench:
                        continue
                    
                    if isinstance(metrics, list) and len(metrics) > 0:
                        validated = self._validate_metrics(metrics)
                        if validated:
                            state.metric_plan[b_name] = validated
                            log.info(f"[{b_name}] ✓ LLM 推荐 Metrics ({len(validated)} 个)")
                        else:
                            log.warning(f"[{b_name}] LLM 推荐 Metrics 验证失败")
                    else:
                        log.warning(f"LLM 返回的 {b_name} metric 格式不正确")
            else:
                log.warning(f"LLM 返回结果格式非 Dict: {parsed_result}")

        # 最终兜底验证
        for bench in state.benches:
            if bench.bench_name not in state.metric_plan:
                log.warning(f"[{bench.bench_name}] Registry和LLM均未命中，使用默认兜底指标。")
                state.metric_plan[bench.bench_name] = [
                    {"name": "exact_match", "priority": "primary", "desc": "Fallback EM"},
                    {"name": "extraction_rate", "priority": "diagnostic", "desc": "Fallback Extraction"}
                ]

        state.result[self.role_name] = {
            "metric_plan": state.metric_plan,
            "registry_hits": list(registry_hits.keys()),
            "llm_analyzed": [b.bench_name for b in unknown_benches] if unknown_benches else []
        }
        
        log.info(f"Metric 推荐完成: 共 {len(state.metric_plan)} 个 Benchmark")
        return state
