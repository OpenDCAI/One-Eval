from typing import List, Any, Dict, Optional
import random
import os
import logging
import asyncio
import json
from one_eval.core.metric_registry import register_metric, MetricCategory
from one_eval.logger import get_logger
from one_eval.serving.custom_llm_caller import CustomLLMCaller
from langchain_core.messages import HumanMessage, SystemMessage
log = get_logger(__name__)

# Mock State for CustomLLMCaller to satisfy initialization requirements
class MockState:
    def __init__(self, model_name: str):
        # LLMCaller accesses self.state.request.model
        self.request = type("MockRequest", (), {"model": model_name})()

@register_metric(
    name="case_study_analyst",
    desc="通用抽样诊断器 (LLM-based)",
    usage="深度分析错误原因/模型行为",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI, MetricCategory.CHOICE_SINGLE]
)
def compute_case_study_analyst(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    CaseStudyAnalyst: 自动抽样并调用 LLM (via CustomLLMCaller) 进行分析。
    
    Args:
        preds: 预测结果列表
        refs: 参考答案列表
        kwargs:
            sample_size (int): 抽样数量，默认 5
            target_group (str): 'positive' | 'negative' | 'mixed'，默认 'negative'
            instruction (str): 分析指令
            auto_prompt (bool): 是否启用自动 Prompt 优化
            model_name (str): LLM 模型名称，默认 "gpt-4o"
            api_key (str): OpenAI API Key
            base_url (str): OpenAI Base URL
    """
    # 1. 参数解析
    sample_size = int(kwargs.get("sample_size", 5))
    target_group = kwargs.get("target_group", "negative")
    instruction = kwargs.get("instruction", "")
    auto_prompt = kwargs.get("auto_prompt", False)
    lang = str(kwargs.get("language", "zh") or "zh").lower()
    is_en = lang.startswith("en")
    
    # LLM Config
    model_name = kwargs.get("model_name", "gpt-4o")
    api_key = kwargs.get("api_key") or os.environ.get("OE_API_KEY")
    base_url = kwargs.get("base_url") or os.environ.get("OE_API_BASE")
    
    # Try to retrieve real state from kwargs (if passed by caller)
    real_state = kwargs.get("state", None)
    
    if not api_key:
        return {"score": 0.0, "error": ("Missing API Key for CaseStudyAnalyst." if is_en else "CaseStudyAnalyst 缺少 API Key。")}

    # 2. 区分正负例
    pos_indices = []
    neg_indices = []
    
    for idx, (p, r) in enumerate(zip(preds, refs)):
        # 简单判断逻辑：宽松匹配 (Loose Match)
        is_correct = False
        p_str = str(p).strip()
        
        r_list = r if isinstance(r, list) else [r]
        for gold in r_list:
            g_str = str(gold).strip()
            if p_str == g_str or (g_str and g_str in p_str):
                is_correct = True
                break
        
        if is_correct:
            pos_indices.append(idx)
        else:
            neg_indices.append(idx)

    # 3. 抽样
    selected_indices = []
    if target_group == "positive":
        candidates = pos_indices
    elif target_group == "negative":
        candidates = neg_indices
    elif target_group == "mixed":
        candidates = pos_indices + neg_indices
    else:
        candidates = neg_indices
        
    if not candidates:
        return {
            "score": 0.0, 
            "analysis": (
                f"No samples found for group '{target_group}'. (Pos: {len(pos_indices)}, Neg: {len(neg_indices)})"
                if is_en else
                f"分组 '{target_group}' 未找到可分析样本。（正例: {len(pos_indices)}，负例: {len(neg_indices)}）"
            )
        }

    if len(candidates) > sample_size:
        selected_indices = random.sample(candidates, sample_size)
    else:
        selected_indices = candidates
        
    # 4. 构建 Analysis Prompt
    cases_text = ""
    for i, idx in enumerate(selected_indices):
        if is_en:
            cases_text += f"\n[Case {i+1}]\n"
            cases_text += f"Prediction: {preds[idx]}\n"
            cases_text += f"Reference: {refs[idx]}\n"
        else:
            cases_text += f"\n[样本 {i+1}]\n"
            cases_text += f"模型输出: {preds[idx]}\n"
            cases_text += f"参考答案: {refs[idx]}\n"

    system_prompt = (
        "You are an expert AI model evaluator. Your goal is to analyze model predictions against reference answers."
        if is_en else
        "你是资深模型评测分析师。你的目标是对照参考答案分析模型输出表现。"
    )

    user_content = (
        f"Here are {len(selected_indices)} sampled cases ({target_group} examples).\n"
        if is_en else
        f"以下是抽样得到的 {len(selected_indices)} 条样本（分组：{target_group}）。\n"
    )
    user_content += cases_text
    user_content += "\n\n"
    
    if auto_prompt:
        if not instruction:
            user_content += (
                "Please automatically identify common patterns, potential error types, and provide a concise summary of model performance on these cases."
                if is_en else
                "请自动识别共性模式、潜在错误类型，并给出简洁的整体表现总结。"
            )
        else:
            user_content += (
                f"User Instruction: {instruction}\n\nBased on the user instruction, please analyze these cases and add any relevant insights."
                if is_en else
                f"用户指令：{instruction}\n\n请基于以上指令分析这些样本，并补充你发现的关键洞察。"
            )
    elif instruction:
        user_content += (
            f"Instruction: {instruction}\n\nPlease analyze the cases strictly following the instruction above."
            if is_en else
            f"指令：{instruction}\n\n请严格依据以上指令分析这些样本。"
        )
    else:
        user_content += (
            "Please analyze these cases and provide a summary."
            if is_en else
            "请分析这些样本并给出总结。"
        )

    # 5. 调用 CustomLLMCaller (Async Wrapper)
    async def _call_llm():
        # Use real state if available, otherwise use MockState
        # Note: real_state should expose request.model for CustomLLMCaller.
        llm_state = real_state if real_state else MockState(model_name)
        
        # Initialize CustomLLMCaller
        caller = CustomLLMCaller(
            state=llm_state,
            tool_manager=None,
            agent_role="case_study_analyst",
            model_name=model_name,
            base_url=base_url or "http://123.129.219.111:3000/v1", # fallback
            api_key=api_key,
            temperature=0.7
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        response = await caller.call(messages, bind_post_tools=False)
        return response.content

    try:
        # Check for existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            # If in a loop (e.g. Jupyter), try nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                analysis_result = asyncio.run(_call_llm())
            except ImportError:
                # Fallback: Try to create a new loop in a separate thread if nest_asyncio is missing?
                # Or just error out.
                return {"score": 0.0, "error": "Running in async loop but nest_asyncio not installed."}
        else:
            # Standard synchronous context
            analysis_result = asyncio.run(_call_llm())
        
        return {
            "score": 1.0, 
            "analysis": analysis_result,
            "details": selected_indices, 
            "artifacts": {
                "instruction": instruction,
                "target_group": target_group,
                "sample_count": len(selected_indices),
                "pos_count": len(pos_indices),
                "neg_count": len(neg_indices)
            }
        }
        
    except Exception as e:
        log.error(f"CaseStudyAnalyst LLM call failed: {e}")
        return {"score": 0.0, "error": str(e)}

@register_metric(
    name="metric_summary_analyst",
    desc="指标汇总分析 (LLM-based)",
    usage="基于已计算的所有指标生成汇总报告 (需置于指标列表末尾)",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI, MetricCategory.CHOICE_SINGLE, MetricCategory.TEXT_SCORE]
)
def compute_metric_summary_analyst(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    MetricSummaryAnalyst: 汇总当前 Bench 上已计算的所有 Metric 结果，并调用 LLM 生成分析报告。
    
    Args:
        preds: 预测结果 (未使用，仅占位)
        refs: 参考答案 (未使用，仅占位)
        kwargs:
            all_metric_results (Dict): 由 MetricRunner 注入的当前已计算指标结果
            model_name (str): LLM 模型名称
            api_key (str): OpenAI API Key
            base_url (str): OpenAI Base URL
    """
    lang = str(kwargs.get("language", "zh") or "zh").lower()
    is_en = lang.startswith("en")

    # 1. 获取上下文中的 Metric 结果
    all_results = kwargs.get("all_metric_results", {})
    if not all_results:
        return {
            "score": 0.0,
            "summary": (
                "No metric results found to summarize. Please ensure this metric is run after other metrics."
                if is_en else
                "未找到可汇总的指标结果，请确保该指标在其他指标之后执行。"
            )
        }
        
    # 2. 准备 LLM 调用
    model_name = kwargs.get("model_name", "gpt-4o")
    api_key = kwargs.get("api_key") or os.environ.get("OE_API_KEY", "sk-xxx")
    base_url = kwargs.get("base_url") or os.environ.get("OE_API_BASE", "http://123.129.219.111:3000/v1")
    
    # Try to retrieve real state from kwargs (if passed by caller)
    real_state = kwargs.get("state", None)

    if not api_key:
        return {"score": 0.0, "error": "Missing API Key for MetricSummaryAnalyst."}

    # 3. 格式化数据
    # 过滤掉 error 的 metric，提取 score 和 details 摘要
    summary_data = {}
    for k, v in all_results.items():
        if "error" in v:
            summary_data[k] = f"Error: {v['error']}"
        else:
            # 仅保留 score 和 desc，避免 details 太长撑爆 Context
            summary_data[k] = {
                "score": v.get("score"),
                "desc": v.get("desc", ""),
                "priority": v.get("priority", "secondary")
            }

    # 4. 构建 Prompt
    if is_en:
        system_prompt = """You are an Expert AI Evaluation Analyst.
Your goal is to analyze benchmark metrics and provide a concise, actionable summary report."""
        user_prompt = f"""Please analyze the following metric results and provide a summary report in English:

Metric Results:
{json.dumps(summary_data, indent=2, ensure_ascii=False)}

Your report should include:
1. Overall Performance: verdict based on primary metrics.
2. Detailed Analysis: strengths and weaknesses.
3. Anomalies: conflicting or unexpected values.
4. Conclusion: final assessment for this task.

Do not output markdown horizontal rules like ---.
"""
    else:
        system_prompt = """你是资深 AI 评测分析师。
你的目标是分析基准测试指标，并输出简洁、可执行的中文汇总报告。"""
        user_prompt = f"""请分析以下指标结果，并输出中文总结报告：

指标结果：
{json.dumps(summary_data, indent=2, ensure_ascii=False)}

报告需包含：
1. 整体表现：基于主指标给出结论。
2. 细项分析：主要优势与短板。
3. 异常点：冲突或异常指标表现。
4. 结论建议：针对该任务的最终判断。

不要输出 markdown 分割线（例如 ---）。
"""

    # 5. 调用 CustomLLMCaller (Async Wrapper)
    async def _call_llm():
        llm_state = real_state if real_state else MockState(model_name)
        
        caller = CustomLLMCaller(
            state=llm_state,
            tool_manager=None,
            agent_role="MetricSummaryAnalyst",
            model_name=model_name,
            base_url=base_url,
            api_key=api_key
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        response = await caller.call(messages, bind_post_tools=False)
        return response.content

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
                analysis_result = asyncio.run(_call_llm())
            except ImportError:
                return {"score": 0.0, "error": "Running in async loop but nest_asyncio not installed."}
        else:
            analysis_result = asyncio.run(_call_llm())
        
        return {
            "score": 1.0, 
            "summary": analysis_result
        }
        
    except Exception as e:
        log.error(f"MetricSummaryAnalyst LLM call failed: {e}")
        return {
            "score": 0.0, 
            "error": f"LLM analysis failed: {str(e)}"
        }
