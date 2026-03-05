from typing import List, Any, Dict, Optional
import random
import os
import asyncio
import json
from one_eval.core.metric_registry import register_metric, MetricCategory
from one_eval.logger import get_logger
from one_eval.serving.custom_llm_caller import CustomLLMCaller
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
log = get_logger(__name__)

# Mock State for CustomLLMCaller to satisfy initialization requirements
class MockState:
    def __init__(self, model_name: str):
        # BaseLLMCaller accesses self.state.request.model
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
    
    # LLM Config
    model_name = kwargs.get("model_name", "gpt-4o")
    api_key = kwargs.get("api_key") or os.environ.get("OE_API_KEY")
    base_url = kwargs.get("base_url") or os.environ.get("OE_API_BASE")
    
    # Try to retrieve real state from kwargs (if passed by caller)
    real_state = kwargs.get("state", None)
    
    if not api_key:
        return {"score": 0.0, "error": "Missing API Key for CaseStudyAnalyst."}

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
            "analysis": f"No samples found for group '{target_group}'. (Pos: {len(pos_indices)}, Neg: {len(neg_indices)})"
        }

    if len(candidates) > sample_size:
        selected_indices = random.sample(candidates, sample_size)
    else:
        selected_indices = candidates
        
    # 4. 构建 Analysis Prompt
    cases_text = ""
    for i, idx in enumerate(selected_indices):
        cases_text += f"\n[Case {i+1}]\n"
        cases_text += f"Prediction: {preds[idx]}\n"
        cases_text += f"Reference: {refs[idx]}\n"
        
    system_prompt = "You are an expert AI model evaluator. Your goal is to analyze model predictions against reference answers."
    
    user_content = f"Here are {len(selected_indices)} sampled cases ({target_group} examples).\n"
    user_content += cases_text
    user_content += "\n\n"
    
    if auto_prompt:
        if not instruction:
            user_content += "Please automatically identify the common patterns, error types (if any), and provide a concise summary of the model's performance on these cases."
        else:
            user_content += f"User Instruction: {instruction}\n\nBased on the user instruction, please analyze these cases. Also, feel free to add any other relevant insights you discover."
    elif instruction:
        user_content += f"Instruction: {instruction}\n\nPlease analyze the cases strictly following the instruction above."
    else:
        user_content += "Please analyze these cases and provide a summary."

    # 5. 调用 CustomLLMCaller (Async Wrapper)
    async def _call_llm():
        # Use real state if available, otherwise use MockState
        # Note: If real_state is passed, it should be an instance of dataflow_agent.state.MainState or similar
        # that CustomLLMCaller expects.
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
        
        # Use bind_post_tools=False for raw API call (faster, no tools)
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
    # 1. 获取上下文中的 Metric 结果
    all_results = kwargs.get("all_metric_results", {})
    if not all_results:
        return {
            "score": 0.0,
            "summary": "No metric results found to summarize. Please ensure this metric is run after other metrics."
        }
        
    # 2. 准备 LLM 调用
    model_name = kwargs.get("model_name", "gpt-4o")
    api_key = kwargs.get("api_key") or os.environ.get("OE_API_KEY")
    base_url = kwargs.get("base_url") or os.environ.get("OE_API_BASE")
    
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
    system_prompt = """You are an Expert AI Evaluation Analyst. 
Your goal is to analyze the performance metrics of an AI model on a specific benchmark and provide a comprehensive summary report."""
    
    user_prompt = f"""Please analyze the following metric results and provide a summary report:

Metric Results:
{json.dumps(summary_data, indent=2)}

Your report should include:
1. **Overall Performance**: A high-level verdict based on primary metrics.
2. **Detailed Analysis**: Breakdown of specific strengths and weaknesses.
3. **Anomalies**: Any conflicting or unexpected metric values (e.g., high Exact Match but low F1, or errors).
4. **Conclusion**: Final thoughts on the model's capability on this task.

Output the report in Markdown format.
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

# if __name__ == "__main__":
#     import sys
#     import os
    
#     # Ensure project root is in path
#     current_dir = os.path.dirname(os.path.abspath(__file__)) 
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)

#     # Real-World Demo (No Mocks)
#     def run_real_demo():
#         print("\n=== Real-World Demo: CaseStudyAnalyst (Production Flow) ===\n")
        
#         # 1. 构造真实的 GSM8K 风格数据 (Math Word Problems)
#         # Context: "A shop sells apples for $2 and oranges for $3. Alice buys 5 apples and 3 oranges."
#         # Correct Answer: 5*2 + 3*3 = 10 + 9 = 19.
        
#         refs = ["19"] * 5
        
#         preds = [
#             # Case 1: Correct
#             "Alice buys 5 apples ($10) and 3 oranges ($9). Total is 19.", 
            
#             # Case 2: Calculation Error (Logic correct, arithmetic wrong)
#             "5 apples cost 5*2=10. 3 oranges cost 3*3=6. Total is 10+6=16.", 
            
#             # Case 3: Semantic/Logic Error (Confused unit prices)
#             "Apples are $3 and oranges are $2. So 5*3 + 3*2 = 15 + 6 = 21.",
            
#             # Case 4: Hallucination/Irrelevant
#             "The weather is nice today. I think the answer might be 20.",
            
#             # Case 5: Correct
#             "19"
#         ]
        
#         print(f"[Data] Loaded {len(preds)} samples. (Expect mix of correct and errors)")
        
#         # 2. 定义明确的用户需求
#         instruction = (
#             "请详细分析这些错误样本（Negative Cases）。"
#             "我特别想知道：模型是简单的计算错误（Calculation Error），"
#             "还是在这个应用题的语义理解（Semantic Understanding）上出了问题？"
#             "请引用具体的错误回复片段来佐证你的分析。"
#         )
#         print(f"\n[User Instruction]\n{instruction}\n")
        
#         # 3. 真实调用
        
#         print("[System] Executing compute_case_study_analyst (Calling Real LLM)...")
        
#         # 直接调用，不 Mock 任何东西
#         result = compute_case_study_analyst(
#             preds=preds, 
#             refs=refs, 
#             sample_size=3,              # 抽样 3 个
#             target_group="negative",    # 只看错题
#             instruction=instruction,
#             model_name="gpt-4o",        # 指定模型
#             # api_key="sk-..."          # 如果环境变量没配，可以在这里硬编码测试
#         )
        
#         print("\n" + "="*40)
#         print("         ANALYSIS RESULT         ")
#         print("="*40)
        
#         if result.get("error"):
#             print(f"❌ Execution Failed: {result['error']}")
#         else:
#             print(f"✅ Score: {result.get('score')}")
#             print(f"📍 Sampled Indices: {result.get('details')}")
#             print("\n📝 [LLM Analysis Report]:\n")
#             print(result.get("analysis"))
#             print("\n" + "-"*40)

#     # Run directly (Sync)
#     run_real_demo()
