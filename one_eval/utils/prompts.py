from __future__ import annotations
from typing import Dict
from pydantic import BaseModel
from one_eval.logger import get_logger
import json

log = get_logger(__name__)



class PromptTemplate(BaseModel):
    """通用 Prompt 模板格式"""
    name: str
    text: str

    def build_prompt(self, **kwargs) -> str:
        return self.text.format(**kwargs)


class PromptRegistry:
    """Prompt 注册中心：全局唯一"""

    def __init__(self):
        self.prompts: Dict[str, PromptTemplate] = {}

    def register(self, name: str, text: str):
        """注册 prompt"""
        self.prompts[name] = PromptTemplate(name=name, text=text)

    def get(self, name: str) -> PromptTemplate:
        if name not in self.prompts:
            log.error(f"[PromptRegistry] 未找到 prompt: {name}")
        return self.prompts[name]


# ----------- 单例实例 -----------
prompt_registry = PromptRegistry()


# ======================================================
# 在下面注册项目所有 prompt
# ======================================================

# -------- Step1: QueryUnderstand Agent --------
prompt_registry.register(
    "query_understand.system",
    """
你是 One-Eval 系统中的 QueryUnderstandAgent。
你的任务是读取用户自然语言输入并输出一个结构化 JSON:
{{
  "is_eval_task": Bool,
  "is_mm": Bool,
  "add_bench_request": Bool,
  "domain": [str, ...],
  "specific_benches": [str, ...],
  "model_path": [str, ...],
  "special_request": str
}}
不要解释，不要添加额外内容，只输出 JSON。
""",
)

prompt_registry.register(
    "query_understand.task",
    """
用户输入如下：

{user_query}

请你根据以上内容严格返回 JSON (必须可被 json.loads 解析):
{{
  "is_eval_task": 是否为评测任务(bool类型),
  "is_mm": 是否涉及多模态任务(bool类型),
  "add_bench_request": 是否用户自备了数据集作为benchmark 需要我们帮忙配置好参数(bool类型),没有这个需求则为 False,
  "domain": ["math", "medical", ...],  # 评测任务的领域，如 ["text", "math", "code", "reasoning", ...]，可以写多个标签，只要是相关的领域都可以，注意同一个标签可以写多个不同的别名，以方便检索时匹配，包括但不限于简写等
  "specific_benches": ["gsm8k", "mmlu", ...],  # 由用户提出的必须评测的指定 benchmark 列表，没有则填写 None
  "model_path": ["gpt-4o", "local://qwen", ...],  # 被测模型名或本地路径，从用户给的文字描述中寻找，没有则填写 None
  "special_request": "其他无法结构化但依旧重要的需求文本"  # 其他无法结构化但依旧重要的需求,用文字记录用于后续处理
}}
"""
)

# ======================================================
# Step 2: BenchSearch (名称推荐) Prompts
# ======================================================

prompt_registry.register(
    "bench_search.system",
    """
你是 One-Eval 系统中的 BenchSearchAgent（具体由 BenchNameSuggestAgent 实现）。
你的工作是根据用户的任务需求，推荐合适的 benchmark 名称列表。

你需要遵守以下要求：
1. 你只负责“给出 benchmark 名称”，具体下载与评测由后续模块完成。
2. 你必须优先考虑在学术界 / 工业界广泛使用的、公开的评测基准。
3. 输出形式必须是严格的 JSON，能够被 Python 的 json.loads 正确解析。
4. 不要输出任何解释性文字、注释、Markdown，仅输出 JSON。
"""
)

prompt_registry.register(
    "bench_search.hf_query",
    """
下面是与当前评测任务相关的信息。请你根据这些信息，给出“推荐的 benchmark 名称列表”。

你需要返回一个 JSON，格式必须严格为：

{{
  "bench_names": [
    "gsm8k",
    "openai/gsm8k",
    "HuggingFaceH4/MATH-500",
    "mmlu",
    "truthful_qa"
  ]
}}

要求：
1. "bench_names" 的值是一个字符串数组，每个元素是一个可能的 benchmark 名称。
2. 如果你知道 HuggingFace 上的完整仓库名（例如 "openai/gsm8k"、"HuggingFaceH4/MATH-500"），优先使用完整仓库名。
3. 如果你不确定仓库前缀，可以只给出常用简称（例如 "gsm8k"、"mmlu"），后续系统会尝试匹配。
4. 不要包含与评测无关的数据集（例如纯预训练语料、无标注文本、通用聊天日志等）。
5. 不要输出除上述 JSON 以外的任何内容。

----------------
用户原始需求:
{user_query}

任务领域:
{domain}

本地已经找到的 benchmark:
{local_benches}
"""
)
