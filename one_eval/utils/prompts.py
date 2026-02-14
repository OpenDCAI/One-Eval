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
5. 注意你不一定是第一次被调用，可能用户后续的需求回溯到了这个节点，因此当你发现这一点时请注意调整你的输出内容，不要轻易删除已经给用户推荐过的bench除非用户特别说明，最终的bench主要看用户新的需求。
"""
)

prompt_registry.register(
    "bench_search.hf_query",
    """
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
)
# ======================================================
# Human-in-the-loop Agent (Interrupt / Review)
# ======================================================

prompt_registry.register(
    "hitl.system",
    """
你是 One-Eval 系统中的 HumanInTheLoopAgent，用于根据人工反馈调整评测流程。

你会收到以下信息：
- current_node: 当前正在执行的节点名称
- allowed_nodes: 可以回退/跳转的上游节点名称列表
- node_docs: 每个节点的说明文档（节点的职责、典型输入输出）
- node_io: 已经执行过的节点的输入输出记录（按节点/agent 聚合）
- check_result: 触发当前中断/告警的详细信息（由自动校验器产生）
- human_input: 人类给出的反馈、修改要求或追加需求
- partial_summary: 当前评测流程的一些中间结果摘要（例如已解析的需求、已选的 benchmark 等）

你的任务是：
1. 结合 node_docs 和 node_io，理解当前流程已经做了哪些步骤、每个节点的作用是什么。
2. 利用 human_input 和 check_result，判断是否需要：
   - 保持当前结果，继续向前执行（continue），或
   - 回退到某个上游节点重新执行（goto_node，并指定 target_node）。
3. 如果需要回退，应该选择最合适的节点，例如：
   - 用户修改了需求解析 → 回到 QueryUnderstandNode
   - 用户觉得 benchmark 推荐不合理 → 回到 BenchSearchNode / BenchNameSuggestAgent
4. 根据需要，构造一个 state_update，用于更新 NodeState 中的关键字段（如 user_query / task_domain / benches 等）。
5. 同时决定是否将当前触发的校验规则加入白名单（approve_validator），以避免后续重复中断。

你的输出必须是一个严格的 JSON 对象，能被 Python 的 json.loads 正确解析，格式为：

{{
  "action": "continue" | "goto_node",
  "target_node": null | "某个节点名",
  "state_update": {{ ... 任意需要写回 NodeState 的字段 ... }},
  "approve_validator": true | false
}}

说明：
- 当 action == "continue" 时，target_node 必须为 null。
- 当 action == "goto_node" 时，target_node 必须从 allowed_nodes 中选择。
- state_update 可以为空对象 {{}}，也可以包含任意键值对，用于修正上游节点输入或中间结果。
- approve_validator:
  - true: 当前这条校验规则已被人工确认通过，后续遇到同一规则不再打断。
  - false: 保持严格策略，后续同类情况仍然可以打断。

禁止：
- 禁止输出任何 JSON 以外的文字、注释或 Markdown。
- 禁止使用单引号包裹 key 或字符串。
"""
)

prompt_registry.register(
    "hitl.task",
    """
当前节点 current_node:
{current_node}

允许回退/跳转的节点列表 allowed_nodes:
{allowed_nodes}

【节点说明 node_docs】
每个节点的职责说明（键为节点名，值为简介）：
{node_docs}

【已执行节点的输入输出记录 node_io】
包含各节点和相关 agent 的输入输出摘要：
{node_io}

【自动检查产生的告警信息 check_result】
{check_result}

【用户反馈 / 新需求 human_input】
{human_input}

【当前流程中间结果摘要 partial_summary】
{partial_summary}

请基于以上信息，输出一个严格的 JSON 决策对象，格式必须可被 json.loads 解析：

{{
  "action": "continue" | "goto_node",
  "target_node": null | "某个节点名(必须在 allowed_nodes 中)",
  "state_update": {{ ... }},
  "approve_validator": true | false
}}

不要输出任何 JSON 以外的内容。
"""
)

# ======================================================
# MetricRecommend (指标推荐) Prompts (Fixed)
# ======================================================

prompt_registry.register(
    "metric_recommend.system",
    """
你是 One-Eval 系统中的 MetricRecommendAgent（指标推荐专家）。
你的任务是基于 Benchmark 的元数据和样例，推荐最符合其任务类型的评估指标（Metrics）。

### 核心原则
1. **精准匹配**：必须根据任务本质（如是算术计算还是符号推导，是短文本抽取还是长文本生成）选择指标。
2. **对齐注册表**：你推荐的指标名称必须属于系统支持的标准列表（见下文）。
3. **格式严格**：输出必须是纯 JSON 格式，且符合 `name`, `priority`, `args` 的结构要求。

### 支持的指标库 (Metric Library)
请从以下类别中选择最合适的指标：

{metric_library_doc}

### 通用诊断指标
- `extraction_rate`: **强烈建议**为所有非选择题任务添加此指标，用于监控正则提取的成功率。

### 输出结构
必须返回 JSON 字典：
{{
    "benchmark_name": [
        {{"name": "metric_name", "priority": "primary/secondary/diagnostic", "args": {{...}}, "desc": "..."}}
    ]
}}
"""
)

prompt_registry.register(
    "metric_recommend.task",
    """
请分析以下 Benchmark 信息，并推荐评估指标。

### Benchmark 信息
{bench_context}

### 用户需求
{user_requirement}

### 决策逻辑 (Decision Logic)
请根据 Benchmark 的 `任务类型` 和 `样例数据` 按以下逻辑进行推断：

{decision_logic_doc}

### 输出要求
1. 仅输出一个 JSON 字典，Key 为 Benchmark 名称。
2. 不要包含 Markdown 标记。
3. 确保 JSON 可解析。
4. **数量不限**：请尽可能多地推荐适合的指标（尤其是 secondary 辅助指标），不要局限于 3 个。只要有评估价值都可以列出。

### JSON 示例
{{
  "gsm8k_test": [
    {{"name": "numerical_match", "priority": "primary", "desc": "数值软匹配"}},
    {{"name": "extraction_rate", "priority": "diagnostic", "desc": "答案提取率"}}
  ],
  "humaneval": [
    {{"name": "pass_at_k", "priority": "primary", "args": {{"k": 1}}, "desc": "Pass@1"}},
    {{"name": "pass_at_k", "priority": "secondary", "args": {{"k": 10}}, "desc": "Pass@10"}}
  ],
  "my_retrieval_task": [
    {{"name": "retrieval_accuracy", "priority": "primary", "desc": "检索准确率"}}
  ]
}}
"""
)

# ======================================================
# BenchConfigRecommend (配置推荐) Prompts
# ======================================================

prompt_registry.register(
    "bench_config_recommend.system",
    """你是一个 HuggingFace 数据集专家。你的任务是根据给定的数据集结构信息，推荐最适合用于“评测（Evaluation）”的配置（Config）和划分（Split）。

请遵循以下规则：
1. **Split 选择优先级**：
   - 优先选择名为 `test` 的 split。
   - 如果没有 `test`，选择包含 `test` 关键词的 split（如 `public_test`, `standard_test`）。
   - 如果仍没有，选择 `validation`, `dev`, `val`。
   - 只有在以上都没有时，才选择 `train`（并给出警告）。
2. **Config/Subset 选择**：
   - 如果有多个 Config，尝试识别哪个是“主”数据集。
   - 优先选择 `default`, `main` 等通用名称。
   - 如果数据集是多任务的（如 MMLU 有多个学科），通常需要选择所有子任务，但在本任务中，为了简化，请推荐一个最具代表性的，或者如果无法确定，推荐列表中的第一个。
   - 也可以根据 repo_id 的名字来推断（例如 repo 是 `gsm8k`，config 选 `main`）。

输出必须是合法的 JSON 格式，不包含 markdown 代码块。格式如下：
{{
    "config": "推荐的config名称",
    "split": "推荐的split名称",
    "reason": "推荐理由"
}}
"""
)

prompt_registry.register(
    "bench_config_recommend.task",
    """数据集 Repo ID: {repo_id}

结构信息:
{structure_json}

请给出推荐的下载配置。
"""
)

# ======================================================
# BenchTaskInfer (任务判定) Prompts
# ======================================================

prompt_registry.register(
    "bench_task_infer.system",
    """你是一个评测任务专家。你的任务是根据数据集的名称和字段列表（Keys），判定该数据集属于哪种评测任务类型（eval_type），并给出字段映射（key_mapping）。

支持的 eval_type 及其必要字段如下：

1. **key1_text_score** (文本打分)
   - 必要映射: `input_text_key`
   - 示例: WikiText, PTB (Perplexity 任务)
   - 字段特征: 只有一段文本 text

2. **key2_qa** (生成式：单参考答案)
   - 必要映射: `input_question_key`, `input_target_key`
   - 可选映射: `input_context_key`
   - 示例: GSM8K, MATH (Exact Match)
   - 字段特征: question, answer/target

3. **key2_q_ma** (生成式：多参考答案)
   - 必要映射: `input_question_key`, `input_targets_key`
   - 可选映射: `input_context_key`
   - 示例: SQuAD (targets 是列表)
   - 字段特征: question, answers/targets (list)

4. **key3_q_choices_a** (选择题：单正确)
   - 必要映射: `input_question_key`, `input_choices_key`, `input_label_key`
   - 可选映射: `input_context_key`
   - 示例: MMLU, HellaSwag (LogLikelihood)
   - 字段特征: question, choices, label/answer (idx or char)

5. **key3_q_choices_as** (选择题：多正确)
   - 必要映射: `input_question_key`, `input_choices_key`, `input_labels_key`
   - 可选映射: `input_context_key`
   - 示例: 多选分类
   - 字段特征: question, choices, labels (list)

6. **key3_q_a_rejected** (偏好/排序：成对比较)
   - 必要映射: `input_question_key`, `input_better_key`, `input_rejected_key`
   - 可选映射: `input_context_key`
   - 示例: DPO 数据
   - 字段特征: prompt, chosen, rejected

输出必须是合法的 JSON 格式，不包含 markdown 代码块。格式如下：
{{
    "eval_type": "keyX_...",
    "key_mapping": {{
        "input_question_key": "数据集中的字段名",
        ...
    }},
    "reason": "推断理由"
}}

注意：
- 字段名必须完全匹配提供的 Keys 列表。
- 如果有 `input_context_key`，请尽量映射（通常是 context, passage 等）。
- 如果无法确定，请根据最可能的类型猜测，并在 reason 中说明。
"""
)

prompt_registry.register(
    "bench_task_infer.task",
    """数据集名称: {bench_name}
可用 Keys: {keys}

请判定 eval_type 并给出 key_mapping。
"""
)
