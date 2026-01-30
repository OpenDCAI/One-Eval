# One-Eval 指标系统指南：注册与推荐机制

本文档详细介绍了 One-Eval 的指标系统，包括如何注册新的评测指标，以及系统是如何自动为不同的数据集推荐合适的指标的。

---

## 1. 指标推荐机制 (Metric Recommendation)

One-Eval 采用了一套 **四层优先级架构 (4-Layer Priority Model)** 来决定最终使用哪些指标进行评测。这套机制由 `MetricRecommendAgent` 编排，结合了静态规则和 LLM 的动态分析能力。

### 推荐流程架构

优先级从高到低如下：

#### **第1层：用户强制指定 (User Override)**
*   **优先级**： **最高 (Highest)**
*   **逻辑**：系统首先检查 `Benchmark` 元数据中是否显式包含了 `metrics` 字段。
*   **行为**：如果用户手动指定了指标（例如在配置文件或代码中硬编码），系统将**直接使用**，完全跳过后续的查表和 LLM 分析步骤。
*   **适用场景**：用户明确知道自己想要什么，或进行特定实验。

#### **第2层：注册表查表 (Registry Lookup)**
*   **优先级**： **静态建议 (System Suggestion)**
*   **逻辑**：系统根据数据集名称（归一化为小写）在 `DATASET_METRIC_MAP_CONFIG` (`one_eval/metrics/config.py`) 中查找预定义的映射关系。
    
    **映射关系示例**：
    系统维护了一个 `Dataset Name -> Group Key` 的映射表。
    ```python
    # one_eval/metrics/config.py
    DATASET_METRIC_MAP_CONFIG = {
        "gsm8k": "numerical",       # GSM8K 映射到 numerical 组
        "humaneval": "code",        # HumanEval 映射到 code 组
        "squad20": "qa_extractive"  # SQuAD 2.0 映射到 qa_extractive 组
    }
    ```
    当评测 `gsm8k` 时，系统会查到 `numerical`，然后拉取所有注册时 `groups` 包含 `numerical` 的 Metric（如 `exact_match`）。
*   **行为**：
    *   查到的结果**不会直接生效**，而是作为 **"上下文建议"** 提供给 LLM 参考。
    *   只有在第3层 LLM 调用失败时，此结果才会作为**首选兜底方案**生效。
*   **适用场景**：处理业界标准的公开数据集，提供最规范的默认配置。

#### **第3层：LLM 智能决策 (LLM Analyst)**
*   **优先级**： **核心决策层 (Main Logic)**
*   **逻辑**：Agent 收集 Benchmark 的全量信息（Metadata、Prompt 模板、数据样例 Preview、以及第2层的查表建议），调用 LLM (`gpt-4o`) 进行综合分析。
*   **行为**：
    *   LLM 会分析数据样例是“选择题”、“代码生成”还是“长文本”，并结合 Prompt 的要求，从指标库中挑选最合适的指标。
    *   LLM 的输出将覆盖查表建议（除非 LLM 显式采纳了建议）。
*   **适用场景**：
    *   **未知/私有数据集**：自动分析数据特征推断指标。
    *   **复杂需求**：理解用户的自然语言指令（如“请用宽松的匹配标准”）。

#### **第4层：最终兜底 (Safe Fallback)**
*   **优先级**：🛡️ **最低 (Safety Net)**
*   **逻辑**：当上述所有步骤都失效（未指定、LLM 挂了、查表无结果）时触发。
*   **行为**：使用系统默认的 **安全模式**：
    *   `exact_match` (Primary)
    *   `extraction_rate` (Diagnostic)
*   **适用场景**：防止 Pipeline 因无指标而崩溃。

---

## 2. 如何注册新的评测指标 (Metric)

One-Eval 采用 **去中心化** 的 Metric 注册机制。你不需要修改中心化的配置文件，只需在函数上添加装饰器。

### Step 1: 确定代码位置
所有的 Metric 实现代码都存放在 `one_eval/metrics/common/` 目录下。
*   **复用现有文件**：如 `classification.py`, `text_gen.py`。
*   **新建文件**：如 `one_eval/metrics/common/my_custom_metric.py`（会自动被扫描加载）。

### Step 2: 编写计算函数并注册

```python
from typing import List, Any, Dict
from one_eval.core.metric_registry import register_metric

@register_metric(
    name="my_accuracy",                      # [必填] 指标唯一名称
    desc="计算预测值与真实值的精确匹配度",      # [必填] 供 Agent 理解的描述
    usage="适用于分类任务或简答题",            # [选填] 适用场景
    groups={                                 # [关键] 推荐模板组
        "numerical": "secondary",            # 在数值任务中作为次要指标
        "qa_extractive": "primary"           # 在抽取式问答中作为主要指标
    },
    aliases=["acc", "match_rate"]            # [选填] 别名
)
def compute_my_accuracy(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Args:
        preds: 预测结果列表
        refs: 真实标签列表
    Returns:
        Dict: 必须包含 'score' 字段
    """
    correct = 0
    # ... 计算逻辑 ...
    score = correct / len(refs) if refs else 0.0
    
    return {
        "score": score,
        "details": [...] # 可选：每个样本的得分
    }
```

### 推荐组 (Groups) 与 优先级 (Priority)

One-Eval 的推荐系统基于“模板组”工作。例如，当你在注册指标时将其归入 `numerical` 组，那么：
1.  如果某个数据集在配置文件中被显式映射到 `numerical` 组（如 GSM8K），该指标会被自动推荐。
2.  LLM 在分析未知数据集时，也可以选择引用整个 `numerical` 组的指标。

*   **`primary` (主要指标)**：会被优先推荐，通常是该组的核心指标。
*   **`secondary` (次要/参考指标)**：作为补充指标推荐。

---

## 3. 附录：现有 Metric Groups (推荐模板) 列表

当你在注册 Metric 时，你需要知道系统支持哪些 `Group Key` (即 `groups` 的 key)，以便正确归类。

这些 Group Key 主要用于：
1.  **静态映射**：在 `metrics/config.py` 中将数据集名称直接映射到这组指标。
2.  **LLM 检索**：LLM 可能会根据任务性质选择加载某一组指标作为参考。

| Group Key (模板名) | 典型应用场景 | 包含的典型指标 |
| :--- | :--- | :--- |
| **`numerical`** | 数学计算、算术题 | `exact_match` (数字/公式匹配) |
| **`symbolic`** | 符号推导、复杂数学 (LaTeX) | `symbolic_match` (支持 LaTeX/Boxed 解析) |
| **`choice`** | 多项选择题 (MMLU, CEval) | `choice_accuracy`, `exact_match` |
| **`code`** | 代码生成 (HumanEval, MBPP) | `pass@k`, `execution_accuracy` |
| **`generation_bleu`** | 机器翻译 (WMT, IWSLT) | `bleu`, `chrf` |
| **`generation_rouge`** | 文本摘要 (CNN/DM, LCSTS) | `rouge-1`, `rouge-2`, `rouge-l` |
| **`qa_extractive`** | 抽取式问答 (SQuAD) | `exact_match`, `f1_score` |
| **`long_context_qa`** | 长文本问答 (LongBench) | `f1_score`, `rouge` (往往加权或特殊处理) |
| **`retrieval`** | 检索增强生成 (RAG) | `recall@k`, `ndcg` |
| **`count`** | 计数类任务 | `count_error`, `exact_match` |
| **`code_sim`** | 代码相似度 | `code_bleu` |
| **`llm_judge`** | 主观题 (Arena, MT-Bench) | `llm_score` (使用 LLM 打分) |
| **`win_rate`** | 两两对抗 (Pairwise) | `win_rate` |
| **`auc_roc`** | 二分类/多分类概率评估 | `auc`, `roc` |
| **`safety_toxicity`** | 安全性检测 | `toxicity_score` |
| **`truthfulness`** | 真实性/幻觉检测 | `truth_score` |

> **提示**：如果你发现现有的 Group 都不适合你的新 Metric，你可以自定义一个新的 Group Key（例如 `"my_special_task"`）。但你需要确保在数据集配置中也能对应上这个 Key，否则该 Metric 只有在被 LLM 显式选中时才会生效。
