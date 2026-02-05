# One-Eval 指标系统指南：注册与推荐机制

本文档详细介绍了 One-Eval 的指标系统，包括如何注册新的评测指标，以及系统是如何自动为不同的数据集推荐合适的指标的。

---

## 1. 指标推荐机制 (Metric Recommendation)

One-Eval 采用了一套 **四层优先级架构 (4-Layer Priority Model)** 来决定最终使用哪些指标进行评测。这套机制由 `MetricRecommendAgent` 编排，结合了静态配置和 LLM 的动态分析能力。

### 推荐流程架构

优先级从高到低如下：

#### **第1层：用户强制指定 (User Override)**
*   **优先级**： **最高 (Highest)**
*   **逻辑**：系统首先检查 `Benchmark` 元数据中是否显式包含了 `metrics` 字段。
*   **行为**：如果用户手动指定了指标（例如在配置文件或代码中硬编码），系统将**直接使用**，完全跳过后续的查表和 LLM 分析步骤。
*   **适用场景**：用户明确知道自己想要什么，或进行特定实验。

#### **第2层：注册表静态映射 (Registry Lookup)**
*   **优先级**： **静态建议 (System Suggestion)**
*   **逻辑**：系统根据数据集名称（归一化为小写）在 `one_eval/metrics/config.py` 的 `DATASET_METRICS` 中查找预定义的映射关系。
    
    **配置示例** (`one_eval/metrics/config.py`)：
    ```python
    DATASET_METRICS = {
        "gsm8k": ["numerical_match", "extraction_rate"],  # 这里的顺序很重要
        "mmlu": ["accuracy"],
        # ...
    }
    ```
*   **运行时优先级推断 (Runtime Priority Inference)**：
    为了简化配置，系统会自动推断指标的优先级：
    1.  **Primary (主要指标)**：列表中的**第一个**指标。
    2.  **Diagnostic (诊断指标)**：系统内置的诊断类指标（如 `extraction_rate`, `format_compliance`）。
    3.  **Secondary (次要指标)**：列表中的其他指标。
*   **行为**：
    *   查到的结果**不会直接生效**，而是作为 **"上下文建议"** 提供给 LLM 参考。
    *   只有在第3层 LLM 调用失败或决定采纳建议时，此结果才会生效。
*   **适用场景**：处理业界标准的公开数据集，提供最规范的默认配置。

#### **第3层：LLM 智能决策 (LLM Analyst)**
*   **优先级**： **核心决策层 (Main Logic)**
*   **逻辑**：Agent 收集 Benchmark 的全量信息（Metadata、Prompt 模板、数据样例 Preview、以及第2层的查表建议），调用 LLM (`gpt-4o`) 进行综合分析。
*   **行为**：
    *   LLM 会分析数据特征，并结合 Prompt 的要求，从指标库中挑选最合适的指标。
    *   LLM 的输出将覆盖查表建议（除非 LLM 显式采纳了建议）。
*   **适用场景**：
    *   **未知/私有数据集**：自动分析数据特征推断指标。
    *   **复杂需求**：理解用户的自然语言指令（如“请用宽松的匹配标准”）。

#### **第4层：最终兜底 (Safe Fallback)**
*   **优先级**： **最低 (Safety Net)**
*   **逻辑**：当上述所有步骤都失效时触发。
*   **行为**：使用系统默认的 **安全模式**：
    *   `exact_match` (Primary)
    *   `extraction_rate` (Diagnostic)

---

## 2. 如何注册新的评测指标 (Metric)

One-Eval 采用 **轻量级去中心化** 的 Metric 注册机制。你只需要关注指标本身的实现，**不再需要**在注册时指定复杂的 Group 或 Priority 信息。

### Step 1: 确定代码位置
所有的 Metric 实现代码都存放在 `one_eval/metrics/common/` 目录下。
*   **复用现有文件**：如 `classification.py`, `text_gen.py`。
*   **新建文件**：如 `one_eval/metrics/common/my_custom_metric.py`（会自动被扫描加载）。

### Step 2: 编写计算函数并注册

使用 `@register_metric` 装饰器，并指定 `MetricCategory`。

```python
from typing import List, Any, Dict
from one_eval.core.metric_registry import register_metric, MetricCategory

@register_metric(
    name="my_accuracy",                      # [必填] 指标唯一名称
    desc="计算预测值与真实值的精确匹配度",      # [必填] 供 Agent 理解的描述
    usage="适用于分类任务或简答题",            # [选填] 适用场景
    categories=[MetricCategory.CHOICE_SINGLE, MetricCategory.QA_SINGLE], # [必填] 指标类别
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
    # ... 计算逻辑 ...
    return {
        "score": 0.95,
        "details": [...] 
    }
```

**注意：**
*   不再需要 `groups` 参数。
*   不再需要 `priority` 参数。
*   `categories` 帮助 LLM 理解该指标适用于哪类任务（如选择题、文本生成、代码等）。

---

## 3. 如何为数据集配置默认指标 (Optional)

如果你希望某个数据集（如 `my_dataset`）在没有 LLM 介入时也能自动使用你注册的指标，或者希望给 LLM 提供一个强参考，你可以修改 `one_eval/metrics/config.py`。

### 修改 `DATASET_METRICS`

在 `one_eval/metrics/config.py` 文件中，找到 `DATASET_METRICS` 字典并添加映射：

```python
DATASET_METRICS = {
    # ... 现有配置 ...
    "gsm8k": ["numerical_match", "extraction_rate"],
    
    # 新增你的数据集
    # 列表第一个元素 "my_accuracy" 会自动被识别为 Primary 指标
    "my_dataset": ["my_accuracy", "extraction_rate"]
}
```

这样，当系统评测 `my_dataset` 时，会自动推荐 `my_accuracy` 作为主要指标。
