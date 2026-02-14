# One-Eval 指标系统指南：推荐机制与注册

本文档基于当前实现说明 One-Eval 的指标推荐流程与注册方式，帮助你理解系统如何选指标、以及如何扩展新的 Metric。

---

## 1. 核心组件概览

- **MetricRecommendAgent**：主导推荐流程，LLM 优先，规则建议只作为参考。
- **MetricDispatcher**：根据数据集名称查表给出规则建议，并补全默认优先级。
- **Metric Registry**：注册与加载指标实现，提供给 LLM 的指标库与描述信息。
- **DATASET_METRICS**：数据集到指标列表的静态映射配置。

---

## 2. 指标推荐流程 (Runtime)

系统采用“LLM 优先 + 规则建议兜底”的流程，优先级从高到低如下。

### 2.1 用户强制指定 (User Override)
- **触发条件**：`Benchmark.meta` 中存在 `metrics` 字段。
- **行为**：直接使用并验证用户指定的指标列表，不再经过 LLM。

示例（在 Benchmark 的 meta 中指定）：
```json
{
  "metrics": [
    {"name": "exact_match", "priority": "primary"},
    {"name": "extraction_rate", "priority": "diagnostic"},
    {"name": "token_f1", "priority": "secondary", "args": {"average": "macro"}}
  ]
}
```

### 2.2 Registry 查表建议 (Rule-based Suggestion)
- **数据来源**：`one_eval/metrics/config.py` 的 `DATASET_METRICS`。
- **匹配方式**：数据集名会被归一化后做“包含匹配”，最长匹配优先。
- **优先级推断规则**：
  - 列表第一个指标为 `primary`。
  - `extraction_rate` / `format_compliance` 为 `diagnostic`。
  - 其他为 `secondary`。
- **注意**：该结果**不会直接生效**，只作为 LLM 的上下文建议。

### 2.3 LLM 智能推荐 (Primary Path)
- **输入上下文**：
  - Benchmark 的元信息（任务类型、领域、描述等）
  - `bench_dataflow_eval_type` / `eval_type`
  - Prompt 模板（截断）
  - 样例数据预览（仅读取 `eval_detail_path` 的前若干条）
  - 规则建议（Registry Suggestion）
- **指标库**：由 Registry 动态生成，包含所有已注册指标及其描述/适用场景。
- **输出格式**：每个 Benchmark 对应一个指标列表（含 `name`、可选 `priority`、可选 `args`）。

### 2.4 最终兜底 (Fallback)
当 LLM 未返回有效结果时：
- 若 Registry 有建议，使用其结果。
- 否则使用默认兜底：`exact_match`(primary) + `extraction_rate`(diagnostic)。

---

## 3. 注册新的评测指标 (Metric)

系统自动扫描 `one_eval/metrics/common/` 下的所有模块并注册指标，因此你只需要实现函数并添加装饰器。

### Step 1: 选择文件位置
- 复用现有文件：如 `classification.py`、`text_gen.py`。
- 新建文件：如 `one_eval/metrics/common/my_custom_metric.py`。

### Step 2: 编写函数并注册

```python
from typing import List, Any, Dict
from one_eval.core.metric_registry import register_metric, MetricCategory

@register_metric(
    name="my_accuracy",
    desc="计算预测值与真实值的精确匹配度",
    usage="适用于分类任务或简答题",
    categories=[MetricCategory.CHOICE_SINGLE, MetricCategory.QA_SINGLE],
    aliases=["acc", "match_rate"]
)
def compute_my_accuracy(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    correct = 0
    total = min(len(preds), len(refs))
    for p, r in zip(preds, refs):
        if p == r:
            correct += 1
    score = correct / total if total else 0.0
    return {"score": score}
```

要点：
- `name` 可省略，默认取函数名（`compute_xxx` 会自动变成 `xxx`）。
- `desc` 和 `usage` 会直接进入 LLM 的指标库说明。
- `categories` 使用 `MetricCategory` 常量，帮助 LLM 判断指标适用的任务类型。
- 返回的字典必须包含 `score` 字段。
- 无需提供 `groups` 或 `priority` 参数。

---

## 4. 为数据集配置默认指标 (可选)

编辑 `one_eval/metrics/config.py` 中的 `DATASET_METRICS`，增加数据集到指标列表的映射。

```python
DATASET_METRICS = {
    "gsm8k": ["numerical_match", "strict_match", "extraction_rate"],
    "my_dataset": ["my_accuracy", "extraction_rate"]
}
```

说明：
- 列表顺序会影响默认优先级推断。
- 数据集名支持包含匹配，`my_dataset_v2` 也能命中 `my_dataset`。
- 该配置只是建议，LLM 仍可能根据任务特征调整最终推荐。
