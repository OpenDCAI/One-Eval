# Eval Types — 6 种评测类型与 key_mapping 契约（硬规则）

One-Eval 只做**纯文本 LLM 评测**。每个 benchmark 必须归入下列 **6 种 eval_type 之一**，
并按该类型填全 `key_mapping` 必填字段。这是评测内核（dataflow）的硬契约 —— 字段缺失或
类型不符，`run_eval.py` 会在构造 `BenchInfo` 时直接报错。

> 与 `one_eval/core/metric_registry.py:MetricCategory`、
> `scripts/_common.py:REQUIRED_KEYS` 完全一致。改动须同步三处。

## 选型决策树（agent 接入新 bench 时按此判断）

1. **没有标准答案、只对一段文本本身打分？**（如流畅度/毒性/质量评分）
   → `key1_text_score`
2. **问答题，单个标准答案？**（开放问答、数学题、填空）
   → `key2_qa`
3. **问答题，多个可接受答案？**（任一命中即算对）
   → `key2_q_ma`
4. **选择题，单个正确选项？**（MMLU 这类 A/B/C/D 单选）
   → `key3_q_choices_a`
5. **选择题，多个正确选项？**（多选，全中才算对）
   → `key3_q_choices_as`
6. **成对偏好，判断哪条更好？**（RLHF chosen/rejected 偏好对）
   → `key3_q_a_rejected`

## 各类型必填 key_mapping 字段

| eval_type | 必填字段 | 含义 |
|---|---|---|
| `key1_text_score` | `input_text_key` | 待打分文本所在列 |
| `key2_qa` | `input_question_key`, `input_target_key` | 问题列、单一答案列 |
| `key2_q_ma` | `input_question_key`, `input_targets_key` | 问题列、答案**列表**列 |
| `key3_q_choices_a` | `input_question_key`, `input_choices_key`, `input_label_key` | 问题、选项列表、正确标签 |
| `key3_q_choices_as` | `input_question_key`, `input_choices_key`, `input_labels_key` | 问题、选项列表、正确标签**列表** |
| `key3_q_a_rejected` | `input_better_key`, `input_rejected_key` | 更优答案列、被拒答案列 |

可选辅助字段（按需）：`input_context_key`（题干上下文/passage）。

## 关键：嵌套字段必须先拍平

`key_mapping` 的值是**顶层列名**。如果 HF 数据集字段是多层嵌套
（如 `extra.choices.text`、`answers[0].text`），dataflow 取不到。

接入流程：
1. 先用 `scripts/prepare_bench.py --repo-id <repo> --split <split>` 下载并**预览嵌套结构**
   （它会用点路径打印出 `a.b.c` / `key[]` 形式的字段树）。
2. 若发现目标字段是嵌套的，**先写一段预处理代码**把它拍平到顶层 jsonl key，
   再把拍平后的列名填进 `key_mapping`。
3. 拍平后用 `prepare_bench.py --preview-only <拍平后的.jsonl>` 复核结构无误，再进入评测。

## 例子

**key2_qa（MATH-500）**：顶层已有 `problem` / `answer`，直接映射。
```yaml
bench_dataflow_eval_type: "key2_qa"
key_mapping:
  input_question_key: "problem"
  input_target_key: "answer"
```

**key3_q_choices_a（MMLU）**：顶层有 `question` / `choices`（list）/ `answer`（int 索引）。
```yaml
bench_dataflow_eval_type: "key3_q_choices_a"
key_mapping:
  input_question_key: "question"
  input_choices_key: "choices"
  input_label_key: "answer"
```
