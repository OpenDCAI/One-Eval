# Metric Registry — 多维度打分与自定义指标

dataflow 主评测给每个 eval_type 一个**默认主分数**。注册表让你在主分数之外，
按需挑选**额外维度**，形成多维评分。用户也可与你聊天，**临时写一个新 metric**。

> 实时查看可用 metric（最权威，按维度分组，含别名/类别）：
> ```bash
> python scripts/run_metrics.py --list
> ```
> 下表是源码快照（`one_eval/metrics/common/`），以 `--list` 为准。

## 维度（dimension）是什么

每个 metric 带一个 `dimension` 标签，表示它衡量**哪条质量轴**，与 `categories`
（适用哪种题型 eval_type）正交。挑指标时先按题型筛 `categories`，再按你关心的
质量轴选 `dimension`。八条质量轴：

| dimension | 含义 | 典型 metric |
|---|---|---|
| `correctness` | 答案对不对（主结果轴） | exact_match / numerical_match / choice_accuracy / math_verify / multilabel_f1 / micro_f1 / soft_code_execution |
| `similarity` | 与参考文本的相似/重叠（生成类） | bleu / rouge_l / chrf / ter / token_f1 |
| `coverage` | 内容覆盖/召回 | keyword_recall |
| `format` | 格式遵循/可抽取性 | extraction_rate / format_compliance_score |
| `efficiency` | 简洁性/推理效率 | reasoning_efficiency |
| `calibration` | 相关性/校准/区分度 | mcc / pearson / spearman / auc_roc |
| `robustness` | 能力均衡性/稳定性 | gini_index |
| `diagnostic` | 纯诊断信号 | missing_answer_rate |

## 内置 metric 一览

**correctness**
- `exact_match`（别名 em）— 完全匹配；`strict=True` 大小写敏感原样匹配，`use_containment=True` 参考被预测包含即算对
- `numerical_match` — 数值软匹配（1.0==1，容忍浮点误差，`atol` 可调）
- `choice_accuracy`（别名 acc / accuracy）— 自动抽取 A/B/C/D 选项准确率
- `multilabel_f1` — 多标签/多选集合 F1
- `micro_f1` — 多选集合 Micro-F1
- `math_verify` — 数学等价性校验（文本匹配 + 符号验证混合）
- `soft_code_execution` — 静态代码分析（AST 语法可解析 + 是否定义函数/类）

**similarity**（text_gen.py）
- `bleu` — sacreBLEU ／ `rouge_l`（别名 rouge）— ROUGE-L F1 ／ `chrf` ／ `ter` ／ `token_f1`（别名 f1）

**coverage / efficiency**
- `keyword_recall` — Ref 关键词在 Pred 中的占比 ／ `reasoning_efficiency` — Ref 长度/Pred 长度

**format / diagnostic**
- `extraction_rate` — 可抽取率（**强烈建议常带上**，诊断是否按格式输出；`extractor` = number/choice/generic）
- `format_compliance_score` — 答案是否被清晰分隔（boxed/####/答案标记，无多余代码块）
- `missing_answer_rate` — 弃答率（= 1 − 可抽取率）

**calibration / robustness**（classification.py）
- `mcc` ／ `pearson` ／ `spearman` ／ `auc_roc` ／ `gini_index`（按类别正确率的基尼系数；refs 需含类别信息）

**LLM 诊断**（analysis.py，需配 judge 模型与 langchain_openai）
- `case_study_analyst` — 抽样诊断器 ／ `metric_summary_analyst` — 指标汇总分析

> 对纯文本打分（key1_text_score）内核无确定性指标，由调用方（你这个 agent）按 rubric 打分；
> `format_compliance_score` 可作辅助诊断。

## 怎么用（evalspec 或 CLI）

evalspec.yaml：
```yaml
metrics:
  - name: "exact_match"
    priority: "primary"      # primary 进主表 / secondary 进附表
  - name: "extraction_rate"
    priority: "secondary"
```
CLI：
```bash
python scripts/run_metrics.py --results eval_outputs/eval_results.json \
    --metrics exact_match:primary,bleu,extraction_rate
```

## 写一个自定义 metric

当用户要的维度注册表里没有：
1. 跟用户问清打分逻辑（输入是什么、怎么算对、要不要 LLM 裁判）。
2. 复制 `assets/custom_metric.template.py` 到 `custom_metrics/<名>.py`。
3. 实现 `compute_xxx(preds, refs, **kwargs) -> {"score": float, "details": List[float]}`，
   用 `@register_metric(name=..., desc=..., usage=..., categories=[...], dimension=...)` 注册。
4. `scripts/run_metrics.py`（`--list` 与实际打分）会通过 `_common.ensure_metrics_loaded()`
   动态 import `custom_metrics/*.py`，触发注册。之后在 evalspec/CLI 用注册名引用即可。

契约要点（违反会导致聚合失败）：
- `details` 长度必须等于 `preds`（并行分块按它合并）。
- 即使出错也返回 `{"score": 0.0, "details": [...]}`，别抛异常吞掉全表。
- `kwargs` 会被注入 `all_metric_results`（已算出的其他 metric），可做依赖型指标。
- `dimension` 选上表八条质量轴之一（仅作分组展示，不影响计算）。

> `custom_metrics/*.py` 是本地资产，不入库（已 gitignore）；以 `_` 开头的文件会被跳过。
