# Metric Registry — 多维度打分与自定义指标

dataflow 主评测给每个 eval_type 一个**默认主分数**。注册表让你在主分数之外，
按需挑选**额外维度**（如 BLEU/ROUGE/EM/数学等价/LLM 诊断），形成多维评分。
用户也可与你聊天，**临时写一个新 metric** 作为新维度。

> 实时查看可用 metric（最权威，含别名/类别）：
> ```bash
> python scripts/run_metrics.py --list
> ```
> 下表是源码快照（`one_eval/metrics/common/`），可能随版本增减，以 `--list` 为准。

## 内置 metric 一览（按用途分组）

**通用匹配 / 抽取**（general.py）
- `exact_match`（别名 em）— 抽取式答案完全匹配
- `containment_match` — 文本包含匹配（DataFlow 风格）
- `strict_match` — 原始字符串严格匹配
- `numerical_match` — 数值软匹配（1.0==1，容忍浮点误差）
- `choice_accuracy` — 选项字母/离散标签准确率
- `multilabel_f1` — 多标签分类 F1
- `extraction_rate` — 正则提取成功率（**强烈建议常带上，诊断模型是否按格式输出**）
- `missing_answer_rate` — 未输出有效选项的比例（诊断）
- `format_compliance_score` — 格式遵循度

**数学**（math_verify.py / symbolic.py）
- `math_verify` — 数学等价性校验（文本匹配 + 符号验证混合）

**文本生成 / 翻译 / 摘要**（text_gen.py）
- `bleu` — sacreBLEU 主指标
- `rouge_l` — ROUGE-L F1
- `chrf` — CHRF
- `ter` — Translation Edit Rate
- `token_f1` — token 级 F1
- `keyword_recall` — 关键词召回率
- `reasoning_efficiency` — CoT 效率（Ref 长度/Pred 长度）

**分类 / 相关性**（classification.py）
- `accuracy` — 通用 accuracy（支持自动选项匹配）
- `micro_f1` — 多选集合 Micro-F1
- `mcc` — Matthews 相关系数
- `pearson` / `spearman` — 相关系数
- `auc_roc` — AUC-ROC ×100
- `gini_index` — 能力协调性（按类别正确率的基尼系数；refs 需含类别信息）

**代码**（code.py）
- `pass_at_k` — Pass@k（需代码执行）
- `code_similarity` — 代码相似度（BLEU-based）
- `soft_code_execution` — 静态代码分析（语法/复杂度）

**LLM 诊断**（analysis.py，需配 judge 模型）
- `case_study_analyst` — 通用抽样诊断器
- `metric_summary_analyst` — 指标汇总分析

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
   用 `@register_metric(name=..., desc=..., usage=..., categories=[...])` 注册。
4. 评测引擎启动时自动扫描 `custom_metrics/`（与内置同级）。在 evalspec/CLI 用注册名引用。

契约要点（违反会导致聚合失败）：
- `details` 长度必须等于 `preds`（并行分块按它合并）。
- 即使出错也返回 `{"score": 0.0, "details": [...]}`，别抛异常吞掉全表。
- `kwargs` 会被注入 `all_metric_results`（已算出的其他 metric），可做依赖型指标。

> `custom_metrics/*.py` 是本地资产，不入库（已 gitignore）。
