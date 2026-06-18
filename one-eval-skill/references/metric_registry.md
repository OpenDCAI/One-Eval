# Metric Registry — 多维度打分与自定义指标

dataflow 主评测给每个 eval_type 一个**默认主分数**。注册表让你在主分数之外，
按需挑选**额外维度**，形成多维评分。用户也可与你聊天，**临时写一个新 metric**。

> 实时查看可用 metric（最权威，按维度分组，含别名/类别）：
> ```bash
> python scripts/run_metrics.py --list
> ```
> 下表是源码快照（`one_eval/metrics/common/`），以 `--list` 为准。

## 维度（dimension）是什么

每个 metric 带一个 `dimension` 标签，表示它衡量**模型哪一面的表现**，与
`categories`（适用哪种题型 eval_type）正交。挑指标时先按题型筛 `categories`，
再按你关心的维度选。

**重要前提**：本框架的推理产物**只有文本**（pred 文本 vs ref 文本），没有概率/
logits/跨类别标注。因此需要那类输入的统计指标（AUC-ROC、Pearson/Spearman、MCC、
基尼系数）**已被剔除**——它们在这里拿不到数据，只会恒返 0 或报错，根本不反映模型
能力。同理删掉了按字符长度比算的“推理效率”（奖励不思考直接给答案，是反逻辑的）、
重复实现的 micro_f1、方向反掉的 ter、以及与 token_f1 重叠的 keyword_recall。

六条真实维度：

| dimension | 衡量什么 | metric |
|---|---|---|
| `correctness` | 答案对不对（主结果轴） | exact_match / numerical_match / choice_accuracy / multilabel_f1 / set_f1 / math_verify |
| `similarity` | 与参考文本的词面相似/重叠（翻译/摘要/长答案） | bleu / rouge_l / chrf / token_f1 / jaccard_similarity |
| `validity` | 产物本身是否合法可用（≠正确） | code_validity / sql_parse_validity |
| `fluency` | 生成健康度（退化重复、乱码/异常编码等失败模式） | repetition_rate / garbled_text_rate |
| `format` | 格式遵循/可抽取性 | extraction_rate / format_compliance_score / json_validity |
| `diagnostic` | 纯诊断信号（不直接代表好坏，用于归因） | missing_answer_rate / empty_or_whitespace_rate |

## 内置 metric 一览

**correctness**
- `exact_match`（别名 em）— 完全匹配；`strict=True` 大小写敏感，`use_containment=True` 参考被预测包含即算对
- `numerical_match` — 数值软匹配（1.0==1，容忍浮点误差，`atol` 可调）
- `choice_accuracy`（别名 acc / accuracy）— 自动抽取 A/B/C/D 选项准确率
- `multilabel_f1` — 多标签/多选集合 F1
- `set_f1` — 开放答案集合 F1，适合列表型 QA、实体抽取、多答案短答
- `math_verify` — 数学等价性校验（文本匹配 + 符号验证混合）

**similarity**（text_gen.py）
- `bleu` — sacreBLEU ／ `rouge_l`（别名 rouge）— ROUGE-L F1 ／ `chrf` ／ `token_f1`（别名 f1）／ `jaccard_similarity`（别名 jaccard）— 去重 token 集合重叠，适合关键词覆盖/检索式 QA/事实列表

**validity / fluency**
- `code_validity`（别名 soft_code_execution）— 代码能否 AST 解析 + 是否定义函数/类。**只验合法性，不代表逻辑正确**；要真 Pass@k 请在受控沙箱自写 custom metric
- `sql_parse_validity` — SQL 是否能被 parser 解析，适合 Text2SQL/SQL 生成。只验语法/方言可解析性，**不代表执行正确或结果正确**；`dialect` 默认 sqlite，可设 postgres/mysql/bigquery/snowflake 等
- `repetition_rate` — 退化重复率（1 − distinct-n）。无需 ref，抓“复读机/卡循环”，**分越低越好**（建议作诊断，`n` 可调，默认 3）
- `garbled_text_rate` — 乱码率。无需 ref，保守检测明显编码损坏；**分越低越好**

**format / diagnostic**
- `extraction_rate` — 可抽取率（**强烈建议常带上**，诊断是否按格式输出；`extractor` = number/choice/generic）
- `format_compliance_score` — 答案是否被显式标记出来（boxed/####/答案标记）。+0.5 非空 +0.5 有标记
- `json_validity` — JSON 合法率；默认要求整段输出可被 `json.loads` 解析，`allow_extraction=True` 时允许从 markdown fenced block 或文本中首个 JSON object/array 提取后解析
- `missing_answer_rate` — 弃答率（= 1 − 可抽取率）
- `empty_or_whitespace_rate` — 空输出率：输出为 None、空字符串或纯空白的比例。用于定位服务/生成链路未产出可评内容，**分越低越好**

> **没有确定性指标的两种题型**：纯文本打分（key1_text_score）和偏好对比
> （key3_q_a_rejected）。后者 runner 只把 better 当 ref 传进来、rejected 取不到，
> 真偏好判断须由调用方（你这个 agent）按 rubric 裁判。退而求其次可用
> `token_f1` / `rouge_l` 量“与更优答案的词面接近度”，但那是弱代理、别当主分。

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
    --metrics exact_match:primary,token_f1,extraction_rate
```

## 写一个自定义 metric

当用户要的维度注册表里没有（比如需要沙箱的真 Pass@k、需要 judge 模型的主观打分）：
1. 跟用户问清打分逻辑（输入是什么、怎么算对、要不要 LLM 裁判）。
2. 复制 `assets/custom_metric.template.py` 到 `custom_metrics/<名>.py`。
3. 实现 `compute_xxx(preds, refs, **kwargs) -> {"score": float, "details": List[float]}`，
   用 `@register_metric(name=..., desc=..., usage=..., categories=[...], dimension=...)` 注册。
4. `scripts/run_metrics.py`（`--list` 与实际打分）通过 `_common.ensure_metrics_loaded()`
   动态 import `custom_metrics/*.py` 触发注册。之后在 evalspec/CLI 用注册名引用即可。

契约要点（违反会导致聚合失败）：
- `details` 长度必须等于 `preds`（并行分块按它合并）。
- 即使出错也返回 `{"score": 0.0, "details": [...]}`，别抛异常吞掉全表。
- metric 只能看到 `preds` 与 `refs` 两条等长文本列表；拿不到原始 logits/概率。
- `dimension` 选上表六条之一，或在 `MetricDimension` 里加新维度（仅作分组展示）。

> `custom_metrics/*.py` 是本地资产，不入库（已 gitignore）；以 `_` 开头的文件会被跳过。
