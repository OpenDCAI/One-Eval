# Report Template — 评测报告结构（图文并茂、有总有详）

评测跑完后分两步交付：先给**初版结果摘要**（对话里），再写**完整报告**落盘
（`eval_outputs/report.md`）。报告要图文并茂、有总有详 —— 先总览结论，再逐项详情。

图表由 `scripts/make_plots.py` 产出到 `eval_outputs/plots/`，在报告里用相对路径引用。

---

## 第一步：初版结果摘要（对话内，简短）

跑完 `run_eval.py` 后立刻给用户一段摘要，不要等完整报告：
- 模型名、评测了哪些 bench、各自主分数
- 一句话结论（如「数学推理强、选择题偏弱」）
- 下一步建议（要不要加 metric / 跑全量 / 看某个 bench 的错例）

示例：
> 已评测 **gpt-4o-mini** 在 MATH-500（3 条 smoke）：score=0.667，valid=3/3。
> 主分数正常，建议跑全量并加 `extraction_rate` 看格式遵循度。

---

## 第二步：完整报告落盘（eval_outputs/report.md）

按下面结构写。`{{...}}` 是占位，按实际数据填。

```markdown
# 评测报告：{{模型名}}

> 生成时间：{{date}} ｜ 数据来源：eval_outputs/eval_results.json (+ metric_results.json)

## 一、总览（TL;DR）
- **模型**：{{model_name_or_path}}（{{API/vLLM}}）
- **评测范围**：{{N}} 个 benchmark，共 {{总样本数}} 条
- **核心结论**：{{2-3 句话概括强弱项与最值得注意的发现}}

![各 Benchmark 主分数对比](plots/bench_scores.png)

| Benchmark | eval_type | 主分数 | 有效/总样本 | 模式 |
|---|---|---|---|---|
| {{bench}} | {{type}} | {{score}} | {{valid}}/{{total}} | {{full/smoke}} |

## 二、多维度 metric（若有）
说明每个补充 metric 衡量什么、为何选它，再给跨 bench 对比。

![Bench × Metric 热力图](plots/metric_heatmap.png)

| Benchmark | {{metric1}} | {{metric2}} | ... |
|---|---|---|---|
| {{bench}} | {{score}} | {{score}} | |

## 三、逐 Benchmark 详情
每个 bench 一小节：
### {{bench_name}}
- **任务类型**：{{eval_type}}，{{一句话描述这个 bench 考什么}}
- **主分数**：{{score}}；**有效样本**：{{valid}}/{{total}}
- **观察**：{{结合分数与有效率，模型在这个 bench 上的表现特征}}
- **典型错例**（可选）：从明细 detail_path 抽 1-2 条错样，简述错在哪

![有效样本占比](plots/sample_validity.png)

## 四、结论与建议
- **强项**：{{...}}
- **弱项 / 风险**：{{...}}（注意区分「能力不足」与「格式/抽取失败」——看 extraction_rate）
- **下一步**：{{加测哪些 bench / 调哪些参数 / 是否需要自定义 metric}}

## 附：运行配置
- evalspec：模型参数、benchmark 列表、metric 选择
- 环境：{{Python 版本 / dataflow 版本 / API or GPU}}
```

---

## 写报告的注意点

- **有总有详**：总览能让人 30 秒读完，详情供深挖。别把所有数字堆在一处。
- **区分能力 vs 格式**：低分先看 `extraction_rate`/`missing_answer_rate` —— 很多「低分」其实是
  模型没按格式输出导致抽取失败，不是真的不会。报告要点明这一点。
- **smoke vs full**：标清每个分数是 3 条 smoke 还是全量，避免误读。
- **图表引用**：用相对路径 `plots/xxx.png`，报告和图在同一 `eval_outputs/` 下才能正确显示。
- 报告是给人读的叙事，脚本只给确定性数字和图 —— 串成结论是你（agent）的职责。
