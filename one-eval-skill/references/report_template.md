# Report Template — 评测报告结构（图文并茂、有总有详）

评测跑完后分两步交付：先给**初版结果摘要**（对话里），再写**完整报告**落盘
（`eval_outputs/report.md`）。报告要图文并茂、有总有详 —— 先总览结论，再逐项详情。

**报告立场（务必摆正）**：这是一份**面向「被评测模型」的性能报告**——讲这个模型考了多少分、
在公开模型里处于什么水位、强在哪弱在哪。**不要**把报告写成对 One-Eval 评测流程本身的复盘或批判
（「我们的抽取逻辑如何」「框架哪里可以改进」之类不属于这里）。评测设置只需如实**附上**，供读者
判断分数可比性，不展开自我评价。

图表由 `scripts/make_plots.py` 产出到 `eval_outputs/plots/`，在报告里用绝对路径引用
（脚本打印的就是绝对路径）。

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
- **核心结论**：{{2-3 句话概括这个模型的强弱项、最值得注意的发现；面向模型能力，不评价流程}}

![各 Benchmark 主分数对比]({{绝对路径}}/plots/bench_scores.png)

| Benchmark | eval_type | 主分数 | 有效/总样本 | 模式 |
|---|---|---|---|---|
| {{bench}} | {{type}} | {{score}} | {{valid}}/{{total}} | {{full/smoke}} |

## 二、Leaderboard：在公开模型中的位置
由 `scripts/render_leaderboard.py` 生成，把本模型分数穿插进同一 bench 的公开模型分数里排名。
直接嵌入脚本输出的排名表（本模型用 ★ 标出）。务必保留每条公开分数的**来源与设置标注**——
排名仅供粗略定位，各家 shot 数/CoT/子集不同，不是严格对标。无公开参照的 bench 会被列出，
可在 `references/leaderboard_scores.json` 按格式（带来源）补充后重跑。

## 三、多维度 metric（若有）
说明每个补充 metric 衡量什么、为何选它，再给跨 bench 对比。

![Bench × Metric 热力图]({{绝对路径}}/plots/metric_heatmap.png)

| Benchmark | {{metric1}} | {{metric2}} | ... |
|---|---|---|---|
| {{bench}} | {{score}} | {{score}} | |

## 四、逐 Benchmark 详情
每个 bench 一小节：
### {{bench_name}}
- **任务类型**：{{eval_type}}，{{一句话描述这个 bench 考什么能力}}
- **主分数**：{{score}}；**有效样本**：{{valid}}/{{total}}
- **表现特征**：{{结合分数与有效率，模型在这个 bench 上展现的能力特征}}
- **典型样例**：从明细 {{detail_path 绝对路径}} 抽 1-2 条对/错样，简述模型答得如何

![有效样本占比]({{绝对路径}}/plots/sample_validity.png)

## 五、结论与建议
- **强项**：{{模型在哪些能力维度表现好}}
- **弱项 / 风险**：{{...}}（区分「能力不足」与「格式/抽取失败」——看 extraction_rate）
- **下一步**：{{加测哪些 bench / 调哪些参数 / 是否需要自定义 metric}}

## 附：评测设置（如实记录，供判断分数可比性）
- **模型生成参数**：temperature={{}}、top_p={{}}、max_tokens={}、seed={{}}
- **采样规模**：每 bench {{max_samples 或 全量}} 条；smoke / full
- **benchmark 子集**：{{各 bench 的 config/split}}
- **使用的 metric**：主分（各 eval_type 默认）+ {{额外 metric 列表}}
- **环境**：{{Python 版本 / API or GPU / 评测时间}}
- **产物绝对路径**：eval_results={{}}、metric_results={{}}、plots 目录={{}}
```

---

## 写报告的注意点

- **面向模型，不复盘流程**：报告讲模型考得怎样、在公开模型里什么水位，不评价 One-Eval 框架本身。
  评测设置只在「附：评测设置」如实附上，供判断分数可比性。
- **有总有详**：总览+leaderboard 让人 30 秒看懂水位，详情供深挖。别把所有数字堆在一处。
- **区分能力 vs 格式**：低分先看 `extraction_rate`/`missing_answer_rate` —— 很多「低分」其实是
  模型没按格式输出导致抽取失败，不是真不会。报告要点明这一点。
- **leaderboard 守住来源**：公开分数原样保留 source/setting/as_of，排名标注「仅供粗略定位」。
- **smoke vs full**：标清每个分数是 3 条 smoke 还是全量，避免误读。
- **图表/产物引用绝对路径**：脚本打印的就是绝对路径，直接引用，用户能点开。
- 报告是给人读的叙事，脚本只给确定性数字、图和排名 —— 串成结论是你（agent）的职责。
