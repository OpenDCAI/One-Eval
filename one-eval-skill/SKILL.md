---
name: one-eval
description: 驱动 One-Eval 对「纯文本 LLM」做端到端评测。当用户想评测一个模型（API 或本地 vLLM）在某些 benchmark 上的表现、对比多个 benchmark 分数、补充多维度 metric、或生成图文评测报告时使用本 skill。
---

# One-Eval Skill

把 One-Eval（原 LangGraph 多节点框架）的 **LLM 编排职责交给你（调用方 agent）**，
skill 只保留**确定性执行内核**（下载/评测/打分/出图脚本）。你负责与用户交互、做决策、
生成 `evalspec.yaml`、调脚本、解读结果、写报告。

## 前置环境（首次使用必读）

本 skill **不自包含**：脚本通过把仓库根加进 `sys.path` 来 `import one_eval` + `dataflow`，
因此运行前必须先装好 One-Eval 主仓库及其依赖。`one-eval-skill/` 是主仓库下的子目录，
不能脱离主仓库单独跑。

**一次性安装**（二选一，需 Python ≥ 3.10）：
```bash
# 方式 A：Conda
conda create -n one-eval python=3.11 -y
conda activate one-eval
pip install -e .          # 在 One-Eval 仓库根执行，读 pyproject.toml/requirements.txt

# 方式 B：uv
uv venv && source .venv/bin/activate
uv pip install -e .
```
依赖含 `datasets` / `dataflow` 等较重的包；装不全会在首次 `run_eval.py` 时报 import 错。

**装完先自检**（确认依赖齐全，避免跑到一半才发现缺包）：
```bash
python scripts/doctor.py     # 必需项齐全则退出码 0；缺啥会列出并给修复命令
```

**装好之后**：用户**直接用自然语言对话即可**，不需要手敲脚本——你（agent）会按下方流程
替用户调脚本。例如用户说「用 gpt-4o-mini 评一下 mmlu-redux 和 polymath，API 地址 xxx、
key xxx」，你就从测连通一路跑到出报告。脚本路径、evalspec 都由你生成与调用。

> 运行脚本统一用主仓库的 Python 环境（上面装的那个），且 cwd 在 `one-eval-skill/` 下时
> 用 `python scripts/xxx.py`。API key 由用户自备，只写进本地 `evalspec.yaml`（已 gitignore），
> 不要回显到对话或入库。

## 标准流程（按序执行，不要跳步）

### 1. 选模型 + 测连通性（强制门槛）
评测数据量大、耗时长，**接入任何模型前必须先测连通**。主动问用户模型在哪：
- **API 模型**（`is_api: true`）：openai_compatible / deepseek
- **本地 vLLM**（`is_api: false`）：需 GPU 环境（本机无 GPU 则交由用户在 GPU 机验证）

测连通：
```bash
python scripts/check_model.py --api --model <名> --api-url <url> --api-key <key>
# 或从已写好的 spec 读：python scripts/check_model.py --spec evalspec.yaml
```
连通失败**不要往下走**，先按 stderr 的可读原因排查（鉴权/端点/网络）。

**连通后，主动与用户确认生成参数**（别用默认值闷头跑）：温度 `temperature`、`top_p`、
`max_tokens`、`seed`。给出推荐并说明影响——评测默认 `temperature=0`+固定 `seed` 求可复现；
`max_tokens` 对数学/CoT 题不要太小（截断会导致抽不出答案、假阴性，宁可放大到 2048+）。
若用户想测模型「发挥上限」或多样性，再调高 temperature 并说明分数会抖动。最终确认值写进
`evalspec.yaml`，并在报告的「评测设置」里如实记录（见 step 8 / report_template）。

### 2. 选 benchmark
- 先看 `references/bench_gallery.md`：**READY 区**（已测通、可直接复用）优先；
  否则从**候选区**（96 个未验证 bench）选，接入前需走 smoke 验证。
- 用户要评测 gallery 之外的新数据集 → 用 `scripts/prepare_bench.py` 下载并**预览嵌套结构**，
  再按 `references/eval_types.md` 判断 eval_type、规划 key_mapping（嵌套字段须先拍平）。
- **自带仓库 / 需特殊环境的 bench**（LiveCodeBench、BFCL、EvalPlus 等需沙箱执行的）→
  走 `references/external_bench.md` 的 `bench_kind=external_repo` 机制：在 gallery 登记
  仓库地址 + 安装/运行/取分说明。`run_eval.py` 会对这类 bench 优雅短路（返回
  `external_repo_pending` + `repo_eval`），由你据此在外部执行后回填分数（本版未内置执行器）。

### 3. 选 metric（默认已给主分，额外维度可选）
**先告诉用户每个 bench 默认用什么主分、它衡量什么能力**（dataflow 内核按 eval_type 自动选）：

| eval_type | 默认主指标 | 衡量的能力 |
|---|---|---|
| key2_qa | math_verify（数值/数学等价+文本匹配，已修假阴性） | 答案正确性（数学/简答 QA） |
| key2_q_ma | any_math_verify | 多参考答案命中任一即对 |
| key3_q_choices_a | ll_choice_acc（API 模型自动退回 parse_choice_acc） | 单选题准确率 |
| key3_q_choices_as | micro_f1 | 多选题集合 F1 |
| key3_q_a_rejected | pairwise_ll_winrate | 偏好对比胜率 |
| key1_text_score | ppl（困惑度） | 语言建模流畅度 |

- **主分够用就够用**；但要**主动问用户是否补充维度**，并解释每个维度查什么：
  正确性（exact_match/numerical_match）、相似度（bleu/rouge_l/chrf/token_f1，翻译摘要长答案）、
  格式遵循（extraction_rate/format_compliance_score，低分会拖累正确性）、
  生成健康度（repetition_rate 抓复读）、弃答率（missing_answer_rate 做正确性归因）、
  代码合法性（code_validity，注意只验能否解析、非逻辑正确）。
- `python scripts/run_metrics.py --list` 查看全部 14 个 metric（按维度分组，含适用场景）。
- 用户想要注册表里没有的维度 → 参考 `references/metric_registry.md` + `assets/custom_metric.template.py`
  跟用户聊清楚需求后写新 metric，落到 `custom_metrics/`。

### 4. 生成 evalspec.yaml
基于 `assets/evalspec.template.yaml` 填写 model / benchmarks / metrics / runtime。
eval_type 与 key_mapping 必须符合 `references/eval_types.md` 的硬契约。

### 5. Smoke 验证（强制，除非已 READY）
正式全量评测前，**每个未 READY 的 bench 先抽 3 条**跑通：
```bash
python scripts/run_eval.py evalspec.yaml --smoke
```
smoke 通过的 bench 会被标记 READY（写入 `.local_state.json`），下次自动跳过 smoke。

### 6. 正式评测
```bash
python scripts/run_eval.py evalspec.yaml            # max_samples 由 runtime 决定
```
产出 `eval_outputs/eval_results.json`。

### 7. 多维度打分（若选了 metric）
```bash
python scripts/run_metrics.py --results eval_outputs/eval_results.json --metrics <名,名:primary>
```
产出 `eval_outputs/metric_results.json`。

### 8. 出图 + 写报告（图文并茂、有总有详）
```bash
python scripts/make_plots.py --results eval_outputs/eval_results.json \
    --metrics eval_outputs/metric_results.json --out eval_outputs/plots
```
然后按 `references/report_template.md`：
- 先给用户一个**初版结果摘要**（核心分数 + 一句话结论）。
- 再写**完整报告**落盘（含图表引用、逐 bench 详情、跨 bench 对比、结论建议）。

**输出落盘与路径规范（务必遵守）**：
- 报告里引用的所有产物路径（`eval_results.json` / `metric_results.json` / 图表 PNG /
  逐样本明细 `detail_path`）一律写**绝对路径**，让用户能直接点开。脚本现在打印的就是绝对路径，直接引用即可。
- **有总有分**：报告先「总」——一张总览表（模型 × 各 bench 主分 + 关键维度），一句话结论；
  再「分」——每个 bench 一节，给该 bench 的分数、有效样本数、典型对/错样例（指向 `detail_path`）、
  以及该 bench 考察的能力维度。让用户既能一眼看懂全局，又能下钻到单题。

## 文件地图
- `references/eval_types.md` — 6 种 eval_type 与 key_mapping 硬契约（**接入 bench 必读**）
- `references/bench_gallery.md` — READY 区 + 候选区 + 外部仓库 bench 区
- `references/external_bench.md` — 自带仓库 / 需特殊环境 bench 的 schema 与接入机制
- `references/metric_registry.md` — metric 注册表说明 + 自定义指引
- `references/model_setup.md` — API / vLLM 模型接入与凭证、HF 下载配置
- `references/report_template.md` — 报告结构模板
- `scripts/` — check_model / prepare_bench / run_eval / run_metrics / make_plots
- `assets/` — evalspec.template.yaml / custom_metric.template.py / external_bench.entry.template.json

## 安全 & 边界
- API key 等凭证只写进本地 `evalspec.yaml`（已 gitignore），不要回显到对话或入库。
- `eval_outputs/`、`cache/`、`.local_state.json`、`custom_metrics/*.py` 均不入库。
- 本机（Mac）只验证 API 路径；vLLM 路径代码完整但需 GPU 机验证。
