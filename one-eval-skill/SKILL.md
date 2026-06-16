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

### 0. 先确认运行环境（环境隔离，别乱动用户环境）
One-Eval 依赖较重，**必须跑在独立环境里**（专用 conda/venv），不要装进系统自带 python
或用户全局 site-packages。接入前**主动问用户**：在哪台机器跑（本机 Mac 只验 API；vLLM 去
GPU 机）、用哪个 Python 环境。拿到后所有脚本一律用**该环境 python 的绝对路径**调用
（如 `/path/to/.venv/bin/python scripts/xxx.py`），别用裸 `python`，避免误用到别的环境。
不确定时先跑 `doctor.py`——它会打印当前解释器路径、是否在隔离环境，发现误用系统 base 会告警。

## 标准流程（按序执行，不要跳步）

### 1. 选模型 + 测连通性（强制门槛）
评测数据量大、耗时长，**接入任何模型前必须先测连通**。

**被测模型必须由用户明确指定，禁止默认填充、禁止张冠李戴。** 你（agent，自身是 Claude）
和「被测模型」是两回事——不要把自己的名字混进被测模型名（曾出现过把 `gpt-4o-mini`
说成 `claude-sonnet-4-6o-mini` 的错误）。主动问用户要测哪个模型、模型在哪：
- **API 模型**（`is_api: true`）：openai_compatible / deepseek
- **本地 vLLM**（`is_api: false`）：需 GPU 环境（本机无 GPU 则交由用户在 GPU 机验证）

测连通：
```bash
python scripts/check_model.py --api --model <名> --api-url <url> --api-key <key>
# 或从已写好的 spec 读：python scripts/check_model.py --spec evalspec.yaml
```
连通失败**不要往下走**，先按 stderr 的可读原因排查（鉴权/端点/网络）。

**写完 evalspec 后，把 model 段回显给用户确认**（"本次被测模型是 X，API/vLLM，参数如下，对吗？"），
得到确认再开跑。全程对话里**始终明说被测模型是谁**，摘要/报告里的模型名以 `evalspec.yaml`
的 `model_name_or_path` 为准——`run_eval.py` 启动首行也会打印 `被测模型: <名>`，以此为准核对。

**连通后，主动与用户确认生成参数**（别用默认值闷头跑）：温度 `temperature`、`top_p`、
`max_tokens`、`seed`。给出推荐并说明影响——评测默认 `temperature=0`+固定 `seed` 求可复现；
`max_tokens` 对数学/CoT 题不要太小（截断会导致抽不出答案、假阴性，宁可放大到 2048+）。
若用户想测模型「发挥上限」或多样性，再调高 temperature 并说明分数会抖动。最终确认值写进
`evalspec.yaml`，会随结果落盘并在报告「评测设置」里如实记录（见 step 8 / report_template）。

### 2. 选 benchmark
- 先看 `references/bench_gallery.md`：**READY 区**（已测通、可直接复用）优先；
  否则从**候选区**（103 个未验证 bench）选，接入前需走 smoke 验证。
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
  正确性（exact_match/numerical_match）、相似度（bleu/rouge_l/chrf/token_f1/jaccard_similarity，翻译摘要长答案/关键词覆盖）、
  格式遵循（extraction_rate/format_compliance_score，低分会拖累正确性）、
  生成健康度（repetition_rate 抓复读）、弃答率（missing_answer_rate 做正确性归因）、
  代码合法性（code_validity，注意只验能否解析、非逻辑正确）。
- `python scripts/run_metrics.py --list` 查看全部 15 个 metric（按维度分组，含适用场景）。
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
**产物隔离**：每次评测自动生成 `run_id`（时间戳），产物落到独立目录
`eval_outputs/runs/<run_id>/`（含 `eval_results.json`，后续 metric/报告也聚此目录），
多次评测互不覆盖；`eval_outputs/latest_run.txt` 始终指向最新 run 目录。脚本首行打印
被测模型名 + run_id + 产物目录路径——记下这个目录，后面几步都用它。

`eval_results.json` 顶层带 `run_id` / `generated_at` / 脱敏的 `model_config`（含生成参数）/
`runtime`，供报告自包含、可复现（api_key 不落盘，只标 `***`）。

**断点续跑**：每跑完一个 bench 立即增量落盘；若中途中断（崩溃/Ctrl-C），修好问题后用
`--resume` 接着跑，已成功的 bench 自动跳过、只补失败/未跑的（**bench 级**续跑；
dataflow 内核内部不支持样本级断点）：
```bash
python scripts/run_eval.py evalspec.yaml --resume eval_outputs/runs/<run_id>
```

### 7. 多维度打分（若选了 metric）
```bash
# results 用上一步那个 run 目录里的；metric_results.json 自动写进同一 run 目录
python scripts/run_metrics.py --results eval_outputs/runs/<run_id>/eval_results.json --metrics <名,名:primary>
```
产出 `eval_outputs/runs/<run_id>/metric_results.json`（与 results 同目录）。

### 8. 生成 HTML 报告并自动打开（图文并茂、有总有详）
```bash
# 主产物：单文件 HTML 报告（内联 CSS/JS、零 CDN、可离线、leaderboard 条形图 + metric 热力图）
# 默认生成后自动在浏览器打开；--out 缺省即落在 results 同目录（同一 run 目录），无需手填
python scripts/render_report.py --results eval_outputs/runs/<run_id>/eval_results.json \
    --metrics eval_outputs/runs/<run_id>/metric_results.json
# 不想自动打开时加 --no-open；公开分数表默认读 references/leaderboard_scores.json，可用 --scores 覆盖
```
**这一步必须由你（agent）在评测+metric 跑完后主动调用**，让用户做完评测直接看到弹出的报告，
而不是把渲染留给用户手动操作。一次产出单文件 `report.html`（总览卡片 + leaderboard 条形图 + metric
热力图 + 逐 bench 详情 + 附录「评测设置」含被测模型/run_id/生成参数真值，零依赖可离线）。

对话里先给**初版摘要**（**明说被测模型名** + 核心分数 + 一句话水位结论 + 强弱 bench），再说明完整报告已生成并自动打开（附绝对路径）。

> 退路（无法用浏览器 / 用户要 markdown 时）：`make_plots.py` 出 PNG、`render_leaderboard.py` 出 markdown 表，
> 再按 `references/report_template.md` 拼 markdown 报告。HTML 是默认主路径。

**报告立场、leaderboard 来源标注、产物绝对路径**等写报告的硬约束见 `references/report_template.md`，
落地前务必照它执行（核心：面向被评测模型性能而非复盘流程；公开分保留来源、排名仅供粗略定位；所有路径写绝对路径）。

## 文件地图
- `references/eval_types.md` — 6 种 eval_type 与 key_mapping 硬契约（**接入 bench 必读**）
- `references/bench_gallery.md` — READY 区 + 候选区 + 外部仓库 bench 区
- `references/external_bench.md` — 自带仓库 / 需特殊环境 bench 的 schema 与接入机制
- `references/metric_registry.md` — metric 注册表说明 + 自定义指引
- `references/model_setup.md` — API / vLLM 模型接入与凭证、HF 下载配置
- `references/report_template.md` — 报告结构模板（面向模型性能 + 评测设置）
- `references/leaderboard_scores.json` — 公开模型分数表（手工维护、带来源），leaderboard 排名用
- `scripts/` — check_model / prepare_bench / run_eval / run_metrics / render_report（HTML 主报告）/ make_plots / render_leaderboard（markdown 退路）
- `assets/` — evalspec.template.yaml / custom_metric.template.py / external_bench.entry.template.json

## 安全 & 边界
- API key 等凭证只写进本地 `evalspec.yaml`（已 gitignore），不要回显到对话或入库。
- `eval_outputs/`、`cache/`、`.local_state.json`、`custom_metrics/*.py` 均不入库。
- 本机（Mac）只验证 API 路径；vLLM 路径代码完整但需 GPU 机验证。
