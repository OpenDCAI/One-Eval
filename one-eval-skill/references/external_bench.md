# External-Repo Bench — 自带仓库 / 需特殊环境的 benchmark（schema + 机制）

有些 benchmark **没法走 One-Eval 的确定性内核**（下载 jsonl → 生成 → 解析 → 打分）：
它们自带评测仓库、自带打分 harness，且打分依赖特殊环境（代码沙箱、工具调用执行器、
受控运行时）。典型：

- **LiveCodeBench** — 代码生成，需在沙箱里真实执行测试用例（Pass@k）
- **BFCL**（Berkeley Function-Calling Leaderboard）— 工具/函数调用，需 AST 校验 + 执行
- **EvalPlus**（HumanEval+/MBPP+）— 代码生成，需扩充测试集在沙箱执行

对这类 bench，我们**不重写它们的打分逻辑**（那会偏离原始精度，违背对齐目标）。
做法是：在 gallery 里**登记仓库地址 + 安装/运行/取分说明**，把它当成一个
「外部可复现的评测单元」挂进 One-Eval。

> **本版状态：只定义 schema + 机制，不内置任何 external_repo 条目，也不内置执行器。**
> 具体 bench 由用户逐个提供（repo、装法、跑法、取分方式），届时按本契约填进 gallery；
> 真正的自动 clone/执行逻辑（`external_runner.py`）留待需要时再补。

---

## 1. 怎么区分两类 bench：`bench_kind`

gallery 条目（`bench_gallery.json` 的 `benches[]`）新增**顶层字段** `bench_kind`：

| bench_kind | 含义 | 执行路径 |
|---|---|---|
| `dataflow`（默认，缺省即此） | 普通纯文本 bench，走内核 | `DataFlowEvalTool` 下载+生成+解析+打分 |
| `external_repo` | 自带仓库、需特殊环境 | 不走内核；按 `meta.repo_eval` 的说明在外部执行后回填分数 |

**缺省值是 `dataflow`** —— 现有 103 个 bench 不带这个字段，行为完全不变。
只有显式写 `"bench_kind": "external_repo"` 的条目才走外部路径。

`run_eval.py` 遇到 `external_repo` 条目会**优雅短路**：不下载、不调内核、不报错，
返回一条 `mode="external_repo_pending"` 的结果，并把 `meta.repo_eval` 原样带出，
供调用方（你这个 agent）据此执行外部评测、再把最终分数回填。

---

## 2. external_repo 条目的字段契约

顶层字段（与 dataflow 条目同级）：

| 字段 | 必填 | 说明 |
|---|---|---|
| `bench_name` | ✓ | 唯一名，如 `livecodebench` |
| `bench_kind` | ✓ | 固定 `"external_repo"` |
| `bench_dataflow_eval_type` | ✗ | external 不走内核，可设 `null`；保留仅为统一 schema |
| `bench_source_url` | ✗ | 数据/榜单主页（人看的入口），可与 repo_url 不同 |
| `meta.repo_eval` | ✓ | 外部评测的全部可执行信息，见下表 |

`meta.repo_eval` 子字段（这是核心，将来执行器/runbook 全靠它）：

| 子字段 | 必填 | 说明 |
|---|---|---|
| `repo_url` | ✓ | git 仓库地址（建议 https） |
| `ref` | ✓ | 固定到 commit SHA 或 tag，**保证可复现**（不要用浮动的 `main`） |
| `license` | ✗ | 仓库 license，注意可商用性/再分发限制 |
| `env_requires` | ✓ | 环境前提清单：python 版本、是否需 GPU、是否需 docker/沙箱、关键系统依赖 |
| `setup` | ✓ | 安装步骤（有序命令列表），如 `pip install -e .`、`docker build` |
| `model_interface` | ✓ | 这个 harness 怎么接被测模型：`api`（OpenAI 兼容 base_url+key）/ `local_vllm` / `custom`；以及它读哪些 env 或 config |
| `run` | ✓ | 跑评测的命令模板（含占位符，见 §3），以及工作目录 |
| `result` | ✓ | 跑完后结果落在哪、什么格式、怎么解析出主分数（见 §4） |
| `data_alignment` | ✓ | **精度对齐说明**：用的是该 repo 的哪个数据版本/release/子集，对应我们要对齐的 tech report 里的哪个设置 |
| `notes` | ✗ | 坑点、限速、已知 flaky 等 |

---

## 3. `run` 命令模板的占位符约定

`run.command` 是字符串或字符串列表，用 `{{占位符}}` 标注运行时注入的值。
约定占位符（将来执行器/agent 负责替换，**禁止把真实 key 写进 gallery**）：

| 占位符 | 替换为 |
|---|---|
| `{{model_name}}` | 被测模型名 |
| `{{api_base_url}}` | OpenAI 兼容 base_url |
| `{{api_key}}` | API key（仅运行时注入，绝不入库） |
| `{{work_dir}}` | clone 后的仓库本地路径 |
| `{{output_dir}}` | 结果输出目录 |
| `{{max_samples}}` | smoke/限量时的样本数（全量时为空） |

示例（仅示意字段形状，非真实条目）：
```json
"run": {
  "work_dir": "{{work_dir}}",
  "command": [
    "python -m livecodebench.runner --model {{model_name}} --base-url {{api_base_url}} --output {{output_dir}}"
  ]
}
```

---

## 4. `result` 取分约定

外部 harness 跑完会自己落结果文件。我们只需说清**从哪取、取哪个数**，
把它归一化成 One-Eval 统一的 `{score, total_samples, valid_samples, metric}`：

| 子字段 | 说明 |
|---|---|
| `path` | 相对 `output_dir` 的结果文件路径（支持 glob，如 `*.json`） |
| `format` | `json` / `jsonl` / `csv` |
| `score_path` | 主分数在结果里的取值路径（点路径，如 `pass@1` 或 `overall.accuracy`） |
| `scale` | 原值量纲：`ratio`（0–1）/ `percent`（0–100，回填时除 100） |
| `metric_name` | 这个分数对应的指标名（写进报告，如 `pass@1`、`acc`） |

---

## 5. 接入一个 external_repo bench 的流程（将来逐个加时）

1. 跟用户确认：repo 地址、固定 ref、装法、跑法、模型怎么接、结果在哪取分、对齐哪个数据版本。
2. 复制 `assets/external_bench.entry.template.json`，按 §2–§4 填全 `meta.repo_eval`。
3. 把条目追加进 `bench_gallery.json` 的 `benches[]`（`bench_kind="external_repo"`）。
4. `python scripts/build_gallery_md.py` 重生成 gallery md（external 条目会单列一区）。
5. 在 evalspec 里引用该 bench → `run_eval.py` 会短路返回 `external_repo_pending` +
   `repo_eval`。当前版本由你（agent）据此**手动执行外部评测**并回填分数；
   待积累几个后再决定是否写统一的 `external_runner.py` 自动化。

## 6. 安全边界（重要）

- 外部 repo 是**第三方代码**，clone + 执行有风险。固定 `ref` 到具体 commit，
  执行前让用户确认；优先在容器/沙箱里跑。
- `api_key` 等凭证**只在运行时注入**，绝不写进 `bench_gallery.json`（会入库）。
- `data_alignment` 必填，是为了守住「评测精度与 tech report 对齐」这条主线 ——
  外部 repo 的默认数据版本可能与报告不同，接入时必须核对。
