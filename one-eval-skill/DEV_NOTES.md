# DEV_NOTES — One-Eval Skill 开发与维护备忘（内部，非 skill 内容）

> 给维护者看的，不是给调用 skill 的 agent 看的。记录数据来源、同步关系、本机环境坑。

## 这个 skill 是什么

把 One-Eval（原 LangGraph 15 节点框架）的 **LLM 编排职责外移给调用方 agent**（Claude Code），
skill 只保留**确定性执行内核**：下载 / 评测 / 打分 / 出图 5 个脚本 + 决策用的参考文档。

三层结构：
- `references/` — agent 读来做决策（eval_types 硬契约、gallery、metric、model setup、报告模板）
- `assets/` — 模板（evalspec.template.yaml、custom_metric.template.py）
- `scripts/` — 确定性脚本，不做任何 LLM 编排

## 数据来源与同步关系（重要）

`references/bench_gallery.md` 的**候选区**由脚本生成，不要手改：
```bash
python scripts/build_gallery_md.py     # 读 one_eval/utils/bench_table/bench_gallery.json
```

主仓库 bench 数据有三处，改一处要想清楚是否要同步其余：
- `one_eval/utils/bench_table/bench_gallery.json` — **候选区的唯一数据源**（96 个 bench）
- `one_eval/utils/bench_table/benchData.ts` — **前端展示用**（gallery 网页）。
  头部注明 “Auto-generated from BenchmarkTable_Filter_with_keys.xlsx，run convert_xlsx_to_ts.py”。
  它与 bench_gallery.json 是**两套独立产物**，字段名/分类可能不完全一致（如 category 前端是
  "General"，json 里是 "Agents & Tools"）。**新增/修改 bench 时两边都要更新**，否则前端与 skill 不一致。
- `bench_config.json` / `*.xlsx` — 上游原始表，convert 脚本的输入。

**READY 区是 skill 独有的**，不在主仓库数据里。本版默认所有 bench 未验证 → 候选区。
bench 被 `run_eval.py --smoke` 测通后写入 `.local_state.json`（运行时自动识别复用），
人工可把稳定的登记到 bench_gallery.md 的 READY 区（手写，不会被 build 脚本覆盖）。

## 本机（Mac）环境坑（2026-06-10 跑通后记录）

- 系统 Python 3.9 < 要求的 3.10；用 brew python@3.12 建了 `.venv`（仓库根）。
- venv 依赖：`open-dataflow datasets pandas langgraph matplotlib numpy`。
  - `langgraph` 是硬依赖：`one_eval/core/state.py` 顶层 import 它（遗留耦合）。
  - 可选 metric 缺 `rouge_score`(text_gen) / `langchain_openai`(analysis LLM 裁判) 会跳过，不阻塞主流程。
- **HF 下载**：huggingface.co 直连不通，`export HF_ENDPOINT=https://hf-mirror.com`。
  但 datasets 4.0.0 的 `load_dataset` 即便设镜像仍可能报 `LocalEntryNotFoundError`（module factory 解析问题）。
  应急：直接 `curl https://hf-mirror.com/datasets/<repo>/resolve/main/<file>` 下原始 jsonl 到 `cache/`，
  文件名按 run_eval 约定 `<repo__双下划线>__<config>__<split>.jsonl`，即可被复用逻辑识别。
- 本机只验 **API 路径**；vLLM 路径代码完整但需 GPU 机验证（check_model.py 的 vLLM 探测会真起 serving）。

## 不入库的东西（.gitignore）

`eval_outputs/`、`cache/`、`.local_state.json`、`custom_metrics/*.py`、`evalspec*.yaml`（含凭证）。
唯一例外是 `assets/evalspec.template.yaml`（白名单）。**API key 只进本地 spec，不回显不入库。**

## 端到端冒烟基线（回归参照）

gpt-4o-mini × MATH-500（3 条 smoke）：dataflow score≈0.667，valid=3/3；
math_verify≈0.667 / exact_match≈0.333（math_verify 能识别 EM 漏判的等价形式）。
make_plots 产出 bench_scores / sample_validity / metric_heatmap 三图。
