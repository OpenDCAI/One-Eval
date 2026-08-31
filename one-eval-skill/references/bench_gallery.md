# Bench Gallery — benchmark 清单（READY 区 + 候选区）

> 本文件由 `scripts/build_gallery_md.py` 从主仓库 `bench_gallery.json` 生成候选区。
> **不要手改候选区**（会被覆盖）；READY 区可手工维护。

## 接入约定（务必先读 `eval_types.md`）
- **READY 区**：已 smoke 测通、key_mapping 已确认、本地数据就绪的 bench，可直接复用（免重测）。
- **候选区**：来自主仓库 gallery 的 dataflow bench，**本版默认都未验证**。接入某个候选 bench 时：
  1. `eval_type` 列只是依据原始字段做的**初步归类**，需按 `eval_types.md` 复核。
  2. `原始字段` 是 HF 上的列名，**不等于** key_mapping —— 嵌套字段要先拍平。
  3. 用 `prepare_bench.py` 下载预览结构 → 填 key_mapping → `run_eval.py --smoke` 验证。
  4. 测通后该 bench 进入 READY（`.local_state.json`），可手工登记到下方 READY 区。

> **候选区仅包含 dataflow bench**。External repo bench（使用独立评测仓库）已单独列在文件末尾 "External Repo Benchmarks" 区。

---

## READY 区（已测通，可直接复用）

> 初始为空。每测通一个 bench，在此登记：bench_name ｜ eval_type ｜ 本地数据路径 ｜ key_mapping。
> 运行时由 `run_eval.py` 通过 `.local_state.json` 自动识别 READY，无需在此手填即可复用；
> 这里的清单仅供人查阅「哪些已稳定可用」。

_（暂无）_


---

## 候选区（未验证，按分类）

### Domain-Specific（14）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| convfinqa | `key2_qa` | https://huggingface.co/datasets/AdaptLLM/finance-tasks | id, input, label |
| cupcase | `key3_q_choices_a` | https://huggingface.co/datasets/ofir408/CupCase | clean_case_presentation, correct_diagnosis, distractor1, distractor2, distractor3 |
| financebench | `key2_qa` | — | question, answer |
| financeqa | `key2_qa` | https://huggingface.co/datasets/AfterQuery/FinanceQA | COMPANY_ID, QUERY, ANSWER, CONTEXT, INDEX |
| lab-bench-language-agent-biology-benchmark | `key3_q_choices_a` | https://huggingface.co/datasets/futurehouse/lab-bench | id, question, ideal, distractors, canary, source, subtask |
| legalbench | `key2_qa` | https://huggingface.co/datasets/Equall/legalbench_instruct | prompt, response |
| livemedbench | `key3_q_choices_a` | — | question, choices, label |
| med-halt | `key3_q_choices_a` | — | question, choices, label |
| medconceptsqa | `key3_q_choices_a` | https://huggingface.co/datasets/ofir408/MedConceptsQA | question_id, answer, answer_id, option1, option2, option3, option4, question, vocab, level |
| medhallu | `key3_q_choices_a` | — | question, choices, label |
| medmcqa | `key3_q_choices_a` | https://huggingface.co/datasets/openlifescienceai/medmcqa | id, question, opa, opb, opc, opd, cop, choice_type, exp, subject_name, topic_name |
| medqa | `key3_q_choices_a` | https://huggingface.co/datasets/truehealth/medqa | question, options, answer |
| medqa-usmle | `key3_q_choices_a` | https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options | question, answer, options, meta_info, answer_idx, metamap_phrases |
| pubmedqa | `key3_q_choices_a` | https://huggingface.co/datasets/qiaojin/PubMedQA | pubid, question, context, long_answer, final_decision |


### Instruction & Chat（16）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| alpacaeval | `key3_q_a_rejected` | — | chosen, rejected |
| arena-hard-auto | `key2_qa` | — | question, answer |
| biggen-bench | `key2_qa` | https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench | id, capability, task, instance_idx, system_prompt, input, reference_answer, score_rubric |
| eq-bench | `key2_qa` | https://huggingface.co/datasets/pbevan11/EQ-Bench | prompt, reference_answer, reference_answer_fullscale |
| ifeval | `key1_text_score` | https://huggingface.co/datasets/google/IFEval | key, prompt, instruction_id_list, kwargs |
| include | `key3_q_choices_a` | https://huggingface.co/datasets/CohereLabs/include-base-44 | language, country, domain, subject, regional_feature, level, question, option_a, option_b, option_c, option_d, answer |
| infobench | `key3_q_choices_a` | https://huggingface.co/datasets/kqsong/InFoBench | id, input, category, instruction, decomposed_questions, subset, question_label |
| judgebench | `key3_q_a_rejected` | https://huggingface.co/datasets/ScalerLab/JudgeBench | pair_id, original_id, source, question, response_model, response_A, response_B, label |
| m-ifeval | `key2_qa` | https://huggingface.co/datasets/BSC-LT/IFEval_es | question, answer |
| mixeval | `key3_q_choices_a` | https://huggingface.co/datasets/MixEval/MixEval | id, problem_type, context, prompt, target, benchmark_name, options |
| mt-bench | `key3_q_a_rejected` | https://huggingface.co/datasets/lmsys/mt_bench_human_judgments | question_id, model_a, model_b, winner, judge, conversation_a, conversation_b, turn |
| multichallenge | `key2_qa` | — | question, answer |
| multinrc | `key2_qa` | https://huggingface.co/datasets/ScaleAI/MultiNRC | task_id, i18n_prompt, i18n_gtfa, english_prompt, english_gtfa, language, category |
| structflowbench | `key2_qa` | — | question, answer |
| wildbench | `key2_q_ma` | https://huggingface.co/datasets/allenai/WildBench | id, session_id, conversation_input, references, length, checklist, intent, primary_tag, secondary_tags |
| wildchat | `key1_text_score` | https://huggingface.co/datasets/allenai/WildChat-1M | conversation_hash, model, timestamp, conversation, turn, language, openai_moderation, detoxify_moderation, toxic, redacted, state, country, hashed_ip, header |


### Knowledge & QA（20）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| arc | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/ai2_arc | id, question, choices, answerKey |
| boolq | `key3_q_choices_a` | https://huggingface.co/datasets/google/boolq | question, answer, passage |
| c-eval | `key3_q_choices_a` | https://huggingface.co/datasets/ceval/ceval-exam | id, question, A, B, C, D, answer, explanation |
| gpqa | `key3_q_choices_a` | https://huggingface.co/datasets/Idavidrein/gpqa | Pre-Revision Question, Pre-Revision Correct Answer, Pre-Revision Incorrect Answer 1, Pre-Revision Incorrect Answer 2, Pre-Revision Incorrect Answer 3, Pre-Revision Explanation, Self-reported question-writing time (minutes), Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Explanation, Revision Comments (from Question Writer), Subdomain, Writer's Difficulty Estimate, Extra Revised Question, Extra Revised Explanation, Extra Revised Correct Answer, Extra Revised Incorrect Answer 1, Extra Revised Incorrect Answer 2, Extra Revised Incorrect Answer 3, Non-Expert Validator Accuracy, Majority Non-Expert Vals Incorrect, Expert Validator Accuracy, Record ID, High-level domain, Question Writer, Feedback_EV_1, Validator Revision Suggestion_EV_1, Is First Validation_EV_1, Post hoc agreement_EV_1, Sufficient Expertise?_EV_1, Understand the question?_EV_1, Question Difficulty_EV_1, Validator Answered Correctly_EV_1, Self-reported time (minutes)_EV_1, Probability Correct_EV_1, Manual Correctness Adjustment_EV_1, Expert Validator_EV_1, Feedback_EV_2, Validator Revision Suggestion_EV_2, Is First Validation_EV_2, Post hoc agreement_EV_2, Sufficient Expertise?_EV_2, Understand the question?_EV_2, Question Difficulty_EV_2, Validator Answered Correctly_EV_2, Self-reported time (minutes)_EV_2, Probability Correct_EV_2, Manual Correctness Adjustment_EV_2, Expert Validator_EV_2, Feedback_NEV_1, Validator Answered Correctly_NEV_1, Explanation_NEV_1, Self-reported time (minutes)_NEV_1, Websites visited_NEV_1, Probability Correct_NEV_1, Manual Correctness Adjustment_NEV_1, Non-Expert Validator_NEV_1, Feedback_NEV_2, Validator Answered Correctly_NEV_2, Explanation_NEV_2, Self-reported time (minutes)_NEV_2, Websites visited_NEV_2, Probability Correct_NEV_2, Manual Correctness Adjustment_NEV_2, Non-Expert Validator_NEV_2, Feedback_NEV_3, Validator Answered Correctly_NEV_3, Explanation_NEV_3, Self-reported time (minutes)_NEV_3, Websites visited_NEV_3, Probability Correct_NEV_3, Manual Correctness Adjustment_NEV_3, Non-Expert Validator_NEV_3, Expert Validator Disagreement Category, Canary String |
| hellaswag | `key3_q_choices_a` | https://huggingface.co/datasets/Rowan/hellaswag | ind, activity_label, ctx_a, ctx_b, ctx, endings, source_id, split, split_type, label |
| livebench | `key3_q_choices_a` | — | question, choices, label |
| megascience | `key2_q_ma` | https://huggingface.co/datasets/MegaScience/MegaScience | question, answer, subject, reference_answer, source |
| mmlu | `key3_q_choices_a` | https://huggingface.co/datasets/cais/mmlu | question, subject, choices, answer |
| mmlu-pro | `key3_q_choices_a` | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | question_id, question, options, answer, answer_index, cot_content, category, src |
| mmlu-redux | `key3_q_choices_a` | https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0 | question, choices, answer, error_type, source |
| mmmlu | `key3_q_choices_a` | https://huggingface.co/datasets/openai/MMMLU | Question, A, B, C, D, Answer, Subject |
| nq-open | `key2_qa` | https://huggingface.co/datasets/nq_open | question, answer |
| openbookqa | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/openbookqa | id, question_stem, choices, answerKey |
| piqa | `key3_q_choices_a` | https://huggingface.co/datasets/baber/piqa | goal, sol1, sol2, label |
| scienceqa | `key3_q_choices_a` | https://huggingface.co/datasets/derek-thomas/ScienceQA | image, question, choices, answer, hint, task, grade, subject, topic, category, skill, lecture, solution |
| sciq | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/sciq | question, distractor3, distractor1, distractor2, correct_answer, support |
| simpleqa | `key2_qa` | https://huggingface.co/datasets/basicv8vc/SimpleQA | metadata, problem, answer |
| supergpqa | `key3_q_choices_a` | https://huggingface.co/datasets/m-a-p/SuperGPQA | uuid, question, options, answer, answer_letter, discipline, field, subfield, difficulty |
| triviaqa | `key2_qa` | https://huggingface.co/datasets/mandarjoshi/trivia_qa | question, question_id, question_source, entity_pages, search_results, answer |
| truthfulqa | `key2_q_ma` | https://huggingface.co/datasets/truthfulqa/truthful_qa | type, category, question, best_answer, correct_answers, incorrect_answers, source |


### Long Context & RAG（14）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| contextualbench | `key2_qa` | https://huggingface.co/datasets/Salesforce/ContextualBench | _id, type, question, context, supporting_facts, evidences, answer |
| facts-grounding | `key3_q_choices_a` | — | question, choices, label |
| frames-factuality-retrieval-and-reasoning-measurement-set | `key2_qa` | https://huggingface.co/datasets/google/frames-benchmark | Unnamed: 0, Prompt, Answer, wikipedia_link_1, wikipedia_link_2, wikipedia_link_3, wikipedia_link_4, wikipedia_link_5, wikipedia_link_6, wikipedia_link_7, wikipedia_link_8, wikipedia_link_9, wikipedia_link_10, wikipedia_link_11+, reasoning_types, wiki_links |
| infinitebench | `key2_q_ma` | — | question, answers |
| l-eval | `key3_q_choices_a` | — | question, choices, label |
| longbench | `key3_q_choices_a` | https://huggingface.co/datasets/THUDM/LongBench-v2 | _id, domain, sub_domain, difficulty, length, question, choice_A, choice_B, choice_C, choice_D, answer, context |
| ms-marco | `key2_q_ma` | https://huggingface.co/datasets/microsoft/ms_marco | answers, passages, query, query_id, query_type, wellFormedAnswers |
| nolima | `key1_text_score` | https://huggingface.co/datasets/amodaresi/NoLiMa | text |
| ragtruth | `key2_qa` | https://huggingface.co/datasets/wandb/RAGTruth-processed | id, query, context, output, task_type, quality, model, temperature, hallucination_labels, hallucination_labels_processed, input_str |
| ruler | `key2_q_ma` | — | question, answers |
| squad-stanford-question-answering-dataset | `key2_q_ma` | https://huggingface.co/datasets/rajpurkar/squad | id, title, context, question, answers |
| squad2-0 | `key2_q_ma` | https://huggingface.co/datasets/bayes-group-diffusion/squad-2.0 | target, source |
| wice | `key3_q_choices_a` | https://huggingface.co/datasets/tasksource/wice | label, supporting_sentences, claim, evidence, meta |
| wixqa | `key2_qa` | https://huggingface.co/datasets/Wix/WixQA | question, answer, article_ids |


### Math（16）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| aime | `key2_qa` | https://huggingface.co/datasets/Maxwell-Jia/AIME_2024 | ID, Problem, Solution, Answer |
| aime2025 | `key2_qa` | https://huggingface.co/datasets/opencompass/AIME2025 | question, answer |
| aqua-rat | `key3_q_choices_a` | https://huggingface.co/datasets/deepmind/aqua_rat | question, options, rationale, correct |
| gsm8k | `key2_qa` | https://huggingface.co/datasets/openai/gsm8k | question, answer |
| gsmhard | `key2_qa` | https://huggingface.co/datasets/reasoning-machines/gsm-hard | input, code, target |
| hendrycks-math | `key2_qa` | https://huggingface.co/datasets/EleutherAI/hendrycks_math | problem, level, type, solution |
| math-500 | `key2_qa` | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | problem, answer, subject, unique_id |
| mathbench | `key3_q_choices_a` | — | question, choices, label |
| mgsm | `key2_qa` | https://huggingface.co/datasets/juletxara/mgsm | question, answer, answer_number, equation_solution |
| olympiadbench | `key2_qa` | https://huggingface.co/datasets/Hothan/OlympiadBench | id, question, solution, final_answer, answer_type, subject, language |
| omni-math | `key2_qa` | — | question, answer |
| polymath | `key2_qa` | https://huggingface.co/datasets/Qwen/PolyMath | id, question, answer |
| putnambench | `key2_qa` | — | question, answer |
| templategsm | `key2_qa` | https://huggingface.co/datasets/math-ai/TemplateGSM | problem, solution_code, result, solution_wocode, source, template_id, problem_id |
| theoremqa | `key2_qa` | https://huggingface.co/datasets/TIGER-Lab/TheoremQA | Question, Answer, Answer_type, Picture |
| we-math | `key3_q_choices_a` | https://huggingface.co/datasets/We-Math/We-Math | ID, split, knowledge concept, question, option, answer, image_path, key, question number, knowledge concept description |


### Reasoning（15）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| agieval | `key3_q_choices_a` | — | question, choices, label |
| anli | `key3_q_choices_a` | https://huggingface.co/datasets/facebook/anli | uid, premise, hypothesis, label, reason |
| arc-agi | `key2_qa` | — | question, answer |
| arc-challenge | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/ai2_arc | id, question, choices, answerKey |
| bbeh | `key3_q_choices_a` | — | question, choices, label |
| bbh | `key2_qa` | https://huggingface.co/datasets/lukaemon/bbh | input, target |
| big-bench-full | `key3_q_choices_a` | — | question, choices, label |
| drop-discrete-reasoning-over-paragraphs | `key2_q_ma` | https://huggingface.co/datasets/ucinlp/drop | section_id, query_id, passage, question, answers_spans |
| glue-general-language-understanding-evaluation | `key3_q_choices_a` | https://huggingface.co/datasets/nyu-mll/glue | premise, hypothesis, label, idx |
| graphwalks | `key2_q_ma` | https://huggingface.co/datasets/openai/graphwalks | prompt, answer_nodes, prompt_chars, problem_type |
| multinli-multi-genre-natural-language-inference | `key3_q_choices_a` | https://huggingface.co/datasets/nyu-mll/multi_nli | promptID, pairID, premise, premise_binary_parse, premise_parse, hypothesis, hypothesis_binary_parse, hypothesis_parse, genre, label |
| planbench | `key2_qa` | https://huggingface.co/datasets/tasksource/planbench | task, prompt_type, domain, instance_id, example_instance_ids, query, ground_truth_plan |
| superglue | `key3_q_choices_a` | https://huggingface.co/datasets/aps/super_glue | question, passage, idx, label |
| winogrande | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/winogrande | sentence, option1, option2, answer |
| zebralogic | `key2_qa` | https://huggingface.co/datasets/WildEval/ZebraLogic | id, size, puzzle, solution, created_at |


### Safety & Alignment（16）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| air-bench | `key3_q_a_rejected` | https://huggingface.co/datasets/stanford-crfm/air-bench-2024 | cate-idx, l2-name, l3-name, l4-name, prompt |
| anthropicredteam | `key3_q_a_rejected` | https://huggingface.co/datasets/Anthropic/hh-rlhf | chosen, rejected |
| backdoorllm | `key2_qa` | https://huggingface.co/datasets/BackdoorLLM/Backdoored_Dataset | instruction, input, output |
| beavertails | `key2_qa` | https://huggingface.co/datasets/PKU-Alignment/BeaverTails | prompt, response, category, is_safe |
| donotanswer | `key3_q_a_rejected` | https://huggingface.co/datasets/LibrAI/do-not-answer | id, risk_area, types_of_harm, specific_harms, question, GPT4_response, GPT4_harmful, GPT4_action, ChatGPT_response, ChatGPT_harmful, ChatGPT_action, Claude_response, Claude_harmful, Claude_action, ChatGLM2_response, ChatGLM2_harmful, ChatGLM2_action, llama2-7b-chat_response, llama2-7b-chat_harmful, llama2-7b-chat_action, vicuna-7b_response, vicuna-7b_harmful, vicuna-7b_action |
| global-mmlu | `key3_q_choices_a` | https://huggingface.co/datasets/CohereForAI/Global-MMLU | sample_id, subject, subject_category, question, option_a, option_b, option_c, option_d, answer, required_knowledge, time_sensitive, reference, culture, region, country, cultural_sensitivity_label, is_annotated |
| harmfulqa | `key3_q_choices_a` | https://huggingface.co/datasets/declare-lab/HarmfulQA | id, topic, subtopic, question, blue_conversations, red_conversations |
| jailbreakbench | `key3_q_a_rejected` | — | chosen, rejected |
| or-bench | `key3_q_a_rejected` | https://huggingface.co/datasets/bench-llm/or-bench | prompt, category |
| realtoxicityprompt | `key3_q_a_rejected` | https://huggingface.co/datasets/allenai/real-toxicity-prompts | filename, begin, end, challenging, prompt, continuation |
| safetybench | `key3_q_choices_a` | https://huggingface.co/datasets/thu-coai/SafetyBench | question, options, category, id |
| simplesafetytests | `key1_text_score` | — | prompt |
| stereoset | `key3_q_choices_a` | https://huggingface.co/datasets/McGill-NLP/stereoset | id, target, bias_type, context, sentences |
| toxigen | `key1_text_score` | https://huggingface.co/datasets/toxigen/toxigen-data | text, target_group, factual?, ingroup_effect, lewd, framing, predicted_group, stereotyping, intent, toxicity_ai, toxicity_human, predicted_author, actual_method |
| winogender | `key3_q_choices_a` | https://huggingface.co/datasets/oskarvanderwal/winogender | sentid, sentence, pronoun, occupation, participant, gender, target, label |
| xstest | `key3_q_a_rejected` | — | chosen, rejected |


---

## External Repo Benchmarks（独立评测仓库）

以下 benchmark 使用官方或指定版本的外部评测仓库，由 `ExternalRepoRunner` 自动完成环境准备、评测执行和结果解析。**这些 bench 不需要填写 eval_type 或 key_mapping，也不通过 prepare_bench 下载数据**；所有配置由 `bench_gallery.json` 提供。

### 使用方式

在 EvalSpec 中：
- 必须填写 `bench_name` 和 `bench_kind: external_repo`
- `params` 只填写希望覆盖的参数；未填写的参数使用 gallery 中的默认值
- `model_name` 自动取自 `model.model_name_or_path`
- `max_samples` 自动取自 `runtime.max_samples`
- `api_url` 和 `api_key` 从 `model` 配置自动注入，不应写入 `params`
- **前置条件**（Docker、Python 版本、HF_TOKEN、联网等）见下表；agent 应在生成 evalspec 前向用户确认这些条件

示例：
```yaml
benchmarks:
  - bench_name: "livecodebench"
    bench_kind: "external_repo"
    params:
      release_version: "release_v6"
      n: 10
```

### 可用的 External Repo Benchmarks

| bench | 能力/主指标 | 前置条件 |
|---|---|---|
| mathvista | 视觉数学推理，accuracy | vision 模型 |
| mmmu | 多模态学科推理，accuracy | vision 模型 |
| mmmu_pro | 强化多模态推理，accuracy | vision 模型 |
| livecodebench | 代码生成，pass@1 | Docker（宿主机需安装并启动 Docker daemon） |
| humanevalplus | 代码生成，pass@1 | Docker（宿主机需安装并启动 Docker daemon） |
| mbppplus | 代码生成，pass@1 | Docker（宿主机需安装并启动 Docker daemon） |
| bfcl | 函数调用，overall accuracy | Docker（宿主机需安装并启动 Docker daemon） |
| tau2_bench | 多轮工具调用，pass rate | 无额外前置条件 |
| gaia | 搜索工具Agent，accuracy | ① Docker（宿主机需安装并启动 Docker daemon） ② HF_TOKEN (gated) ③ 联网环境 |
| omniabench | 通用Agent，pass@1 | 无额外前置条件 |

> **重要**：表格中展示的参数信息力求完整，但**参数的类型、默认值、choices 等细节以 `bench_gallery.json` 中对应条目的 `meta.repo_eval.params` 为最终依据**。生成 evalspec 前，不要根据其他 benchmark 推测参数。

### 参数详细说明

#### mathvista

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| temperature | float | 0.0 | 采样温度 |
| max_tokens | int | 1024 | 最大生成长度 |
| shot_num | int | 0 | few-shot 样本数 |
| shot_type | string | solution | solution/code — few-shot 示例类型 |

#### mmmu

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| split | string | validation | validation/test — 数据集划分 |
| temperature | float | 0.0 | 采样温度 |
| max_tokens | int | 1024 | 最大生成长度 |

#### mmmu_pro

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| setting | string | vision | vision/standard (10 options)/standard (4 options) — 输入设置 |
| mode | string | direct | direct/cot — 提示模式（直接回答/思维链） |
| max_workers | int | 16 | 并行推理 worker 数 |
| max_tokens | int | 4096 | 最大生成长度 |

#### livecodebench

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| temperature | float | 0.2 | 采样温度 |
| release_version | string | release_latest | release_v1/v2/v3/v4/v5/v6/release_latest — 评测版本 |
| n | int | 10 | 每题生成样本数（pass@k） |
| scenario | string | codegeneration | codegeneration/selfrepair/testoutputprediction/codeexecution — 评测场景 |
| max_workers | int | -1 | 并行 worker 数（-1 = 自动检测） |
| num_process_evaluate | int | 1 | 评测进程数 |
| timeout | int | 6 | 代码执行超时（秒） |
| eval_batch_size | int | 32 | 每批评测的问题数 |

#### humanevalplus

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| temperature | float | 0.0 | 采样温度 |
| n_samples | int | 1 | 每题生成样本数（pass@k） |

#### mbppplus

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| temperature | float | 0.0 | 采样温度 |
| n_samples | int | 1 | 每题生成样本数（pass@k） |

#### bfcl

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| temperature | float | 0.001 | 采样温度 |
| test_category | string | all | ast/live/multi_turn/non_live/single_turn/all/default — 测试类别或预设 |
| num_threads | int | 4 | 并发生成线程数 |

#### tau2_bench

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| domains | string | airline,retail,telecom | 逗号分隔的领域列表（airline/retail/telecom/banking_knowledge） |
| user_model | string | null | 用户模拟器模型（默认使用 agent 模型） |
| num_trials | int | 4 | 每个任务的评测试次 |
| task_seed | int | 300 | 任务采样/排序的随机种子 |
| max_concurrency | int | 5 | 最大并发评测任务数 |
| agent_reasoning_effort | string | none | none/low/medium/high — agent LLM 推理努力程度 |
| user_reasoning_effort | string | low | none/low/medium/high — 用户模拟器 LLM 推理努力程度 |
| eval_model | string | null | 自然语言评估器模型（默认使用 agent 模型） |

#### gaia

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| concurrency | int | 4 | 并发任务数 |

#### omniabench

| 参数 | 类型 | 默认值 | 可选值/说明 |
|---|---|---|---|
| temperature | float | 0.0 | 采样温度 |
| routes | string | route1,route2,route3,route4 | 逗号分隔的评测路线 |
| pass_k | int | 1 | pass@k：每个任务的尝试次数 |
| max_task_workers | int | 8 | 最大并发任务 worker 数 |
| judge_model | string | null | 评分 judge 模型（默认使用 agent 模型） |
| user_model | string | null | 用户模拟器模型（默认使用 agent 模型） |
| infer_mode | string | fc | fc/prompt — 推理模式（函数调用/提示） |
| reasoning_effort | string | high | high/medium/low — agent 推理努力程度 |

