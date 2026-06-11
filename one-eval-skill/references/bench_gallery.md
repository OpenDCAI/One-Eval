# Bench Gallery — benchmark 清单（READY 区 + 候选区）

> 本文件由 `scripts/build_gallery_md.py` 从主仓库 `bench_gallery.json` 生成候选区。
> **不要手改候选区**（会被覆盖）；READY 区可手工维护。

## 接入约定（务必先读 `eval_types.md`）
- **READY 区**：已 smoke 测通、key_mapping 已确认、本地数据就绪的 bench，可直接复用（免重测）。
- **候选区**：来自主仓库 gallery 的 103 个 bench，**本版默认都未验证**。接入某个候选 bench 时：
  1. `eval_type` 列只是依据原始字段做的**初步归类**，需按 `eval_types.md` 复核。
  2. `原始字段` 是 HF 上的列名，**不等于** key_mapping —— 嵌套字段要先拍平。
  3. 用 `prepare_bench.py` 下载预览结构 → 填 key_mapping → `run_eval.py --smoke` 验证。
  4. 测通后该 bench 进入 READY（`.local_state.json`），可手工登记到下方 READY 区。


---

## READY 区（已测通，可直接复用）

> 初始为空。每测通一个 bench，在此登记：bench_name ｜ eval_type ｜ 本地数据路径 ｜ key_mapping。
> 运行时由 `run_eval.py` 通过 `.local_state.json` 自动识别 READY，无需在此手填即可复用；
> 这里的清单仅供人查阅「哪些已稳定可用」。

| bench_name | eval_type | 本地数据路径(cache/) | key_mapping | 备注 |
|---|---|---|---|---|
| gsm8k | `key2_qa` | openai__gsm8k__main__test.jsonl | question / answer | 小学数学，含 math_verify 假阴性翻正 |
| MATH-500 | `key2_qa` | HuggingFaceH4__MATH-500__None__test.jsonl | problem / answer | 竞赛数学 500 题 |
| AIME_2024 | `key2_qa` | Maxwell-Jia__AIME_2024__None__train.jsonl | question / answer | AIME 2024，仅 30 题、整数答案、方差大 |
| MMLU-Pro | `key3_q_choices_a` | TIGER-Lab__MMLU-Pro__None__test.jsonl | question / choices / answer | 10 选项进阶知识，12032 题，14 学科 |
| mmlu-redux | `key3_q_choices_a` | edinburgh-dawg__mmlu-redux-2.0__abstract_algebra__test.jsonl | question / choices / answer | 子集 abstract_algebra |
| supergpqa | `key3_q_choices_a` | m-a-p__SuperGPQA__default__train.jsonl | (见 spec) | 研究生级跨学科 285 学科 |
| c-eval | `key3_q_choices_a` | ceval__ceval-exam__accountant__val.jsonl | question / choices / answer | 中文，子集 accountant |
| polymath | `key2_qa` | Qwen__PolyMath__en__high.jsonl | (见 spec) | 子集 en/high |

> MMLU-Pro / AIME_2024 因 datasets 4.0 下载层对镜像不稳，改用 `curl` 直拉
> `https://hf-mirror.com/datasets/<repo>/resolve/main/<file>` 后转 jsonl 入 cache，按
> `{repo//→__}__{config}__{split}.jsonl` 命名即被 `run_eval.py` 自动复用、免下载。



---

## 候选区（未验证，按分类）


### Agents & Tools（6）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| acpbench | `key2_qa` | https://huggingface.co/datasets/ibm-research/acp_bench | id, group, context, question, answer |
| agentharm | `key2_qa` | https://huggingface.co/datasets/ai-safety-institute/AgentHarm | id, id_original, detailed_prompt, hint_included, name, category, prompt, target_functions, grading_function |
| crmarena | `key2_qa` | https://huggingface.co/datasets/Salesforce/CRMArena | idx, answer, metadata, reward_metric, query, task |
| crmarena-pro | `key2_qa` | https://huggingface.co/datasets/Salesforce/CRMArenaPro | idx, answer, task, persona, metadata, reward_metric, query |
| gaia | `key2_qa` | https://huggingface.co/datasets/gaia-benchmark/GAIA | task_id, Question, Level, Final answer, file_name, file_path, Annotator Metadata |
| scigym | `key2_qa` | https://huggingface.co/datasets/h4duan/scigym-sbml | folder_name, truth_sedml, partial, truth_xml |


### Coding（10）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| bigcodebench | `key2_qa` | https://huggingface.co/datasets/bigcode/bigcodebench | task_id, complete_prompt, instruct_prompt, canonical_solution, code_prompt, test, entry_point, doc_struct, libs |
| classeval | `key2_qa` | https://huggingface.co/datasets/FudanSELab/ClassEval | task_id, skeleton, test, solution_code, import_statement, class_description, methods_info, class_name, test_classes, class_constructor, fields |
| codeelo | `key2_qa` | https://huggingface.co/datasets/Qwen/CodeElo | problem_id, url, title, rating, tags, div, time_limit_ms, memory_limit_mb, description, input, output, interaction, examples, note |
| cruxeval-code-reasoning-understanding-and-execution-evaluation | `key2_qa` | https://huggingface.co/datasets/cruxeval-org/cruxeval | code, input, output, id |
| ds-1000 | `key2_qa` | https://huggingface.co/datasets/xlangai/DS-1000 | prompt, reference_code, metadata, code_context |
| humaneval | `key2_qa` | https://huggingface.co/datasets/openai/openai_humaneval | task_id, prompt, canonical_solution, test, entry_point |
| mbpp | `key2_qa` | https://huggingface.co/datasets/google-research-datasets/mbpp | task_id, text, code, test_list, test_setup_code, challenge_test_list |
| scicode | `key2_qa` | https://huggingface.co/datasets/SciCode1/SciCode | problem_name, problem_id, problem_description_main, problem_background_main, problem_io, required_dependencies, sub_steps, general_solution, general_tests |
| swe-bench | `key2_qa` | https://huggingface.co/datasets/princeton-nlp/SWE-bench | repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit |
| swe-bench-verified | `key3_q_a_rejected` | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified | repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit, difficulty |


### Domain-Specific（10）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| convfinqa | `key2_qa` | https://huggingface.co/datasets/AdaptLLM/finance-tasks | id, input, label |
| cupcase | `key3_q_choices_a` | https://huggingface.co/datasets/ofir408/CupCase | clean_case_presentation, correct_diagnosis, distractor1, distractor2, distractor3 |
| financeqa | `key2_qa` | https://huggingface.co/datasets/AfterQuery/FinanceQA | COMPANY_ID, QUERY, ANSWER, CONTEXT, INDEX |
| lab-bench-language-agent-biology-benchmark | `key3_q_choices_a` | https://huggingface.co/datasets/futurehouse/lab-bench | id, question, ideal, distractors, canary, source, subtask |
| legalbench | `key2_qa` | https://huggingface.co/datasets/Equall/legalbench_instruct | prompt, response |
| medconceptsqa | `key3_q_choices_a` | https://huggingface.co/datasets/ofir408/MedConceptsQA | question_id, answer, answer_id, option1, option2, option3, option4, question, vocab, level |
| medmcqa | `key3_q_choices_a` | https://huggingface.co/datasets/openlifescienceai/medmcqa | id, question, opa, opb, opc, opd, cop, choice_type, exp, subject_name, topic_name |
| medqa | `key3_q_choices_a` | https://huggingface.co/datasets/truehealth/medqa | question, options, answer |
| medqa-usmle | `key3_q_choices_a` | https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options | question, answer, options, meta_info, answer_idx, metamap_phrases |
| pubmedqa | `key3_q_choices_a` | https://huggingface.co/datasets/qiaojin/PubMedQA | pubid, question, context, long_answer, final_decision |


### Instruction & Chat（11）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| biggen-bench | `key2_qa` | https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench | id, capability, task, instance_idx, system_prompt, input, reference_answer, score_rubric |
| eq-bench | `key2_qa` | https://huggingface.co/datasets/pbevan11/EQ-Bench | prompt, reference_answer, reference_answer_fullscale |
| ifeval | `key1_text_score` | https://huggingface.co/datasets/google/IFEval | key, prompt, instruction_id_list, kwargs |
| include | `key3_q_choices_a` | https://huggingface.co/datasets/CohereLabs/include-base-44 | language, country, domain, subject, regional_feature, level, question, option_a, option_b, option_c, option_d, answer |
| infobench | `key3_q_choices_a` | https://huggingface.co/datasets/kqsong/InFoBench | id, input, category, instruction, decomposed_questions, subset, question_label |
| judgebench | `key3_q_a_rejected` | https://huggingface.co/datasets/ScalerLab/JudgeBench | pair_id, original_id, source, question, response_model, response_A, response_B, label |
| mixeval | `key3_q_choices_a` | https://huggingface.co/datasets/MixEval/MixEval | id, problem_type, context, prompt, target, benchmark_name, options |
| mt-bench | `key3_q_a_rejected` | https://huggingface.co/datasets/lmsys/mt_bench_human_judgments | question_id, model_a, model_b, winner, judge, conversation_a, conversation_b, turn |
| multinrc | `key2_qa` | https://huggingface.co/datasets/ScaleAI/MultiNRC | task_id, i18n_prompt, i18n_gtfa, english_prompt, english_gtfa, language, category |
| wildbench | `key2_q_ma` | https://huggingface.co/datasets/allenai/WildBench | id, session_id, conversation_input, references, length, checklist, intent, primary_tag, secondary_tags |
| wildchat | `key1_text_score` | https://huggingface.co/datasets/allenai/WildChat-1M | conversation_hash, model, timestamp, conversation, turn, language, openai_moderation, detoxify_moderation, toxic, redacted, state, country, hashed_ip, header |


### Knowledge & QA（18）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| arc | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/ai2_arc | id, question, choices, answerKey |
| boolq | `key3_q_choices_a` | https://huggingface.co/datasets/google/boolq | question, answer, passage |
| c-eval | `key3_q_choices_a` | https://huggingface.co/datasets/ceval/ceval-exam | id, question, A, B, C, D, answer, explanation |
| gpqa | `key3_q_choices_a` | https://huggingface.co/datasets/Idavidrein/gpqa | Pre-Revision Question, Pre-Revision Correct Answer, Pre-Revision Incorrect Answer 1, Pre-Revision Incorrect Answer 2, Pre-Revision Incorrect Answer 3, Pre-Revision Explanation, Self-reported question-writing time (minutes), Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Explanation, Revision Comments (from Question Writer), Subdomain, Writer's Difficulty Estimate, Extra Revised Question, Extra Revised Explanation, Extra Revised Correct Answer, Extra Revised Incorrect Answer 1, Extra Revised Incorrect Answer 2, Extra Revised Incorrect Answer 3, Non-Expert Validator Accuracy, Majority Non-Expert Vals Incorrect, Expert Validator Accuracy, Record ID, High-level domain, Question Writer, Feedback_EV_1, Validator Revision Suggestion_EV_1, Is First Validation_EV_1, Post hoc agreement_EV_1, Sufficient Expertise?_EV_1, Understand the question?_EV_1, Question Difficulty_EV_1, Validator Answered Correctly_EV_1, Self-reported time (minutes)_EV_1, Probability Correct_EV_1, Manual Correctness Adjustment_EV_1, Expert Validator_EV_1, Feedback_EV_2, Validator Revision Suggestion_EV_2, Is First Validation_EV_2, Post hoc agreement_EV_2, Sufficient Expertise?_EV_2, Understand the question?_EV_2, Question Difficulty_EV_2, Validator Answered Correctly_EV_2, Self-reported time (minutes)_EV_2, Probability Correct_EV_2, Manual Correctness Adjustment_EV_2, Expert Validator_EV_2, Feedback_NEV_1, Validator Answered Correctly_NEV_1, Explanation_NEV_1, Self-reported time (minutes)_NEV_1, Websites visited_NEV_1, Probability Correct_NEV_1, Manual Correctness Adjustment_NEV_1, Non-Expert Validator_NEV_1, Feedback_NEV_2, Validator Answered Correctly_NEV_2, Explanation_NEV_2, Self-reported time (minutes)_NEV_2, Websites visited_NEV_2, Probability Correct_NEV_2, Manual Correctness Adjustment_NEV_2, Non-Expert Validator_NEV_2, Feedback_NEV_3, Validator Answered Correctly_NEV_3, Explanation_NEV_3, Self-reported time (minutes)_NEV_3, Websites visited_NEV_3, Probability Correct_NEV_3, Manual Correctness Adjustment_NEV_3, Non-Expert Validator_NEV_3, Expert Validator Disagreement Category, Canary String |
| hellaswag | `key3_q_choices_a` | https://huggingface.co/datasets/Rowan/hellaswag | ind, activity_label, ctx_a, ctx_b, ctx, endings, source_id, split, split_type, label |
| megascience | `key2_q_ma` | https://huggingface.co/datasets/MegaScience/MegaScience | question, answer, subject, reference_answer, source |
| mmlu | `key3_q_choices_a` | https://huggingface.co/datasets/cais/mmlu | question, subject, choices, answer |
| mmlu-pro | `key3_q_choices_a` | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | question_id, question, options, answer, answer_index, cot_content, category, src |
| mmlu-redux | `key3_q_choices_a` | https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0 | question, choices, answer, error_type, source |
| mmmlu | `key3_q_choices_a` | https://huggingface.co/datasets/openai/MMMLU | Question, A, B, C, D, Answer, Subject |
| nq-open | `key2_qa` | https://huggingface.co/datasets/nq_open | question, answer |
| openbookqa | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/openbookqa | id, question_stem, choices, answerKey |
| scienceqa | `key3_q_choices_a` | https://huggingface.co/datasets/derek-thomas/ScienceQA | image, question, choices, answer, hint, task, grade, subject, topic, category, skill, lecture, solution |
| sciq | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/sciq | question, distractor3, distractor1, distractor2, correct_answer, support |
| simpleqa | `key2_qa` | https://huggingface.co/datasets/basicv8vc/SimpleQA | metadata, problem, answer |
| supergpqa | `key3_q_choices_a` | https://huggingface.co/datasets/m-a-p/SuperGPQA | uuid, question, options, answer, answer_letter, discipline, field, subfield, difficulty |
| triviaqa | `key2_qa` | https://huggingface.co/datasets/mandarjoshi/trivia_qa | question, question_id, question_source, entity_pages, search_results, answer |
| truthfulqa | `key2_q_ma` | https://huggingface.co/datasets/truthfulqa/truthful_qa | type, category, question, best_answer, correct_answers, incorrect_answers, source |


### Long Context & RAG（10）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| contextualbench | `key2_qa` | https://huggingface.co/datasets/Salesforce/ContextualBench | _id, type, question, context, supporting_facts, evidences, answer |
| frames-factuality-retrieval-and-reasoning-measurement-set | `key2_qa` | https://huggingface.co/datasets/google/frames-benchmark | Unnamed: 0, Prompt, Answer, wikipedia_link_1, wikipedia_link_2, wikipedia_link_3, wikipedia_link_4, wikipedia_link_5, wikipedia_link_6, wikipedia_link_7, wikipedia_link_8, wikipedia_link_9, wikipedia_link_10, wikipedia_link_11+, reasoning_types, wiki_links |
| longbench | `key3_q_choices_a` | https://huggingface.co/datasets/THUDM/LongBench-v2 | _id, domain, sub_domain, difficulty, length, question, choice_A, choice_B, choice_C, choice_D, answer, context |
| ms-marco | `key2_q_ma` | https://huggingface.co/datasets/microsoft/ms_marco | answers, passages, query, query_id, query_type, wellFormedAnswers |
| nolima | `key1_text_score` | https://huggingface.co/datasets/amodaresi/NoLiMa | text |
| ragtruth | `key2_qa` | https://huggingface.co/datasets/wandb/RAGTruth-processed | id, query, context, output, task_type, quality, model, temperature, hallucination_labels, hallucination_labels_processed, input_str |
| squad-stanford-question-answering-dataset | `key2_q_ma` | https://huggingface.co/datasets/rajpurkar/squad | id, title, context, question, answers |
| squad2-0 | `key2_q_ma` | https://huggingface.co/datasets/bayes-group-diffusion/squad-2.0 | target, source |
| wice | `key3_q_choices_a` | https://huggingface.co/datasets/tasksource/wice | label, supporting_sentences, claim, evidence, meta |
| wixqa | `key2_qa` | https://huggingface.co/datasets/Wix/WixQA | question, answer, article_ids |


### Math（12）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| aime | `key2_qa` | https://huggingface.co/datasets/Maxwell-Jia/AIME_2024 | ID, Problem, Solution, Answer |
| aime2025 | `key2_qa` | https://huggingface.co/datasets/opencompass/AIME2025 | question, answer |
| aqua-rat | `key3_q_choices_a` | https://huggingface.co/datasets/deepmind/aqua_rat | question, options, rationale, correct |
| gsm8k | `key2_qa` | https://huggingface.co/datasets/openai/gsm8k | question, answer |
| gsmhard | `key2_qa` | https://huggingface.co/datasets/reasoning-machines/gsm-hard | input, code, target |
| hendrycks-math | `key2_qa` | https://huggingface.co/datasets/EleutherAI/hendrycks_math | problem, level, type, solution |
| mgsm | `key2_qa` | https://huggingface.co/datasets/juletxara/mgsm | question, answer, answer_number, equation_solution |
| olympiadbench | `key2_qa` | https://huggingface.co/datasets/Hothan/OlympiadBench | id, question, solution, final_answer, answer_type, subject, language |
| polymath | `key2_qa` | https://huggingface.co/datasets/Qwen/PolyMath | id, question, answer |
| templategsm | `key2_qa` | https://huggingface.co/datasets/math-ai/TemplateGSM | problem, solution_code, result, solution_wocode, source, template_id, problem_id |
| theoremqa | `key2_qa` | https://huggingface.co/datasets/TIGER-Lab/TheoremQA | Question, Answer, Answer_type, Picture |
| we-math | `key3_q_choices_a` | https://huggingface.co/datasets/We-Math/We-Math | ID, split, knowledge concept, question, option, answer, image_path, key, question number, knowledge concept description |


### Reasoning（12）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| anli | `key3_q_choices_a` | https://huggingface.co/datasets/facebook/anli | uid, premise, hypothesis, label, reason |
| arc-challenge | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/ai2_arc | id, question, choices, answerKey |
| bbh | `key2_qa` | https://huggingface.co/datasets/lukaemon/bbh | input, target |
| drop-discrete-reasoning-over-paragraphs | `key2_q_ma` | https://huggingface.co/datasets/ucinlp/drop | section_id, query_id, passage, question, answers_spans |
| glue-general-language-understanding-evaluation | `key3_q_choices_a` | https://huggingface.co/datasets/nyu-mll/glue | premise, hypothesis, label, idx |
| graphwalks | `key2_q_ma` | https://huggingface.co/datasets/openai/graphwalks | prompt, answer_nodes, prompt_chars, problem_type |
| lambada-language-modelling-broadened-to-account-for-discourse-aspects | `key1_text_score` | https://huggingface.co/datasets/cimec/lambada | text, domain |
| multinli-multi-genre-natural-language-inference | `key3_q_choices_a` | https://huggingface.co/datasets/nyu-mll/multi_nli | promptID, pairID, premise, premise_binary_parse, premise_parse, hypothesis, hypothesis_binary_parse, hypothesis_parse, genre, label |
| planbench | `key2_qa` | https://huggingface.co/datasets/tasksource/planbench | task, prompt_type, domain, instance_id, example_instance_ids, query, ground_truth_plan |
| superglue | `key3_q_choices_a` | https://huggingface.co/datasets/aps/super_glue | question, passage, idx, label |
| winogrande | `key3_q_choices_a` | https://huggingface.co/datasets/allenai/winogrande | sentence, option1, option2, answer |
| zebralogic | `key2_qa` | https://huggingface.co/datasets/WildEval/ZebraLogic | id, size, puzzle, solution, created_at |


### Safety & Alignment（14）

| bench_name | eval_type(初判) | source_url | 原始字段 |
|---|---|---|---|
| air-bench | `key3_q_a_rejected` | https://huggingface.co/datasets/stanford-crfm/air-bench-2024 | cate-idx, l2-name, l3-name, l4-name, prompt |
| anthropicredteam | `key3_q_a_rejected` | https://huggingface.co/datasets/Anthropic/hh-rlhf | chosen, rejected |
| backdoorllm | `key2_qa` | https://huggingface.co/datasets/BackdoorLLM/Backdoored_Dataset | instruction, input, output |
| beavertails | `key2_qa` | https://huggingface.co/datasets/PKU-Alignment/BeaverTails | prompt, response, category, is_safe |
| civil-comments | `key1_text_score` | https://huggingface.co/datasets/google/civil_comments | text, toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit |
| donotanswer | `key3_q_a_rejected` | https://huggingface.co/datasets/LibrAI/do-not-answer | id, risk_area, types_of_harm, specific_harms, question, GPT4_response, GPT4_harmful, GPT4_action, ChatGPT_response, ChatGPT_harmful, ChatGPT_action, Claude_response, Claude_harmful, Claude_action, ChatGLM2_response, ChatGLM2_harmful, ChatGLM2_action, llama2-7b-chat_response, llama2-7b-chat_harmful, llama2-7b-chat_action, vicuna-7b_response, vicuna-7b_harmful, vicuna-7b_action |
| global-mmlu | `key3_q_choices_a` | https://huggingface.co/datasets/CohereForAI/Global-MMLU | sample_id, subject, subject_category, question, option_a, option_b, option_c, option_d, answer, required_knowledge, time_sensitive, reference, culture, region, country, cultural_sensitivity_label, is_annotated |
| harmfulqa | `key3_q_choices_a` | https://huggingface.co/datasets/declare-lab/HarmfulQA | id, topic, subtopic, question, blue_conversations, red_conversations |
| or-bench | `key3_q_a_rejected` | https://huggingface.co/datasets/bench-llm/or-bench | prompt, category |
| realtoxicityprompt | `key3_q_a_rejected` | https://huggingface.co/datasets/allenai/real-toxicity-prompts | filename, begin, end, challenging, prompt, continuation |
| safetybench | `key3_q_choices_a` | https://huggingface.co/datasets/thu-coai/SafetyBench | question, options, category, id |
| stereoset | `key3_q_choices_a` | https://huggingface.co/datasets/McGill-NLP/stereoset | id, target, bias_type, context, sentences |
| toxigen | `key1_text_score` | https://huggingface.co/datasets/toxigen/toxigen-data | text, target_group, factual?, ingroup_effect, lewd, framing, predicted_group, stereotyping, intent, toxicity_ai, toxicity_human, predicted_author, actual_method |
| winogender | `key3_q_choices_a` | https://huggingface.co/datasets/oskarvanderwal/winogender | sentid, sentence, pronoun, occupation, participant, gender, target, label |

