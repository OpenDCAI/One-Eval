import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen,
  Brain,
  Code2,
  GraduationCap,
  ShieldCheck,
  Search,
  SlidersHorizontal,
  Tag,
  X,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

type BenchCategory = "Math" | "Reasoning" | "Knowledge" | "Safety" | "Coding" | "General";

type BenchMeta = {
  category: BenchCategory;
  tags: string[];
  description: string;
  sampleCount?: number;
  datasetUrl?: string;
  datasetKeys?: string[];
};

type BenchItem = {
  id: string;
  name: string;
  meta: BenchMeta;
};

const DEFAULT_BENCHES: BenchItem[] = [
  { id: "crmarena", name: "CRMArena", meta: { category: "General", tags: ["agents & tools use"], description: "A benchmark designed to evaluate AI agents on realistic tasks grounded in professional work environments.", datasetUrl: "https://huggingface.co/datasets/Salesforce/CRMArena", datasetKeys: ["idx", "answer", "metadata", "reward_metric", "query", "task"] } },
  { id: "crmarena-pro", name: "CRMArena-Pro", meta: { category: "General", tags: ["agents & tools use"], description: "A benchmark developed by Salesforce AI Research to evaluate LLM agents in realistic CRM (Customer Relationship Management) tasks", datasetUrl: "https://huggingface.co/datasets/Salesforce/CRMArenaPro", datasetKeys: ["idx", "answer", "task", "persona", "metadata", "reward_metric", "query"] } },
  { id: "gaia", name: "GAIA", meta: { category: "General", tags: ["agents & tools use"], description: "Presents real-world questions requiring reasoning, multi-modality handling, and tool-use proficiency to evaluate general AI assistants.", datasetUrl: "https://huggingface.co/datasets/gaia-benchmark/GAIA", datasetKeys: ["task_id", "Question", "Level", "Final answer", "file_name", "file_path", "Annotator Metadata"] } },
  { id: "scigym", name: "SciGym", meta: { category: "General", tags: ["agents & tools use", "domain-specific"], description: "A benchmark that assesses LLMs' iterative experiment design and analysis abilities in open-ended scientific discovery tasks. It challenges models to uncover biological mechanisms by designing and interpreting simulated experiments.", datasetUrl: "https://huggingface.co/datasets/h4duan/scigym-sbml", datasetKeys: ["folder_name", "truth_sedml", "partial", "truth_xml"] } },
  { id: "acpbench", name: "ACPBench", meta: { category: "Reasoning", tags: ["agents & tools use", "language & reasoning"], description: "A benchmark for evaluating the reasoning tasks in the field of planning. The benchmark consists of 7 reasoning tasks over 13 planning domains.", datasetUrl: "https://huggingface.co/datasets/ibm-research/acp_bench", datasetKeys: ["id", "group", "context", "question", "answer"] } },
  { id: "global-mmlu", name: "Global MMLU", meta: { category: "Safety", tags: ["bias & ethics"], description: "Translated MMLU, that also includes cultural sensitivity annotations for a subset of the questions, with evaluation coverage across 42 languages.", datasetUrl: "https://huggingface.co/datasets/CohereForAI/Global-MMLU", datasetKeys: ["sample_id", "subject", "subject_category", "question", "option_a", "option_b", "option_c", "option_d", "answer", "required_knowledge", "time_sensitive", "reference", "culture", "region", "country", "cultural_sensitivity_label", "is_annotated"] } },
  { id: "civil-comments", name: "Civil Comments", meta: { category: "Safety", tags: ["bias & ethics"], description: "A suite of threshold-agnostic metrics for unintended bias and a test set of online comments with crowd-sourced annotations for identity references.", datasetUrl: "https://huggingface.co/datasets/google/civil_comments", datasetKeys: ["text", "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"] } },
  { id: "scicode", name: "SciCode", meta: { category: "Coding", tags: ["coding"], description: "A benchmark that challenges language models to code solutions for scientific problems.", datasetUrl: "https://huggingface.co/datasets/SciCode1/SciCode", datasetKeys: ["problem_name", "problem_id", "problem_description_main", "problem_background_main", "problem_io", "required_dependencies", "sub_steps", "general_solution", "general_tests"] } },
  { id: "classeval", name: "ClassEval", meta: { category: "Coding", tags: ["coding"], description: "Class-level Python code generation tasks.", datasetUrl: "https://huggingface.co/datasets/FudanSELab/ClassEval", datasetKeys: ["task_id", "skeleton", "test", "solution_code", "import_statement", "class_description", "methods_info", "class_name", "test_classes", "class_constructor", "fields"] } },
  { id: "swe-bench-verified", name: "SWE-bench verified", meta: { category: "Coding", tags: ["coding"], description: "A subset of SWE-bench, consisting of 500 samples verified to be non-problematic by our human annotators.", datasetUrl: "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified", datasetKeys: ["repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "created_at", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit", "difficulty"] } },
  { id: "swe-bench", name: "SWE-bench", meta: { category: "Coding", tags: ["coding"], description: "Real-world software issues collected from GitHub.", datasetUrl: "https://huggingface.co/datasets/princeton-nlp/SWE-bench", datasetKeys: ["repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "created_at", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"] } },
  { id: "cruxeval-code-reasoning-understanding-and-execution-evaluation", name: "CRUXEval", meta: { category: "Coding", tags: ["coding"], description: "A set of Python functions and input-output pairs that consists of two tasks: input prediction and output prediction.", datasetUrl: "https://huggingface.co/datasets/cruxeval-org/cruxeval", datasetKeys: ["code", "input", "output", "id"] } },
  { id: "ds-1000", name: "DS-1000", meta: { category: "Coding", tags: ["coding"], description: "Code generation benchmark with data science problems spanning seven Python libraries, such as NumPy and Pandas.", datasetUrl: "https://huggingface.co/datasets/xlangai/DS-1000", datasetKeys: ["prompt", "reference_code", "metadata", "code_context"] } },
  { id: "humaneval", name: "HumanEval", meta: { category: "Coding", tags: ["coding"], description: "Programming tasks and unit tests to check model-generated code.", datasetUrl: "https://huggingface.co/datasets/openai/openai_humaneval", datasetKeys: ["task_id", "prompt", "canonical_solution", "test", "entry_point"] } },
  { id: "codeelo", name: "CodeElo", meta: { category: "Coding", tags: ["coding"], description: "A standardized competition-level code generation benchmark.", datasetUrl: "https://huggingface.co/datasets/Qwen/CodeElo", datasetKeys: ["problem_id", "url", "title", "rating", "tags", "div", "time_limit_ms", "memory_limit_mb", "description", "input", "output", "interaction", "examples", "note"] } },
  { id: "wildchat", name: "WildChat", meta: { category: "General", tags: ["conversation & chatbots"], description: "A collection of 1 million conversations between human users and ChatGPT, alongside demographic data.", datasetUrl: "https://huggingface.co/datasets/allenai/WildChat-1M", datasetKeys: ["conversation_hash", "model", "timestamp", "conversation", "turn", "language", "openai_moderation", "detoxify_moderation", "toxic", "redacted", "state", "country", "hashed_ip", "header"] } },
  { id: "mixeval", name: "MixEval", meta: { category: "General", tags: ["conversation & chatbots"], description: "A ground-truth-based dynamic benchmark derived from off-the-shelf benchmark mixtures.", datasetUrl: "https://huggingface.co/datasets/MixEval/MixEval", datasetKeys: ["id", "problem_type", "context", "prompt", "target", "benchmark_name", "options"] } },
  { id: "mt-bench", name: "MT-Bench", meta: { category: "General", tags: ["conversation & chatbots"], description: "Multi-turn questions: an open-ended question and a follow-up question.", datasetUrl: "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments", datasetKeys: ["question_id", "model_a", "model_b", "winner", "judge", "conversation_a", "conversation_b", "turn"] } },
  { id: "wildbench", name: "Wildbench", meta: { category: "General", tags: ["conversation & chatbots"], description: "An automated evaluation framework designed to benchmark LLMs on real-world user queries. It consists of 1,024 tasks selected from over one million human-chatbot conversation logs.", datasetUrl: "https://huggingface.co/datasets/allenai/WildBench", datasetKeys: ["id", "session_id", "conversation_input", "references", "length", "checklist", "intent", "primary_tag", "secondary_tags"] } },
  { id: "lab-bench-language-agent-biology-benchmark", name: "LAB-Bench", meta: { category: "General", tags: ["domain-specific"], description: "An evaluation dataset for AI systems intended to benchmark capabilities foundational to scientific research in biology.", datasetUrl: "https://huggingface.co/datasets/futurehouse/lab-bench", datasetKeys: ["id", "question", "ideal", "distractors", "canary", "source", "subtask"] } },
  { id: "cupcase", name: "CUPCase", meta: { category: "General", tags: ["domain-specific"], description: "CUPCase is based on 3,563 real-world clinical case reports formulated into diagnoses in open-ended textual format and as multiple-choice options with distractors.", datasetUrl: "https://huggingface.co/datasets/ofir408/CupCase", datasetKeys: ["clean_case_presentation", "correct_diagnosis", "distractor1", "distractor2", "distractor3"] } },
  { id: "medconceptsqa", name: "MedConceptsQA", meta: { category: "General", tags: ["domain-specific"], description: "MedConceptsQA measures the ability of models to interpret and distinguish between medical codes for diagnoses, procedures, and drugs.", datasetUrl: "https://huggingface.co/datasets/ofir408/MedConceptsQA", datasetKeys: ["question_id", "answer", "answer_id", "option1", "option2", "option3", "option4", "question", "vocab", "level"] } },
  { id: "eq-bench", name: "EQ-Bench", meta: { category: "General", tags: ["empathy"], description: "Assesses the ability of LLMs to understand complex emotions and social interactions by asking them to predict the intensity of emotional states of characters in a dialogue.", datasetUrl: "https://huggingface.co/datasets/pbevan11/EQ-Bench", datasetKeys: ["prompt", "reference_answer", "reference_answer_fullscale"] } },
  { id: "wice", name: "WiCE", meta: { category: "General", tags: ["information retrieval & RAG"], description: "Textual entailment dataset built on natural claim and evidence pairs extracted from Wikipedia.", datasetUrl: "https://huggingface.co/datasets/tasksource/wice", datasetKeys: ["label", "supporting_sentences", "claim", "evidence", "meta"] } },
  { id: "contextualbench", name: "ContextualBench", meta: { category: "General", tags: ["information retrieval & RAG"], description: "A compilation of 7 popular contextual question answering benchmarks to evaluate LLMs in RAG application.", datasetUrl: "https://huggingface.co/datasets/Salesforce/ContextualBench", datasetKeys: ["_id", "type", "question", "context", "supporting_facts", "evidences", "answer"] } },
  { id: "nolima", name: "NoLiMa", meta: { category: "General", tags: ["information retrieval & RAG"], description: "Extended NIAH, where questions and needles have minimal lexical overlap, requiring models to infer latent associations to locate the needle within the haystack.", datasetUrl: "https://huggingface.co/datasets/amodaresi/NoLiMa", datasetKeys: ["text"] } },
  { id: "wixqa", name: "WixQA", meta: { category: "General", tags: ["information retrieval & RAG"], description: "A benchmark suite featuring QA datasets grounded in the released knowledge base corpus, enabling holistic evaluation of retrieval and generation components.", datasetUrl: "https://huggingface.co/datasets/Wix/WixQA", datasetKeys: ["question", "answer", "article_ids"] } },
  { id: "frames-factuality-retrieval-and-reasoning-measurement-set", name: "FRAMES", meta: { category: "Safety", tags: ["information retrieval & RAG", "language & reasoning", "safety"], description: "Tests the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.", datasetUrl: "https://huggingface.co/datasets/google/frames-benchmark", datasetKeys: ["Prompt", "Answer", "reasoning_types", "wiki_links"] } },
  { id: "ragtruth", name: "RAGTruth", meta: { category: "Safety", tags: ["information retrieval & RAG", "safety"], description: "A corpus tailored for analyzing word-level hallucinations within the standard RAG frameworks for LLM applications.", datasetUrl: "https://huggingface.co/datasets/wandb/RAGTruth-processed", datasetKeys: ["id", "query", "context", "output", "task_type", "quality", "model", "temperature", "hallucination_labels"] } },
  { id: "infobench", name: "Infobench", meta: { category: "General", tags: ["instruction-following"], description: "Evaluating Large Language Models' (LLMs) ability to follow instructions by breaking complex instructions into simpler criteria.", datasetUrl: "https://huggingface.co/datasets/kqsong/InFoBench", datasetKeys: ["id", "input", "category", "instruction", "decomposed_questions", "subset", "question_label"] } },
  { id: "megascience", name: "MegaScience", meta: { category: "Knowledge", tags: ["knowledge", "language & reasoning"], description: "A large-scale mixture of high-quality open-source datasets totaling 1.25 million instances.", datasetUrl: "https://huggingface.co/datasets/MegaScience/MegaScience", datasetKeys: ["question", "answer", "subject", "reference_answer", "source"] } },
  { id: "mmlu", name: "MMLU", meta: { category: "Knowledge", tags: ["knowledge", "language & reasoning"], description: "Multi-choice tasks across 57 subjects, high school to expert level.", datasetUrl: "https://huggingface.co/datasets/cais/mmlu", datasetKeys: ["question", "subject", "choices", "answer"] } },
  { id: "arc", name: "ARC", meta: { category: "Knowledge", tags: ["knowledge", "language & reasoning"], description: "Grade-school level, multiple-choice science questions.", datasetUrl: "https://huggingface.co/datasets/allenai/ai2_arc", datasetKeys: ["id", "question", "choices", "answerKey"] } },
  { id: "mmlu-pro", name: "MMLU Pro", meta: { category: "Knowledge", tags: ["knowledge", "language & reasoning"], description: "An enhanced dataset designed to extend the MMLU benchmark. More challenging questions, the choice set of ten options.", datasetUrl: "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro", datasetKeys: ["question_id", "question", "options", "answer", "answer_index", "cot_content", "category", "src"] } },
  { id: "truthfulqa", name: "TruthfulQA", meta: { category: "Safety", tags: ["knowledge", "language & reasoning", "safety"], description: "Evaluates how well models generate truthful responses.", datasetUrl: "https://huggingface.co/datasets/truthfulqa/truthful_qa", datasetKeys: ["type", "category", "question", "best_answer", "correct_answers", "incorrect_answers", "source"] } },
  { id: "superglue", name: "SuperGLUE", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Improved and more challenging version of GLUE benchmark.", datasetUrl: "https://huggingface.co/datasets/aps/super_glue", datasetKeys: ["question", "passage", "idx", "label"] } },
  { id: "drop-discrete-reasoning-over-paragraphs", name: "DROP", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Tasks to resolve references in a question and perform discrete operations over them (such as addition, counting, or sorting).", datasetUrl: "https://huggingface.co/datasets/ucinlp/drop", datasetKeys: ["section_id", "query_id", "passage", "question", "answers_spans"] } },
  { id: "winogrande", name: "Winogrande", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Fill-in-a-blank tasks resolving ambiguities in pronoun references with binary options.", datasetUrl: "https://huggingface.co/datasets/allenai/winogrande", datasetKeys: ["sentence", "option1", "option2", "answer"] } },
  { id: "anli", name: "ANLI", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Large-scale NLI benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure.", datasetUrl: "https://huggingface.co/datasets/facebook/anli", datasetKeys: ["uid", "premise", "hypothesis", "label", "reason"] } },
  { id: "multinli-multi-genre-natural-language-inference", name: "MultiNLI", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A crowdsourced collection of sentence pairs annotated with textual entailment information.", datasetUrl: "https://huggingface.co/datasets/nyu-mll/multi_nli", datasetKeys: ["promptID", "pairID", "premise", "hypothesis", "genre", "label"] } },
  { id: "squad-stanford-question-answering-dataset", name: "SQuAD", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A reading comprehension dataset consisting of 100,000 questions posed by crowdworkers on a set of Wikipedia articles.", datasetUrl: "https://huggingface.co/datasets/rajpurkar/squad", datasetKeys: ["id", "title", "context", "question", "answers"] } },
  { id: "openbookqa", name: "OpenBookQA", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Question answering dataset, modeled after open book exams.", datasetUrl: "https://huggingface.co/datasets/allenai/openbookqa", datasetKeys: ["id", "question_stem", "choices", "answerKey"] } },
  { id: "squad2-0", name: "SQuAD2.0", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers.", datasetUrl: "https://huggingface.co/datasets/bayes-group-diffusion/squad-2.0", datasetKeys: ["target", "source"] } },
  { id: "ms-marco", name: "MS MARCO", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Questions sampled from Bing's search query logs and passages from web documents.", datasetUrl: "https://huggingface.co/datasets/microsoft/ms_marco", datasetKeys: ["answers", "passages", "query", "query_id", "query_type", "wellFormedAnswers"] } },
  { id: "sciq", name: "SciQ", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Multiple choice science exam questions.", datasetUrl: "https://huggingface.co/datasets/allenai/sciq", datasetKeys: ["question", "distractor3", "distractor1", "distractor2", "correct_answer", "support"] } },
  { id: "triviaqa", name: "TriviaQA", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A large-scale question-answering dataset.", datasetUrl: "https://huggingface.co/datasets/mandarjoshi/trivia_qa", datasetKeys: ["question", "question_id", "question_source", "entity_pages", "search_results", "answer"] } },
  { id: "lambada-language-modelling-broadened-to-account-for-discourse-aspects", name: "LAMBADA", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A set of passages composed of a context and a target sentence. The task is to guess the last word of the target sentence.", datasetUrl: "https://huggingface.co/datasets/cimec/lambada", datasetKeys: ["text", "domain"] } },
  { id: "glue-general-language-understanding-evaluation", name: "GLUE", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Tool for evaluating and analyzing the performance of models on NLU tasks. Was quickly outperformed by LLMs and replaced by SuperGLUE.", datasetUrl: "https://huggingface.co/datasets/nyu-mll/glue", datasetKeys: ["premise", "hypothesis", "label", "idx"] } },
  { id: "graphwalks", name: "Graphwalks", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A dataset for evaluating multi-hop long-context reasoning. In Graphwalks, the model is given a graph represented by its edge list and asked to perform an operation.", datasetUrl: "https://huggingface.co/datasets/openai/graphwalks", datasetKeys: ["prompt", "answer_nodes", "prompt_chars", "problem_type"] } },
  { id: "planbench", name: "PlanBench", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A benchmark designed to evaluate the ability of LLMs to generate plans of action and reason about change.", datasetUrl: "https://huggingface.co/datasets/tasksource/planbench", datasetKeys: ["task", "prompt_type", "domain", "instance_id", "example_instance_ids", "query", "ground_truth_plan"] } },
  { id: "longbench", name: "LongBench", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Assesses the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks.", datasetUrl: "https://huggingface.co/datasets/THUDM/LongBench-v2", datasetKeys: ["_id", "domain", "sub_domain", "difficulty", "length", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context"] } },
  { id: "zebralogic", name: "Zebralogic", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "Evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs).", datasetUrl: "https://huggingface.co/datasets/WildEval/ZebraLogic", datasetKeys: ["id", "size", "puzzle", "solution", "created_at"] } },
  { id: "gpqa", name: "GPQA", meta: { category: "Reasoning", tags: ["language & reasoning"], description: "A set of multiple-choice questions written by domain experts in biology, physics, and chemistry.", datasetUrl: "https://huggingface.co/datasets/Idavidrein/gpqa", datasetKeys: ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Explanation", "Subdomain", "High-level domain"] } },
  { id: "biggen-bench", name: "BiGGen-Bench", meta: { category: "Safety", tags: ["language & reasoning", "agents & tools use", "safety", "instruction-following"], description: "Evaluates nine distinct capabilities of LMs, including instruction following, reasoning, tool usage, and safety.", datasetUrl: "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench", datasetKeys: ["id", "capability", "task", "instance_idx", "system_prompt", "input", "reference_answer", "score_rubric"] } },
  { id: "include", name: "Include", meta: { category: "Reasoning", tags: ["language & reasoning", "multilingual"], description: "An evaluation suite to measure the capabilities of multilingual LLMs in a variety of regional contexts across 44 written languages.", datasetUrl: "https://huggingface.co/datasets/CohereLabs/include-base-44", datasetKeys: ["language", "country", "domain", "subject", "regional_feature", "level", "question", "option_a", "option_b", "option_c", "option_d", "answer"] } },
  { id: "multinrc", name: "MultiNRC", meta: { category: "Reasoning", tags: ["language & reasoning", "multilingual"], description: "Assesses LLMs on reasoning questions written by native speakers in French, Spanish, and Chinese.", datasetUrl: "https://huggingface.co/datasets/ScaleAI/MultiNRC", datasetKeys: ["task_id", "i18n_prompt", "i18n_gtfa", "english_prompt", "english_gtfa", "language", "category"] } },
  { id: "judgebench", name: "JudgeBench", meta: { category: "General", tags: ["llm judge evaluation"], description: "A benchmark for evaluating LLM-based judges on challenging response pairs spanning knowledge, reasoning, math, and coding.", datasetUrl: "https://huggingface.co/datasets/ScalerLab/JudgeBench", datasetKeys: ["pair_id", "original_id", "source", "question", "response_model", "response_A", "response_B", "label"] } },
  { id: "aime", name: "AIME", meta: { category: "Math", tags: ["math"], description: "This dataset contains problems from the American Invitational Mathematics Examination (AIME) 2024.", datasetUrl: "https://huggingface.co/datasets/Maxwell-Jia/AIME_2024", datasetKeys: ["ID", "Problem", "Solution", "Answer"] } },
  { id: "templategsm", name: "TemplateGSM", meta: { category: "Math", tags: ["math"], description: "A dataset comprising over 7 million synthetically generated grade school math problems, each accompanied by code-based and natural language solutions.", datasetUrl: "https://huggingface.co/datasets/math-ai/TemplateGSM", datasetKeys: ["problem", "solution_code", "result", "solution_wocode", "source", "template_id", "problem_id"] } },
  { id: "we-math", name: "We-Math", meta: { category: "Math", tags: ["math"], description: "A benchmark that evaluates the problem-solving principles in knowledge acquisition and generalization for math tasks.", datasetUrl: "https://huggingface.co/datasets/We-Math/We-Math", datasetKeys: ["ID", "split", "knowledge concept", "question", "option", "answer", "image_path", "key", "question number", "knowledge concept description"] } },
  { id: "gsmhard", name: "GSMHard", meta: { category: "Math", tags: ["math"], description: "The harder version of the GSM8K math reasoning dataset. Numbers in the questions of GSM8K are replaced with larger numbers that are less common.", datasetUrl: "https://huggingface.co/datasets/reasoning-machines/gsm-hard", datasetKeys: ["input", "code", "target"] } },
  { id: "aqua-rat", name: "AQUA-RAT", meta: { category: "Math", tags: ["math"], description: "An algebraic word problem dataset, with multiple choice questions annotated with rationales.", datasetUrl: "https://huggingface.co/datasets/deepmind/aqua_rat", datasetKeys: ["question", "options", "rationale", "correct"] } },
  { id: "theoremqa", name: "TheoremQA", meta: { category: "Math", tags: ["math"], description: "Theorem-driven QA dataset that evaluates LLMs capabilities to apply theorems to solve science problems. Contains 800 questions covering 350 theorems from math, physics, EE&CS, and finance.", datasetUrl: "https://huggingface.co/datasets/TIGER-Lab/TheoremQA", datasetKeys: ["Question", "Answer", "Answer_type", "Picture"] } },
  { id: "simpleqa", name: "SimpleQA", meta: { category: "Safety", tags: ["safety"], description: "Measures the ability for language models to answer short, fact-seeking questions to reduce hallucinations.", datasetUrl: "https://huggingface.co/datasets/basicv8vc/SimpleQA", datasetKeys: ["metadata", "problem", "answer"] } },
  { id: "air-bench", name: "AIR-Bench", meta: { category: "Safety", tags: ["safety"], description: "AI safety benchmark aligned with emerging regulations. Considers operational, content safety, legal and societal risks.", datasetUrl: "https://huggingface.co/datasets/stanford-crfm/air-bench-2024", datasetKeys: ["cate-idx", "l2-name", "l3-name", "l4-name", "prompt"] } },
  { id: "or-bench", name: "OR-Bench", meta: { category: "Safety", tags: ["safety"], description: "80,000 benign prompts likely rejected by LLMs across 10 common rejection categories.", datasetUrl: "https://huggingface.co/datasets/bench-llm/or-bench", datasetKeys: ["prompt", "category"] } },
  { id: "agentharm", name: "AgentHarm", meta: { category: "Safety", tags: ["safety"], description: "Explicitly malicious agent tasks, including fraud, cybercrime, and harassment.", datasetUrl: "https://huggingface.co/datasets/ai-safety-institute/AgentHarm", datasetKeys: ["id", "id_original", "detailed_prompt", "hint_included", "name", "category", "prompt", "target_functions", "grading_function"] } },
  { id: "backdoorllm", name: "BackdoorLLM", meta: { category: "Safety", tags: ["safety"], description: "A benchmark for backdoor attacks in text generation.", datasetUrl: "https://huggingface.co/datasets/BackdoorLLM/Backdoored_Dataset", datasetKeys: ["instruction", "input", "output"] } },
  { id: "safetybench", name: "SafetyBench", meta: { category: "Safety", tags: ["safety"], description: "Multiple-choice questions concerning offensive content, bias, illegal activities, and mental health.", datasetUrl: "https://huggingface.co/datasets/thu-coai/SafetyBench", datasetKeys: ["question", "options", "category", "id"] } },
  { id: "harmfulqa", name: "HarmfulQA", meta: { category: "Safety", tags: ["safety"], description: "Harmful questions covering 10 topics and ~10 subtopics each.", datasetUrl: "https://huggingface.co/datasets/declare-lab/HarmfulQA", datasetKeys: ["id", "topic", "subtopic", "question", "blue_conversations", "red_conversations"] } },
  { id: "beavertails", name: "BeaverTails", meta: { category: "Safety", tags: ["safety"], description: "A set of prompts sampled from AnthropicRedTeam that cover 14 harm categories.", datasetUrl: "https://huggingface.co/datasets/PKU-Alignment/BeaverTails", datasetKeys: ["prompt", "response", "category", "is_safe"] } },
  { id: "donotanswer", name: "DoNotAnswer", meta: { category: "Safety", tags: ["safety"], description: "The dataset consists of prompts across 12 harm types to which responsible LLMs do not answer.", datasetUrl: "https://huggingface.co/datasets/LibrAI/do-not-answer", datasetKeys: ["id", "risk_area", "types_of_harm", "specific_harms", "question"] } },
  { id: "toxigen", name: "ToxiGen", meta: { category: "Safety", tags: ["safety"], description: "A set of toxic and benign statements about minority groups.", datasetUrl: "https://huggingface.co/datasets/toxigen/toxigen-data", datasetKeys: ["text", "target_group", "factual?", "ingroup_effect", "lewd", "framing", "predicted_group", "stereotyping", "intent", "toxicity_ai", "toxicity_human"] } },
  { id: "anthropicredteam", name: "AnthropicRedTeam", meta: { category: "Safety", tags: ["safety"], description: "Human-generated and annotated red teaming dialogues.", datasetUrl: "https://huggingface.co/datasets/Anthropic/hh-rlhf", datasetKeys: ["chosen", "rejected"] } },
  { id: "realtoxicityprompt", name: "RealToxicityPrompt", meta: { category: "Safety", tags: ["safety"], description: "A dataset of 100K naturally occurring, sentence-level prompts derived from a large corpus of English web text, paired with toxicity scores.", datasetUrl: "https://huggingface.co/datasets/allenai/real-toxicity-prompts", datasetKeys: ["filename", "begin", "end", "challenging", "prompt", "continuation"] } },
  { id: "stereoset", name: "StereoSet", meta: { category: "Safety", tags: ["safety", "bias & ethics"], description: "A large-scale natural dataset in English to measure stereotypical biases in four domains: gender, profession, race, and religion.", datasetUrl: "https://huggingface.co/datasets/McGill-NLP/stereoset", datasetKeys: ["id", "target", "bias_type", "context", "sentences"] } },
  { id: "winogender", name: "WinoGender", meta: { category: "Safety", tags: ["safety", "bias & ethics"], description: "Pairs of sentences that differ only by the gender of one pronoun in the sentence, designed to test for the presence of gender bias.", datasetUrl: "https://huggingface.co/datasets/oskarvanderwal/winogender", datasetKeys: ["sentid", "sentence", "pronoun", "occupation", "participant", "gender", "target", "label"] } },
];

const CATEGORIES: Array<{ id: BenchCategory | "All"; label: string }> = [
  { id: "All", label: "All" },
  { id: "Math", label: "Math" },
  { id: "Reasoning", label: "Reasoning" },
  { id: "Knowledge", label: "Knowledge" },
  { id: "Safety", label: "Safety" },
  { id: "Coding", label: "Coding" },
  { id: "General", label: "General" },
];

function getBenchIcon(category: BenchCategory) {
  switch (category) {
    case "Math":
      return { Icon: BookOpen, bg: "bg-emerald-50", fg: "text-emerald-600" };
    case "Reasoning":
      return { Icon: Brain, bg: "bg-indigo-50", fg: "text-indigo-600" };
    case "Knowledge":
      return { Icon: GraduationCap, bg: "bg-sky-50", fg: "text-sky-600" };
    case "Safety":
      return { Icon: ShieldCheck, bg: "bg-amber-50", fg: "text-amber-700" };
    case "Coding":
      return { Icon: Code2, bg: "bg-violet-50", fg: "text-violet-600" };
    default:
      return { Icon: Tag, bg: "bg-slate-50", fg: "text-slate-600" };
  }
}

function loadGalleryBenches(): BenchItem[] {
  try {
    const raw = localStorage.getItem("oneEval.gallery.benches");
    if (!raw) return DEFAULT_BENCHES;
    const parsed = JSON.parse(raw) as BenchItem[];
    if (!Array.isArray(parsed) || parsed.length === 0) return DEFAULT_BENCHES;
    return parsed;
  } catch {
    return DEFAULT_BENCHES;
  }
}

function saveGalleryBenches(items: BenchItem[]) {
  localStorage.setItem("oneEval.gallery.benches", JSON.stringify(items));
}

export const Gallery = () => {
  const navigate = useNavigate();
  const [benches, setBenches] = useState<BenchItem[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]["id"]>("All");
  const [activeBenchId, setActiveBenchId] = useState<string | null>(null);

  useEffect(() => {
    setBenches(loadGalleryBenches());
  }, []);

  const activeBench = useMemo(() => benches.find((b) => b.id === activeBenchId) ?? null, [benches, activeBenchId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return benches
      .filter((b) => (category === "All" ? true : b.meta.category === category))
      .filter((b) => {
        if (!q) return true;
        const hay = `${b.name} ${b.meta.description} ${b.meta.tags.join(" ")} ${b.meta.category}`.toLowerCase();
        return hay.includes(q);
      });
  }, [benches, query, category]);

  const handleUseBench = (benchId: string) => {
    navigate("/eval", { state: { preSelectedBench: benchId } });
  };

  const handleUpdateBench = (updated: BenchItem) => {
    setBenches((prev) => {
      const next = prev.map((b) => (b.id === updated.id ? updated : b));
      saveGalleryBenches(next);
      return next;
    });
  };

  const handleReset = () => {
    setBenches(DEFAULT_BENCHES);
    saveGalleryBenches(DEFAULT_BENCHES);
  };

  return (
    <div className="p-12 max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">Benchmark Gallery</h1>
          <p className="text-slate-600 text-lg">Search, filter, and configure your curated benchmarks.</p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" className="border-slate-200" onClick={handleReset}>
            Reset to Defaults
          </Button>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-3 md:items-center md:justify-between">
          <div className="relative w-full md:max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search benches, tags, categories..."
              className="pl-9 bg-white border-slate-200"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {CATEGORIES.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setCategory(c.id)}
                    className={cn(
                      "px-3 py-1.5 text-sm rounded-full border transition-colors",
                      c.id === category
                    ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white border-transparent shadow-sm shadow-blue-600/20"
                    : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                    )}
                  >
                    {c.label}
                  </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filtered.map((bench, idx) => {
          const { Icon, bg, fg } = getBenchIcon(bench.meta.category);
          return (
            <motion.div key={bench.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.04 }}>
              <Card className="h-full flex flex-col border-slate-200 hover:shadow-lg transition-shadow duration-300">
                <CardHeader>
                  <div className="flex justify-between items-start gap-4">
                    <div className="flex items-center gap-3">
                      <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center", bg)}>
                        <Icon className={cn("w-6 h-6", fg)} />
                      </div>
                      <div>
                        <CardTitle className="text-xl text-slate-900">{bench.name}</CardTitle>
                        <div className="text-xs text-slate-500 mt-0.5">{bench.meta.category}</div>
                      </div>
                    </div>
                    {bench.meta.sampleCount != null && (
                      <div className="px-2 py-1 bg-slate-50 border border-slate-200 rounded text-xs font-mono text-slate-500">
                        {bench.meta.sampleCount.toLocaleString()} samples
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap gap-2 mt-4">
                    {bench.meta.tags.slice(0, 4).map((tag) => (
                      <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-50 text-slate-600 border border-slate-200">
                        {tag}
                      </span>
                    ))}
                    {bench.meta.tags.length > 4 && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-slate-50 text-slate-500 border border-slate-200">
                        +{bench.meta.tags.length - 4}
                      </span>
                    )}
                  </div>
                </CardHeader>

                <CardContent className="flex-1">
                  <CardDescription className="text-sm text-slate-600 line-clamp-3">{bench.meta.description}</CardDescription>
                </CardContent>

                <CardFooter className="pt-4 border-t border-slate-100 bg-slate-50/30 flex gap-2">
                  <Button
                    className="flex-1 text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                    onClick={() => handleUseBench(bench.id)}
                  >
                    Use
                  </Button>
                  {bench.meta.datasetUrl && (
                    <Button
                      variant="outline"
                      className="border-slate-200"
                      onClick={() => window.open(bench.meta.datasetUrl, "_blank")}
                    >
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  )}
                  <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(bench.id)}>
                    <SlidersHorizontal className="w-4 h-4" />
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <AnimatePresence>
        {activeBench && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50"
          >
            <div className="absolute inset-0 bg-black/20" onClick={() => setActiveBenchId(null)} />
            <motion.div
              initial={{ x: 420 }}
              animate={{ x: 0 }}
              exit={{ x: 420 }}
              transition={{ type: "spring", stiffness: 280, damping: 30 }}
              className="absolute right-0 top-0 bottom-0 w-full max-w-md bg-white border-l border-slate-200 shadow-2xl p-6 overflow-y-auto"
              role="dialog"
              aria-modal="true"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wider">Configure Bench</div>
                  <div className="text-2xl font-bold text-slate-900 mt-1">{activeBench.name}</div>
                </div>
                <button
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                  onClick={() => setActiveBenchId(null)}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="mt-6 space-y-5">
                <div className="space-y-2">
                  <Label>Display Name</Label>
                  <Input
                    value={activeBench.name}
                    onChange={(e) => handleUpdateBench({ ...activeBench, name: e.target.value })}
                    className="border-slate-200"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Description</Label>
                  <textarea
                    value={activeBench.meta.description}
                    onChange={(e) =>
                      handleUpdateBench({ ...activeBench, meta: { ...activeBench.meta, description: e.target.value } })
                    }
                    className="w-full min-h-[120px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Category</Label>
                  <select
                    value={activeBench.meta.category}
                    onChange={(e) =>
                      handleUpdateBench({
                        ...activeBench,
                        meta: { ...activeBench.meta, category: e.target.value as BenchCategory },
                      })
                    }
                    className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    {CATEGORIES.filter((c) => c.id !== "All").map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>Tags (comma-separated)</Label>
                  <Input
                    value={activeBench.meta.tags.join(", ")}
                    onChange={(e) =>
                      handleUpdateBench({
                        ...activeBench,
                        meta: {
                          ...activeBench.meta,
                          tags: e.target.value
                            .split(",")
                            .map((t) => t.trim())
                            .filter(Boolean),
                        },
                      })
                    }
                    className="border-slate-200"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Sample Count</Label>
                  <Input
                    type="number"
                    value={activeBench.meta.sampleCount ?? ""}
                    onChange={(e) =>
                      handleUpdateBench({
                        ...activeBench,
                        meta: {
                          ...activeBench.meta,
                          sampleCount: e.target.value ? Number(e.target.value) : undefined,
                        },
                      })
                    }
                    className="border-slate-200"
                  />
                </div>

                {activeBench.meta.datasetUrl && (
                  <div className="space-y-2">
                    <Label>Dataset URL</Label>
                    <div className="flex gap-2">
                      <Input
                        value={activeBench.meta.datasetUrl}
                        readOnly
                        className="border-slate-200 bg-slate-50 text-slate-600 flex-1"
                      />
                      <Button
                        variant="outline"
                        className="border-slate-200 shrink-0"
                        onClick={() => window.open(activeBench.meta.datasetUrl, "_blank")}
                      >
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                )}

                {activeBench.meta.datasetKeys && activeBench.meta.datasetKeys.length > 0 && (
                  <div className="space-y-2">
                    <Label>Dataset Keys</Label>
                    <div className="flex flex-wrap gap-1.5 p-3 rounded-md border border-slate-200 bg-slate-50">
                      {activeBench.meta.datasetKeys.map((key) => (
                        <span key={key} className="text-xs px-2 py-1 rounded bg-white border border-slate-200 text-slate-600 font-mono">
                          {key}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-8 flex gap-2">
                <Button className="flex-1 bg-slate-900 text-white hover:bg-slate-800" onClick={() => handleUseBench(activeBench.id)}>
                  Use This Bench
                </Button>
                <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(null)}>
                  Close
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
