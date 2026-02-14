# one_eval/metrics/config.py
from typing import Dict, List

# --- Reusable Metric Suites (Names Only) ---

<<<<<<< HEAD
_SUITE_NUMERICAL = ["numerical_match", "strict_match", "extraction_rate"]
_SUITE_SYMBOLIC = ["symbolic_match", "strict_match", "extraction_rate"]
=======
_SUITE_NUMERICAL = ["numerical_match", "extraction_rate"]
_SUITE_SYMBOLIC = ["math_verify", "extraction_rate"]
>>>>>>> hyh_0214
_SUITE_CHOICE = ["choice_accuracy", "extraction_rate"]
_SUITE_CODE = ["pass_at_k", "code_similarity", "soft_code_execution"]
_SUITE_GEN_BLEU = ["bleu", "chrf"]
_SUITE_GEN_ROUGE = ["rouge_l"]
_SUITE_QA_EXTRACTIVE = ["exact_match", "token_f1", "extraction_rate"]
_SUITE_QA_LONG = ["token_f1", "exact_match"]
_SUITE_RETRIEVAL = ["retrieval_recall", "retrieval_ndcg"]
_SUITE_JUDGE = ["llm_judge_score"]
_SUITE_WIN_RATE = ["win_rate_against_baseline"]
_SUITE_AUC_ROC = ["auc_roc", "accuracy"]

# --- Dataset Metric Configuration ---
# Direct mapping: Dataset Name -> List of Metric Names
# Priority is inferred at runtime (1st=primary, others=secondary/diagnostic) or decided by LLM.

DATASET_METRICS: Dict[str, List[str]] = {
    # --- Numerical ---
    "gsm8k": _SUITE_NUMERICAL,
    "svamp": _SUITE_NUMERICAL,
    "calc-ape210k": _SUITE_NUMERICAL,
    "calc-mawps": _SUITE_NUMERICAL,
    "calc-asdiv_a": _SUITE_NUMERICAL,

    # --- Symbolic / Math ---
    "math": _SUITE_SYMBOLIC,
    "hendrycks_math": _SUITE_SYMBOLIC,
    "math-500": _SUITE_SYMBOLIC,
    "competition_math": _SUITE_SYMBOLIC,

    # --- Choice ---
    "aqua-rat": _SUITE_CHOICE,
    "mmlu": _SUITE_CHOICE,
    "agieval-gaokao-mathqa": _SUITE_CHOICE,
    "math-qa": _SUITE_CHOICE,

    # --- Code ---
    "humaneval": _SUITE_CODE,
    "mbpp": _SUITE_CODE,

    # --- Generation ---
    "general_qa": _SUITE_GEN_ROUGE,
    "summscreen": _SUITE_GEN_ROUGE,
    "lcsts": _SUITE_GEN_ROUGE,
    "iwslt2017": _SUITE_GEN_BLEU,
    "flores": _SUITE_GEN_BLEU,

    # --- Extractive QA ---
    "squad20": _SUITE_QA_EXTRACTIVE,
    "tydiqa": _SUITE_QA_EXTRACTIVE,
    "nq": _SUITE_QA_EXTRACTIVE,
    "nq_cn": _SUITE_QA_EXTRACTIVE,
    "qasper": _SUITE_QA_EXTRACTIVE,

    # --- Long Context ---
    "longbench": _SUITE_QA_LONG,
    "lveval": _SUITE_QA_LONG,

    # --- Retrieval / Count ---
    "needlebench": _SUITE_RETRIEVAL,
    "needlebench_v2": _SUITE_RETRIEVAL,
    "longbench_retrieval": _SUITE_RETRIEVAL,
    # "longbench_count": ["count"],
    # "longbench_codesim": ["code_sim"],

    # --- Judge ---
    "subjective": _SUITE_JUDGE,
    "arena": _SUITE_JUDGE,
    "mtbench": _SUITE_JUDGE,
    "promptbench": _SUITE_JUDGE,
    "teval": _SUITE_JUDGE,
    "omni_math_judge": _SUITE_JUDGE,
    
    # --- Pairwise ---
    "leval": _SUITE_WIN_RATE,
    
    # --- Other ---
    "llm_compression": _SUITE_AUC_ROC,
}