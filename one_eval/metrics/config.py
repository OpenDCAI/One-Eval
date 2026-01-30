# one_eval/metrics/config.py
from typing import Dict, List, Any

# Dataset name to Template mapping
# Key: Dataset name (lowercase)
# Value: List of metric configurations (or just template name, but here we store full config list for now as per original code structure)
# Ideally, this should be in a YAML file, but for now we keep it as a Python dictionary.

DATASET_METRIC_MAP_CONFIG: Dict[str, str] = {
    # --- Group A: Numerical ---
    "gsm8k": "numerical",
    "svamp": "numerical",
    "calc-ape210k": "numerical",
    "calc-mawps": "numerical",
    "calc-asdiv_a": "numerical",

    # --- Group B: Symbolic / Math ---
    "math": "symbolic",
    "hendrycks_math": "symbolic",
    "math-500": "symbolic",
    "competition_math": "symbolic",

    # --- Group C: Choice ---
    "aqua-rat": "choice",
    "mmlu": "choice",
    "agieval-gaokao-mathqa": "choice",
    "math-qa": "choice",

    # --- Group D: Code ---
    "humaneval": "code",
    "mbpp": "code",

    # --- Group E: Generation ---
    "general_qa": "generation_rouge",
    "summscreen": "generation_rouge",
    "lcsts": "generation_rouge",
    "iwslt2017": "generation_bleu",
    "flores": "generation_bleu",

    # --- Group F: Extractive QA ---
    "squad20": "qa_extractive",
    "tydiqa": "qa_extractive",
    "nq": "qa_extractive",
    "nq_cn": "qa_extractive",
    "qasper": "qa_extractive",

    # --- Group G: Long Context ---
    "longbench": "long_context_qa",
    "lveval": "long_context_qa",

    # --- Group H: Retrieval / Count ---
    "needlebench": "retrieval",
    "needlebench_v2": "retrieval",
    "longbench_retrieval": "retrieval",
    "longbench_count": "count",
    "longbench_codesim": "code_sim",

    # --- Group I: Judge ---
    "subjective": "llm_judge",
    "arena": "llm_judge",
    "mtbench": "llm_judge",
    "promptbench": "llm_judge",
    "teval": "llm_judge",
    "omni_math_judge": "llm_judge",
    
    # --- Group J: Pairwise ---
    "leval": "win_rate",
    
    # --- Group K: Other ---
    "llm_compression": "auc_roc",
}