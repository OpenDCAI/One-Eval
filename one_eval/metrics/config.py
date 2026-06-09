# one_eval/metrics/config.py
from typing import Dict, List

# --- Reusable Metric Suites (Names Only) ---

# 仅引用确定性内核里真实注册的指标名。LLM-judge 类数据集没有确定性指标，
# 由调用方(外部 agent)接管打分；此处给出可跑的确定性兜底(文本相似/覆盖)。
_SUITE_NUMERICAL = ["numerical_match", "extraction_rate"]
_SUITE_SYMBOLIC = ["math_verify", "extraction_rate"]
_SUITE_CHOICE = ["choice_accuracy", "extraction_rate"]
_SUITE_CODE = ["soft_code_execution"]
_SUITE_GEN_BLEU = ["bleu", "chrf"]
_SUITE_GEN_ROUGE = ["rouge_l"]
_SUITE_QA_EXTRACTIVE = ["exact_match", "token_f1", "extraction_rate"]
_SUITE_QA_LONG = ["token_f1", "exact_match"]
_SUITE_RETRIEVAL = ["token_f1", "exact_match"]
_SUITE_JUDGE = ["rouge_l", "token_f1"]
# pairwise(better/rejected)与纯文本打分一样,没有确定性指标:
# runner 只把 better 当 ref 传进来,rejected 取不到,真偏好判断须由调用方(agent)裁判。
# 这里只给“与更优答案的词面接近度”作可跑的弱代理(诊断用,别当主分)。
_SUITE_WIN_RATE = ["token_f1", "rouge_l"]
_SUITE_CLASSIFY = ["choice_accuracy", "extraction_rate"]

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
    "llm_compression": _SUITE_CLASSIFY,
}