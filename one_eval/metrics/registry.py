# one_eval/metrics/registry.py
from .lib.basic import compute_exact_match
from .lib.numerical import compute_numerical_match
from .lib.choice import compute_choice_accuracy
from .lib.diagnostic import compute_extraction_rate, compute_missing_answer_rate

METRIC_FUNCTIONS = {
    "exact_match": compute_exact_match,
    "strict_match": lambda p, r, **k: compute_exact_match(p, r, strict=True, **k),
    "numerical_match": compute_numerical_match,
    "choice_accuracy": compute_choice_accuracy,
    "extraction_rate": compute_extraction_rate,
    "missing_answer_rate": compute_missing_answer_rate,
}

def get_metric_fn(name: str):
    return METRIC_FUNCTIONS.get(name)
