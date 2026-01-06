# one_eval/utils/extractor.py
import re
import math
from typing import Any, Optional


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if math.isfinite(float(x)):
            return float(x)
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = float(s)
            if math.isfinite(v):
                return v
            return None
        except Exception:
            return None
    return None


def extract_first_number(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    return safe_float(m.group(0))


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_choice(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().upper()
    if not s:
        return None
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"^\(?\s*([A-Z])\s*\)?", s)
    if m:
        return m.group(1)
    return None

