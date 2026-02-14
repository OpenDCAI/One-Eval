# one_eval/utils/extractor.py
import re
import math
import json
import unicodedata
from typing import Any, Optional, List, Set, Union, Dict
from word2number import w2n


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

    # --- Pre-processing for CoT (Chain-of-Thought) ---
    # 1. GSM8K style: ".... #### 42" -> "42"
    if "####" in s:
        parts = s.split("####")
        if len(parts) > 1:
            s = parts[-1] 
    
    # 2. MATH style: ".... \boxed{42} ..." -> "42"
    # Basic regex for \boxed{...}.
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", s)
    if boxed_matches:
        s = boxed_matches[-1]

    # Remove commas from numbers (e.g. 1,000 -> 1000)
    s = s.replace(",", "")
    
    # 1. Try to find fractions (e.g. "1/2", "3 / 4")
    # Priority: High (to avoid matching '1' from '1/2')
    fraction_pattern = r"([-+]?\d+)\s*/\s*(\d+)"
    m_frac = re.search(fraction_pattern, s)
    if m_frac:
        try:
            numerator = float(m_frac.group(1))
            denominator = float(m_frac.group(2))
            if denominator != 0:
                return numerator / denominator
        except:
            pass

    # 2. Try to find percentages (e.g. "50%", "33.3%")
    # Priority: High (to avoid matching '50' from '50%')
    percent_pattern = r"([-+]?\d+(?:\.\d+)?)\s*%"
    m_percent = re.search(percent_pattern, s)
    if m_percent:
        try:
            val = float(m_percent.group(1))
            return val / 100.0
        except:
            pass

    # 3. Standard float extraction
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    return safe_float(m.group(0))


def normalize_text(x: Any) -> str:
    """
    Standard text normalization for metrics:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Fix whitespace
    """
    if x is None:
        return ""
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = str(x)
    
    # --- CoT Extraction for Text Generation Metrics ---
    # If explicit answer marker exists, use it.
    if "####" in s:
        s = s.split("####")[-1]
    elif "\\boxed{" in s:
        boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", s)
        if boxed_matches:
            s = boxed_matches[-1]
    elif len(s) > 50:
        # 3. Common CoT patterns: "The answer is X", "Answer: X"
        # Improved regex: capture until punctuation or end of line, not just one word
        m = re.search(r"(?:answer|result)\s+is\s+:?\s*([^!.?\n]+)", s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_choice(text: Any) -> Optional[str]:
    if text is None:
        return None
        
    # 0. Handle integer input (0 -> 'A')
    if isinstance(text, int):
        if 0 <= text <= 25:
            return chr(65 + text) # 0->A, 1->B
        return None

    s = str(text)
    
    # 1. CoT: #### A
    if "####" in s:
        s = s.split("####")[-1]
    
    # 2. CoT: \boxed{A}
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", s)
    if boxed_matches:
        s = boxed_matches[-1]

    s = s.strip().upper()
    if not s:
        return None
        
    # 3. Explicit "Answer: A" or "Answer is A" pattern
    # Matches "Answer:" followed by optional text then a single letter A-Z
    m_ans = re.search(r"(?:answer|option)\s*[:is]\s*\(?([A-Z])\)?(?:\W|$)", s, re.IGNORECASE)
    if m_ans:
        return m_ans.group(1).upper()

    # 4. Standard patterns (Fallbacks)
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"^\(?\s*([A-Z])\s*\)?", s)
    if m:
        return m.group(1)
    return None


def extract_multi_choice(text: Any) -> set:
    """
    提取多个选项标签 (e.g. "A, C", "AC", "A and B")
    返回集合 (e.g. {"A", "C"})
    """
    if text is None:
        return set()
    s = str(text).strip().upper()
    if not s:
        return set()
        
    # 策略1: 逗号/空格/和 分隔的单字母
    # e.g. "A, B", "A B", "A and B"
    # 查找所有独立的 A-Z
    candidates = re.findall(r"\b([A-Z])\b", s)
    if candidates:
        return set(candidates)
        
    # 策略2: 连续大写字母 (如果策略1没找到)
    # e.g. "AC" -> {"A", "C"}
    # 仅当整个字符串很短且全是大写字母时才启用，防止 "ANSWER" 被拆成 A,N,S,W,E,R
    if len(s) <= 5 and s.isalpha():
        return set(s)
        
    return set()


# -------------------------------------------------------------------------
# Ported from DataFlow (dataflow/utils/reasoning/AnswerExtraction.py)
# -------------------------------------------------------------------------

class StringProcessor:
    """
    A class that encapsulates various string processing functions for mathematical expressions.
    """

    @staticmethod
    def _fix_fracs(string):
        """
        Fixes fraction expressions in the string, ensuring they are properly formatted as \frac{a}{b}.
        """
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            for substr in substrs[1:]:
                new_str += "\\frac"
                if len(substr) > 0 and substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) >= 2:
                        a, b = substr[0], substr[1]
                        if b != "{":
                            new_str += f"{{{a}}}{{{b}}}{substr[2:]}" if len(substr) > 2 else f"{{{a}}}{{{b}}}"
                        else:
                            new_str += f"{{{a}}}{b}{substr[2:]}" if len(substr) > 2 else f"{{{a}}}{b}"
                    else:
                        return string
        return new_str

    @staticmethod
    def _fix_a_slash_b(string):
        """
        Fixes cases where a fraction is represented as a simple division (e.g., a/b) and converts it to \frac{a}{b}.
        """
        if len(string.split("/")) != 2:
            return string
        a, b = string.split("/")
        try:
            a, b = int(a) if "sqrt" not in a else a, int(b) if "sqrt" not in b else b
            # assert string == f"{a}/{b}" 
            return f"\\frac{{{a}}}{{{b}}}"
        except:
            return string

    @staticmethod
    def _fix_sqrt(string):
        """
        Ensures that square root expressions are properly formatted as \sqrt{...}.
        """
        return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)

    @staticmethod
    def convert_word_number(text: str) -> str:
        """
        Converts a word representation of a number to a digit.
        """
        try:
            return str(w2n.word_to_num(text))
        except Exception:
            return text
        

class UnitTextManager:
    """
    A class that encapsulates unit text management to remove unwanted unit terms from strings.
    """

    def __init__(self):
        """
        Initializes the unit texts and their plural forms.
        """
        self.unit_texts = [
            "east", "degree", "mph", "kmph", "ft", "m sqaure", "m east", "sq m", "deg", "mile", "q .", "monkey", "prime",
            "ratio", "profit of rs", "rd", "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
            "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h", "no change", "men", "soldier", "pie", "bc",
            "excess", "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th", "π r 2", "sq",
            "mark", "l", "toy", "coin", "sq . m", "gallon", "° f", "profit", "minw", "yr", "women", "feet", "am", "pm", "hr",
            "cu cm", "square", "v â € ™", "are", "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
            "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³", "hectare",
            "more", "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm", "apple", "liter",
            "loss", "yard", "pure", "year", "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .", "metre",
            "meter", "inch",
        ]
        self.unit_texts.extend([t + "s" for t in self.unit_texts])

    def clean_units(self, string: str):
        """
        Cleans the string by removing unit terms from it.
        """
        for unit_text in self.unit_texts:
            string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        return string


class StringCleaner:
    """
    A class responsible for cleaning and formatting strings in mathematical expressions.
    """

    def __init__(self, unit_manager: UnitTextManager = None):
        """
        Initializes the StringCleaner class with a unit manager.
        """
        self.unit_manager = unit_manager or UnitTextManager()

    def strip_string(self, string, skip_unit=False):
        """
        Strips unwanted characters and units from the string.
        """
        string = str(string).strip().replace("\n", "").rstrip(".").replace("\\!", "")
        string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
        string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string).replace("bmatrix", "pmatrix")
        string = string.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
        string = string.replace("\\left", "").replace("\\right", "").replace("\\{", "{").replace("\\}", "}")
        
        # Clean unit texts if needed
        if not skip_unit:
            string = self.unit_manager.clean_units(string)

        string = string.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "").replace("\\(", "").replace("\\)", "")
        string = StringProcessor.convert_word_number(string)
        string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
        
        for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
            string = string.replace(key, "")
        
        string = string.replace("\\emptyset", r"{}").replace("(-\\infty,\\infty)", "\\mathbb{R}")
        string = string.replace("%", "").replace(" .", " 0.").replace("{.", "{0.")
        
        return string


class AnswerExtractor:
    """
    A class responsible for extracting the final answer from a prediction string.
    """

    def __init__(self, string_cleaner: StringCleaner = None):
        """
        Initializes the AnswerExtractor class with a string cleaner.
        """
        self.string_cleaner = string_cleaner or StringCleaner()

    def extract_answer(self, pred_str, data_name=None, use_last_number=True):
        """
        Extracts the final answer from the prediction string, processing various formats.
        """
        if not pred_str:
            pred_str = ""
        pred_str = str(pred_str).replace("\u043a\u0438", "")
        
        # Handle special cases based on data_name or pattern
        if "final answer is $" in pred_str and "$. I hope" in pred_str:
            pred = pred_str.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
        elif "boxed" in pred_str:
            pred = self._extract_boxed_answer(pred_str)
        elif "he answer is" in pred_str:
            pred = pred_str.split("he answer is")[-1].strip()
        else:
            pred = self._get_last_number_answer(pred_str, use_last_number)
        
        pred = self.string_cleaner.strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
        return pred

    def _extract_boxed_answer(self, pred_str):
        """
        Extracts answers enclosed in 'boxed' notation.
        """
        ans = pred_str.split("boxed")[-1]
        if ans.startswith("{"):
            return self._extract_bracketed_answer(ans)
        else:
            return ans.split("$")[0].strip()

    def _extract_bracketed_answer(self, ans):
        """
        Handles answers that are enclosed within brackets.
        """
        stack = 1
        result = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                result += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                result += c
            else:
                result += c
        return result

    def _get_last_number_answer(self, pred_str, use_last_number):
        """
        Extracts the last number from the string if use_last_number is True.
        Otherwise returns the cleaned string.
        """
        if use_last_number:
            pattern = "-?\\d*\\.?\\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            return pred[-1] if pred else ""
        return pred_str

    @staticmethod
    def normalize_text_for_match(text: Any) -> str:
        if text is None:
            return ""
        s = unicodedata.normalize("NFKC", str(text))
        s = s.translate(str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        }))
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        if s.endswith((".", "。", "!", "！", "?", "？")):
            s = s[:-1].strip()
        return s.casefold()

    @staticmethod
    def text_contains_match(pred: Any, ref: Any) -> bool:
        p = AnswerExtractor.normalize_text_for_match(pred)
        r = AnswerExtractor.normalize_text_for_match(ref)
        if not p or not r:
            return False
        return (r in p) or (p in r)

    @staticmethod
    def parse_choice_from_text(text: str, num_choices: int) -> Optional[int]:
        if text is None:
            return None
        t = str(text).strip()
        if not t:
            return None

        # A/B/C...
        m = re.search(r"\b([A-Za-z])\b", t)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < num_choices:
                return idx

        # Number
        m = re.search(r"\b(\d+)\b", t)
        if m:
            val = int(m.group(1))
            if 0 <= val < num_choices:
                return val
            if 1 <= val <= num_choices:
                return val - 1

        return None

    @staticmethod
    def parse_multiselect_set(text: str, num_choices: int) -> Optional[Set[int]]:
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return None

        # json list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    res = set()
                    for x in obj:
                        if isinstance(x, str):
                            x = x.strip()
                            if len(x) == 1 and x.isalpha():
                                idx = ord(x.upper()) - ord("A")
                                if 0 <= idx < num_choices:
                                    res.add(idx)
                            elif x.isdigit():
                                v = int(x)
                                if 0 <= v < num_choices:
                                    res.add(v)
                                elif 1 <= v <= num_choices:
                                    res.add(v - 1)
                        elif isinstance(x, int):
                            if 0 <= x < num_choices:
                                res.add(x)
                            elif 1 <= x <= num_choices:
                                res.add(x - 1)
                    return res
            except Exception:
                pass

        # Letters: "A,C,D" / "B D"
        letters = re.findall(r"\b([A-Za-z])\b", s)
        if letters:
            res = set()
            for ch in letters:
                idx = ord(ch.upper()) - ord("A")
                if 0 <= idx < num_choices:
                    res.add(idx)
            return res if res else None

        # Numbers: "1,3,4"
        nums = re.findall(r"\b(\d+)\b", s)
        if nums:
            res = set()
            for n in nums:
                v = int(n)
                if 0 <= v < num_choices:
                    res.add(v)
                elif 1 <= v <= num_choices:
                    res.add(v - 1)
            return res if res else None

        return None