from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import ParseResult, Parser


def normalize_choice_labels(value: Any, record: Optional[Dict[str, Any]] = None) -> List[str]:
    if isinstance(value, list):
        labels = []
        for item in value:
            raw = str(item).strip().upper()
            labels.append(raw if len(raw) == 1 and raw.isalpha() else chr(65 + len(labels)))
        return labels

    if isinstance(value, str):
        raw = value.strip().upper()
        m = re.fullmatch(r"([A-Z])\s*-\s*([A-Z])", raw)
        if m:
            start, end = ord(m.group(1)), ord(m.group(2))
            if start <= end:
                return [chr(i) for i in range(start, end + 1)]
        if "," in raw:
            labels = [p.strip() for p in raw.split(",") if p.strip()]
            if labels:
                return labels

    if record:
        choices = record.get("choices") or record.get("normalized_choices") or record.get("merged_choices")
        if isinstance(choices, list) and choices:
            return [chr(65 + i) for i in range(len(choices))]

    return ["A", "B", "C", "D"]


def _get_path(record: Dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _flatten_choice_values(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], list):
            return value["text"]
        labels = value.get("label")
        texts = value.get("text")
        if isinstance(labels, list) and isinstance(texts, list):
            return texts
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_choice_texts(config: Optional[Dict[str, Any]], record: Optional[Dict[str, Any]]) -> List[str]:
    cfg = config or {}
    values: List[Any] = []

    literal_choices = cfg.get("choice_texts")
    if isinstance(literal_choices, list):
        values.extend(literal_choices)

    if record:
        for field in cfg.get("choice_text_fields") or []:
            values.extend(_flatten_choice_values(_get_path(record, str(field))))

        choices_key = cfg.get("choices_key")
        if isinstance(choices_key, str):
            values.extend(_flatten_choice_values(_get_path(record, choices_key)))

        for key in ("choices", "normalized_choices", "merged_choices", "options", "endings"):
            if key in record:
                values.extend(_flatten_choice_values(record.get(key)))

    out: List[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        norm = _normalize_choice_text(text)
        if text and norm not in seen:
            out.append(text)
            seen.add(norm)
    return out


def _normalize_choice_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


class ChoiceLetterParser(Parser):
    def parse(
        self,
        value: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        record: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        cfg = config or {}
        labels = normalize_choice_labels(cfg.get("choices", "A-D"), record)
        valid = set(labels)

        if value is None:
            return ParseResult(raw=value, ok=False, error="empty_output")

        if isinstance(value, int) and not isinstance(value, bool):
            label = self._label_from_index(value, labels)
            if label:
                return ParseResult(raw=value, normalized=label, ok=True, evidence=str(value), strategy="index")
            return ParseResult(raw=value, ok=False, error="invalid_choice")

        text = str(value).strip()
        if not text:
            return ParseResult(raw=value, ok=False, error="empty_output")

        numeric = self._parse_numeric_index(text, labels)
        if numeric:
            return numeric

        patterns = [
            ("tagged_answer", r"<answer>\s*([A-Za-z])\s*</answer>"),
            ("boxed_answer", r"\\boxed\{\s*([A-Za-z])\s*\}"),
            ("final_answer_line", r"(?:final\s+answer|answer|option|答案|答案是|答案为)\s*(?:is|是|为)?\s*[:：]?\s*\(?\s*([A-Za-z])\s*\)?"),
        ]
        for strategy, pattern in patterns:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            for match in reversed(matches):
                label = match.group(1).upper()
                if label in valid:
                    return ParseResult(
                        raw=value,
                        normalized=label,
                        ok=True,
                        evidence=match.group(0),
                        strategy=strategy,
                    )

        text_match = self._label_from_choice_text(text, labels, cfg, record)
        if text_match:
            label, evidence = text_match
            return ParseResult(raw=value, normalized=label, ok=True, evidence=evidence, strategy="choice_text")

        tail = self._last_nonempty_lines(text, limit=3)
        tail_match = self._last_standalone_choice(tail, valid)
        if tail_match:
            label, evidence = tail_match
            return ParseResult(raw=value, normalized=label, ok=True, evidence=evidence, strategy="last_standalone_choice")

        if cfg.get("allow_full_text_scan", True):
            full_match = self._last_standalone_choice(text, valid)
            if full_match:
                label, evidence = full_match
                return ParseResult(raw=value, normalized=label, ok=True, evidence=evidence, strategy="full_text_scan")

        return ParseResult(raw=value, ok=False, error="parse_failed")

    def _parse_numeric_index(self, text: str, labels: List[str]) -> Optional[ParseResult]:
        raw = text.strip()
        if not re.fullmatch(r"\(?\s*\d+\s*\)?", raw):
            return None
        idx = int(re.sub(r"\D", "", raw))
        label = self._label_from_index(idx, labels)
        if label:
            return ParseResult(raw=text, normalized=label, ok=True, evidence=raw, strategy="index")
        return ParseResult(raw=text, ok=False, error="invalid_choice")

    def _label_from_index(self, idx: int, labels: List[str]) -> Optional[str]:
        if 0 <= idx < len(labels):
            return labels[idx]
        if 1 <= idx <= len(labels):
            return labels[idx - 1]
        return None

    def _label_from_choice_text(
        self,
        text: str,
        labels: List[str],
        config: Optional[Dict[str, Any]],
        record: Optional[Dict[str, Any]],
    ) -> Optional[tuple[str, str]]:
        choice_texts = resolve_choice_texts(config, record)
        if not choice_texts:
            return None

        normalized = _normalize_choice_text(text)
        for idx, choice in enumerate(choice_texts):
            if idx >= len(labels):
                break
            choice_norm = _normalize_choice_text(choice)
            if normalized == choice_norm:
                return labels[idx], str(choice)

        if len(text) <= 256:
            for idx, choice in enumerate(choice_texts):
                if idx >= len(labels):
                    break
                choice_norm = _normalize_choice_text(choice)
                if len(choice_norm) >= 3 and choice_norm in normalized:
                    return labels[idx], str(choice)
        return None

    def _last_nonempty_lines(self, text: str, limit: int) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-limit:]) if lines else text

    def _last_standalone_choice(self, text: str, valid: set[str]) -> Optional[tuple[str, str]]:
        matches = []
        for match in re.finditer(r"(?<![A-Za-z0-9])\(?\s*([A-Za-z])\s*\)?(?![A-Za-z0-9])", text):
            label = match.group(1).upper()
            if label not in valid:
                continue
            after = text[match.end(): match.end() + 2]
            if after.startswith((".", "．")):
                continue
            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            suffix = text[match.end():line_end].strip()
            if suffix and not re.fullmatch(r"[\).,;:!?。！？、，；：]*", suffix):
                continue
            matches.append((label, match.group(0).strip()))
        return matches[-1] if matches else None
