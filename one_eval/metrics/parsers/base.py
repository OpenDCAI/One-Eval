from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ParseResult:
    raw: Any
    normalized: Any = None
    ok: bool = False
    error: Optional[str] = None
    evidence: Optional[str] = None
    strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "ok": self.ok,
            "error": self.error,
            "evidence": self.evidence,
            "strategy": self.strategy,
        }


class Parser:
    def parse(
        self,
        value: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        record: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        raise NotImplementedError
