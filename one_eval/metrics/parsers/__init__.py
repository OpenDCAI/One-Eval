from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ParseResult, Parser
from .choice import ChoiceLetterParser


class NoParseParser(Parser):
    def parse(
        self,
        value: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        record: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        if value is None or (isinstance(value, str) and not value.strip()):
            return ParseResult(raw=value, ok=False, error="empty_output")
        return ParseResult(raw=value, normalized=value, ok=True, evidence=str(value), strategy="no_parse")


PARSER_REGISTRY = {
    "choice_letter": ChoiceLetterParser(),
    "no_parse": NoParseParser(),
}


def get_parser(parser_type: Optional[str]) -> Parser:
    return PARSER_REGISTRY.get(parser_type or "no_parse", PARSER_REGISTRY["no_parse"])


def parse_value(
    value: Any,
    parser_config: Optional[Dict[str, Any]],
    record: Optional[Dict[str, Any]] = None,
) -> ParseResult:
    cfg = parser_config or {"type": "no_parse"}
    parser = get_parser(cfg.get("type"))
    return parser.parse(value, config=cfg, record=record)
