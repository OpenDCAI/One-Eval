"""
sql.py —— Text2SQL / SQL 产物合法性指标。

只做静态 parse，不连接数据库、不检查 schema/table/column，也不代表执行正确。
"""
import re
from typing import List, Any, Dict, Tuple

from one_eval.core.metric_registry import register_metric, MetricCategory, MetricDimension


_SQL_FENCE_RE = re.compile(r"```[ \t]*(?:sql|SQL)\s*\n?([\s\S]*?)```")
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n?([\s\S]*?)```")


def _extract_sql_candidate(pred: Any) -> Tuple[str, bool]:
    """优先取 markdown SQL 代码块；没有代码块时使用整段输出。"""
    text = "" if pred is None else str(pred).strip()
    if not text:
        return "", False

    match = _SQL_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
    if match:
        return match.group(1).strip(), True
    return text, False


@register_metric(
    name="sql_parse_validity",
    desc="SQL 解析合法率：输出是否能被 SQL parser 解析",
    usage="Text2SQL、SQL 生成任务。只验 SQL 语法/方言可解析性，不代表执行正确或结果正确；dialect 可设 sqlite/postgres/mysql/bigquery/snowflake 等",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    dimension=MetricDimension.VALIDITY,
)
def compute_sql_parse_validity(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """逐条检查预测 SQL 是否可被 sqlglot 解析。

    kwargs.dialect:
      SQL 方言，默认 sqlite。传空字符串时使用 sqlglot 的通用解析。
    """
    import sqlglot

    dialect = str(kwargs.get("dialect", "sqlite") or "").strip() or None
    scores, artifacts = [], []

    for p in preds:
        candidate, extracted = _extract_sql_candidate(p)
        if not candidate:
            scores.append(0.0)
            artifacts.append({"valid": False, "error": "empty", "extracted": extracted})
            continue

        error = None
        try:
            valid = bool(sqlglot.parse(candidate, read=dialect))
        except Exception as e:
            valid = False
            error = str(e)

        scores.append(1.0 if valid else 0.0)
        item = {"valid": valid, "extracted": extracted}
        if dialect:
            item["dialect"] = dialect
        if error:
            item["error"] = error
        artifacts.append(item)

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": artifacts,
    }
