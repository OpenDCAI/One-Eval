from __future__ import annotations
from typing import List, Dict, Any
from huggingface_hub import list_datasets, DatasetCard
from langchain_core.tools import tool

@tool
def hf_search_tool(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    在 HuggingFace Hub 搜索与 query 匹配的数据集，并返回结构化信息。
    可被 LLM 作为 function_call 调用。
    """
    results = []

    datasets = list_datasets(search=query, limit=limit)

    for d in datasets:
        repo_id = d.id
        try:
            card = DatasetCard.load(repo_id)
            item = {
                "id": repo_id,
                "card_text": card.text or "",
                "meta": card.data or {},
                "tags": card.data.get("tags", []) if card.data else []
            }
            results.append(item)
        except Exception as e:
            results.append({
                "id": repo_id,
                "error": str(e),
                "card_text": "",
                "meta": {},
                "tags": []
            })

    return results
