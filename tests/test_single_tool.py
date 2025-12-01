from one_eval.toolkits.hf_search_tool import hf_search_tool
from one_eval.logger import get_logger

log = get_logger("test_hf_tool")

if __name__ == "__main__":
    result = hf_search_tool.func(
        query="text",
        limit=5
    )

    for item in result:
        log.info(f"{item['id']} | result: {item} \n\n")
