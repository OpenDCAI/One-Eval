"""
BenchmarkRetriever 简单测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from one_eval.agents.bench_name_suggest_agent import BenchmarkRetriever


def test_tfidf():
    """测试 TF-IDF 模式"""
    print("=" * 50)
    print("TF-IDF 模式测试")
    print("=" * 50)

    retriever = BenchmarkRetriever(use_rag=False)
    retriever.build_index()

    query = "评估大语言模型的数学推理能力"
    print(f"\n查询: {query}\n")

    results = retriever.search(query, top_k=3)

    for r in results:
        print(f"【{r['rank']}】{r['name']}")
        print(f"    类型: {r['type']}")
        print(f"    相似度: {r['score']:.4f}")
        print()


def test_rag():
    """测试 RAG 模式"""
    print("=" * 50)
    print("RAG 模式测试")
    print("=" * 50)

    retriever = BenchmarkRetriever(use_rag=True)
    retriever.build_index()

    query = "评估大语言模型的数学推理能力"
    print(f"\n查询: {query}\n")

    results = retriever.search(query, top_k=3)

    for r in results:
        print(f"【{r['rank']}】{r['name']}")
        print(f"    类型: {r['type']}")
        print(f"    相似度: {r['score']:.4f}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rag', action='store_true', help='使用RAG模式')
    args = parser.parse_args()

    if args.rag:
        test_rag()
    else:
        test_tfidf()
