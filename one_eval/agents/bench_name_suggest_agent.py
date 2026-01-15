from __future__ import annotations
import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import Counter

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger

log = get_logger("BenchNameSuggestAgent")


# ==================== BenchmarkRetriever ====================

class BenchmarkRetriever:
    """基于语义相似度的Benchmark检索器，支持RAG和非RAG模式"""

    def __init__(
        self,
        xlsx_path: str = None,
        cache_dir: str = None,
        batch_size: int = 50,
        use_rag: bool = False,
        api_base: str = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        初始化检索器

        Args:
            xlsx_path: xlsx文件路径
            cache_dir: 缓存目录
            batch_size: 批量调用API时的批大小
            use_rag: 是否使用RAG模式（embedding检索），False则使用TF-IDF+关键词匹配
            api_base: OpenAI兼容API的基础URL
            embedding_model: embedding模型名称
        """
        # 默认路径指向 one_eval/utils/bench_table 目录
        self.base_dir = Path(__file__).parent.parent / "utils" / "bench_table"
        self.xlsx_path = Path(xlsx_path) if xlsx_path else self.base_dir / "BenchmarkTable.xlsx"
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / "cache"
        self.batch_size = batch_size
        self.use_rag = use_rag
        self.api_base = api_base
        self.embedding_model = embedding_model

        self.client = None
        self.df = None
        self.embeddings = None
        self.meta_data = None

        # TF-IDF相关（非RAG模式）
        self.tfidf_matrix = None
        self.vocabulary = None
        self.idf_values = None
        self.doc_texts = None

        # 缓存文件路径
        self.meta_path = self.cache_dir / "benchmarks_meta.json"
        self.embeddings_path = self.cache_dir / "benchmarks_embeddings.npy"
        self.tfidf_path = self.cache_dir / "benchmarks_tfidf.json"

    def _get_client(self):
        """获取OpenAI兼容客户端"""
        if self.client is None:
            from openai import OpenAI
            api_key = os.environ.get("GENERAL_API_KEY")
            if not api_key:
                raise ValueError("请设置环境变量 GENERAL_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url=self.api_base)
        return self.client

    def _get_embedding(self, texts: List[str]) -> np.ndarray:
        """调用OpenAI兼容API获取embedding"""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]
        return np.array(embeddings)

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """批量获取embedding"""
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            log.info(f"正在处理 {i+1}-{min(i+len(batch), total)}/{total}...")
            batch_embeddings = self._get_embedding(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _load_xlsx(self) -> pd.DataFrame:
        """加载xlsx数据"""
        log.info(f"正在加载数据: {self.xlsx_path}")
        df = pd.read_excel(self.xlsx_path)
        df.columns = [col.strip() for col in df.columns]
        log.info(f"加载了 {len(df)} 条benchmark记录")
        return df

    def _build_texts(self, df: pd.DataFrame) -> List[str]:
        """构建用于embedding的文本"""
        texts = []
        for _, row in df.iterrows():
            text_parts = []
            if pd.notna(row.get('Name')):
                text_parts.append(f"Name: {row['Name']}")
            if pd.notna(row.get('Type')):
                text_parts.append(f"Type: {row['Type']}")
            if pd.notna(row.get('Description')):
                text_parts.append(f"Description: {row['Description']}")
            texts.append(" | ".join(text_parts))
        return texts

    def _build_meta(self, df: pd.DataFrame) -> List[Dict]:
        """构建元数据"""
        meta = []
        for _, row in df.iterrows():
            meta.append({
                'name': row.get('Name', ''),
                'type': row.get('Type', ''),
                'description': row.get('Description', ''),
                'dataset_url': row.get('Dataset', '')
            })
        return meta

    # ==================== TF-IDF 相关方法 ====================

    def _tokenize(self, text: str) -> List[str]:
        """分词：支持中英文混合"""
        text = str(text).lower()
        english_words = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        numbers = re.findall(r'\d+', text)
        return english_words + chinese_chars + numbers

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """计算词频（TF）"""
        counter = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {word: count / total for word, count in counter.items()}

    def _build_tfidf_index(self, texts: List[str]):
        """构建TF-IDF索引"""
        self.doc_texts = texts
        n_docs = len(texts)

        tokenized_docs = [self._tokenize(text) for text in texts]

        doc_freq = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        self.vocabulary = list(doc_freq.keys())
        self.idf_values = {}
        for word, df in doc_freq.items():
            self.idf_values[word] = math.log(n_docs / (df + 1)) + 1

        self.tfidf_matrix = []
        for tokens in tokenized_docs:
            tf = self._compute_tf(tokens)
            tfidf_vec = {}
            for word, tf_val in tf.items():
                if word in self.idf_values:
                    tfidf_vec[word] = tf_val * self.idf_values[word]
            self.tfidf_matrix.append(tfidf_vec)

        log.info(f"TF-IDF索引构建完成，词表大小: {len(self.vocabulary)}")

    def _compute_tfidf_similarity(self, query: str, doc_tfidf: Dict[str, float]) -> float:
        """计算查询与文档的TF-IDF相似度"""
        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)

        query_tfidf = {}
        for word, tf_val in query_tf.items():
            idf = self.idf_values.get(word, 1.0)
            query_tfidf[word] = tf_val * idf

        dot_product = 0.0
        query_norm = 0.0
        doc_norm = 0.0

        for word, val in query_tfidf.items():
            query_norm += val * val
            if word in doc_tfidf:
                dot_product += val * doc_tfidf[word]

        for val in doc_tfidf.values():
            doc_norm += val * val

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        cosine_sim = dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))

        query_words = set(query_tokens)
        doc_words = set(doc_tfidf.keys())
        overlap = query_words & doc_words
        keyword_bonus = len(overlap) / (len(query_words) + 1) * 0.3

        return cosine_sim + keyword_bonus

    def _save_cache(self):
        """保存缓存到文件"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta_data, f, ensure_ascii=False, indent=2)
        log.info(f"元数据已保存到: {self.meta_path}")

        if self.use_rag:
            np.save(self.embeddings_path, self.embeddings)
            log.info(f"Embeddings已保存到: {self.embeddings_path}")
        else:
            tfidf_data = {
                'vocabulary': self.vocabulary,
                'idf_values': self.idf_values,
                'tfidf_matrix': self.tfidf_matrix,
                'doc_texts': self.doc_texts
            }
            with open(self.tfidf_path, 'w', encoding='utf-8') as f:
                json.dump(tfidf_data, f, ensure_ascii=False)
            log.info(f"TF-IDF索引已保存到: {self.tfidf_path}")

    def _load_cache(self) -> bool:
        """从缓存加载数据"""
        if not self.meta_path.exists():
            return False

        if self.use_rag:
            if not self.embeddings_path.exists():
                return False
            log.info("从缓存加载RAG数据...")
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.meta_data = json.load(f)
            self.embeddings = np.load(self.embeddings_path)

            try:
                test_embedding = self._get_embedding(["test"])[0]
                expected_dim = len(test_embedding)
                cached_dim = self.embeddings.shape[1]
                if expected_dim != cached_dim:
                    log.warning(f"缓存维度({cached_dim})与当前模型维度({expected_dim})不匹配，需要重建索引")
                    self.embeddings = None
                    self.meta_data = None
                    return False
            except Exception as e:
                log.warning(f"维度检查失败: {e}，将重建索引")
                self.embeddings = None
                self.meta_data = None
                return False

            log.info(f"已加载 {len(self.meta_data)} 条记录的RAG缓存")
        else:
            if not self.tfidf_path.exists():
                return False
            log.info("从缓存加载TF-IDF数据...")
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.meta_data = json.load(f)
            with open(self.tfidf_path, 'r', encoding='utf-8') as f:
                tfidf_data = json.load(f)
            self.vocabulary = tfidf_data['vocabulary']
            self.idf_values = tfidf_data['idf_values']
            self.tfidf_matrix = tfidf_data['tfidf_matrix']
            self.doc_texts = tfidf_data['doc_texts']
            log.info(f"已加载 {len(self.meta_data)} 条记录的TF-IDF缓存")

        return True

    def build_index(self, force_rebuild: bool = False):
        """构建索引（RAG模式用embedding，非RAG模式用TF-IDF）"""
        if not force_rebuild and self._load_cache():
            return

        df = self._load_xlsx()
        texts = self._build_texts(df)
        self.meta_data = self._build_meta(df)

        if self.use_rag:
            log.info("正在调用OpenAI兼容API生成embeddings...")
            self.embeddings = self._get_embeddings_batch(texts)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norms
        else:
            log.info("正在构建TF-IDF索引...")
            self._build_tfidf_index(texts)

        self._save_cache()

    def search(self, query: str, top_k: int = 5, return_scores: bool = True) -> List[Dict]:
        """检索（支持RAG和非RAG模式）"""
        if self.meta_data is None:
            self.build_index()

        if self.use_rag:
            if self.embeddings is None:
                self.build_index()

            query_embedding = self._get_embedding([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = [float(similarities[idx]) for idx in top_indices]
        else:
            if self.tfidf_matrix is None:
                self.build_index()

            similarities = []
            for doc_tfidf in self.tfidf_matrix:
                sim = self._compute_tfidf_similarity(query, doc_tfidf)
                similarities.append(sim)

            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = [float(similarities[idx]) for idx in top_indices]

        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': len(results) + 1,
                'name': self.meta_data[idx]['name'],
                'type': self.meta_data[idx]['type'],
                'description': self.meta_data[idx]['description'],
                'dataset_url': self.meta_data[idx]['dataset_url'],
            }
            if return_scores:
                result['score'] = scores[i]
            results.append(result)

        return results


# ==================== BenchNameSuggestAgent ====================

class BenchNameSuggestAgent(CustomAgent):
    """
    Benchmark名称推荐Agent

    支持两种检索模式：
    - RAG模式：使用embedding语义检索（需要API调用）
    - TF-IDF模式：使用TF-IDF + 关键词匹配（本地计算，无需API）

    数据源：one_eval/utils/bench_table/BenchmarkTable.xlsx
    """

    def __init__(
            self,
            use_rag: bool = False,
            api_base: str = None",
            embedding_model: str = "text-embedding-3-small",
            top_k: int = 3,
            **kwargs
    ):
        """
        初始化Agent

        Args:
            use_rag: 是否使用RAG模式（embedding检索），默认False使用TF-IDF
            api_base: OpenAI兼容API的基础URL（RAG模式需要）
            embedding_model: embedding模型名称（RAG模式需要）
            top_k: 检索返回的结果数，默认3
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(**kwargs)
        self.use_rag = use_rag
        self.api_base = api_base
        self.embedding_model = embedding_model
        self.top_k = top_k

        self._retriever: Optional[BenchmarkRetriever] = None

    def _get_retriever(self) -> BenchmarkRetriever:
        """获取或创建检索器（懒加载）"""
        if self._retriever is None:
            self._retriever = BenchmarkRetriever(
                use_rag=self.use_rag,
                api_base=self.api_base,
                embedding_model=self.embedding_model
            )
            self._retriever.build_index()

            mode = "RAG (Embedding)" if self.use_rag else "TF-IDF"
            log.info(f"BenchmarkRetriever 已初始化，模式: {mode}")

        return self._retriever

    @property
    def role_name(self) -> str:
        return "BenchNameSuggestAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return ""  # 不再使用 LLM

    @property
    def task_prompt_template_name(self) -> str:
        return ""  # 不再使用 LLM

    def _extract_query_info(self, state: NodeState) -> Dict[str, Any]:
        """从 QueryUnderstandAgent 的输出中抽取必要信息"""
        q = {}
        if isinstance(state.result, dict):
            q = state.result.get("QueryUnderstandAgent", {}) or {}

        return {
            "domain": q.get("domain") or [],
            "specific_benches": q.get("specific_benches") or [],
            "user_query": getattr(state, "user_query", ""),
        }

    def _build_search_query(self, info: Dict[str, Any]) -> str:
        """构建检索查询字符串"""
        parts = []

        user_query = info.get("user_query", "")
        if user_query:
            parts.append(user_query)

        domains = info.get("domain", [])
        if domains:
            parts.append(f"领域: {', '.join(domains)}")

        specific_benches = info.get("specific_benches", [])
        if specific_benches:
            parts.append(f"benchmark: {', '.join(specific_benches)}")

        return " ".join(parts) if parts else "benchmark evaluation"

    async def run(self, state: NodeState) -> NodeState:
        """
        执行 benchmark 检索推荐

        流程：
        1. 从 state 中提取查询信息
        2. 使用 BenchmarkRetriever 检索（RAG 或 TF-IDF）
        3. 返回 top_k 个结果，设置 skip_resolve=True 跳过后续 HF 解析
        """
        info = self._extract_query_info(state)

        # 使用 BenchmarkRetriever 检索
        retriever = self._get_retriever()
        search_query = self._build_search_query(info)
        log.info(f"检索查询: {search_query}")

        search_results = retriever.search(search_query, top_k=self.top_k, return_scores=True)

        mode = "RAG" if self.use_rag else "TF-IDF"
        log.info(f"[{mode}模式] 检索到 {len(search_results)} 个 benchmark")

        # 转换检索结果
        bench_info: Dict[str, Dict[str, Any]] = {}
        local_matches = []

        for result in search_results:
            name = result.get('name', '')
            if not name:
                continue

            bench_data = {
                'bench_name': name,
                'type': result.get('type', ''),
                'description': result.get('description', ''),
                'dataset_url': result.get('dataset_url', ''),
                'score': result.get('score', 0.0),
                'source': 'retrieval',
            }
            bench_info[name] = bench_data
            local_matches.append(bench_data)

        # 构建 BenchInfo 列表
        state.benches = [
            BenchInfo(
                bench_name=name,
                bench_table_exist=True,
                bench_source_url=bench_info[name].get('dataset_url'),
                meta=bench_info[name],
            )
            for name in bench_info.keys()
        ]

        state.bench_info = bench_info

        state.agent_results["BenchNameSuggestAgent"] = {
            "local_matches": local_matches,
            "bench_names": [],
            "skip_resolve": True,
            "retrieval_mode": "rag" if self.use_rag else "tfidf",
            "search_query": search_query,
        }

        # 跳过后续 BenchResolveAgent
        state.temp_data["skip_resolve"] = True

        log.info(f"检索完成，返回 {len(local_matches)} 个 benchmark")
        return state
