"""
Benchmark 名称推荐 Node

基于 TF-IDF 或 RAG (Embedding) 的 Benchmark 检索，
用于根据用户查询推荐合适的评测基准。

数据源：one_eval/utils/bench_table/BenchmarkTable_Filter.xlsx
"""
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

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState, BenchInfo
from one_eval.serving.custom_llm_caller import EmbeddingCaller
from one_eval.logger import get_logger

log = get_logger("BenchNameSuggestNode")


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
        api_key: str = None,
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
            api_key: API密钥（RAG模式必需）
            embedding_model: embedding模型名称
        """
        # 默认路径指向 one_eval/utils/bench_table 目录
        self.base_dir = Path(__file__).parent.parent / "utils" / "bench_table"
        self.xlsx_path = Path(xlsx_path) if xlsx_path else self.base_dir / "BenchmarkTable_Filter.xlsx"
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / "cache"
        self.batch_size = batch_size
        self.use_rag = use_rag
        self.api_base = api_base
        self.api_key = api_key
        self.embedding_model = embedding_model

        self._embedding_caller = None
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

    def _get_embedding_caller(self) -> EmbeddingCaller:
        """获取 EmbeddingCaller（懒加载）"""
        if self._embedding_caller is None:
            if not self.api_key:
                raise ValueError("RAG模式需要提供 api_key 参数")
            if not self.api_base:
                raise ValueError("RAG模式需要提供 api_base 参数")
            self._embedding_caller = EmbeddingCaller(
                base_url=self.api_base,
                api_key=self.api_key,
                model=self.embedding_model
            )
        return self._embedding_caller

    def _get_embedding(self, texts: List[str]) -> np.ndarray:
        """调用 EmbeddingCaller 获取 embedding"""
        caller = self._get_embedding_caller()
        embeddings = caller.get_embedding(texts)
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

    def _load_gallery_extra(self) -> tuple:
        """从 bench_gallery.json 加载 xlsx 里没有的 bench，返回 (extra_meta, extra_texts)"""
        gallery_path = Path(__file__).parent.parent / "utils" / "bench_table" / "bench_gallery.json"
        if not gallery_path.exists():
            return [], []

        gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
        extra_meta, extra_texts = [], []
        for b in gallery.get("benches", []):
            name = b.get("bench_name", "")
            if not name:
                continue
            meta_section = b.get("meta") or {}
            description = meta_section.get("description", "")
            category = meta_section.get("category", "")
            aliases = meta_section.get("aliases") or []
            tags = meta_section.get("tags") or []
            source_url = b.get("bench_source_url", "")

            extra_meta.append({
                "name": name,
                "type": category,
                "description": description,
                "dataset_url": source_url,
            })
            alias_str = " ".join(aliases)
            tag_str = " ".join(tags)
            text_parts = [f"Name: {name}"]
            if alias_str:
                text_parts.append(f"Aliases: {alias_str}")
            if category:
                text_parts.append(f"Type: {category}")
            if tag_str:
                text_parts.append(f"Tags: {tag_str}")
            if description:
                text_parts.append(f"Description: {description}")
            extra_texts.append(" | ".join(text_parts))

        return extra_meta, extra_texts

    def build_index(self, force_rebuild: bool = False):
        """构建索引（RAG模式用embedding，非RAG模式用TF-IDF）"""
        if not force_rebuild and self._load_cache():
            return

        df = self._load_xlsx()
        texts = self._build_texts(df)
        self.meta_data = self._build_meta(df)

        # 把 gallery.json 里有但 xlsx 里没有的 bench 追加进索引
        xlsx_names_lower = {str(m.get("name", "")).lower() for m in self.meta_data}
        extra_meta, extra_texts = self._load_gallery_extra()
        for m, t in zip(extra_meta, extra_texts):
            if m["name"].lower() not in xlsx_names_lower:
                self.meta_data.append(m)
                texts.append(t)
        log.info(f"追加了 {len(extra_meta) - sum(1 for m in extra_meta if m['name'].lower() in xlsx_names_lower)} 条 gallery-only bench 进入索引")

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


# ==================== BenchNameSuggestNode ====================

class BenchNameSuggestNode(BaseNode):
    """
    Benchmark 名称推荐 Node

    基于 TF-IDF 或 RAG (Embedding) 检索 benchmark，
    不调用 LLM，纯硬逻辑检索。

    支持两种检索模式：
    - RAG模式：使用 embedding 语义检索（需要 API 调用）
    - TF-IDF模式：使用 TF-IDF + 关键词匹配（本地计算，无需 API）
    """

    def __init__(
        self,
        use_rag: bool = False,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
    ):
        """
        初始化 Node

        Args:
            use_rag: 是否使用 RAG 模式（embedding 检索），默认 False 使用 TF-IDF
            embedding_model: embedding 模型名称（RAG 模式需要）
            top_k: 检索返回的结果数，默认 8

        Note:
            RAG 模式的 api_base 和 api_key 从环境变量 OE_API_BASE 和 OE_API_KEY 读取
        """
        super().__init__(name="BenchNameSuggestNode")
        self.use_rag = use_rag
        self.embedding_model = embedding_model
        self.top_k = top_k

        # API 配置从环境变量读取（与 CustomAgent 保持一致）
        self.api_url = os.getenv(
            "OE_API_BASE",
            "http://123.129.219.111:3000/v1",
        )
        self.api_key = os.getenv(
            "OE_API_KEY",
            "",
        )

        self._retriever: Optional[BenchmarkRetriever] = None

        # 加载 gallery 索引，用于检索结果补全
        self._gallery_index: Dict[str, Dict] = self._load_gallery_index()

    def _load_gallery_index(self) -> Dict[str, Dict]:
        """加载 bench_gallery.json，构建 bench_name -> 完整配置 的索引"""
        gallery_path = Path(__file__).parent.parent / "utils" / "bench_table" / "bench_gallery.json"
        if not gallery_path.exists():
            log.warning(f"bench_gallery.json 不存在: {gallery_path}")
            return {}
        with open(gallery_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        index: Dict[str, Dict] = {}
        for bench in data.get("benches", []):
            name = bench.get("bench_name", "")
            if name:
                index[name.lower()] = bench
                # 同时索引 aliases
                for alias in (bench.get("meta") or {}).get("aliases", []):
                    if isinstance(alias, str) and alias:
                        index[alias.lower()] = bench
        log.info(f"已加载 gallery 索引，共 {len(data.get('benches', []))} 条")
        return index

    def _lookup_gallery(self, bench_name: str) -> Optional[Dict]:
        """在 gallery 中查找 bench，返回完整配置或 None"""
        if not bench_name:
            return None
        return self._gallery_index.get(bench_name.lower())

    def _get_retriever(self) -> BenchmarkRetriever:
        """获取或创建检索器（懒加载）"""
        if self._retriever is None:
            self._retriever = BenchmarkRetriever(
                use_rag=self.use_rag,
                api_base=self.api_url,
                api_key=self.api_key,
                embedding_model=self.embedding_model
            )
            self._retriever.build_index()

            mode = "RAG (Embedding)" if self.use_rag else "TF-IDF"
            log.info(f"BenchmarkRetriever 已初始化，模式: {mode}")

        return self._retriever

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

    def _extract_hf_repo_from_url(self, url: str) -> str:
        """从 HuggingFace URL 提取 repo ID

        Examples:
            https://huggingface.co/datasets/ought/raft → ought/raft
            https://huggingface.co/datasets/RAFT → RAFT
            ought/raft → ought/raft
        """
        if not url:
            return ""

        # 去掉协议和域名
        if "huggingface.co/datasets/" in url:
            parts = url.split("huggingface.co/datasets/")[1]
            # 去掉查询参数和锚点
            repo_id = parts.split("?")[0].split("#")[0]
            return repo_id.strip("/")

        # 如果已经是 repo ID 格式，直接返回
        return url

    # 质量阈值：低于此分数的本地结果视为不够匹配
    SCORE_THRESHOLD_TFIDF = 0.1
    SCORE_THRESHOLD_RAG = 0.25

    async def run(self, state: NodeState) -> NodeState:
        """
        执行 benchmark 检索推荐

        流程：
        1. 从 state 中提取查询信息
        2. 使用 BenchmarkRetriever 检索（RAG 或 TF-IDF）
        3. 按 local_count 配额截取本地结果；将搜索查询传递给 BenchResolveAgent 供 HF 配额使用
        """
        info = self._extract_query_info(state)

        # 从 state 读取配额设置
        local_count = int(getattr(state, 'local_count', 3) or 3)
        hf_count = int(getattr(state, 'hf_count', 2) or 2)

        # local_count == 0 表示用户不需要本地结果，跳过本地检索
        if local_count == 0:
            log.info("local_count=0，跳过本地检索，全部交由 HF 搜索")
            state.benches = []
            state.bench_info = {}
            search_query = self._build_search_query(info)
            state.temp_data["skip_resolve"] = False
            state.temp_data["hf_search_query"] = search_query
            state.temp_data["bench_names_suggested"] = []
            state.agent_results["BenchNameSuggestNode"] = {
                "local_matches": [],
                "quality_matches": [],
                "gallery_hits": [],
                "bench_names": [],
                "skip_resolve": False,
                "retrieval_mode": "skipped",
                "search_query": search_query,
                "local_count": local_count,
                "hf_count": hf_count,
            }
            return state

        # 从 state 读取 use_rag 设置（优先使用 state 中的值）
        use_rag = getattr(state, 'use_rag', self.use_rag)

        # 如果 use_rag 设置变化，需要重新创建 retriever
        if self._retriever is not None and self._retriever.use_rag != use_rag:
            self._retriever = None

        # 临时覆盖 self.use_rag 以便 _get_retriever 使用正确的模式
        original_use_rag = self.use_rag
        self.use_rag = use_rag

        # 使用 BenchmarkRetriever 检索
        retriever = self._get_retriever()

        # 恢复原值
        self.use_rag = original_use_rag

        search_query = self._build_search_query(info)
        log.info(f"检索查询: {search_query}")

        # 多检索一些候选，以便按配额截取
        fetch_k = max(self.top_k, local_count + 5)
        search_results = retriever.search(search_query, top_k=fetch_k, return_scores=True)

        mode = "RAG" if use_rag else "TF-IDF"
        log.info(f"[{mode}模式] 检索到 {len(search_results)} 个 benchmark，本地配额: {local_count}")

        # 按分数阈值区分高质量 / 低质量结果
        score_threshold = self.SCORE_THRESHOLD_RAG if use_rag else self.SCORE_THRESHOLD_TFIDF

        # 转换检索结果
        bench_info: Dict[str, Dict[str, Any]] = {}
        local_matches = []
        quality_matches = []  # 高于阈值的结果

        # 用户明确指定的 specific_benches，在 gallery 里直接命中，不依赖检索分数
        specific_benches_early: List[str] = info.get("specific_benches") or []
        for name in specific_benches_early:
            if not name:
                continue
            gallery_entry = self._lookup_gallery(name)
            if gallery_entry and gallery_entry['bench_name'] not in bench_info:
                repo_id = gallery_entry['bench_name']
                bench_data = {
                    'bench_name': repo_id,
                    'type': gallery_entry.get('meta', {}).get('category', ''),
                    'description': gallery_entry.get('meta', {}).get('description', ''),
                    'dataset_url': gallery_entry.get('bench_source_url', ''),
                    'score': 1.0,  # 直接命中，视为满分
                    'source': 'gallery_direct',
                    'from_gallery': True,
                    '_gallery_entry': gallery_entry,
                }
                bench_info[repo_id] = bench_data
                local_matches.append(bench_data)
                quality_matches.append(bench_data)
                log.info(f"[gallery直接命中] specific_bench={name} → {repo_id}")

        for result in search_results:
            name = result.get('name', '')
            dataset_url = result.get('dataset_url', '')
            if not name:
                continue

            # 从 URL 提取 HF repo ID（用于下载）
            hf_repo_id = self._extract_hf_repo_from_url(dataset_url) or name
            score = result.get('score', 0.0)

            # 已经通过 specific_benches_early 直接命中的，不用检索结果覆盖
            if hf_repo_id in bench_info:
                continue

            # 查 gallery，有则直接用完整配置（保存 entry 供后续构建 BenchInfo 使用）
            gallery_entry = self._lookup_gallery(name) or self._lookup_gallery(hf_repo_id)
            from_gallery = gallery_entry is not None

            bench_data = {
                'bench_name': hf_repo_id,
                'type': result.get('type', ''),
                'description': result.get('description', ''),
                'dataset_url': dataset_url,
                'score': score,
                'source': 'retrieval',
                'from_gallery': from_gallery,
                '_gallery_entry': gallery_entry,  # 保留引用，避免二次查找 key 不一致
            }
            bench_info[hf_repo_id] = bench_data
            local_matches.append(bench_data)
            if score >= score_threshold:
                quality_matches.append(bench_data)

        # ========== 按 local_count 配额构建本地结果 ==========
        # 优先保留 gallery_direct 命中的，然后按分数取高质量结果，最后兜底
        only_gallery_direct = bool(info.get("specific_benches"))

        # 候选池：按优先级排序
        candidates = []
        # 1) gallery_direct（用户明确指定的）
        for m in local_matches:
            if m.get('source') == 'gallery_direct':
                candidates.append(m)
        # 2) 高分结果（gallery 优先）
        for m in local_matches:
            if m in candidates:
                continue
            if m.get('score', 0.0) >= score_threshold:
                candidates.append(m)
        # 3) 兜底：剩余的低分结果
        for m in local_matches:
            if m in candidates:
                continue
            candidates.append(m)

        if only_gallery_direct:
            # 用户指定了 specific_benches，只保留 gallery_direct
            candidates = [m for m in candidates if m.get('source') == 'gallery_direct']

        # 按 local_count 截取
        selected_local = candidates[:local_count]

        built_benches = []
        built_bench_info = {}
        for data in selected_local:
            repo_id = data['bench_name']
            gallery_entry = data.get('_gallery_entry')
            if gallery_entry:
                bench = BenchInfo(
                    bench_name=gallery_entry['bench_name'],
                    bench_table_exist=gallery_entry.get('bench_table_exist', True),
                    bench_source_url=gallery_entry.get('bench_source_url'),
                    bench_dataflow_eval_type=gallery_entry.get('bench_dataflow_eval_type'),
                    bench_prompt_template=gallery_entry.get('bench_prompt_template'),
                    bench_keys=gallery_entry.get('bench_keys', []),
                    meta={**gallery_entry.get('meta', {}), 'from_gallery': True, 'retrieval_score': data['score']},
                )
                log.info(f"[gallery命中] {repo_id} → 使用完整配置（eval_type={bench.bench_dataflow_eval_type}）")
            else:
                bench = BenchInfo(
                    bench_name=repo_id,
                    bench_table_exist=True,
                    bench_source_url=data.get('dataset_url'),
                    meta={**data, 'from_gallery': False},
                )
            built_benches.append(bench)
            built_bench_info[repo_id] = data

        state.benches = built_benches
        state.bench_info = built_bench_info

        # ========== 为 BenchResolveAgent 准备 HF 搜索信息 ==========
        # 传递搜索查询，让 BenchResolveAgent 根据 hf_count 主动搜索 HF
        specific_benches: List[str] = info.get("specific_benches") or []
        selected_names = {m['bench_name'] for m in selected_local}

        names_for_hf: List[str] = []
        # 1) 用户明确指定但不在 gallery 的
        for name in specific_benches:
            if name and not self._lookup_gallery(name):
                names_for_hf.append(name)

        skip_resolve = (hf_count == 0)
        state.temp_data["skip_resolve"] = skip_resolve
        state.temp_data["bench_names_suggested"] = names_for_hf
        state.temp_data["hf_search_query"] = search_query
        state.temp_data["local_bench_names"] = list(selected_names)

        gallery_hits = [m['bench_name'] for m in selected_local if m.get('from_gallery')]
        state.agent_results["BenchNameSuggestNode"] = {
            "local_matches": local_matches,
            "quality_matches": [m['bench_name'] for m in quality_matches],
            "gallery_hits": gallery_hits,
            "bench_names": names_for_hf,
            "skip_resolve": skip_resolve,
            "retrieval_mode": "rag" if use_rag else "tfidf",
            "search_query": search_query,
            "score_threshold": score_threshold,
            "local_count": local_count,
            "hf_count": hf_count,
        }

        log.info(
            f"检索完成，本地配额 {local_count}，实际选中 {len(selected_local)} 个"
            f"（gallery命中: {len(gallery_hits)}），HF 配额: {hf_count}"
        )

        return state
