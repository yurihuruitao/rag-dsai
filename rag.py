"""底层模块：PDF加载、索引管理、混合检索、重排序"""

import os
import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rank_bm25 import BM25Okapi
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.readers.file import PyMuPDFReader



# ===== BM25 检索器 =====
class BM25Retriever(BaseRetriever):
    """基于 BM25 的关键词检索器"""

    def __init__(self, nodes, top_k=5):
        super().__init__()
        self._nodes = nodes
        self._top_k = top_k
        corpus = [node.get_content().lower().split() for node in nodes]
        self._bm25 = BM25Okapi(corpus)

    def _retrieve(self, query_bundle: QueryBundle):
        query_tokens = query_bundle.query_str.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-self._top_k :][::-1]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(
                    NodeWithScore(node=self._nodes[i], score=float(scores[i]))
                )
        return results


# ===== 初始化 =====
def init_settings(config):
    """初始化全局 LLM 和 Embedding 设置

    Args:
        config: 配置对象
    """
    model = config.DEFAULT_MODEL
    if model not in config.MODELS:
        raise ValueError(f"未知模型: {model}，可选: {list(config.MODELS.keys())}")

    Settings.llm = OpenAILike(
        api_key=config.DEEPSEEK_API_KEY,
        api_base=config.DEEPSEEK_BASE_URL,
        model=config.MODELS[model],
        is_chat_model=True,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
    )
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    chunk_strategy = getattr(config, "CHUNK_STRATEGY", "sentence")
    if chunk_strategy == "paragraph":
        Settings.node_parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            paragraph_separator="\n\n",
        )
    elif chunk_strategy == "character":
        Settings.node_parser = TokenTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separator="",
            backup_separators=[" "],
        )
    else:
        Settings.node_parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    return model


# ===== 文档加载 =====
def load_documents(pdf_dir):
    """从 PDF 目录加载所有文档（PyMuPDF + 多进程）"""
    reader = SimpleDirectoryReader(
        input_dir=pdf_dir,
        file_extractor={".pdf": PyMuPDFReader()}
    )
    documents = reader.load_data(num_workers=1, show_progress=True)

    # 清理非法 surrogate 字符，防止 UTF-8 编码报错
    for doc in documents:
        clean = doc.text.encode("utf-8", errors="surrogatepass").decode(
            "utf-8", errors="replace"
        )
        doc.set_content(clean)

    print(f"✅ 已加载 {len(documents)} 个文档块")
    return documents


# ===== 索引管理 =====
def build_index(config):
    """构建 FAISS 向量索引并持久化"""
    documents = load_documents(config.PDF_DIR)

    faiss_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )

    storage_dir = getattr(config, "STORAGE_DIR", "storage")
    os.makedirs(storage_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"✅ 索引已保存到 {storage_dir}/")
    return index


def load_index(storage_dir):
    """从磁盘加载已有索引"""
    vector_store = FaissVectorStore.from_persist_dir(storage_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=storage_dir
    )
    index = load_index_from_storage(storage_context)
    print("✅ 已加载已有索引")
    return index


# ===== 混合检索 + 重排序 =====
def build_hybrid_retriever(index, config):
    """构建混合检索器：向量检索 + BM25 关键词检索"""
    vector_retriever = index.as_retriever(similarity_top_k=config.VECTOR_TOP_K)

    all_nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever(all_nodes, top_k=config.BM25_TOP_K)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        num_queries=1,
        use_async=False,
        similarity_top_k=config.VECTOR_TOP_K + config.BM25_TOP_K,
    )
    print("✅ 混合检索器已构建（向量 + BM25）")
    return hybrid_retriever


def get_reranker(config):
    """创建重排序后处理器"""
    use_reranker = getattr(config, "USE_RERANKER", True)
    if not use_reranker:
        print("ℹ️ 未启用重排序模型 (USE_RERANKER=False)")
        return None

    reranker = SentenceTransformerRerank(
        model=config.RERANK_MODEL,
        top_n=config.RERANK_TOP_N,
    )
    print("✅ 重排序模型已加载")
    return reranker
