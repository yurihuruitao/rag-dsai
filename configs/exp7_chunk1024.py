import os
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}
DEFAULT_MODEL = "deepseek-reasoner"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

USE_RERANKER = True
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = 3

PDF_DIR = "ourbench/pdf"

# Experiment settings
EXP_NAME = "chunk1024"
STORAGE_DIR = f"storage/{EXP_NAME}"

# chunk_strategy can be "sentence", "paragraph", or "token"
CHUNK_STRATEGY = "sentence"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

VECTOR_TOP_K = 5
BM25_TOP_K = 5

CHAT_MEMORY_TOKEN_LIMIT = 3000

BENCH_QUERIES = "ourbench/Q&A/queries_filtered.json"
BENCH_ANSWERS = "ourbench/Q&A/answers_filtered.json"
BENCH_OUTPUT = f"results/bench_{EXP_NAME}.json"
EVAL_OUTPUT = f"results/eval_{EXP_NAME}.json"

SYSTEM_PROMPT = """You are a professional document QA assistant. Answer the user's question based on the retrieved document content.

Requirements:
1. Answer strictly based on the provided document content, do not fabricate information
2. If there is no relevant information in the documents, clearly inform the user
3. Keep answers well-structured, use bullet points when necessary
4. Answer in English"""
