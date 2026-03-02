# test_config.py
DEEPSEEK_API_KEY = "sk-550759a782c941e3b59a47bcd6f8e2aa"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}
DEFAULT_MODEL = "deepseek-reasoner"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = 3

PDF_DIR = "bench_pdf"
STORAGE_DIR = "storage"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

VECTOR_TOP_K = 5
BM25_TOP_K = 5

CHAT_MEMORY_TOKEN_LIMIT = 3000

BENCH_QUERIES = "openragbench/pdf/arxiv/queries.json"
BENCH_ANSWERS = "openragbench/pdf/arxiv/answers.json"
BENCH_OUTPUT = "bench_results.json"

SYSTEM_PROMPT = """You are a professional document QA assistant. Answer the user's question based on the retrieved document content.

Requirements:
1. Answer strictly based on the provided document content, do not fabricate information
2. If there is no relevant information in the documents, clearly inform the user
3. Keep answers well-structured, use bullet points when necessary
4. Answer in English"""
