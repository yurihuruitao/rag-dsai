# configs/exp2_no_reranker.py
DEEPSEEK_API_KEY = "sk-550759a782c941e3b59a47bcd6f8e2aa"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}
DEFAULT_MODEL = "deepseek-reasoner"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Experiment settings
EXP_NAME = "no_reranker"
INDEX_SOURCE_EXP = "baseline"

# Toggle Reranker off
USE_RERANKER = False

# We still set these so the script won't crash when loading imports,
# but they won't be used because USE_RERANKER is False.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = 3

PDF_DIR = "bench_pdf"

# chunk_strategy can be "sentence", "paragraph", or "character"
CHUNK_STRATEGY = "sentence"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

VECTOR_TOP_K = 5
BM25_TOP_K = 5

CHAT_MEMORY_TOKEN_LIMIT = 3000

BENCH_QUERIES = "openragbench/pdf/arxiv/queries.json"
BENCH_ANSWERS = "openragbench/pdf/arxiv/answers.json"
BENCH_OUTPUT = f"results/bench_{EXP_NAME}.json"
EVAL_OUTPUT = f"results/eval_{EXP_NAME}.json"

SYSTEM_PROMPT = """You are a professional document QA assistant. Answer the user's question based on the retrieved document content.

Requirements:
1. Answer strictly based on the provided document content, do not fabricate information
2. If there is no relevant information in the documents, clearly inform the user
3. Keep answers well-structured, use bullet points when necessary
4. Answer in English"""
