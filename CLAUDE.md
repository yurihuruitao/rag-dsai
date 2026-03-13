# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **RAG (Retrieval-Augmented Generation) evaluation system** built on LlamaIndex. It indexes PDF documents (85 papers from OpenRAGBench), runs QA benchmarks with hybrid retrieval, and scores results using an LLM judge. The primary goal is comparing different chunking strategies and retrieval configurations.

## Setup

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-key-here"  # or add to .env
```

Download PDFs once before running (downloads to `ourbench/pdf/`):
```bash
python download/download_pdfs.py
```

## Common Commands

### Run full pipeline (build index → benchmark → evaluate)
```bash
python run_pipeline.py --config configs/exp1_baseline.py
python run_pipeline.py --config configs/exp1_baseline.py --rebuild --limit 10 --workers 16
```

### Run steps individually
```bash
python build_index.py --config configs/exp1_baseline.py [--rebuild]
python benchmark.py --config configs/exp1_baseline.py [--limit 10] [--workers 8] [--no-resume]
python evaluate.py --config configs/exp1_baseline.py [--workers 8]
python no_rag_bench.py --config configs/exp1_baseline.py  # baseline without RAG
```

### Analyze results across all experiments
```bash
python analyze_eval_results.py --input-dir results --baseline baseline
```
Outputs to `results/analysis/`: CSV summaries, significance tests, and a PNG visualization.

## Architecture

### Data Flow
```
PDFs → build_index.py → storage/{EXP_NAME}/   (FAISS + BM25 indexes)
                              ↓
benchmark.py → results/bench_{EXP_NAME}.json   (responses + timing)
                              ↓
evaluate.py  → results/eval_{EXP_NAME}.json    (0-10 LLM scores)
                              ↓
analyze_eval_results.py → results/analysis/    (stats + visualization)
```

### Core Module: `rag.py`
All shared RAG logic lives here:
- **Chunking strategies**: `sentence` (default, SentenceSplitter), `paragraph` (newline-separated), `token` (TokenTextSplitter) — controlled by `CHUNK_STRATEGY` config key
- **Hybrid retrieval**: Vector search (FAISS + `bge-m3` embeddings) + BM25, fused via QueryFusionRetriever
- **Re-ranking**: Optional `bge-reranker-v2-m3` post-processor controlled by `USE_RERANKER`

### Configuration System
Each experiment is a Python module in `configs/`. Scripts load configs with `--config configs/expN_name.py`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `EXP_NAME` | Experiment ID; used for storage path and output filenames |
| `DEFAULT_MODEL` | `"deepseek-chat"` or `"deepseek-reasoner"` |
| `CHUNK_STRATEGY` | `"sentence"`, `"paragraph"`, or `"token"` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (default: 512/50) |
| `VECTOR_TOP_K` / `BM25_TOP_K` | Retrieval top-k (default: 5 each) |
| `USE_RERANKER` | Boolean; enables sentence-transformer re-ranking |
| `RERANK_TOP_N` | Final results after re-ranking (default: 3) |
| `INDEX_SOURCE_EXP` | Reuse index from another experiment instead of building |

### Benchmark & Evaluate Design
- **Multi-threaded**: Thread pool with thread-local chat engines (prevents context contamination across queries)
- **Checkpoint/resume**: Results saved periodically; re-running continues from where it stopped. Use `--no-resume` to restart
- **Evaluation scoring**: DeepSeek LLM scores each answer 0-10; score extracted via regex from response

### Existing Experiments
| Config | Key Difference |
|--------|---------------|
| `exp1_baseline.py` | Sentence chunking, chunk_size=512, with reranker |
| `exp2_no_reranker.py` | Same as baseline but `USE_RERANKER=False`; reuses baseline index |
| `exp3_token.py` | Token-based chunking |
| `exp4_paragraph.py` | Paragraph-based chunking |
| `exp5_chunk256.py` | chunk_size=256 (smaller chunks) |
| `exp6_chat_model.py` | `deepseek-chat` instead of `deepseek-reasoner` |

## Key Paths
- `ourbench/Q&A/queries_filtered.json` — 821 test queries
- `ourbench/Q&A/answers_filtered.json` — ground truth answers
- `storage/{EXP_NAME}/` — persisted FAISS indexes
- `results/bench_{EXP_NAME}.json` — benchmark outputs
- `results/eval_{EXP_NAME}.json` — evaluation scores
