# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAG evaluation system built on LlamaIndex. Indexes 500 PDF papers (OurBench, our self-constructed benchmark), runs QA benchmarks with hybrid retrieval (FAISS + BM25), and scores results using an LLM judge. The primary goal is comparing different chunking strategies and retrieval configurations across 8 controlled experiments.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# API keys go in .env (DEEPSEEK_API_KEY and GLM_API_KEY)
python download/download_pdfs.py   # one-time PDF download to ourbench/pdf/
```

## Common Commands

```bash
# Full pipeline (build index → benchmark → evaluate)
python pipeline/run_pipeline.py --config configs/exp1_baseline.py
python pipeline/run_pipeline.py --config configs/exp1_baseline.py --rebuild --limit 10 --workers 16

# Individual steps
python pipeline/build_index.py --config configs/exp1_baseline.py [--rebuild]
python pipeline/benchmark.py --config configs/exp1_baseline.py [--limit 10] [--workers 8] [--no-resume]
python pipeline/evaluate.py --config configs/exp1_baseline.py [--workers 8]
python pipeline/no_rag_bench.py --config configs/exp1_baseline.py  # baseline without RAG

# Analysis
python analysis/stats_analysis.py   # outputs to analysis/output/
python analysis/plot_results.py     # outputs figures to analysis/output/figures/

# Tests
python tests/check_env.py   # verify environment completeness
python tests/test_run.py    # end-to-end pipeline test (cleans up after itself)
```

## Architecture

### Data Flow

```
PDFs → build_index.py → storage/{EXP_NAME}/      (FAISS + BM25 indexes)
                                  ↓
       benchmark.py  → results/bench_{EXP_NAME}.json  (responses + timing)
                                  ↓
       evaluate.py   → results/eval_{EXP_NAME}.json   (0-10 LLM scores)
                                  ↓
       stats_analysis.py → analysis/output/           (stats + visualization)
```

### Core Module: `pipeline/rag.py`

All shared RAG logic lives here — this is the most important file to understand:
- **Chunking strategies**: `sentence` (SentenceSplitter), `paragraph` (newline-separated), `token` (TokenTextSplitter) — controlled by `CHUNK_STRATEGY` config key
- **Hybrid retrieval**: Vector search (FAISS + `bge-m3` embeddings) + BM25, fused via `QueryFusionRetriever`
- **Re-ranking**: Optional `bge-reranker-v2-m3` post-processor controlled by `USE_RERANKER`
- **`BM25Retriever` class**: Custom retriever wrapping `rank_bm25.BM25Okapi`, implements LlamaIndex's `BaseRetriever` interface

### Configuration System

Each experiment is a Python module in `configs/`. Scripts load configs with `--config configs/expN_name.py` and dynamically import them via `importlib`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `EXP_NAME` | Experiment ID; determines storage path and output filenames |
| `DEFAULT_MODEL` | `"deepseek-chat"`, `"deepseek-reasoner"`, or `"glm-4-plus"` |
| `CHUNK_STRATEGY` | `"sentence"`, `"paragraph"`, or `"token"` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (default: 512/50) |
| `VECTOR_TOP_K` / `BM25_TOP_K` | Retrieval top-k (default: 5 each) |
| `USE_RERANKER` | Boolean; enables cross-encoder re-ranking |
| `INDEX_SOURCE_EXP` | Reuse index from another experiment (e.g., `"baseline"`) |

### Concurrency & Resume Design

- **benchmark.py** and **evaluate.py** use `ThreadPoolExecutor` with thread-local chat engines to prevent context contamination
- Results checkpoint periodically (every 10 queries for benchmark, every 16 for evaluate)
- Re-running automatically resumes from checkpoint; use `--no-resume` to restart

### Experiments

Experiments E2, E6, E8 reuse the baseline index (`INDEX_SOURCE_EXP="baseline"`) since they only vary retrieval/generation settings. E3, E4, E5, E7 build their own indexes due to different chunking configs.

## Key Paths

- `ourbench/Q&A/queries_filtered.json` — 821 test queries (520 extractive, 301 abstractive)
- `ourbench/Q&A/answers_filtered.json` — ground truth answers
- `ourbench/Q&A/qrels_filtered.json` — query-to-document mapping (doc_id + section_id)
- `storage/{EXP_NAME}/` — persisted FAISS indexes
- `results/bench_{EXP_NAME}.json` — benchmark outputs
- `results/eval_{EXP_NAME}.json` — evaluation scores
- `analysis/output/stats_summary.csv` — experiment comparison summary
- `methodology.md` — paper methodology section with generated flowcharts in `methodology_figures/`
