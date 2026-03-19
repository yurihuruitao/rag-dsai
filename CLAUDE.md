# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **RAG (Retrieval-Augmented Generation) evaluation system** built on LlamaIndex. It indexes PDF documents (500 papers from OpenRAGBench), runs QA benchmarks with hybrid retrieval, and scores results using an LLM judge. The primary goal is comparing different chunking strategies and retrieval configurations.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# API keys go in .env (DEEPSEEK_API_KEY and GLM_API_KEY)
```

Download PDFs once before running (downloads to `ourbench/pdf/`):
```bash
python download/download_pdfs.py
```

## Common Commands

### Run full pipeline (build index → benchmark → evaluate)
```bash
python pipeline/run_pipeline.py --config configs/exp1_baseline.py
python pipeline/run_pipeline.py --config configs/exp1_baseline.py --rebuild --limit 10 --workers 16
```

### Run steps individually
```bash
python pipeline/build_index.py --config configs/exp1_baseline.py [--rebuild]
python pipeline/benchmark.py --config configs/exp1_baseline.py [--limit 10] [--workers 8] [--no-resume]
python pipeline/evaluate.py --config configs/exp1_baseline.py [--workers 8]
python pipeline/no_rag_bench.py --config configs/exp1_baseline.py  # baseline without RAG
```

### Analyze results
```bash
python analysis/stats_analysis.py   # outputs to analysis/output/
python analysis/plot_results.py     # outputs figures to analysis/output/figures/
```

### Check environment and run tests
```bash
python tests/check_env.py   # verify environment completeness
python tests/test_run.py    # end-to-end pipeline test (no traces left)
```

## Architecture

### Directory Structure
```
rag-dsai/
├── pipeline/               # Core pipeline scripts
│   ├── rag.py              # Shared RAG logic (chunking, indexing, retrieval)
│   ├── build_index.py      # Build FAISS vector indexes
│   ├── benchmark.py        # Run QA benchmark
│   ├── evaluate.py         # Score answers with LLM judge
│   ├── run_pipeline.py     # Orchestrate full pipeline
│   └── no_rag_bench.py     # No-RAG baseline
├── configs/                # Experiment configuration files
├── analysis/               # Statistical analysis and visualization
│   ├── stats_analysis.py
│   ├── plot_results.py
│   └── output/             # Generated CSVs, JSONs, figures
├── tests/                  # Environment check and E2E test
├── download/               # PDF download utilities
├── ourbench/               # Benchmark dataset (PDFs and Q&A)
├── storage/                # Persisted FAISS indexes (one dir per experiment)
├── results/                # Benchmark and evaluation outputs
└── logs/                   # Run logs
```

### Data Flow
```
PDFs → pipeline/build_index.py → storage/{EXP_NAME}/      (FAISS + BM25 indexes)
                                          ↓
       pipeline/benchmark.py  → results/bench_{EXP_NAME}.json  (responses + timing)
                                          ↓
       pipeline/evaluate.py   → results/eval_{EXP_NAME}.json   (0-10 LLM scores)
                                          ↓
       analysis/stats_analysis.py → analysis/output/           (stats + visualization)
```

### Core Module: `pipeline/rag.py`
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
| `exp6_chat_model.py` | `deepseek-chat` instead of `deepseek-reasoner`; reuses baseline index |
| `exp7_chunk1024.py` | chunk_size=1024 (larger chunks) |
| `exp8_glm.py` | GLM-4-Plus model (Zhipu AI); reuses baseline index |

## Key Paths
- `ourbench/Q&A/queries_filtered.json` — 821 test queries
- `ourbench/Q&A/answers_filtered.json` — ground truth answers
- `storage/{EXP_NAME}/` — persisted FAISS indexes
- `results/bench_{EXP_NAME}.json` — benchmark outputs
- `results/eval_{EXP_NAME}.json` — evaluation scores
- `analysis/output/stats_summary.csv` — experiment comparison summary
