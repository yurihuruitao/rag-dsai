# RAG-DSAI: RAG Evaluation System for Document QA

A retrieval-augmented generation (RAG) evaluation framework built with LlamaIndex for academic paper question answering. It systematically compares how chunking strategies, retrieval settings, and generation models affect answer quality.

Chinese documentation is available in [README_ZH.md](/workspace/projects/rag-dsai/README_ZH.md).

## System Architecture

```text
OurBench (500 PDFs + 821 QA pairs)
        |
        v
+- Stage 1: Index Construction --------------------------------+
| PDF -> PyMuPDF parsing -> Text chunking -> BGE-M3 embedding  |
|                     |                    |                    |
|                     v                    v                    |
|              FAISS vector index    BM25 inverted index       |
+--------------------------------------------------------------+
        |
        v
+- Stage 2: Benchmarking --------------------------------------+
| 821 queries -> Hybrid retrieval (Dense+Sparse) -> Rerank     |
| (optional) -> LLM answer generation -> bench_{EXP}.json      |
+--------------------------------------------------------------+
        |
        v
+- Stage 3: Automatic Evaluation ------------------------------+
| bench_{EXP}.json -> deepseek-reasoner scoring (0-10)         |
|                   -> eval_{EXP}.json                         |
+--------------------------------------------------------------+
        |
        v
Statistical analysis (Wilcoxon / Bootstrap CI / Cohen's d)
```

Detailed methodology and diagrams: [`report/methodology.md`](/workspace/projects/rag-dsai/report/methodology.md)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your API keys:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
GLM_API_KEY=your-glm-api-key   # only required for exp8_glm
```

Download the OurBench corpus once. This fetches 500 PDFs into `ourbench/pdf/`:

```bash
python download/download_pdfs.py
```

## Quick Start

### Run the full pipeline

```bash
python pipeline/run_pipeline.py --config configs/exp1_baseline.py
```

Supported options:

```bash
python pipeline/run_pipeline.py --config configs/exp1_baseline.py \
    --rebuild       # force index rebuild
    --limit 10      # benchmark only the first 10 queries
    --workers 16    # number of concurrent workers
    --no-resume     # disable checkpoint resume and restart from scratch
```

### Run step by step

```bash
# 1. Build the vector index
python pipeline/build_index.py --config configs/exp1_baseline.py [--rebuild]

# 2. Run QA benchmarking
python pipeline/benchmark.py --config configs/exp1_baseline.py [--limit 10] [--workers 8]

# 3. LLM-based evaluation
python pipeline/evaluate.py --config configs/exp1_baseline.py [--workers 8]

# No-RAG baseline (direct LLM answering without retrieval)
python pipeline/no_rag_bench.py --config configs/exp1_baseline.py
```

### Statistical analysis and plotting

```bash
python analysis/stats_analysis.py   # writes statistical summaries to analysis/output/
python analysis/plot_results.py     # writes publication-style figures to analysis/output/figures/
```

## OurBench Dataset

OurBench is a self-built academic paper QA benchmark. The complete dataset construction notebook is [`Dataset_Creation_Code.ipynb`](/workspace/projects/rag-dsai/Dataset_Creation_Code.ipynb).

| Item | Count |
|------|-------|
| Academic paper PDFs | 500 |
| Raw QA pairs | 1,000 |
| Filtered QA pairs | 821 |
| Extractive questions | 520 (63.3%) |
| Abstractive questions | 301 (36.7%) |

### Construction Pipeline

OurBench is built in five stages, all executed in Google Colab:

```text
Stage 1                Stage 2              Stage 3              Stage 4              Stage 5
Document Collection -> Document Processing -> Content Segmentation -> QA Generation -> Quality Filtering
   (arXiv API)           (Mistral OCR)         (Markdown -> JSON)     (DeepSeek)        (Rules + LLM)
```

#### Stage 1: Document Collection

Paper metadata is retrieved in batches through the arXiv API by subject category such as `cs.CL`, and PDFs are downloaded. Results are sorted by `lastUpdatedDate` in descending order, with pagination and retry support built in.

#### Stage 2: Document Processing

Mistral OCR (`pixtral`) converts PDFs into structured Markdown while preserving headings, equations, tables, and similar structure. Each paper is exported as an individual JSON file, with 4-way parallel processing support.

#### Stage 3: Content Segmentation

The Markdown produced by Mistral is normalized into a standard corpus format:

- Split content into sections by heading hierarchy
- Extract and associate tables and figure captions
- Produce structured `{paper_id}.json` corpus files and `pdf_urls.json` as the PDF URL index

#### Stage 4: QA Pair Generation

1,000 QA pairs are generated from the 500-paper corpus:

1. Corpus split: randomly divide the papers into 200 positive papers for QA generation and 300 hard negative papers for retrieval interference
2. Section sampling: sample sections from positive papers with balanced coverage across content types (`text` / `text-table`)
3. LLM generation: use `deepseek-chat` to generate one QA pair per section with JSON-constrained output:

- Question: 10 to 25 words, answerable only from the section
- Answer: one sentence for extractive questions, 2 to 4 sentences for abstractive questions
- Labels: `query_type` (`abstractive` / `extractive`) and `source_type` (`text` / `text-table`)

Target distribution: about 450 abstractive + 550 extractive, with 800 text and 200 text-table examples.

#### Stage 5: Quality Filtering

Two-stage filtering reduces the 1,000 generated pairs to 821:

Stage 5a, rule-based filtering:

- Question length check (8 to 40 words)
- Minimum answer length check (at least 5 words)
- Non-retrievable phrase filtering, such as "this section" or "the above text"
- Model refusal detection, such as "I cannot" or "as an AI"
- Fuzzy deduplication with RapidFuzz, where similarity >= 90% counts as duplicate

Stage 5b, LLM-based filtering with DeepSeek API:

- Score each QA pair on three dimensions from 1 to 3
- Clarity: whether the question is clear and unambiguous
- Accuracy: whether the answer is supported by the source
- Retrievability: whether the source section can be located from the question
- Threshold: all three scores must be >= 2

### Data Files

- `ourbench/Q&A/queries_filtered.json`: filtered queries with query, type, and source fields
- `ourbench/Q&A/answers_filtered.json`: reference answers
- `ourbench/Q&A/qrels_filtered.json`: query-document mappings with `doc_id` and `section_id`
- `download/pdf_urls.json`: source PDF URLs

## Experiment Configurations

There are eight controlled experiments, each changing only one variable:

| Config File | Experiment | Independent Variable | Changed Value | Index |
|-------------|------------|----------------------|---------------|-------|
| `exp1_baseline.py` | `baseline` | - | Sentence/512/Reranker/Reasoner | dedicated |
| `exp2_no_reranker.py` | `no_reranker` | Reranking | `USE_RERANKER=False` | reuse baseline |
| `exp3_token.py` | `token` | Chunking strategy | `CHUNK_STRATEGY="token"` | dedicated |
| `exp4_paragraph.py` | `paragraph` | Chunking strategy | `CHUNK_STRATEGY="paragraph"` | dedicated |
| `exp5_chunk256.py` | `chunk256` | Chunk size | `CHUNK_SIZE=256` | dedicated |
| `exp6_chat_model.py` | `chat_model` | Generation model | `DEFAULT_MODEL="deepseek-chat"` | reuse baseline |
| `exp7_chunk1024.py` | `chunk1024` | Chunk size | `CHUNK_SIZE=1024` | dedicated |
| `exp8_glm.py` | `glm` | Generation model | `DEFAULT_MODEL="glm-4-plus"` | reuse baseline |

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EXP_NAME` | Experiment ID used in index and result paths | - |
| `DEFAULT_MODEL` | Generation model | `"deepseek-reasoner"` |
| `CHUNK_STRATEGY` | Chunking strategy: `"sentence"` / `"paragraph"` / `"token"` | `"sentence"` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk size and overlap token count | `512` / `50` |
| `VECTOR_TOP_K` / `BM25_TOP_K` | Top-k values for vector and BM25 retrieval | `5` / `5` |
| `USE_RERANKER` | Whether to enable BGE-Reranker-v2-M3 reranking | `True` |
| `RERANK_TOP_N` | Number of chunks kept after reranking | `3` |
| `INDEX_SOURCE_EXP` | Reuse the index from another experiment | - |

## Results Overview

| Experiment | Mean Score | 95% CI | vs Baseline (p) | Cohen's d | Avg Latency |
|------------|------------|--------|-----------------|-----------|-------------|
| Baseline (Sentence-512, Reranker) | 8.20 | [7.96, 8.44] | - | - | 15.3s |
| No Reranker | 7.10 | [6.81, 7.37] | p<0.001 | -0.284 | 28.9s |
| Chunk-256 | 7.81 | [7.56, 8.06] | p<0.001 | -0.109 | 27.0s |
| Chunk-1024 | 7.34 | [7.07, 7.62] | p<0.001 | -0.224 | 20.7s |
| Token Chunking | 8.07 | [7.83, 8.31] | p=0.094 | -0.036 | 18.2s |
| Paragraph Chunking | 8.20 | [7.95, 8.44] | p=0.948 | -0.001 | 24.8s |
| DeepSeek-Chat | 8.16 | [7.92, 8.40] | p=0.989 | -0.012 | 14.2s |
| GLM-4 | 8.02 | [7.77, 8.26] | p=0.003 | -0.051 | 7.0s |

Key findings:

- Reranking is the most critical component; disabling it causes a statistically significant drop in score
- Sentence and paragraph chunking perform similarly, and token chunking shows no significant difference from baseline
- A chunk size of 512 performs best; both 256 and 1024 reduce quality significantly
- GLM-4 is the fastest model at 7.0 seconds average latency, with only a small quality tradeoff

Full analysis report: [`report/summary.pdf`](/workspace/projects/rag-dsai/report/summary.pdf)

## Core Modules

### `pipeline/rag.py`: low-level RAG engine

This module contains all shared logic:

- Document loading: PyMuPDF parsing and UTF-8 cleanup
- Chunking strategies: `SentenceSplitter`, `TokenTextSplitter`, and paragraph-based chunking
- Vector index: FAISS `IndexFlatL2` (1024 dimensions, BGE-M3)
- Hybrid retrieval: `QueryFusionRetriever` combining dense retrieval with a custom `BM25Retriever`
- Reranking: `SentenceTransformerRerank` with BGE-Reranker-v2-M3

### Models Used

| Component | Model | Source |
|-----------|-------|--------|
| Text embedding | `BAAI/bge-m3` (1024 dims) | local HuggingFace |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | local HuggingFace |
| Generation model (baseline) | `deepseek-reasoner` | DeepSeek API |
| Generation model (variant) | `deepseek-chat` | DeepSeek API |
| Generation model (variant) | `glm-4-plus` | Zhipu AI API |
| Evaluation judge | `deepseek-reasoner` | DeepSeek API |

## Checkpoint Resume

Both benchmark and evaluation support checkpoint resume. Progress is auto-saved every 10 queries and 16 queries respectively. Re-running the command skips completed items automatically. To restart from scratch:

```bash
python pipeline/benchmark.py --config configs/exp1_baseline.py --no-resume
```

## Project Structure

```text
rag-dsai/
├── pipeline/                  # Core pipeline scripts
│   ├── rag.py                 # Low-level modules: loading, indexing, hybrid retrieval, reranking
│   ├── build_index.py         # Build the FAISS vector index
│   ├── benchmark.py           # Multithreaded QA benchmarking
│   ├── evaluate.py            # LLM-based scoring from 0 to 10
│   ├── run_pipeline.py        # One-command full pipeline runner
│   └── no_rag_bench.py        # No-RAG baseline
├── configs/                   # Eight experiment config files
├── analysis/                  # Statistical analysis and visualization
│   ├── stats_analysis.py      # Wilcoxon / Bootstrap CI / Cohen's d / Mann-Whitney U
│   ├── plot_results.py        # Seven publication-style figures
│   └── output/                # Generated CSV, JSON, and figures
├── report/                    # Summary reports used for the paper
│   ├── methodology.md         # Methodology chapter
│   ├── summary.pdf / .tex     # Analysis report
│   ├── stats_summary.csv      # Statistical comparison summary
│   └── figures/               # Methodology diagrams and analysis charts
├── tests/                     # Environment checks and end-to-end tests
├── download/                  # PDF download utilities
├── ourbench/                  # OurBench dataset
│   ├── Q&A/                   # 821 queries, reference answers, and mappings
│   └── pdf/                   # 500 academic paper PDFs
├── Dataset_Creation_Code.ipynb # OurBench dataset construction code
├── storage/                   # Persisted FAISS indexes by experiment
├── results/                   # Benchmark and evaluation JSON outputs
└── requirements.txt
```

## Environment Checks

```bash
python tests/check_env.py     # Check dependencies, API keys, data, and indexes
python tests/test_run.py      # End-to-end test with real API calls and cleanup
```
