# RAG-DSAI: RAG 文档问答评测系统

基于 LlamaIndex 构建的检索增强生成（RAG）评测框架，针对学术论文问答场景，系统性比较不同分块策略、检索配置与生成模型对回答质量的影响。

## 系统架构

```
OurBench (500 PDF + 821 QA)
        │
        ▼
┌─ Stage 1: 索引构建 ─────────────────────────────────┐
│  PDF → PyMuPDF 解析 → 文本分块 → BGE-M3 编码        │
│                    ↓                ↓                │
│            FAISS 向量索引      BM25 倒排索引          │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ Stage 2: 基准测试 ─────────────────────────────────┐
│  821 条查询 → 混合检索(Dense+Sparse) → 重排序(可选)  │
│            → LLM 生成回答 → bench_{EXP}.json         │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ Stage 3: 自动评测 ─────────────────────────────────┐
│  bench_{EXP}.json → deepseek-reasoner 打分(0-10)    │
│                   → eval_{EXP}.json                  │
└─────────────────────────────────────────────────────┘
        │
        ▼
  统计分析 (Wilcoxon / Bootstrap CI / Cohen's d)
```

**详细方法论与流程图**: 见 [`report/methodology.md`](report/methodology.md)

## 环境准备

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

在根目录创建 `.env` 文件，填入 API Key：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
GLM_API_KEY=your-glm-api-key   # 仅 exp8_glm 需要
```

下载 OurBench 语料（一次性，500 篇 PDF 下载到 `ourbench/pdf/`）：

```bash
python download/download_pdfs.py
```

## 快速开始

### 一键运行完整流水线

```bash
python pipeline/run_pipeline.py --config configs/exp1_baseline.py
```

支持的参数：

```bash
python pipeline/run_pipeline.py --config configs/exp1_baseline.py \
    --rebuild       # 强制重建索引
    --limit 10      # 只测前 10 条（快速验证）
    --workers 16    # 并发线程数
    --no-resume     # 不使用断点续传，从头开始
```

### 分步执行

```bash
# 1. 构建向量索引
python pipeline/build_index.py --config configs/exp1_baseline.py [--rebuild]

# 2. 运行问答测试
python pipeline/benchmark.py --config configs/exp1_baseline.py [--limit 10] [--workers 8]

# 3. LLM 评分
python pipeline/evaluate.py --config configs/exp1_baseline.py [--workers 8]

# 无 RAG 基准对照（直接用 LLM 回答，不检索）
python pipeline/no_rag_bench.py --config configs/exp1_baseline.py
```

### 统计分析与可视化

```bash
python analysis/stats_analysis.py   # 输出统计摘要到 analysis/output/
python analysis/plot_results.py     # 输出论文质量图表到 analysis/output/figures/
```

## OurBench 数据集

自建学术论文问答基准，完整构建代码见 [`Dataset_Creation_Code.ipynb`](Dataset_Creation_Code.ipynb)。

| 项目 | 数量 |
|------|------|
| 学术论文 PDF | 500 篇 |
| 原始 QA 对 | 1,000 条 |
| 过滤后 QA 对 | 821 条 |
| 抽取式问题 (extractive) | 520 条 (63.3%) |
| 推理式问题 (abstractive) | 301 条 (36.7%) |

### 构建流程

OurBench 的构建分为 5 个阶段，全部在 Google Colab 中运行：

```
Stage 1                Stage 2              Stage 3              Stage 4              Stage 5
Document Collection → Document Processing → Content Segmentation → QA Generation → Quality Filtering
   (arXiv API)        (Mistral OCR)         (Markdown → JSON)    (DeepSeek)        (规则 + LLM)
```

#### Stage 1: Document Collection — 论文采集

通过 **arXiv API** 按学科分类（如 `cs.CL`）批量检索论文元数据，下载 PDF 文件。按 `lastUpdatedDate` 降序排列，支持分页抓取，内置重试机制。

#### Stage 2: Document Processing — 文档解析

使用 **Mistral OCR**（`pixtral` 视觉模型）将 PDF 转换为结构化 Markdown 文本，保留标题层级、公式、表格等结构信息。每篇论文输出为独立 JSON 文件，支持 4 路并行处理。

#### Stage 3: Content Segmentation — 内容分段

将 Mistral 输出的 Markdown 解析为标准化语料格式：
- 按标题层级切分为独立 section
- 提取并关联表格、图片描述
- 生成 `{paper_id}.json` 结构化语料 + `pdf_urls.json` 下载链接索引

#### Stage 4: QA Pairs Generation — 问答对生成

从 500 篇论文中生成 1,000 条 QA 对：

1. **语料拆分**: 随机将 500 篇论文分为 200 篇 positive（用于生成 QA）+ 300 篇 hard negative（用于检索干扰）
2. **段落采样**: 从 positive 论文中按内容类型（text / text-table）平衡采样 section
3. **LLM 生成**: 使用 `deepseek-chat` 为每个 section 生成一组 QA 对，通过 JSON 格式约束输出：
   - **问题**: 10-25 词，必须仅基于该 section 可回答
   - **答案**: extractive 类型 1 句话，abstractive 类型 2-4 句话
   - **分类标注**: `query_type`（abstractive/extractive）+ `source_type`（text/text-table）

目标分布：~450 abstractive + ~550 extractive，800 text + 200 text-table。

#### Stage 5: Quality Filtering — 质量过滤

两阶段过滤，从 1,000 条筛选至 821 条：

**Stage 5a — 规则过滤**（免费，即时）：
- 问题长度检查（8-40 词）
- 答案最短长度检查（≥5 词）
- 不可检索短语过滤（如 "this section"、"the above text" 等指代表述）
- 模型拒答检测（如 "I cannot"、"as an AI" 等）
- 模糊去重（RapidFuzz，相似度 ≥ 90% 视为重复）

**Stage 5b — LLM 过滤**（DeepSeek API）：
- 对每条 QA 对在 3 个维度上评分（1-3 分）：
  - **Clarity**: 问题是否清晰无歧义
  - **Accuracy**: 答案是否有据可依
  - **Retrievability**: 问题是否能定位到源 section
- 阈值: 三项评分均 ≥ 2 方通过

### 数据文件

- `ourbench/Q&A/queries_filtered.json` — 过滤后查询（含 query、type、source 字段）
- `ourbench/Q&A/answers_filtered.json` — 标准答案
- `ourbench/Q&A/qrels_filtered.json` — 查询-文档映射（doc_id + section_id）
- `download/pdf_urls.json` — 论文 PDF 下载链接

## 实验配置

8 组对照实验，每组仅改变一个变量：

| 配置文件 | 实验名 | 自变量 | 改变值 | 索引 |
|----------|--------|--------|--------|------|
| `exp1_baseline.py` | `baseline` | — | Sentence/512/Reranker/Reasoner | 自建 |
| `exp2_no_reranker.py` | `no_reranker` | 重排序 | `USE_RERANKER=False` | 复用 baseline |
| `exp3_token.py` | `token` | 分块策略 | `CHUNK_STRATEGY="token"` | 自建 |
| `exp4_paragraph.py` | `paragraph` | 分块策略 | `CHUNK_STRATEGY="paragraph"` | 自建 |
| `exp5_chunk256.py` | `chunk256` | 分块大小 | `CHUNK_SIZE=256` | 自建 |
| `exp6_chat_model.py` | `chat_model` | 生成模型 | `DEFAULT_MODEL="deepseek-chat"` | 复用 baseline |
| `exp7_chunk1024.py` | `chunk1024` | 分块大小 | `CHUNK_SIZE=1024` | 自建 |
| `exp8_glm.py` | `glm` | 生成模型 | `DEFAULT_MODEL="glm-4-plus"` | 复用 baseline |

### 关键配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EXP_NAME` | 实验 ID，决定索引/结果文件路径 | — |
| `DEFAULT_MODEL` | 生成模型 | `"deepseek-reasoner"` |
| `CHUNK_STRATEGY` | 分块策略：`"sentence"` / `"paragraph"` / `"token"` | `"sentence"` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小 / 重叠 token 数 | `512` / `50` |
| `VECTOR_TOP_K` / `BM25_TOP_K` | 向量 / BM25 各取 top-k | `5` / `5` |
| `USE_RERANKER` | 是否启用 BGE-Reranker-v2-M3 重排序 | `True` |
| `RERANK_TOP_N` | 重排序后保留条数 | `3` |
| `INDEX_SOURCE_EXP` | 复用其他实验的索引 | — |

## 实验结果概览

| 实验 | 均分 | 95% CI | vs Baseline (p) | Cohen's d | 平均延迟 |
|------|------|--------|-----------------|-----------|---------|
| **Baseline** (Sentence-512, Reranker) | **8.20** | [7.96, 8.44] | — | — | 15.3s |
| No Reranker | 7.10 | [6.81, 7.37] | p<0.001 \*\*\* | -0.284 | 28.9s |
| Chunk-256 | 7.81 | [7.56, 8.06] | p<0.001 \*\*\* | -0.109 | 27.0s |
| Chunk-1024 | 7.34 | [7.07, 7.62] | p<0.001 \*\*\* | -0.224 | 20.7s |
| Token Chunking | 8.07 | [7.83, 8.31] | p=0.094 ns | -0.036 | 18.2s |
| Paragraph Chunking | 8.20 | [7.95, 8.44] | p=0.948 ns | -0.001 | 24.8s |
| DeepSeek-Chat | 8.16 | [7.92, 8.40] | p=0.989 ns | -0.012 | 14.2s |
| GLM-4 | 8.02 | [7.77, 8.26] | p=0.003 \*\* | -0.051 | 7.0s |

**主要发现**：
- 重排序是最关键的组件，关闭后得分显著下降（Cohen's d = -0.284）
- Sentence 和 Paragraph 分块策略效果相当，Token 分块无显著差异
- 分块大小 512 最优，256 和 1024 均显著下降
- GLM-4 速度最快（7.0s），仅有微小质量损失

**完整分析报告**: 见 [`report/summary.pdf`](report/summary.pdf)

## 核心模块说明

### `pipeline/rag.py` — 底层 RAG 引擎

所有共享逻辑的核心模块：
- **文档加载**: PyMuPDF 解析 + UTF-8 清洗
- **分块策略**: SentenceSplitter / TokenTextSplitter / 段落分块
- **向量索引**: FAISS `IndexFlatL2` (1024 维, BGE-M3)
- **混合检索**: `QueryFusionRetriever` 融合向量检索 + 自定义 `BM25Retriever`
- **重排序**: `SentenceTransformerRerank` (BGE-Reranker-v2-M3)

### 使用的模型

| 组件 | 模型 | 来源 |
|------|------|------|
| 文本嵌入 | `BAAI/bge-m3` (1024 维) | HuggingFace 本地 |
| 交叉编码重排序 | `BAAI/bge-reranker-v2-m3` | HuggingFace 本地 |
| 生成模型 (基线) | `deepseek-reasoner` | DeepSeek API |
| 生成模型 (变体) | `deepseek-chat` | DeepSeek API |
| 生成模型 (变体) | `glm-4-plus` | 智谱 AI API |
| 评测裁判 | `deepseek-reasoner` | DeepSeek API |

## 断点续传

Benchmark 和 Evaluate 均支持断点续传（分别每 10 条和 16 条自动保存检查点），中断后重新运行会自动跳过已完成的查询。如需从头重跑：

```bash
python pipeline/benchmark.py --config configs/exp1_baseline.py --no-resume
```

## 项目结构

```
rag-dsai/
├── pipeline/                  # 核心流程脚本
│   ├── rag.py                 # 底层模块：加载、索引、混合检索、重排序
│   ├── build_index.py         # 构建 FAISS 向量索引
│   ├── benchmark.py           # 多线程问答测试（线程隔离 chat engine）
│   ├── evaluate.py            # LLM 自动打分（0-10 分）
│   ├── run_pipeline.py        # 一键运行完整流水线
│   └── no_rag_bench.py        # 无 RAG 基准对照
├── configs/                   # 8 组实验配置文件
├── analysis/                  # 统计分析与可视化
│   ├── stats_analysis.py      # Wilcoxon / Bootstrap CI / Cohen's d / Mann-Whitney U
│   ├── plot_results.py        # 7 张论文质量图表
│   └── output/                # 生成的 CSV、JSON、图表
├── report/                    # 汇总报告（论文用）
│   ├── methodology.md         # 方法论章节
│   ├── summary.pdf / .tex     # 分析报告
│   ├── stats_summary.csv      # 实验对比统计摘要
│   └── figures/               # 方法论流程图 + 分析图表
├── tests/                     # 环境检查与端到端测试
├── download/                  # PDF 下载工具
├── ourbench/                  # OurBench 数据集
│   ├── Q&A/                   # 821 条查询 + 标准答案 + 文档映射
│   └── pdf/                   # 500 篇论文 PDF
├── Dataset_Creation_Code.ipynb # OurBench 数据集构建代码
├── storage/                   # FAISS 索引持久化（按实验分目录）
├── results/                   # Benchmark 和评测输出 JSON
└── requirements.txt
```

## 检查环境

```bash
python tests/check_env.py     # 检查依赖、API Key、数据、索引是否完整
python tests/test_run.py      # 端到端测试（实际调用 API，自动清理）
```
