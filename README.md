# RAG 文档问答评测系统

基于 LlamaIndex 的 PDF 文档检索增强生成（RAG）评测系统，支持向量检索 + BM25 混合检索 + 重排序，用于比较不同分块策略和检索配置的效果。

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

下载测试 PDF（一次性，下载到 `ourbench/pdf/`）：

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

# 无 RAG 基准对照
python pipeline/no_rag_bench.py --config configs/exp1_baseline.py
```

### 统计分析

```bash
python analysis/stats_analysis.py   # 生成统计摘要
python analysis/plot_results.py     # 生成可视化图表
```

输出到 `analysis/output/`。

## 项目结构

```
rag-dsai/
├── pipeline/               # 核心流程脚本
│   ├── rag.py              # 底层模块：加载、索引、混合检索、重排序
│   ├── build_index.py      # 构建 FAISS 向量索引
│   ├── benchmark.py        # 自动问答测试
│   ├── evaluate.py         # LLM 自动打分（0-10 分）
│   ├── run_pipeline.py     # 一键运行完整流水线
│   └── no_rag_bench.py     # 无 RAG 基准对照
├── configs/                # 实验配置文件
│   ├── exp1_baseline.py
│   └── ...
├── analysis/               # 统计分析与可视化
│   ├── stats_analysis.py   # Wilcoxon 检验、Cohen's d、置信区间
│   ├── plot_results.py     # 生成论文质量图表
│   └── output/             # 生成的 CSV、JSON、图表
├── tests/                  # 测试脚本
│   ├── check_env.py        # 环境完整性检查
│   └── test_run.py         # 端到端运行测试
├── download/               # PDF 下载工具
├── ourbench/               # 测试数据集
│   ├── Q&A/                # 821 条查询 + 标准答案
│   └── pdf/                # 500 篇论文 PDF
├── storage/                # FAISS 索引（按实验分目录）
├── results/                # Benchmark 和评测输出
├── logs/                   # 运行日志
├── .env                    # API Keys（不提交到 git）
└── requirements.txt
```

## 实验配置

| 配置文件 | 实验名 | 关键差异 |
|----------|--------|----------|
| `exp1_baseline.py` | `baseline` | Sentence 分块，chunk_size=512，开启重排序 |
| `exp2_no_reranker.py` | `no_reranker` | 同 baseline，关闭重排序；复用 baseline 索引 |
| `exp3_token.py` | `token` | Token 分块策略 |
| `exp4_paragraph.py` | `paragraph` | Paragraph 分块策略 |
| `exp5_chunk256.py` | `chunk256` | 更小分块，chunk_size=256 |
| `exp6_chat_model.py` | `chat_model` | 使用 deepseek-chat；复用 baseline 索引 |
| `exp7_chunk1024.py` | `chunk1024` | 更大分块，chunk_size=1024 |
| `exp8_glm.py` | `glm` | 使用 GLM-4-Plus（智谱 AI）；复用 baseline 索引 |

### 关键配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EXP_NAME` | 实验名称，决定索引和结果文件的路径 | — |
| `CHUNK_STRATEGY` | 分块策略：`"sentence"` / `"paragraph"` / `"token"` | `"sentence"` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小 / 重叠 token 数 | `512` / `50` |
| `VECTOR_TOP_K` / `BM25_TOP_K` | 向量检索 / BM25 检索各取 top-k | `5` / `5` |
| `USE_RERANKER` | 是否启用重排序模型 | `True` |
| `RERANK_TOP_N` | 重排序后保留条数 | `3` |
| `INDEX_SOURCE_EXP` | 复用其他实验的索引（不重新构建） | — |

## 断点续传

Benchmark 和 Evaluate 均支持断点续传，中断后重新运行会自动跳过已完成的查询。如需从头重跑：

```bash
python pipeline/benchmark.py --config configs/exp1_baseline.py --no-resume
```

## 检查环境

```bash
python tests/check_env.py     # 检查依赖、API Key、数据、索引是否完整
python tests/test_run.py      # 实际调用 API 跑通完整流程（自动清理，不留痕迹）
```
