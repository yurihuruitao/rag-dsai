# RAG 文档问答系统

基于 LlamaIndex 的 PDF 文档检索增强生成（RAG）系统，支持向量检索 + BM25 混合检索 + 重排序。

## 环境准备

```bash
pip install -r requirements.txt
```

## 配置

实验配置文件存放在 `configs/` 目录下（例如 `configs/exp1_baseline.py`）。每个实验拥有独立的配置，支持动态调整重排序模型、分块策略等。

关键参数说明：

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `EXP_NAME` | **必填**。实验名称，如 `"baseline"`。系统会自动生成对应的存储路径 `storage/{EXP_NAME}` | `"baseline"` |
| `INDEX_SOURCE_EXP` | **选填**。如果只想复用其他实验建好的索引库（例如跨模型评测同一批文本），设置为该实验的名称 | `"baseline"` |
| `CHUNK_STRATEGY` | 文本切块策略，支持 `"sentence"`, `"paragraph"`, 或 `"token"` | `"Sentence"` |
| `USE_RERANKER` | 是否开启重排序模型 (`True` 或 `False`) | `True` |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-xxx` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-m3` |
| `RERANK_MODEL` | 重排序模型 | `BAAI/bge-reranker-v2-m3` |

### 文本分块策略 (CHUNK_STRATEGY)

系统目前在 `rag.py` 中实现了三种不同的文本切分（Chunking）策略，以适配不同的检索需求：

1. **`"sentence"`**（默认策略）：
   - **代码实现**：使用 LlamaIndex 的 `SentenceSplitter`。
   - **策略说明**：优先按照单句边界进行文本切分。它会在满足 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 长度限制的前提下，尽量保证一句话的完整性，适合大多数常规文本检索场景。

2. **`"paragraph"`**：
   - **代码实现**：使用 `SentenceSplitter` 并指定强制段落分隔符 `paragraph_separator="\n\n"`。
   - **策略说明**：优先按照段落（连续两个换行符）进行文本分块。这种策略能够最大程度保留单个段落上下文的连贯性，对于排版规范、段落逻辑严密（如 PDF 论文）的文档非常有效。

3. **`"token"`**：
   - **代码实现**：使用 LlamaIndex 的 `TokenTextSplitter`，以空格 `separator=" "` 为主要切分符，换行符 `backup_separators=["\n"]` 为备用切分符。
   - **策略说明**：严格按照 Token（或词元边界）进行硬切分。这种方式能最精确地控制每个分块的大小上限，但在切分时可能会破坏句子的物理结构甚至截断语义。

## 使用流程

### 🚀 一键运行完整评测流水线

我们提供了一个串联脚本 `run_pipeline.py`，可以一键执行：**构建索引 -> Benchmark测试 -> 模型打分评测**的完整流程：

```bash
# 基础用法：只需要指定配置文件
python run_pipeline.py --config configs/exp1_baseline.py

# 进阶用法：支持限制条数、并发数、强制重建索引等
python run_pipeline.py --config configs/exp1_baseline.py \
    --rebuild \
    --limit 10 \
    --workers 16 \
    --no-resume
```

---

如果你想**分步进行调试和执行**，请参考以下单步说明：

### 1. 下载测试 PDF

从 OpenRAGBench 下载 benchmark 所需的 PDF 文件：

```bash
python download_pdfs.py
```

PDF 文件将保存到 `ourbench/pdf/` 目录。

### 2. 构建向量索引

将 PDF 文档解析、分块并根据指定的配置文件构建 FAISS 向量索引：

```bash
python build_index.py --config configs/exp1_baseline.py
```

如需**重建索引**（删除旧索引后重新构建）：

```bash
python build_index.py --config configs/exp1_baseline.py --rebuild
```

索引文件保存在 `storage/` 目录。

### 3. 运行 Benchmark 测试

使用 OpenRAGBench 数据集对该配置进行自动问答评测：

```bash
# 完整测试
python benchmark.py --config configs/exp1_baseline.py

# 只测前 10 条（快速验证）
python benchmark.py --config configs/exp1_baseline.py --limit 10

# 指定并发数和输出路径（默认自动保存为 results/bench_{EXP_NAME}.json）
python benchmark.py --config configs/exp1_baseline.py --workers 4
```

> **断点续传**：默认开启。如果测试中断，重新运行同样的命令即可自动跳过已完成的查询，继续测试剩余部分。如需从头开始，加 `--no-resume`。

结果保存到 `bench_results.json`。

### 4. 评分评测

使用 DeepSeek 对 RAG 回答进行自动打分（0-10 分）：

```bash
python evaluate.py --config configs/exp1_baseline.py --workers 8
```

评测完成后会输出平均得分，结果默认保存到 `results/eval_{EXP_NAME}.json`。

## 项目结构

```text
├── rag.py              # 核心模块：PDF 加载、索引、混合检索、重排序
├── run_pipeline.py     # 完整流水线 (一键构建、测试、打分)
├── build_index.py      # 构建向量索引
├── benchmark.py        # Benchmark 自动测试
├── evaluate.py         # LLM 自动评分
├── download_pdfs.py    # 下载测试 PDF
├── configs/            # 存放各个实验的配置文件
│   ├── exp1_baseline.py
│   └── ...
├── requirements.txt    # 依赖列表
├── ourbench/           # 测试数据集 (PDFs和Q&A)
├── storage/            # 向量索引存储 (按实验分组)
└── results/            # Benchmark 和评测输出 (按实验分组)
```
