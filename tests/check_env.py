#!/usr/bin/env python3
"""
环境与仓库完整性检查脚本
运行: python check_env.py
"""

import sys
import os
import json
import importlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"

results = []

def check(label, ok, detail="", warn=False):
    symbol = WARN if (warn and not ok) else (PASS if ok else FAIL)
    line = f"  {symbol}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append((label, ok, warn))

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ── 1. Python 版本 ─────────────────────────────────────────
section("1. Python 环境")
major, minor = sys.version_info[:2]
check("Python 版本 >= 3.10", major == 3 and minor >= 10,
      f"当前 {major}.{minor}")

# ── 2. 依赖包 ──────────────────────────────────────────────
section("2. 依赖包")
PACKAGES = {
    "llama_index.core":                    "llama-index-core",
    "llama_index.readers.file":            "llama-index-readers-file",
    "llama_index.llms.openai_like":        "llama-index-llms-openai-like",
    "llama_index.embeddings.huggingface":  "llama-index-embeddings-huggingface",
    "llama_index.vector_stores.faiss":     "llama-index-vector-stores-faiss",
    "faiss":                               "faiss-cpu",
    "rank_bm25":                           "rank_bm25",
    "sentence_transformers":               "sentence-transformers",
    "dotenv":                              "python-dotenv",
    "openai":                              "openai",
    "tqdm":                                "tqdm",
    "numpy":                               "numpy",
    "pandas":                              "pandas",
    "scipy":                               "scipy",
    "matplotlib":                          "matplotlib",
    "seaborn":                             "seaborn",
    "fitz":                                "PyMuPDF",
}
for module, pkg in PACKAGES.items():
    try:
        importlib.import_module(module)
        check(pkg, True)
    except ImportError as e:
        check(pkg, False, str(e))

# ── 3. 环境变量 / .env ─────────────────────────────────────
section("3. 环境变量")
from dotenv import load_dotenv
load_dotenv()

deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
glm_key = os.environ.get("GLM_API_KEY", "")

check(".env 文件存在", (ROOT / ".env").exists())
check("DEEPSEEK_API_KEY 已设置", bool(deepseek_key),
      f"长度={len(deepseek_key)}" if deepseek_key else "未设置")
check("GLM_API_KEY 已设置", bool(glm_key),
      f"长度={len(glm_key)}" if glm_key else "未设置（exp8_glm 需要）", warn=True)

# ── 4. 核心文件 ────────────────────────────────────────────
section("4. 核心脚本")
CORE_FILES = [
    "pipeline/rag.py",
    "pipeline/build_index.py",
    "pipeline/benchmark.py",
    "pipeline/evaluate.py",
    "pipeline/run_pipeline.py",
    "pipeline/no_rag_bench.py",
    "analysis/stats_analysis.py",
    "analysis/plot_results.py",
]
for f in CORE_FILES:
    check(f, (ROOT / f).exists())

# ── 5. 配置文件 ────────────────────────────────────────────
section("5. 实验配置 (configs/)")
CONFIGS = [
    "exp1_baseline.py",
    "exp2_no_reranker.py",
    "exp3_token.py",
    "exp4_paragraph.py",
    "exp5_chunk256.py",
    "exp6_chat_model.py",
    "exp7_chunk1024.py",
    "exp8_glm.py",
]
for cfg in CONFIGS:
    path = ROOT / "configs" / cfg
    if path.exists():
        # 验证能动态加载
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_cfg", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            required = ["EXP_NAME", "CHUNK_STRATEGY", "CHUNK_SIZE",
                        "VECTOR_TOP_K", "BM25_TOP_K", "BENCH_OUTPUT"]
            missing = [k for k in required if not hasattr(mod, k)]
            if missing:
                check(cfg, False, f"缺少字段: {missing}")
            else:
                check(cfg, True, f"EXP_NAME={mod.EXP_NAME}")
        except Exception as e:
            check(cfg, False, str(e))
    else:
        check(cfg, False, "文件不存在")

# ── 6. Benchmark 数据 ─────────────────────────────────────
section("6. Benchmark 数据 (ourbench/)")
queries_path = ROOT / "ourbench/Q&A/queries_filtered.json"
answers_path = ROOT / "ourbench/Q&A/answers_filtered.json"

check("queries_filtered.json", queries_path.exists())
check("answers_filtered.json", answers_path.exists())

if queries_path.exists() and answers_path.exists():
    queries = json.loads(queries_path.read_text())
    answers = json.loads(answers_path.read_text())
    check("查询数量 == 821", len(queries) == 821, f"实际 {len(queries)}")
    check("答案数量 == 821", len(answers) == 821, f"实际 {len(answers)}")
    q_ids = set(queries.keys())
    a_ids = set(answers.keys())
    check("查询与答案 ID 一致", q_ids == a_ids,
          f"不匹配 {len(q_ids - a_ids)} 条" if q_ids != a_ids else "")

# ── 7. PDF 文件 ────────────────────────────────────────────
section("7. PDF 文件 (ourbench/pdf/)")
pdf_dir = ROOT / "ourbench/pdf"
if pdf_dir.exists():
    pdfs = list(pdf_dir.glob("*.pdf"))
    pdf_count = len(pdfs)
    check("PDF 目录存在", True)
    check("PDF 数量 >= 85",  pdf_count >= 85,  f"当前 {pdf_count} 个")
    # 检查有无空文件
    empty = [p.name for p in pdfs if p.stat().st_size == 0]
    check("无空 PDF 文件", len(empty) == 0,
          f"{len(empty)} 个空文件: {empty[:3]}" if empty else "")
else:
    check("PDF 目录存在", False, "请运行 python download/download_pdfs.py")

# ── 8. Storage 索引 ────────────────────────────────────────
section("8. FAISS 索引 (storage/)")
# 需要独立索引的实验（不复用其他实验）
STANDALONE_EXPS = ["baseline", "token", "paragraph", "chunk256", "chunk1024"]
REUSE_EXPS = {
    "no_reranker": "baseline",
    "chat_model":  "baseline",
    "glm":         "baseline",
}
REQUIRED_FILES = [
    "docstore.json",
    "default__vector_store.json",
    "index_store.json",
]
for exp in STANDALONE_EXPS:
    storage = ROOT / "storage" / exp
    if storage.exists():
        missing_files = [f for f in REQUIRED_FILES if not (storage / f).exists()]
        if missing_files:
            check(f"storage/{exp}", False, f"缺少: {missing_files}")
        else:
            size_mb = sum(f.stat().st_size for f in storage.iterdir()) / 1e6
            check(f"storage/{exp}", True, f"{size_mb:.1f} MB")
    else:
        check(f"storage/{exp}", False,
              f"运行 python build_index.py --config configs/exp_{exp}.py")

for exp, source in REUSE_EXPS.items():
    check(f"storage/{exp} (复用 {source})", (ROOT / "storage" / source).exists(),
          "依赖源索引存在" if (ROOT / "storage" / source).exists() else f"需先建 {source} 索引")

# ── 9. 结果文件（可选）────────────────────────────────────
section("9. 已有结果 (results/)  [可选]")
results_dir = ROOT / "results"
EXPS = ["baseline", "no_reranker", "token", "paragraph",
        "chunk256", "chunk1024", "chat_model", "glm"]
for exp in EXPS:
    bench = results_dir / f"bench_{exp}.json"
    evalf = results_dir / f"eval_{exp}.json"
    if bench.exists():
        data = json.loads(bench.read_text())
        n = len(data) if isinstance(data, list) else 0
        scored = evalf.exists()
        detail = f"bench={n}条" + (" eval=✓" if scored else " eval=未完成")
        check(exp, True, detail)
    else:
        check(exp, False, "尚未运行 benchmark", warn=True)

# ── 10. rag.py 可导入 ──────────────────────────────────────
section("10. 核心模块导入")
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("rag", ROOT / "pipeline/rag.py")
    rag = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(rag)
    check("pipeline/rag.py 导入成功", True)
    for fn in ["init_settings", "load_documents", "build_index",
               "load_index", "build_hybrid_retriever", "get_reranker"]:
        check(f"  rag.{fn}", hasattr(rag, fn))
except Exception as e:
    check("pipeline/rag.py 导入", False, str(e))

# ── 汇总 ──────────────────────────────────────────────────
section("汇总")
hard_fails  = [(l, ok) for l, ok, warn in results if not ok and not warn]
soft_fails  = [(l, ok) for l, ok, warn in results if not ok and warn]
total       = len(results)
passed      = sum(1 for _, ok, _ in results if ok)

print(f"  通过: {passed}/{total}")
if soft_fails:
    print(f"  警告: {len(soft_fails)} 项（不影响主流程）")
if hard_fails:
    print(f"\n  \033[31m以下项目需修复：\033[0m")
    for label, _ in hard_fails:
        print(f"    - {label}")
    sys.exit(1)
else:
    print(f"\n  \033[32m环境完整，项目可以运行。\033[0m")
