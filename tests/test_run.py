#!/usr/bin/env python3
"""
端到端运行测试脚本
测试完整 pipeline 能否正常执行：index加载 → benchmark → evaluate
所有输出写入 /tmp，测试结束后自动清理，不留任何痕迹。

运行: python test_run.py
"""

import sys
import os
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

PYTHON = str(ROOT / ".venv/bin/python")
PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

# 所有临时文件都写这里
TMPDIR = Path(tempfile.mkdtemp(prefix="rag_test_"))

passed = []
failed = []

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def ok(label, detail=""):
    msg = f"  {PASS}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    passed.append(label)

def fail(label, detail=""):
    msg = f"  {FAIL}  {label}"
    if detail:
        msg += f"\n      {detail}"
    print(msg)
    failed.append(label)

def run(cmd, timeout=300):
    """运行子进程，返回 (returncode, stdout, stderr)"""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT
    )
    return result.returncode, result.stdout, result.stderr


# ── 1. 临时配置文件 ────────────────────────────────────────
section("1. 创建临时测试配置")

BENCH_OUTPUT = str(TMPDIR / "bench_test.json")
EVAL_OUTPUT  = str(TMPDIR / "eval_test.json")

# 基于 baseline 配置，复用已有索引，输出到 /tmp
CONFIG_CONTENT = f"""\
import os
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODELS = {{"deepseek-chat": "deepseek-chat", "deepseek-reasoner": "deepseek-reasoner"}}
DEFAULT_MODEL     = "deepseek-reasoner"

EMBEDDING_MODEL   = "BAAI/bge-m3"
EMBEDDING_DIM     = 1024

USE_RERANKER      = True
RERANK_MODEL      = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N      = 3

PDF_DIR           = "ourbench/pdf"
EXP_NAME          = "test_tmp"
STORAGE_DIR       = "storage/baseline"   # 复用已有索引

CHUNK_STRATEGY    = "sentence"
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 50
VECTOR_TOP_K      = 5
BM25_TOP_K        = 5
CHAT_MEMORY_TOKEN_LIMIT = 3000

BENCH_QUERIES     = "ourbench/Q&A/queries_filtered.json"
BENCH_ANSWERS     = "ourbench/Q&A/answers_filtered.json"
BENCH_OUTPUT      = {repr(BENCH_OUTPUT)}
EVAL_OUTPUT       = {repr(EVAL_OUTPUT)}

SYSTEM_PROMPT = "You are a professional document QA assistant. Answer based on the retrieved content."
"""

CONFIG_PATH = TMPDIR / "test_config.py"
CONFIG_PATH.write_text(CONFIG_CONTENT)
ok("临时配置写入", str(CONFIG_PATH))


# ── 2. 加载索引 ────────────────────────────────────────────
section("2. 加载 FAISS 索引（storage/baseline）")

load_script = f"""\
import sys
sys.path.insert(0, 'pipeline')
import importlib.util, types
from dotenv import load_dotenv
load_dotenv()

spec = importlib.util.spec_from_file_location("cfg", "{CONFIG_PATH}")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg  = types.SimpleNamespace(**{{k: getattr(mod, k) for k in dir(mod) if not k.startswith('__')}})

from rag import init_settings, load_index
init_settings(cfg)
index = load_index(cfg.STORAGE_DIR)

# 简单验证
all_nodes = list(index.docstore.docs.values())
assert len(all_nodes) > 0, "索引中没有文档"
print(f"OK nodes={{len(all_nodes)}}")
"""

rc, out, err = run([PYTHON, "-c", load_script], timeout=120)
if rc == 0 and "OK" in out:
    node_count = out.strip().split("nodes=")[-1]
    ok("索引加载成功", f"{node_count} 个节点")
else:
    fail("索引加载失败", (err or out)[-500:])
    # 索引加载失败则后续无意义，直接退出
    shutil.rmtree(TMPDIR, ignore_errors=True)
    print("\n\033[31m索引加载失败，终止测试。\033[0m")
    sys.exit(1)


# ── 3. Benchmark（1条）─────────────────────────────────────
section("3. Benchmark 测试（limit=1）")

t0 = time.time()
rc, out, err = run([
    PYTHON, "pipeline/benchmark.py",
    "--config", str(CONFIG_PATH),
    "--limit", "1",
    "--workers", "1",
    "--no-resume",
    "--output", BENCH_OUTPUT,
], timeout=120)
elapsed = round(time.time() - t0, 1)

if rc != 0:
    fail("benchmark.py 执行失败", (err or out)[-600:])
else:
    bench_path = Path(BENCH_OUTPUT)
    if not bench_path.exists():
        fail("benchmark 输出文件未生成")
    else:
        try:
            data = json.loads(bench_path.read_text())
            assert isinstance(data, list) and len(data) == 1
            item = data[0]
            assert "rag_answer" in item and "query" in item
            answer_preview = item["rag_answer"][:60].replace("\n", " ")
            ok("benchmark.py 运行成功", f"{elapsed}s，1条结果")
            ok("输出格式正确", f'query="{item["query"][:40]}..."')
            ok("RAG 回答非空" if item["rag_answer"] and not item["rag_answer"].startswith("ERROR") else "RAG 回答有错误",
               f'"{answer_preview}..."')
            if item.get("sources"):
                ok("检索源非空", f"{len(item['sources'])} 个来源")
            else:
                fail("检索源为空", "未能检索到相关文档片段")
        except Exception as e:
            fail("输出文件格式异常", str(e))


# ── 4. Evaluate（1条）────────────────────────────────────
section("4. Evaluate 评分（1条）")

if Path(BENCH_OUTPUT).exists():
    t0 = time.time()
    rc, out, err = run([
        PYTHON, "pipeline/evaluate.py",
        "--config", str(CONFIG_PATH),
        "--input",  BENCH_OUTPUT,
        "--output", EVAL_OUTPUT,
        "--workers", "1",
        "--no-resume",
    ], timeout=120)
    elapsed = round(time.time() - t0, 1)

    if rc != 0:
        fail("evaluate.py 执行失败", (err or out)[-600:])
    else:
        eval_path = Path(EVAL_OUTPUT)
        if not eval_path.exists():
            fail("evaluate 输出文件未生成")
        else:
            try:
                edata = json.loads(eval_path.read_text())
                assert isinstance(edata, list) and len(edata) == 1
                item = edata[0]
                assert "score" in item, "缺少 score 字段"
                score = item["score"]
                assert isinstance(score, int) and 0 <= score <= 10
                ok("evaluate.py 运行成功", f"{elapsed}s")
                ok("评分字段存在", f"score={score}/10")
            except Exception as e:
                fail("evaluate 输出格式异常", str(e))
else:
    fail("跳过 evaluate（benchmark 未成功）")


# ── 5. 清理 ────────────────────────────────────────────────
section("5. 清理临时文件")

shutil.rmtree(TMPDIR, ignore_errors=True)
if not TMPDIR.exists():
    ok("临时目录已删除", str(TMPDIR))
else:
    fail("临时目录删除失败", str(TMPDIR))


# ── 汇总 ──────────────────────────────────────────────────
section("汇总")
total = len(passed) + len(failed)
print(f"  通过: {len(passed)}/{total}")
if failed:
    print(f"\n  \033[31m失败项目：\033[0m")
    for f in failed:
        print(f"    - {f}")
    sys.exit(1)
else:
    print(f"\n  \033[32m所有测试通过，pipeline 可正常运行。\033[0m")
