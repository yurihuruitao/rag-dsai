"""
no_rag_bench.py —— 不用 RAG，直接用 deepseek-chat 回答问题，输出格式与 benchmark.py 相同。

用法:
    python no_rag_bench.py --config configs/exp1_baseline.py
    python no_rag_bench.py --config configs/exp1_baseline.py --output results/bench_no_rag.json

评分:
    python evaluate.py --config configs/exp1_baseline.py --input results/bench_no_rag.json --output results/eval_no_rag.json
"""

import json
import os
import time
import argparse
import concurrent.futures
import threading
import importlib.util
import types

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 里的 DEEPSEEK_API_KEY


def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = types.SimpleNamespace()
    for k in dir(mod):
        if not k.startswith("__"):
            setattr(cfg, k, getattr(mod, k))
    return cfg


def ask(client, query):
    """直接问 deepseek-chat，不带任何文档上下文。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": query}],
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="No-RAG baseline benchmark")
    parser.add_argument("--config", required=True, help="配置文件路径，如 configs/exp1_baseline.py")
    parser.add_argument("--output", default=None, help="输出 JSON 路径（默认 results/bench_no_rag.json）")
    parser.add_argument("--limit", type=int, default=None, help="只测前 N 条")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    parser.add_argument("--no-resume", action="store_true", help="从头开始，忽略已有结果")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or "results/bench_no_rag.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    # 加载问题和答案
    with open(config.BENCH_QUERIES, encoding="utf-8") as f:
        queries = json.load(f)
    with open(config.BENCH_ANSWERS, encoding="utf-8") as f:
        answers = json.load(f)

    query_ids = list(queries.keys())
    if args.limit:
        query_ids = query_ids[: args.limit]

    global_index_map = {qid: i + 1 for i, qid in enumerate(query_ids)}

    # 断点续传
    existing_results = []
    if not args.no_resume and os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                existing_results = json.load(f)
            done_ids = {r["id"] for r in existing_results}
            before = len(query_ids)
            query_ids = [qid for qid in query_ids if qid not in done_ids]
            print(f"🔄 断点续传：已完成 {before - len(query_ids)} 条，剩余 {len(query_ids)} 条")
        except Exception:
            existing_results = []

    if not query_ids:
        print("✅ 所有查询均已完成。")
        return

    print(f"🚀 No-RAG baseline，共 {len(query_ids)} 条，并发数 {args.workers}")

    results = []
    save_lock = threading.Lock()
    completed_count = 0

    def process(qid):
        q = queries[qid]["query"]
        gt = answers.get(qid, "")
        idx = global_index_map[qid]
        try:
            t0 = time.time()
            answer = ask(client, q)
            elapsed = round(time.time() - t0, 2)
        except Exception as e:
            answer = f"ERROR: {e}"
            elapsed = 0
        return {
            "index": idx,
            "id": qid,
            "query": q,
            "query_type": queries[qid].get("type", ""),
            "rag_answer": answer,   # evaluate.py 读这个字段
            "ground_truth": gt,
            "sources": [],
            "time_sec": elapsed,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, qid): qid for qid in query_ids}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(query_ids), desc="No-RAG"):
            res = future.result()
            with save_lock:
                results.append(res)
                completed_count += 1
                if completed_count % 10 == 0:
                    merged = existing_results + sorted(results, key=lambda x: x["index"])
                    merged.sort(key=lambda x: x["index"])
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(merged, f, ensure_ascii=False, indent=2)

    all_results = existing_results + sorted(results, key=lambda x: x["index"])
    all_results.sort(key=lambda x: x["index"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成！共 {len(all_results)} 条，已保存至 {output_path}")


if __name__ == "__main__":
    main()
