"""OpenRAGBench 自动测试脚本

用法:
    python3 benchmark.py --config test_config.py --limit 10
"""

import json
import time
import os
import importlib.util
import types
import concurrent.futures
import threading

from tqdm import tqdm
import argparse

from rag import init_settings, load_index


def get_args():
    parser = argparse.ArgumentParser(description="OpenRAGBench 自动测试脚本")
    parser.add_argument(
        "--config",
        type=str,
        help="自定义配置文件路径 (例如: test_config.py)",
        required=True,
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="只测前 N 条数据 (例如: --limit 10)"
    )
    parser.add_argument(
        "--output", type=str, help="保存结果的 JSON 路径，如果不用配置里的"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="并发执行的线程数，默认为 8"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="不使用断点续传，从头开始测试"
    )
    return parser.parse_args()


def load_custom_config(config_path):
    """动态加载自定义配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)

    config_obj = types.SimpleNamespace()
    for key in dir(custom_config):
        if not key.startswith("__"):
            setattr(config_obj, key, getattr(custom_config, key))

    return config_obj


def load_bench_data(config):
    """加载 benchmark 数据"""
    with open(config.BENCH_QUERIES) as f:
        queries = json.load(f)
    with open(config.BENCH_ANSWERS) as f:
        answers = json.load(f)
    return queries, answers


def save_results(results, path):
    """保存结果到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_benchmark(args):
    """执行 benchmark 测试"""
    print(f"🔧 加载自定义配置: {args.config}")
    config = load_custom_config(args.config)

    output_path = (
        args.output
        if args.output
        else getattr(config, "BENCH_OUTPUT", "bench_results.json")
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    storage_dir = getattr(config, "STORAGE_DIR", "storage")
    index_source_exp = getattr(config, "INDEX_SOURCE_EXP", None)
    if index_source_exp:
        storage_dir = f"storage/{index_source_exp}"

    print("==================================================")
    print("🚀 启动 Benchmark")
    print(f"🔹 模型: {config.DEFAULT_MODEL}")
    print(f"🔹 Embedding 模型: {config.EMBEDDING_MODEL}")
    print(f"🔹 Rerank 模型: {config.RERANK_MODEL}")
    print(f"🔹 文档切块: {config.CHUNK_SIZE} / 重叠 {config.CHUNK_OVERLAP}")
    print(f"🔹 粗检索数: 向量 {config.VECTOR_TOP_K} / BM25 {config.BM25_TOP_K}")
    print(f"🔹 重排序保留: {config.RERANK_TOP_N}")
    if index_source_exp:
        print(f"🔹 索引来源 (INDEX_SOURCE_EXP): {index_source_exp} ({storage_dir})")
    print(f"🔹 并发线程数: {args.workers}")
    print(f"🔹 输出文件: {output_path}")
    print("==================================================\n")

    # 初始化
    print("⏳ 正在初始化模型...")
    init_settings(config)

    if not os.path.exists(storage_dir):
        print(
            f"❌ 索引目录不存在: {storage_dir}。请先运行 build_index.py 建立向量数据库。"
        )
        return

    # 加载索引
    print("⏳ 加载索引...")
    index = load_index(storage_dir)

    # 构建检索器和重排序器
    from rag import build_hybrid_retriever, get_reranker
    from llama_index.core.chat_engine import CondensePlusContextChatEngine
    from llama_index.core.memory import ChatMemoryBuffer

    retriever = build_hybrid_retriever(index, config)
    reranker = get_reranker(config)
    node_postprocessors = [reranker] if reranker else []

    # 加载数据
    queries, answers = load_bench_data(config)
    query_ids = list(queries.keys())
    if args.limit:
        query_ids = query_ids[: args.limit]

    # 断点续传：加载已有结果，跳过已完成的查询
    existing_results = []
    if not args.no_resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            done_ids = {r["id"] for r in existing_results}
            before = len(query_ids)
            query_ids = [qid for qid in query_ids if qid not in done_ids]
            print(
                f"\n🔄 断点续传：已完成 {before - len(query_ids)} 条，跳过，剩余 {len(query_ids)} 条"
            )
        except (json.JSONDecodeError, KeyError):
            print("\n⚠️  已有结果文件损坏，将从头开始")
            existing_results = []

    if not query_ids:
        print("\n✅ 所有查询均已完成，无需继续。如需重跑请加 --no-resume")
        return

    total = len(query_ids)
    print(f"📊 共 {total} 条待测试\n")

    results = []
    save_lock = threading.Lock()
    completed_count = 0

    def process_query(i, qid):
        q = queries[qid]["query"]
        gt = answers.get(qid, "")

        # 每条独立对话引擎，避免上下文干扰
        # 由于我们使用多线程，需要确保每个线程拥有独立的 chat_engine 和 memory
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=config.CHAT_MEMORY_TOKEN_LIMIT
        )
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            memory=memory,
            system_prompt=getattr(config, "SYSTEM_PROMPT", ""),
            verbose=False,
        )

        try:
            t0 = time.time()
            response = chat_engine.chat(q)
            elapsed = round(time.time() - t0, 2)

            # 解析来源
            sources = []
            if response.source_nodes:
                seen = set()
                for node in response.source_nodes:
                    meta = node.node.metadata
                    fn = meta.get("file_name", "")
                    page = meta.get("page_label", "?")
                    key = f"{fn}-{page}"
                    if key not in seen:
                        seen.add(key)
                        sources.append(
                            {
                                "filename": fn,
                                "page": page,
                                "score": (
                                    round(float(node.score), 4) if node.score else 0.0
                                ),
                            }
                        )

            return {
                "index": i,
                "id": qid,
                "query": q,
                "query_type": queries[qid].get("type", ""),
                "rag_answer": str(response),
                "ground_truth": gt,
                "sources": sources,
                "time_sec": elapsed,
            }, None

        except Exception as e:
            return {
                "index": i,
                "id": qid,
                "query": q,
                "query_type": queries[qid].get("type", ""),
                "rag_answer": f"ERROR: {e}",
                "ground_truth": gt,
                "sources": [],
                "time_sec": 0,
            }, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_qid = {
            executor.submit(process_query, i, qid): (i, qid)
            for i, qid in enumerate(query_ids)
        }

        with tqdm(total=total, desc="Benchmark") as pbar:
            for future in concurrent.futures.as_completed(future_to_qid):
                i, qid = future_to_qid[future]
                try:
                    res, err = future.result()
                    if err:
                        tqdm.write(f"  ❌ [{i+1}] {err}")
                    else:
                        pbar.set_postfix_str(res["query"][:40])

                    with save_lock:
                        results.append(res)
                        completed_count += 1
                        # 每 10 条保存一次（合并已有结果）
                        if completed_count % 10 == 0:
                            merged = existing_results + sorted(
                                results, key=lambda x: x["index"]
                            )
                            merged.sort(key=lambda x: x["index"])
                            save_results(merged, output_path)

                except Exception as exc:
                    tqdm.write(f"  ❌ [{i+1}] {exc}")

                pbar.update(1)

    # 最终保存（合并已有结果 + 新结果）
    all_results = existing_results + sorted(results, key=lambda x: x["index"])
    all_results.sort(key=lambda x: x["index"])
    save_results(all_results, output_path)
    print(
        f"\n🎉 测试完成！本次新增 {len(results)} 条，共 {len(all_results)} 条结果已保存到 {output_path}"
    )


def main():
    args = get_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
