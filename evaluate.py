import json
import argparse
import concurrent.futures
import re
from tqdm import tqdm
from openai import OpenAI

import os
import types
import importlib.util
import threading


def load_custom_config(config_path):
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


def get_args():
    parser = argparse.ArgumentParser(description="使用 deepseek-reasoner 评测 RAG 回答")
    parser.add_argument(
        "--config",
        type=str,
        help="自定义配置文件路径 (例如: test_config.py)",
        required=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入的 benchmark 结果文件，如果不填默认使用 config 里的",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出包含打分结果的文件，如果不填默认使用 config 里的",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="并发执行的线程数，默认为 8"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="不使用断点续传，从头开始测试"
    )
    return parser.parse_args()


def evaluate_single(item, client):
    """
    对单条数据进行评测
    """
    prompt = f"""You are an expert evaluator.
Please evaluate the RAG answer based on the given query and the ground truth.
Score the answer from 0 to 10, where 10 means perfectly matching the ground truth, and 0 means completely wrong.
Return ONLY an integer score, no other text.

Query: {item["query"]}
Ground Truth: {item["ground_truth"]}
RAG Answer: {item["rag_answer"]}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}]
        )

        # deepseek-reasoner 最终的回复在这个字段
        msg = response.choices[0].message
        content = msg.content.strip() if msg.content else ""
        reasoning = getattr(msg, "reasoning_content", "")

        # 提取 0-10 之间的数字
        match = re.search(r"\b(10|[0-9])\b", content)
        item["score"] = int(match.group(1)) if match else 0
        item["eval_raw"] = content
        if reasoning:
            item["eval_reasoning"] = reasoning

    except Exception as e:
        item["score"] = 0
        item["eval_error"] = str(e)

    return item


def main():
    args = get_args()
    print(f"🔧 加载自定义配置: {args.config}")
    config = load_custom_config(args.config)

    input_path = (
        args.input
        if args.input
        else getattr(config, "BENCH_OUTPUT", "bench_results.json")
    )
    output_path = (
        args.output
        if args.output
        else getattr(config, "EVAL_OUTPUT", "eval_results.json")
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    print(f"📂 加载评测数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 断点续传：加载已经评测过的结果
    existing_results = []
    if not args.no_resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            # 根据 id 判断是否已完成（含有 score 字段则认为已评测）
            done_ids = {r.get("id") for r in existing_results if "score" in r or "eval_error" in r}
            before = len(data)
            data = [item for item in data if item.get("id") not in done_ids]
            print(f"\n🔄 断点续传：已完成 {before - len(data)} 条，跳过，剩余 {len(data)} 条")
        except (json.JSONDecodeError, KeyError):
            print("\n⚠️  已有评测结果文件损坏，将从头开始")
            existing_results = []

    if not data:
        print("\n✅ 所有评测均已完成，无需继续。如需重跑请使用 --no-resume")
        if existing_results:
            total_score = sum(r.get("score", 0) for r in existing_results)
            avg_score = total_score / len(existing_results) if len(existing_results) > 0 else 0
            print(f"🏆 平均得分: {avg_score:.2f} / 10.00")
        return

    print(f"🚀 开始使用 deepseek-reasoner 进行打分评测 (并发数: {args.workers})")

    results = []
    save_lock = threading.Lock()
    completed_count = 0

    # 使用多线程进行并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_single, item, client) for item in data]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(data), desc="Evaluating"
        ):
            res = future.result()
            with save_lock:
                results.append(res)
                completed_count += 1
                # 每完成 16 条保存一次（合并已有结果）
                if completed_count % 16 == 0:
                    merged = existing_results + sorted(results, key=lambda x: x.get("index", 0))
                    merged.sort(key=lambda x: x.get("index", 0))
                    with open(output_path, "w", encoding="utf-8") as temp_f:
                        json.dump(merged, temp_f, ensure_ascii=False, indent=2)

    # 最终合并结果
    all_results = existing_results + sorted(results, key=lambda x: x.get("index", 0))
    # 恢复原始顺序
    all_results.sort(key=lambda x: x.get("index", 0))

    # 计算平均分
    total_score = sum(r.get("score", 0) for r in all_results)
    avg_score = total_score / len(all_results) if len(all_results) > 0 else 0

    print(f"\n✅ 评测完成！本次新增计算 {len(results)} 条数据，共 {len(all_results)} 条。")
    print(f"🏆 平均得分: {avg_score:.2f} / 10.00")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"💾 包含打分的最终结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
