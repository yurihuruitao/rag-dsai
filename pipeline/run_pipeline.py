"""
OpenRAGBench 完整测试流水线脚本
按照顺序依次执行:
1. build_index.py 构建向量数据库
2. benchmark.py 执行测试
3. evaluate.py 对测试结果进行打分评估

用法:
    python3 run_pipeline.py --config configs/exp1_baseline.py [--rebuild] [--limit N] [--workers N] [--no-resume]
"""

import argparse
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv() 
def main():
    parser = argparse.ArgumentParser(description="完整运行 RAG 评测流水线")
    parser.add_argument("--config", type=str, required=True, help="自定义配置文件路径")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引 (传给 build_index.py)")
    parser.add_argument("--limit", type=int, help="只测前 N 条数据 (传给 benchmark.py)")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--no-resume", action="store_true", help="不使用断点续传")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 启动 OpenRAGBench 完整数据流水线")
    print(f"🔹 配置文件: {args.config}")
    print("=" * 60)
    
    # 1. 运行 build_index.py
    print("\n\n" + "-" * 50)
    print("👉 阶段 1: 构建向量索引 (build_index.py)")
    print("-" * 50)
    build_cmd = [sys.executable, "pipeline/build_index.py", "--config", args.config]
    if args.rebuild:
        build_cmd.append("--rebuild")
        
    res = subprocess.run(build_cmd)
    if res.returncode != 0:
        print("\n❌ 阶段 1 (build_index.py) 执行失败，流水线终止！")
        sys.exit(res.returncode)
        
    # 2. 运行 benchmark.py
    print("\n\n" + "-" * 50)
    print("👉 阶段 2: 运行测试并生成回答 (benchmark.py)")
    print("-" * 50)
    bench_cmd = [
        sys.executable, "pipeline/benchmark.py",
        "--config", args.config,
        "--workers", str(args.workers)
    ]
    if args.limit is not None:
        bench_cmd.extend(["--limit", str(args.limit)])
    if args.no_resume:
        bench_cmd.append("--no-resume")
        
    res = subprocess.run(bench_cmd)
    if res.returncode != 0:
        print("\n❌ 阶段 2 (benchmark.py) 执行失败，流水线终止！")
        sys.exit(res.returncode)
        
    # 3. 运行 evaluate.py
    print("\n\n" + "-" * 50)
    print("👉 阶段 3: 使用大模型进行评估 (evaluate.py)")
    print("-" * 50)
    eval_cmd = [
        sys.executable, "pipeline/evaluate.py",
        "--config", args.config,
        "--workers", str(args.workers)
    ]
    if args.no_resume:
        eval_cmd.append("--no-resume")
        
    res = subprocess.run(eval_cmd)
    if res.returncode != 0:
        print("\n❌ 阶段 3 (evaluate.py) 执行失败，流水线终止！")
        sys.exit(res.returncode)
        
    print("\n" + "=" * 60)
    print("🎉 OpenRAGBench 流水线全部执行成功完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
