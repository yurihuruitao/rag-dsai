"""构建向量索引的独立脚本"""

import argparse
import os
import shutil
import importlib.util
import types

from rag import init_settings, build_index, load_index


def load_custom_config(config_path):
    """动态加载自定义配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)

    # 转换为一个简单的对象，方便属性访问
    config_obj = types.SimpleNamespace()
    for key in dir(custom_config):
        if not key.startswith("__"):
            setattr(config_obj, key, getattr(custom_config, key))

    return config_obj


def main():
    parser = argparse.ArgumentParser(description="独立构建 RAG 向量索引")
    parser.add_argument(
        "--config",
        type=str,
        help="自定义配置文件路径 (例如: test_config.py)",
        required=True,
    )
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()

    print(f"🔧 加载自定义配置: {args.config}")
    config = load_custom_config(args.config)

    storage_dir = getattr(config, "STORAGE_DIR", "storage")
    index_source_exp = getattr(config, "INDEX_SOURCE_EXP", None)
    if index_source_exp:
        storage_dir = f"storage/{index_source_exp}"

    print("==================================================")
    print("🚀 启动索引构建")
    print(f"🔹 Embedding 模型: {config.EMBEDDING_MODEL}")
    print(f"🔹 PDF 目录: {config.PDF_DIR}")
    print(f"🔹 存储目录: {storage_dir}")
    if index_source_exp:
        print(f"🔹 复用索引来源 (INDEX_SOURCE_EXP): {index_source_exp}")
    print(f"🔹 文档切块: {config.CHUNK_SIZE} / 重叠 {config.CHUNK_OVERLAP}")
    print(f"🔹 分块策略: {getattr(config, 'CHUNK_STRATEGY', 'sentence')}")
    print("==================================================\n")

    print("⏳ 初始化设置...")
    init_settings(config)

    if args.rebuild and os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
        print("🗑️  已删除旧索引")

    if os.path.exists(storage_dir):
        print("ℹ️  索引已存在，如果需要重建请使用 --rebuild 参数。加载已有索引验证...")
        load_index(storage_dir)
    else:
        print(f"⏳ 开始构建索引，输出到: {storage_dir}")
        build_index(config)
        print("✅ 索引构建完成！")


if __name__ == "__main__":
    main()
