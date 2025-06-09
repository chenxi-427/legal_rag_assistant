#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
法律RAG助手启动脚本
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

# 配置路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


def process_embeddings():
    """处理文本嵌入"""
    print("开始处理文档嵌入...")
    try:
        from embedding.process import main as embedding_main
        embedding_main()
        print("文档嵌入处理完成！")
    except Exception as e:
        print(f"处理文档嵌入时出错: {e}")
        sys.exit(1)


def run_app():
    """运行Streamlit应用"""
    print("启动Streamlit应用...")
    app_path = BASE_DIR / "app" / "main.py"
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动Streamlit应用失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("错误: 未找到Streamlit。请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="法律RAG助手")
    parser.add_argument("--process", action="store_true", help="处理文档嵌入")
    parser.add_argument("--run", action="store_true", help="启动Streamlit应用")
    args = parser.parse_args()

    # 使用自定义LLM，不需要API密钥
    print("正在使用简单的演示模式，无需API密钥")
    
    # 检查向量数据库目录
    if not VECTOR_STORE_DIR.exists():
        VECTOR_STORE_DIR.mkdir(parents=True)
    
    # 执行操作
    if args.process:
        process_embeddings()
    elif args.run:
        run_app()
    else:
        # 默认行为: 如果向量数据库为空，先处理嵌入，然后启动应用
        if not any(VECTOR_STORE_DIR.iterdir()) or len(list(VECTOR_STORE_DIR.glob("*"))) <= 1:
            print("向量数据库为空，先处理嵌入...")
            process_embeddings()
        run_app()


if __name__ == "__main__":
    main() 