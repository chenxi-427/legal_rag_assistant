#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本向量化处理 - 使用真实的句向量模型
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb

# 尝试导入SentenceTransformer，如果失败则准备使用备用方案
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# --- 配置 ---
# 配置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
# 使用一个轻量级、高效的中文句向量模型
EMBEDDING_MODEL_NAME = 'infgrad/stella-base-zh-v2'

# 确保向量存储目录存在
VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)

# ----------------- 智能文本分割器 -----------------

def split_law_text_by_article(text: str) -> List[str]:
    """
    使用正则表达式按"第X条"来分割法律文本。
    每个分块包含一条完整的法条。
    """
    # 正则表达式匹配"第X条"模式，X可以是中文数字或阿拉伯数字
    # 使用正向先行断言 (?=...) 来保留分隔符（即"第X条"）在每个分块的开头
    pattern = r'(?=第[一二三四五六七八九十百\\d]+条)'
    
    # 使用 re.split 来分割文本
    chunks = re.split(pattern, text)
    
    # 过滤掉可能产生的空字符串（通常是第一个）
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return cleaned_chunks


# ----------------- 旧的文本分割器配置 (不再使用) -----------------
# TEXT_SPLITTER = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     separators=["\\n\\n", "\\n", "。", "，", " ", ""]
# )


# 向量维度
VECTOR_DIMENSION = 384  # 通常的嵌入维度


def load_and_split_text(file_path: str) -> List[Tuple[str, str, str]]:
    """加载文本文件并按法条进行分割"""
    try:
        file_path = str(file_path)
        file_name = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用新的、基于规则的分割函数
        chunks = split_law_text_by_article(content)
        
        # 返回 (chunk_id, chunk_text, source) 元组列表
        return [(f"{file_name}_{i}", chunk, file_name) 
                for i, chunk in enumerate(chunks)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def create_embeddings(chunks: List[Tuple[str, str, str]], model: Any) -> List[Dict[str, Any]]:
    """为文本块创建嵌入向量"""
    
    # 提取所有文本内容以进行批量编码
    texts_to_embed = [text for _, text, _ in chunks]
    
    if hasattr(model, 'encode'):
        # 使用真实的SentenceTransformer模型
        print(f"  正在使用模型 '{EMBEDDING_MODEL_NAME}' 生成 {len(texts_to_embed)} 个嵌入...")
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    else:
        # 使用简单的随机嵌入（备用方案）
        print(f"  使用备用随机嵌入生成 {len(texts_to_embed)} 个嵌入...")
        # 为每个文本生成一个随机向量，但保持向量维度一致
        # 注意：这只是一个备用方案，实际效果会很差
        embeddings = [np.random.randn(VECTOR_DIMENSION).astype(np.float32) for _ in texts_to_embed]
    
    embedded_docs = []
    for i, (chunk_id, text, source) in enumerate(chunks):
        embedded_docs.append({
            'id': chunk_id,
            'text': text,
            'source': source,
            'embedding': embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else embeddings[i] # 转换为list以便JSON序列化
        })
    
    return embedded_docs


def save_to_chroma(records: List[Dict[str, Any]]) -> None:
    """将文档及其嵌入保存到Chroma向量数据库"""
    
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    
    # 在添加前，先清空旧的集合，确保数据最新
    try:
        client.delete_collection(name="legal_documents")
    except ValueError:
        print("集合 'legal_documents' 不存在，将创建新集合。")

    collection = client.get_or_create_collection(
        name="legal_documents",
        metadata={"description": "法律文档集合"}
    )
    
    ids = [r['id'] for r in records]
    documents = [r['text'] for r in records]
    metadatas = [{'source': r['source']} for r in records]
    embeddings = [r['embedding'] for r in records]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    
    print(f"成功保存 {len(ids)} 条文档到Chroma数据库")


def main():
    """主函数：处理文本嵌入"""
    
    # 加载嵌入模型
    try:
        print(f"正在加载句向量模型: {EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("模型加载完毕。")
    except Exception as e:
        print(f"加载句向量模型失败: {e}")
        print("使用备用随机嵌入模型...")
        # 创建一个简单的备用"模型"对象，没有encode方法
        class SimpleEmbeddingModel:
            pass
        model = SimpleEmbeddingModel()
    
    # 获取所有文本文件路径
    file_paths = [str(f) for f in DATA_DIR.glob("*.txt")]
    print(f"找到 {len(file_paths)} 个文本文件")
    
    # 处理所有文档
    all_documents = []
    for file_path in file_paths:
        print(f"处理文件: {os.path.basename(file_path)}")
        # 加载和分割文本
        chunks = load_and_split_text(file_path)
        print(f"  文件被分割为 {len(chunks)} 个文本块")
        
        # 创建嵌入
        embedded_docs = create_embeddings(chunks, model)
        all_documents.extend(embedded_docs)
        print(f"  成功创建了 {len(embedded_docs)} 个文本嵌入")
    
    # 保存到Chroma
    print(f"保存 {len(all_documents)} 个嵌入到向量数据库...")
    save_to_chroma(all_documents)
    print("处理完成!")


if __name__ == "__main__":
    main() 