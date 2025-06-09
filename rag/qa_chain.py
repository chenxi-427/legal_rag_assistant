#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
定义问答链 - RAG的核心
"""
import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings  # 添加备用嵌入模型

# 本地模块
from .llm import SimpleLLM

# --- 全局配置 ---
# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
# 定义向量数据库路径
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
# 定义嵌入模型
EMBEDDING_MODEL_NAME = 'infgrad/stella-base-zh-v2'
# 定义集合名称
COLLECTION_NAME = "legal_documents"

# --- 提示模板 ---
TEMPLATE = """
[任务]
你是一个专业的中国劳动法律师。请根据以下已知信息，简洁、准确地回答用户的问题。
严格禁止在已知信息之外进行任何补充或想象。
如果已知信息与问题不相关，或者无法从已知信息中找到答案，请直接回答："根据提供的法律法规信息，无法找到与您问题直接相关的具体条款。建议您咨询专业的法律顾问获取更准确的信息。"

[已知信息]
{context}

[问题]
{question}

[回答]
"""
QA_PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])


class LegalQAChain:
    """
    法律问答链，封装了RAG的整个流程：
    1. 接收用户问题
    2. 使用向量数据库检索相关文档
    3. 将问题和相关文档组合成提示
    4. 使用LLM生成答案
    """

    def __init__(self, llm_model: Any, temperature: float = 0.1, top_k: int = 4):
        """
        初始化问答链

        Args:
            llm_model: 用于生成答案的语言模型实例 (例如 SimpleLLM)
            temperature: 模型的温度参数
            top_k: 从向量数据库中检索的文档数量
        """
        self.llm = llm_model
        self.llm.temperature = temperature
        
        # 初始化嵌入模型
        try:
            print("正在尝试加载在线嵌入模型...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'} # 或者 'cuda' 如果有GPU
            )
        except Exception as e:
            print(f"无法加载在线模型，错误: {e}")
            print("使用本地备用嵌入模型...")
            # 使用简单的备用嵌入模型
            self.embedding_model = FakeEmbeddings(size=384)  # 使用与原模型相同的维度

        # 初始化向量数据库
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=self.embedding_model
        )
        
        # 初始化检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': top_k}
        )

        # 构建问答链 (LCEL - LangChain Expression Language)
        self.chain = (
            RunnableParallel(
                context=(self.retriever | self._format_docs),
                question=RunnablePassthrough()
            )
            | QA_PROMPT
            | self.llm
        )

    @staticmethod
    def _format_docs(docs: List[Any]) -> str:
        """格式化检索到的文档，确保法条内容清晰可读"""
        formatted_texts = []
        
        for doc in docs:
            # 清理文本，去除多余空格和换行
            content = doc.page_content.strip()
            
            # 检查是否包含法条编号
            if content.startswith("第") and "条" in content[:15]:
                # 如果是法条，保持原格式
                formatted_texts.append(content)
            else:
                # 普通文本，确保没有奇怪的格式
                cleaned = content.replace("\\n\\n", " ").replace("\\n", " ")
                formatted_texts.append(cleaned)
                
        return "\\n\\n".join(formatted_texts)

    def _format_source_docs(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """格式化源文档，以便在前端显示"""
        source_docs = []
        for doc in docs:
            # 清理和格式化文本内容
            content = re.sub(r'\\s+', ' ', doc.page_content).strip()
            source = doc.metadata.get('source', '未知来源')
            # 提取法条编号
            article_match = re.search(r'^(第[一二三四五六七八九十百\\d]+条)', content)
            article = article_match.group(1) if article_match else "相关条款"
            
            source_docs.append({
                "article": article,
                "content": content,
                "source": source
            })
        return source_docs

    def run(self, question: str, show_source: bool = True) -> Dict[str, Any]:
        """
        执行问答链

        Args:
            question: 用户提出的问题
            show_source: 是否在结果中包含源文档

        Returns:
            一个包含答案和可选源文档的字典
        """
        # 调用检索器获取相关文档
        retrieved_docs = self.retriever.invoke(question)
        
        # 调用链生成答案
        answer = self.chain.invoke(question)

        result = {"answer": answer, "source_documents": []}
        if show_source:
            result["source_documents"] = self._format_source_docs(retrieved_docs)
            
        return result 