#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
法律RAG助手 - Streamlit前端应用
"""

import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# 将项目根目录添加到sys.path，以便导入rag模块
# D:/code/legal-rag-assistant
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from rag.qa_chain import LegalQAChain
from rag.llm import SimpleLLM

# 加载环境变量
load_dotenv()

# 配置页面
st.set_page_config(
    page_title="法律RAG助手",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 会话状态管理 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- 应用侧边栏 ---
with st.sidebar:
    st.header("设置")
    
    # 选择LLM模型
    st.selectbox(
        "选择LLM模型",
        ["simple-llm"],
        key="selected_model",
    )
    
    # 设置模型温度
    temperature = st.slider(
        "模型温度", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        help="温度越高，回答越有创造性但可能不准确；温度越低，回答越保守和一致。"
    )

    # 显示相关法律条文的选项
    show_source = st.checkbox("显示相关法律条文", value=True)
    
    st.info(
        """
        **关于**
        
        法律RAG助手是一个基于检索增强生成（RAG）的法律问答系统，旨在为用户提供准确的法律咨询。
        
        系统基于LangChain和Streamlit构建。
        """
    )
    st.markdown("© 2023 法律RAG助手 - 基于LangChain和Streamlit构建")

# --- 初始化问答链 ---
# 仅在需要时（首次运行或参数更改时）重新创建实例
if st.session_state.qa_chain is None or st.session_state.qa_chain.llm.temperature != temperature:
    llm = SimpleLLM()
    st.session_state.qa_chain = LegalQAChain(
        llm_model=llm,
        temperature=temperature
    )

# --- 主界面 ---
st.title("⚖️ 法律RAG助手")

# 使用Tabs来组织界面
tab_qa, tab_instructions = st.tabs(["问答", "使用说明"])

with tab_qa:
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 如果是助手的回答，并且有源文档，则显示
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("参考法条", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"**来源: {source['source']} - {source['article']}**")
                        st.markdown(f"> {source['content']}")
                        st.markdown("---")

    # 聊天输入框
    if prompt := st.chat_input("请输入您的法律问题..."):
        # 将用户问题添加到消息历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 显示"思考中"的动画
        with st.chat_message("assistant"):
            with st.spinner("正在检索法律条文并生成回答..."):
                # 运行问答链
                response = st.session_state.qa_chain.run(prompt, show_source=show_source)
                
                answer = response["answer"]
                source_documents = response.get("source_documents", [])
                
                # 创建助手的完整回复
                full_response_content = answer
                
                st.markdown(full_response_content)
                
                # 如果有源文档，则在可折叠区域中显示
                if show_source and source_documents:
                    with st.expander("参考法条", expanded=True):
                        for source in source_documents:
                            # 使用更醒目的样式显示法条信息
                            st.markdown(f"**来源: {source['source']} - {source['article']}**")
                            
                            # 使用引用块和更好的格式显示法条内容
                            content = source['content']
                            # 如果内容太长，截断显示
                            if len(content) > 500:
                                content = content[:500] + "..."
                                
                            # 添加额外的格式增强可读性
                            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
                            st.markdown("---")

                # 将助手的回复（包括源文档信息）添加到消息历史
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response_content,
                    "sources": source_documents if show_source else []
                })

with tab_instructions:
    st.header("使用说明")
    st.markdown(
        """
        欢迎使用法律RAG助手！

        **如何提问:**
        1. 在下方的聊天输入框中输入您关于中国劳动法的任何问题。
        2. 按回车键或点击发送按钮。
        3. 系统将检索相关的法律条文，并生成精准的回答。

        **功能设置:**
        - **选择LLM模型**: 目前仅支持`simple-llm`用于演示。
        - **模型温度**: 调整滑块可以改变回答的风格。较低的温度使回答更严谨，较高的温度则更具创造性。
        - **显示相关法律条文**: 勾选此项，系统会在回答下方附上引用的具体法律条款，方便您核对来源。

        **示例问题:**
        - `劳动法适用于哪些单位和个人？`
        - `劳动法第60条是什么？`
        - `用人单位有什么义务？`
        """
    )

# 添加页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>© 2023 法律RAG助手 - 基于LangChain和Streamlit构建</div>", 
    unsafe_allow_html=True
)