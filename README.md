# Legal RAG Assistant

基于法律文本的检索增强生成 (RAG) 问答系统。

## 项目结构

- `data/`: 存储法律原始数据（如劳动法）
- `embedding/`: 使用 Flink 处理文本向量化的脚本
- `vector_store/`: 向量数据库（Chroma）的本地持久化目录
- `rag/`: 使用 LangChain 构建 RAG 问答链
- `app/`: 使用 Streamlit 构建可视化前端

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 将法律文本放入 `data/` 目录
2. 运行向量化处理脚本：`python -m embedding.process`
3. 启动前端应用：`streamlit run app/main.py`

## 功能特点

- 基于大语言模型的法律文本理解
- 高效的向量检索技术，提高回答准确性
- 友好的用户界面，便于法律咨询 