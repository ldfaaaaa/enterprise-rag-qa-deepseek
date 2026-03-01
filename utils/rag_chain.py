"""
utils/rag_chain.py
RAG 问答链：Embedding 模型初始化 + RetrievalQA 链构建
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# 加载 .env 文件
load_dotenv()

# ─────────────────────────────────────────────
# 中文 RAG Prompt 模板
# ─────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """你是企业知识库助手。请根据以下上下文回答问题，回答要准确、简洁。\
如果上下文中没有相关信息，请说"抱歉，知识库中没有找到相关信息"，不要编造答案。

上下文：
{context}

问题：{question}

回答："""


def get_embeddings():
    """
    初始化 Embedding 模型。

    优先使用 HuggingFace 本地模型（BAAI/bge-small-zh-v1.5），无需 API Key。
    若环境变量 USE_OPENAI_EMBEDDINGS=true，则改用 OpenAIEmbeddings。

    Returns:
        Embeddings 实例
    """
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

    if use_openai:
        from langchain_openai import OpenAIEmbeddings

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "USE_OPENAI_EMBEDDINGS=true 但未设置 OPENAI_API_KEY，"
                "请在 .env 文件中配置。"
            )
        print("[INFO] 使用 OpenAI Embeddings 进行向量化。")
        return OpenAIEmbeddings(
            api_key=openai_key,
            model="text-embedding-3-small",
        )
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("[INFO] 使用 HuggingFace 本地模型（BAAI/bge-small-zh-v1.5）进行向量化。")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def get_llm() -> ChatOpenAI:
    """
    初始化 DeepSeek LLM（通过 OpenAI 兼容接口）。

    Returns:
        ChatOpenAI 实例，指向 DeepSeek API
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 DEEPSEEK_API_KEY，请在项目根目录创建 .env 文件并填写。\n"
            "参考 .env.example 文件格式。"
        )

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_tokens=2048,
        streaming=False,
    )
    return llm


def build_rag_chain(vector_store: FAISS, top_k: int = 4) -> RetrievalQA:
    """
    构建 RetrievalQA 问答链。

    Args:
        vector_store: 已加载的 FAISS 向量库实例
        top_k: 检索时返回的相关段落数量，默认 4

    Returns:
        RetrievalQA 链实例（return_source_documents=True）
    """
    llm = get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return rag_chain


def ask_question(rag_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    使用 RAG 链回答问题。

    Args:
        rag_chain: 已构建的 RetrievalQA 链
        question: 用户问题字符串

    Returns:
        dict，包含：
            - "answer": str，模型回答
            - "source_documents": List[Document]，引用的原始文档片段
    """
    result = rag_chain.invoke({"query": question})
    return {
        "answer": result.get("result", ""),
        "source_documents": result.get("source_documents", []),
    }
