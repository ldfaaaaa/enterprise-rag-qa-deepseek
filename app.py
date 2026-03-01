"""
app.py
企业知识库智能问答系统（RAG 版）
技术栈：Streamlit + LangChain + FAISS + DeepSeek + HuggingFace Embeddings
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ─────────────────────────────────────────────
# 页面基础配置（必须在所有 st.* 调用之前）
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 延迟导入工具模块（避免启动时因缺少依赖报错）
# ─────────────────────────────────────────────
from utils.document_loader import load_documents  # noqa: E402
from utils.vector_store import (  # noqa: E402
    DEFAULT_FAISS_PATH,
    create_vector_store,
    delete_vector_store,
    save_vector_store,
)
from utils.rag_chain import ask_question, build_rag_chain, get_embeddings  # noqa: E402

# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history": [],          # List[dict]：{"role": "user"/"assistant", "content": str, "sources": list}
        "vector_store": None,        # FAISS 向量库实例
        "rag_chain": None,           # RetrievalQA 链实例
        "embeddings": None,          # Embedding 模型实例（复用，避免重复加载）
        "doc_count": 0,              # 已加载的原始文件数
        "chunk_count": 0,            # 向量库中的总段落数
        "knowledge_base_ready": False,  # 知识库是否已就绪
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ 正在加载向量化模型（首次运行需下载，请稍候）...")
def load_embeddings_model():
    """加载 Embedding 模型，使用 st.cache_resource 缓存，全局只加载一次。"""
    return get_embeddings()


def build_knowledge_base(uploaded_files):
    """
    构建知识库完整流程：
    1. 加载并分块文档
    2. 创建 FAISS 向量库
    3. 保存到本地
    4. 构建 RAG 链
    """
    progress_bar = st.progress(0, text="📄 正在读取文档...")

    # Step 1：加载文档
    with st.spinner("📄 正在解析文档..."):
        documents = load_documents(uploaded_files)
    if not documents:
        st.error("⚠️ 未能从上传文件中提取到任何内容，请检查文件格式。")
        return False

    progress_bar.progress(30, text="🔢 正在向量化文档（首次运行较慢）...")

    # Step 2：获取 Embedding 模型（缓存）
    embeddings = load_embeddings_model()
    st.session_state["embeddings"] = embeddings

    # Step 3：创建向量库
    with st.spinner("🔢 正在构建 FAISS 向量索引..."):
        vector_store = create_vector_store(documents, embeddings)

    progress_bar.progress(70, text="💾 正在保存向量库...")

    # Step 4：保存到本地
    save_vector_store(vector_store, DEFAULT_FAISS_PATH)

    progress_bar.progress(90, text="🔗 正在构建问答链...")

    # Step 5：构建 RAG 链
    try:
        rag_chain = build_rag_chain(vector_store, top_k=4)
    except ValueError as e:
        st.error(f"❌ {e}")
        return False

    # 更新 session state
    st.session_state["vector_store"] = vector_store
    st.session_state["rag_chain"] = rag_chain
    st.session_state["doc_count"] = len(uploaded_files)
    st.session_state["chunk_count"] = len(documents)
    st.session_state["knowledge_base_ready"] = True

    progress_bar.progress(100, text="✅ 知识库构建完成！")
    return True


def clear_knowledge_base():
    """清除向量库、RAG 链和聊天记录。"""
    # 删除本地向量库文件
    delete_vector_store(DEFAULT_FAISS_PATH)

    # 重置 session state
    st.session_state["vector_store"] = None
    st.session_state["rag_chain"] = None
    st.session_state["doc_count"] = 0
    st.session_state["chunk_count"] = 0
    st.session_state["knowledge_base_ready"] = False
    st.session_state["chat_history"] = []


# ─────────────────────────────────────────────
# 页面标题
# ─────────────────────────────────────────────
st.title("🧠 企业知识库智能问答系统")
st.caption("基于 RAG 架构 · DeepSeek LLM · FAISS 向量检索 · HuggingFace Embeddings")
st.divider()

# ─────────────────────────────────────────────
# 左侧边栏
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📚 知识库管理")
    st.markdown("---")

    # 文件上传组件
    st.subheader("1️⃣ 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF、TXT、Markdown 格式",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="可同时上传多个文件。上传后点击「建立知识库」按钮开始处理。",
    )

    st.markdown("---")

    # 建立知识库按钮
    st.subheader("2️⃣ 构建索引")
    build_btn = st.button(
        "🚀 建立知识库",
        type="primary",
        disabled=(not uploaded_files),
        use_container_width=True,
        help="上传文件后点击此按钮，系统将解析文档并建立向量索引。",
    )

    if build_btn and uploaded_files:
        success = build_knowledge_base(uploaded_files)
        if success:
            st.success(
                f"✅ 知识库已就绪！\n\n"
                f"- 加载文件：{st.session_state['doc_count']} 个\n"
                f"- 文本段落：{st.session_state['chunk_count']} 块"
            )

    st.markdown("---")

    # 知识库状态显示
    st.subheader("📊 当前状态")
    if st.session_state["knowledge_base_ready"]:
        st.success("🟢 知识库已就绪")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文件数", st.session_state["doc_count"])
        with col2:
            st.metric("段落数", st.session_state["chunk_count"])
    else:
        st.warning("🔴 知识库未建立")
        st.info("请上传文档并点击「建立知识库」")

    st.markdown("---")

    # 清除知识库按钮
    st.subheader("3️⃣ 清除数据")
    if st.button(
        "🗑️ 清除知识库",
        type="secondary",
        use_container_width=True,
        help="清除当前向量库和所有聊天记录。",
    ):
        clear_knowledge_base()
        st.success("✅ 知识库和聊天记录已清除。")
        st.rerun()

    st.markdown("---")
    st.caption("💡 提示：向量化模型首次运行时会自动下载，约 100MB，请耐心等待。")

# ─────────────────────────────────────────────
# 主区域：聊天界面
# ─────────────────────────────────────────────

# 展示聊天历史
for message in st.session_state["chat_history"]:
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)

        # 如果是 AI 回答且有参考来源，展示来源
        if role == "assistant" and sources:
            with st.expander("📎 参考来源", expanded=False):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get("source", "未知来源")
                    page = doc.metadata.get("page", None)
                    page_info = f" (第 {page + 1} 页)" if page is not None else ""
                    st.markdown(f"**片段 {i}** · 来源：`{source_name}`{page_info}")
                    st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    if i < len(sources):
                        st.markdown("---")

# ─────────────────────────────────────────────
# 底部输入框
# ─────────────────────────────────────────────
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 检查知识库是否就绪
    if not st.session_state["knowledge_base_ready"] or st.session_state["rag_chain"] is None:
        st.warning("⚠️ 请先在左侧上传文档并建立知识库，然后再提问。")
    else:
        # 展示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)

        # 将用户消息添加到历史
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input, "sources": []}
        )

        # 调用 RAG 链生成回答
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在检索知识库并生成回答..."):
                try:
                    result = ask_question(
                        st.session_state["rag_chain"],
                        user_input,
                    )
                    answer = result["answer"]
                    source_docs = result["source_documents"]
                except ValueError as e:
                    answer = f"❌ 配置错误：{e}"
                    source_docs = []
                except Exception as e:
                    answer = f"❌ 生成回答时发生错误：{str(e)}"
                    source_docs = []

            # 展示回答
            st.markdown(answer)

            # 展示参考来源
            if source_docs:
                with st.expander("📎 参考来源", expanded=False):
                    for i, doc in enumerate(source_docs, 1):
                        source_name = doc.metadata.get("source", "未知来源")
                        page = doc.metadata.get("page", None)
                        page_info = f" (第 {page + 1} 页)" if page is not None else ""
                        st.markdown(f"**片段 {i}** · 来源：`{source_name}`{page_info}")
                        st.text(
                            doc.page_content[:300]
                            + ("..." if len(doc.page_content) > 300 else "")
                        )
                        if i < len(source_docs):
                            st.markdown("---")

        # 将 AI 回答添加到历史（存储 source_docs 用于后续重新渲染）
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_docs,
            }
        )
