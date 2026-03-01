"""
utils/document_loader.py
文档加载与分块工具

支持格式：PDF（pypdf）、TXT、Markdown（.md）
"""

import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_documents(uploaded_files) -> List[Document]:
    """
    接收 Streamlit UploadedFile 列表，加载并分块，返回 Document 列表。

    Args:
        uploaded_files: Streamlit file_uploader 返回的文件对象列表

    Returns:
        List[Document]: 分块后的文档列表
    """
    all_documents: List[Document] = []

    # 文本分割器：chunk_size=500，chunk_overlap=50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()

        try:
            if file_ext == ".pdf":
                # PDF 文件：写入临时文件后用 PyPDFLoader 加载
                docs = _load_pdf(uploaded_file, file_name)
            elif file_ext in (".txt", ".md"):
                # TXT / Markdown：直接读取字节内容
                docs = _load_text(uploaded_file, file_name)
            else:
                # 不支持的格式跳过
                print(f"[WARNING] 不支持的文件格式，已跳过：{file_name}")
                continue

            # 分块
            split_docs = text_splitter.split_documents(docs)

            # 为每个分块打上来源元数据
            for doc in split_docs:
                doc.metadata["source"] = file_name

            all_documents.extend(split_docs)

        except Exception as e:
            print(f"[ERROR] 加载文件 {file_name} 时发生错误：{e}")
            raise RuntimeError(f"加载文件 {file_name} 失败：{str(e)}")

    return all_documents


def _load_pdf(uploaded_file, file_name: str) -> List[Document]:
    """将 PDF UploadedFile 写入临时文件后用 PyPDFLoader 加载。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        # 补充原始文件名到 metadata
        for doc in documents:
            doc.metadata["source"] = file_name
        return documents
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_text(uploaded_file, file_name: str) -> List[Document]:
    """加载 TXT / Markdown 文件，返回单个 Document。"""
    # 尝试 UTF-8，失败则用 GBK
    raw_bytes = uploaded_file.read()
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            content = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        content = raw_bytes.decode("utf-8", errors="replace")

    document = Document(
        page_content=content,
        metadata={"source": file_name},
    )
    return [document]
