"""
utils/vector_store.py
FAISS 向量库管理：创建、保存、加载
"""

import os
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# 默认保存路径
DEFAULT_FAISS_PATH = "./faiss_index"


def create_vector_store(documents: list[Document], embeddings) -> FAISS:
    """
    根据文档列表和 Embedding 模型创建 FAISS 向量库。

    Args:
        documents: 分块后的 Document 列表
        embeddings: Embedding 模型实例（HuggingFaceEmbeddings 或 OpenAIEmbeddings）

    Returns:
        FAISS: 创建好的向量库实例
    """
    if not documents:
        raise ValueError("文档列表为空，无法创建向量库。")

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def save_vector_store(
    vector_store: FAISS,
    path: str = DEFAULT_FAISS_PATH,
) -> None:
    """
    将 FAISS 向量库持久化到本地磁盘。

    Args:
        vector_store: FAISS 向量库实例
        path: 保存目录路径（默认 ./faiss_index/）
    """
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f"[INFO] 向量库已保存至：{path}")


def load_vector_store(
    path: str = DEFAULT_FAISS_PATH,
    embeddings=None,
) -> Optional[FAISS]:
    """
    从本地磁盘加载 FAISS 向量库。

    Args:
        path: 向量库目录路径
        embeddings: 与创建时相同的 Embedding 模型实例

    Returns:
        FAISS 实例，若路径不存在则返回 None
    """
    index_file = os.path.join(path, "index.faiss")
    if not os.path.exists(index_file):
        print(f"[INFO] 未找到已保存的向量库：{index_file}")
        return None

    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[INFO] 向量库已从 {path} 加载。")
    return vector_store


def delete_vector_store(path: str = DEFAULT_FAISS_PATH) -> None:
    """
    删除本地保存的向量库文件。

    Args:
        path: 向量库目录路径
    """
    import shutil

    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[INFO] 向量库目录已删除：{path}")
    else:
        print(f"[INFO] 向量库目录不存在，无需删除：{path}")
