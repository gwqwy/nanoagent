"""RAG 知识库子包：分块、向量库（numpy/faiss 双后端）、检索。"""

from .chunking import split_text
from .faiss_store import FaissVectorStore
from .retriever import KnowledgeBase
from .store import VectorStore, create_store

__all__ = ["KnowledgeBase", "VectorStore", "FaissVectorStore", "create_store", "split_text"]
