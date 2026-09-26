"""本地向量库：numpy 余弦相似度检索，支持 JSON 持久化。

面向 demo 与中小规模知识库（几千条以内）够用；
数据量大时应换 faiss / milvus 等专用引擎。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class VectorStore:
    """内存向量库：texts / vectors / metadata 三列对齐存储。"""

    def __init__(self) -> None:
        self.texts: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # shape: (n, dim)
        self.metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.texts)

    # ------------------------------------------------------------------
    def add(self, texts: List[str], vectors: np.ndarray, metadata: Optional[List[dict]] = None) -> None:
        """批量追加。vectors 形状必须是 (len(texts), dim)。"""
        if len(texts) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("vectors 形状必须为 (len(texts), dim)")
        if self.vectors is not None and self.vectors.shape[1] != vectors.shape[1]:
            raise ValueError(
                f"向量维度不一致: 已有 {self.vectors.shape[1]}，新传入 {vectors.shape[1]}"
            )
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata 数量必须与 texts 一致")
        self.texts.extend(texts)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, query_vector, k: int = 4, where: Optional[Dict[str, Any]] = None) -> List[dict]:
        """余弦相似度检索，返回 [{text, score, metadata}, ...]，按分数降序。

        where: 元数据过滤条件（等值匹配），如 {"source": "manual.md"}；
        先过滤后取 top-k，保证返回数量不被无关条目挤占。
        """
        if self.vectors is None or not self.texts:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("查询向量维度与库内不一致")

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        matrix_norm = np.linalg.norm(self.vectors, axis=1)
        matrix_norm[matrix_norm == 0] = 1.0
        scores = (self.vectors @ query) / (matrix_norm * query_norm)

        candidate_indices = range(len(self.texts))
        if where:
            candidate_indices = [
                i for i in candidate_indices
                if all(self.metadata[i].get(key) == value for key, value in where.items())
            ]
        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)
        return [
            {
                "text": self.texts[i],
                "score": float(scores[i]),
                "metadata": self.metadata[i],
            }
            for i in ranked[:k]
        ]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """持久化为 JSON（向量转 list，规模不大时简单可靠）。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "texts": self.texts,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
            "metadata": self.metadata,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"向量库文件不存在: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"向量库文件不是合法 JSON: {source}（{exc}）") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"向量库文件结构错误（应为对象）: {source}")
        store = cls()
        vectors = payload.get("vectors") or []
        store.texts = list(payload.get("texts") or [])
        metadata = payload.get("metadata")
        store.metadata = list(metadata) if isinstance(metadata, list) else []
        # 形状校验：三者长度必须一致，否则 search() 会在 metadata[i] 上 IndexError
        if len(store.metadata) < len(store.texts):
            store.metadata = store.metadata + [{} for _ in range(len(store.texts) - len(store.metadata))]
        if vectors:
            array = np.asarray(vectors, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError(f"向量库 vectors 形状非法（应为二维矩阵）: {source}")
            if array.shape[0] != len(store.texts):
                raise ValueError(
                    f"向量库数据不一致：{len(store.texts)} 条文本 vs {array.shape[0]} 条向量（{source}）"
                )
            store.vectors = array
        elif store.texts:
            store.vectors = None
        return store


def create_store(backend: str = "numpy"):
    """向量库工厂：按 backend 名称创建对应后端实例。

    - "numpy"（默认）：纯 numpy 暴力余弦，零额外依赖，适合几千条以内
    - "faiss"：faiss IndexFlatIP 精确检索，大数据量更快，
      需要先安装可选依赖: pip install nanoagent[faiss]
    两个后端接口完全一致、检索结果一致，可随时互换。
    """
    if backend == "numpy":
        return VectorStore()
    if backend == "faiss":
        try:
            from .faiss_store import FaissVectorStore
        except ImportError as exc:
            raise ImportError(
                "使用 faiss 后端需要先安装: pip install nanoagent[faiss] （或 pip install faiss-cpu）"
            ) from exc
        return FaissVectorStore()
    raise ValueError(f"未知向量库后端: {backend!r}，可选 numpy / faiss")
