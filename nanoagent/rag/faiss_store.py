"""faiss 向量检索后端（可选依赖 nanoagent[faiss]）。

与 numpy 版 VectorStore 完全同接口，可互换：
    - 索引用 IndexFlatIP（内积），入库前对向量做 L2 归一化，
      内积在归一化向量上数学等价于余弦相似度，检索结果与 numpy 后端一致
    - 持久化：faiss.write_index 存向量索引，texts/metadata 存同名 .meta.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class FaissVectorStore:
    """基于 faiss IndexFlatIP 的精确余弦检索向量库。"""

    def __init__(self) -> None:
        import faiss

        self._faiss = faiss
        self.index = None  # faiss.Index，首次 add 时按维度惰性创建
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    def __len__(self) -> int:
        return len(self.texts)

    # ------------------------------------------------------------------
    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = self._faiss.IndexFlatIP(dim)
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(f"向量维度不一致: 已有 {self._dim}，新传入 {dim}")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        normalized = vectors.copy()
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized /= norms
        return normalized

    def add(self, texts: List[str], vectors: np.ndarray, metadata: Optional[List[dict]] = None) -> None:
        """批量追加。vectors 形状必须是 (len(texts), dim)，入库前自动 L2 归一化。"""
        if len(texts) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("vectors 形状必须为 (len(texts), dim)")
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata 数量必须与 texts 一致")
        self._ensure_index(vectors.shape[1])
        self.index.add(self._normalize(vectors))
        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, query_vector, k: int = 4, where: Optional[dict] = None) -> List[dict]:
        """余弦相似度检索，返回 [{text, score, metadata}, ...]，按分数降序。

        where 为元数据等值过滤：faiss 不支持原生过滤，先取全量再过滤取 top-k
        （与 numpy 后端语义一致，仅大数据量时性能有差异）。
        """
        if self.index is None or not self.texts:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dim:
            raise ValueError("查询向量维度与库内不一致")
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        k = min(k, len(self.texts))
        fetch = len(self.texts) if where else k  # 有过滤时需全量召回再筛
        scores, ids = self.index.search(self._normalize(query), fetch)
        results = [
            {"text": self.texts[i], "score": float(scores[0][j]), "metadata": self.metadata[i]}
            for j, i in enumerate(ids[0])
            if i != -1 and (where is None or all(self.metadata[i].get(key) == value for key, value in where.items()))
        ]
        return results[:k]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """索引存为二进制文件，texts/metadata 存为 <path>.meta.json。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            self._faiss.write_index(self.index, str(target))
        meta_path = Path(f"{target}.meta.json")
        meta_path.write_text(
            json.dumps(
                {"texts": self.texts, "metadata": self.metadata, "dim": self._dim},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":
        source = Path(path)
        meta_path = Path(f"{source}.meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"向量库元数据文件不存在: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls()
        store.texts = list(meta.get("texts") or [])
        store.metadata = list(meta.get("metadata") or [{} for _ in store.texts])
        store._dim = meta.get("dim")
        if store._dim and source.exists():
            store.index = store._faiss.read_index(str(source))
            if store.index.ntotal != len(store.texts):
                raise ValueError(
                    f"索引与元数据不一致: 索引 {store.index.ntotal} 条，texts {len(store.texts)} 条"
                )
        return store
