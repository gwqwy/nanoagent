"""知识库：embedding 入库 + 检索，并可一键变成 agent 可用的工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..config import settings
from ..llm import LLM
from ..tools import tool as tool_decorator
from .chunking import split_text
from .store import VectorStore, create_store


class KnowledgeBase:
    """基于 OpenAI 兼容 /embeddings 接口的本地知识库。

    backend: "numpy"（默认，零额外依赖）或 "faiss"（大数据量更快，
    需安装 nanoagent[faiss]）。注意 persist_path 与后端绑定：
    两个后端的存档格式不同，切换后端后需重新入库。
    """

    def __init__(
        self,
        llm: LLM | None = None,
        embedding_model: str | None = None,
        persist_path: str | Path | None = None,
        backend: str = "numpy",
    ):
        self.llm = llm or LLM()
        self.embedding_model = embedding_model or settings()["embedding_model"]
        self.backend = backend
        self.store = create_store(backend)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path:
            if backend == "numpy" and self.persist_path.exists():
                self.store = VectorStore.load(self.persist_path)
            elif backend == "faiss" and Path(f"{self.persist_path}.meta.json").exists():
                self.store = type(self.store).load(self.persist_path)

    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self.llm.embeddings(texts, model=self.embedding_model)
        return np.asarray(vectors, dtype=np.float32)

    def add_text(self, text: str, metadata: Optional[dict] = None) -> int:
        """切分并入库一段文本，返回新增 chunk 数。"""
        chunks = split_text(text)
        if not chunks:
            return 0
        vectors = self._embed(chunks)
        metas = [dict(metadata or {}) for _ in chunks]
        self.store.add(chunks, vectors, metas)
        self._autosave()
        return len(chunks)

    def add_file(self, path: str | Path, metadata: Optional[dict] = None) -> int:
        """读取 txt/markdown 文件入库。"""
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        meta = dict(metadata or {})
        meta.setdefault("source", source.name)
        return self.add_text(text, meta)

    def query(self, question: str, k: int = 4, where: Optional[dict] = None,
              rerank: bool = False) -> List[dict]:
        """检索最相关的 k 条 chunk。

        where: 元数据等值过滤（如 {"source": "a.md"}）
        rerank: True 时先向量召回 k*3 条，再让 LLM 按相关性重排序取前 k
        （多一次 LLM 调用换精度，适合知识库较大、向量召回噪声多的场景）
        """
        if len(self.store) == 0:
            return []
        vector = self._embed([question])[0]
        fetch = k * 3 if rerank else k
        hits = self.store.search(vector, k=fetch, where=where)
        if rerank and len(hits) > 1:
            hits = self._llm_rerank(question, hits)[:k]
        return hits[:k]

    RERANK_PROMPT = (
        "你是检索重排序器。根据问题与各候选片段的相关性，把最相关的排前面。\n"
        "只输出一个 JSON 数组，元素为候选编号（如 [2,0,1]），不要解释。\n\n"
        "问题：{question}\n\n候选：\n{candidates}"
    )

    def _llm_rerank(self, question: str, hits: List[dict]) -> List[dict]:
        """LLM 重排序：解析失败或输出不完整时原样返回向量序（保守降级）。"""
        from ..agent import extract_json

        candidates = "\n".join(f"{i}: {hit['text'][:200]}" for i, hit in enumerate(hits))
        prompt = self.RERANK_PROMPT.format(question=question, candidates=candidates)
        try:
            data = extract_json(self.llm.chat([{"role": "user", "content": prompt}]).content)
            if not isinstance(data, list):
                return hits
            order = [int(i) for i in data if isinstance(i, (int, float)) and 0 <= int(i) < len(hits)]
            if not order:
                return hits
            seen: set = set()
            ordered: List[dict] = []
            for i in order:
                if i not in seen:
                    ordered.append(hits[i])
                    seen.add(i)
            ordered.extend(hit for i, hit in enumerate(hits) if i not in seen)
            return ordered
        except Exception:  # noqa: BLE001
            return hits

    # ------------------------------------------------------------------
    def _autosave(self) -> None:
        if self.persist_path:
            self.store.save(self.persist_path)

    def save(self) -> Optional[Path]:
        self._autosave()
        return self.persist_path

    def as_tool(
        self,
        k: int = 4,
        name: str = "search_knowledge_base",
        description: str = "在本地知识库中检索与 query 最相关的内容并返回原文片段。",
    ) -> Callable:
        """把知识库检索包装成工具，注册进 Agent 即可让模型按需查询。"""

        @tool_decorator(name=name, description=description)
        def search_knowledge_base(query: str) -> str:
            results = self.query(query, k=k)
            if not results:
                return "知识库中没有找到相关内容。"
            return "\n\n".join(
                f"[{i + 1}] (相关度 {r['score']:.3f}) {r['text']}"
                for i, r in enumerate(results)
            )

        return search_knowledge_base
