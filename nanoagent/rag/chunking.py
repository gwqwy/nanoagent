"""文本分块：为 RAG 入库做准备，优先按段落聚合，超长段落再硬切。"""

from __future__ import annotations

from typing import List


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """把长文本切成 chunk。

    - 短段落互相聚合，尽量让每个 chunk 接近 chunk_size
    - 单段超过 chunk_size 时按字符硬切，相邻切点之间保留 overlap 重叠，
      避免关键句恰好被切在边界上
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            step = chunk_size - overlap
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size])
                if start + chunk_size >= len(para):
                    break
                start += step
            continue
        if not buffer:
            buffer = para
        elif len(buffer) + 1 + len(para) <= chunk_size:
            buffer += "\n" + para
        else:
            chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if c.strip()]
