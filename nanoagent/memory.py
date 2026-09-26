"""会话记忆：多会话消息存储、滑动窗口/token 双维裁剪、JSON 文件持久化。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# 视觉模型对每张图片的计费上限（DeepSeek 实测值），用于估算多模态消息
IMAGE_TOKEN_ESTIMATE = 1024


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """原子写：临时文件 + fsync + os.replace，避免写到一半被读到/崩溃后留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def estimate_tokens(text: str) -> int:
    """token 估算：装有 tiktoken 时用 cl100k_base 精确计数，否则零依赖启发式
    （ASCII 约 4 字符/token，CJK 等全角字符约 1 字符/token）。

    用途是窗口裁剪的触发依据，不追求计费级精确。
    """
    if not text:
        return 0
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is not False:  # None=未尝试，False=不可用
        try:
            if _TIKTOKEN_ENCODER is None:
                import tiktoken

                _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
            return len(_TIKTOKEN_ENCODER.encode(text, disallowed_special=()))
        except Exception:  # noqa: BLE001 —— tiktoken 缺失/坏编码时回退启发式
            _TIKTOKEN_ENCODER = False
    wide = sum(1 for ch in text if ord(ch) > 0x2E7F)  # CJK/全角区
    return (len(text) - wide + 3) // 4 + wide


_TIKTOKEN_ENCODER: Any = None  # None=未尝试；False=不可用；否则为编码器


def message_tokens(content) -> int:
    """估算一条消息的 token 数；content 为多模态数组时按文本+图片上限计。"""
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                total += estimate_tokens(part.get("text", ""))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                total += IMAGE_TOKEN_ESTIMATE
        return total
    return estimate_tokens(str(content))


class Memory:
    """按 session_id 组织的对话历史，可按条数与 token 双维限窗并持久化到磁盘。"""

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        persist_path: str | Path | None = None,
    ):
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages 必须 >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens 必须 >= 1")
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.persist_path = Path(persist_path) if persist_path else None
        self._sessions: Dict[str, List[dict]] = {}
        # 持久化文件损坏时置位：禁止 save() 用空历史覆盖用户的原始数据
        self._persist_blocked = False
        if self.persist_path and self.persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _over_limit(self, history: List[dict]) -> bool:
        if self.max_messages and len(history) > self.max_messages:
            return True
        if self.max_tokens and self._history_tokens(history) > self.max_tokens:
            return True
        return False

    def _history_tokens(self, history: List[dict]) -> int:
        return sum(message_tokens(m.get("content", "")) for m in history)

    def add(self, session_id: str, role: str, content) -> None:
        """追加一条消息，超过条数/token 任一上限时调用 _trim 裁剪。"""
        history = self._sessions.setdefault(session_id, [])
        history.append(
            {"role": role, "content": content, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        )
        if self._over_limit(history):
            self._trim(session_id, history)

    def _trim(self, session_id: str, history: List[dict]) -> None:
        """超窗时的裁剪策略：从最旧开始丢，直到条数与 token 都回到限内。

        子类可覆盖（如 SummaryMemory 改为压缩）。单条消息自身超 token 上限时
        也会被裁掉（此时该会话历史可能为空，由调用方决定是否继续写入）。
        """
        while history and self._over_limit(history):
            del history[0]

    def tokens(self, session_id: str) -> int:
        """返回指定会话当前历史的估算 token 总数。"""
        return self._history_tokens(self._sessions.get(session_id, []))

    def history(self, session_id: str) -> List[dict]:
        """返回指定会话的消息副本（去掉内部时间戳字段）。"""
        return [{"role": m["role"], "content": m["content"]} for m in self._sessions.get(session_id, [])]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def sessions(self) -> List[str]:
        return list(self._sessions)

    # ------------------------------------------------------------------
    def save(self) -> Optional[Path]:
        """把全部会话原子写入 JSON 文件，返回文件路径；未配置持久化路径时返回 None。

        若加载时发现文件已损坏（_persist_blocked），这里**拒绝写入** ——
        否则会用当前的空历史覆盖掉用户那份仍然可人工修复的原始数据。
        """
        if not self.persist_path:
            return None
        if self._persist_blocked:
            _logger.warning(
                "记忆文件此前解析失败，已阻止写入以避免覆盖原始数据: %s", self.persist_path
            )
            return None
        _atomic_write_text(
            self.persist_path,
            json.dumps(self._sessions, ensure_ascii=False, indent=2),
        )
        return self.persist_path

    def _load(self) -> None:
        try:
            raw = self.persist_path.read_text(encoding="utf-8")
        except OSError as exc:
            _logger.warning("记忆文件读取失败: %s（%s）", self.persist_path, exc)
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # 绝不静默清空历史：备份原文件、告警、并锁定写入
            backup = self._backup_corrupt()
            self._persist_blocked = True
            _logger.warning(
                "记忆文件无法解析（%s），原文件已备份到 %s；本次不加载历史，"
                "且在修复前不会写回，以免覆盖原始数据。",
                exc, backup,
            )
            return
        if isinstance(data, dict):
            self._sessions = {
                sid: [m for m in msgs if isinstance(m, dict) and "role" in m and "content" in m]
                for sid, msgs in data.items()
                if isinstance(msgs, list)
            }

    def _backup_corrupt(self) -> Path:
        """把损坏的持久化文件另存一份，返回备份路径（失败时返回原路径）。"""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.persist_path.with_name(f"{self.persist_path.name}.corrupt-{stamp}")
        try:
            shutil.copy2(self.persist_path, backup)
        except OSError:
            return self.persist_path
        return backup


class SummaryMemory(Memory):
    """带自动摘要的记忆：超窗的旧消息先经 LLM 压缩成摘要，而不是直接丢弃。

    工作方式（惰性触发，add 时压缩）：
        - 历史超过 max_messages（条数）或 max_tokens（token 估算，自动压缩触发器）
          时，把最旧的若干条连同旧摘要一起交给 LLM 生成新摘要
        - history() 返回：[摘要 system 消息] + 最近 keep_recent 条消息
    Agent 侧无需任何改动，它只依赖 history()。
    """

    DEFAULT_SUMMARY_PROMPT = (
        "请把以下对话记录压缩成一份要点摘要（中文，300 字以内），"
        "保留用户的关键信息、明确的事实与已达成的结论，去掉寒暄与重复。\n\n对话记录：\n{transcript}"
    )

    def __init__(
        self,
        max_messages: int = 20,
        keep_recent: int = 6,
        max_tokens: int | None = None,
        llm=None,
        summary_prompt: str | None = None,
        persist_path: str | Path | None = None,
    ):
        """初始化。注意 super().__init__ 会触发 _load，因此 _summaries 必须先初始化。"""
        if keep_recent < 1:
            raise ValueError("keep_recent 必须 >= 1")
        if keep_recent >= max_messages:
            raise ValueError("keep_recent 必须小于 max_messages，否则永远不触发压缩")
        self.keep_recent = keep_recent
        self._summaries: Dict[str, str] = {}
        super().__init__(max_messages=max_messages, max_tokens=max_tokens, persist_path=persist_path)
        self.llm = llm  # 惰性创建，避免仅用 Memory 时就要求能连上模型服务
        self.summary_prompt = summary_prompt or self.DEFAULT_SUMMARY_PROMPT

    # ------------------------------------------------------------------
    def _trim(self, session_id: str, history: List[dict]) -> None:
        """覆盖父类的滑窗裁剪：把被裁掉的旧消息压缩进摘要而不是丢弃。

        keep_recent 是**软约束**（上限内尽量多保留），不是硬豁免：
        超过 token 上限时，连保留窗口内的最旧消息也会被并入摘要，
        否则 keep_recent 自身超标时裁剪就变成 no-op，上下文会持续膨胀。
        """
        if len(history) > self.keep_recent:
            self._compress(session_id, history)
        # 二次压缩：保留窗口自身仍超 token 预算时，逐步把最旧的一条并入摘要
        while len(history) > 1 and self._over_limit_with_summary(session_id, history):
            before = len(history)
            self._compress(session_id, history, keep=before - 1)
            if len(history) >= before:   # 防御：压缩没有减少条数时退化为直接丢弃
                del history[0]

    def _over_limit_with_summary(self, session_id: str, history: List[dict]) -> bool:
        """在父类上限判定之外，把摘要自身占用的 token 也算进来。"""
        if self._over_limit(history):
            return True
        if not self.max_tokens:
            return False
        summary = self._summaries.get(session_id) or ""
        if not summary:
            return False
        return self._history_tokens(history) + estimate_tokens(summary) > self.max_tokens

    def _compress(self, session_id: str, history: List[dict], keep: int | None = None) -> None:
        """把超窗的旧消息压缩进摘要，只保留最近 keep 条（默认 keep_recent）。"""
        keep = self.keep_recent if keep is None else max(1, min(keep, len(history) - 1))
        evicted = history[: len(history) - keep]
        if not evicted:
            return
        history[:] = history[-keep:]

        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in evicted)
        if self._summaries.get(session_id):
            transcript = f"已有摘要：\n{self._summaries[session_id]}\n\n新增对话：\n{transcript}"
        prompt = self.summary_prompt.format(transcript=transcript)
        self._summaries[session_id] = self._summarize(prompt).strip()

    def _summarize(self, prompt: str) -> str:
        if self.llm is None:
            from .llm import LLM

            self.llm = LLM()
        return self.llm.chat([{"role": "user", "content": prompt}]).content

    # ------------------------------------------------------------------
    def history(self, session_id: str) -> List[dict]:
        """返回 [摘要] + 最近消息；摘要以 system 角色注入历史开头。"""
        messages = super().history(session_id)
        summary = self._summaries.get(session_id)
        if summary:
            messages.insert(0, {"role": "system", "content": f"以下是此前对话的摘要：{summary}"})
        return messages

    def summary(self, session_id: str) -> str:
        """返回指定会话的当前摘要（无则空串）。"""
        return self._summaries.get(session_id, "")

    def clear(self, session_id: str) -> None:
        super().clear(session_id)
        self._summaries.pop(session_id, None)

    # ------------------------------------------------------------------
    def save(self) -> Optional[Path]:
        if not self.persist_path:
            return None
        if self._persist_blocked:
            _logger.warning(
                "记忆文件此前解析失败，已阻止写入以避免覆盖原始数据: %s", self.persist_path
            )
            return None
        payload = {
            "sessions": self._sessions,
            "summaries": self._summaries,
        }
        _atomic_write_text(
            self.persist_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        return self.persist_path

    def _load(self) -> None:
        try:
            raw = self.persist_path.read_text(encoding="utf-8")
        except OSError as exc:
            _logger.warning("记忆文件读取失败: %s（%s）", self.persist_path, exc)
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            backup = self._backup_corrupt()
            self._persist_blocked = True
            _logger.warning(
                "记忆文件无法解析（%s），原文件已备份到 %s；本次不加载历史，"
                "且在修复前不会写回，以免覆盖原始数据。",
                exc, backup,
            )
            return
        if isinstance(data, dict) and "sessions" in data:
            self._summaries = dict(data.get("summaries") or {})
            data = data["sessions"]
        if isinstance(data, dict):
            self._sessions = {
                sid: [m for m in msgs if isinstance(m, dict) and "role" in m and "content" in m]
                for sid, msgs in data.items()
                if isinstance(msgs, list)
            }
