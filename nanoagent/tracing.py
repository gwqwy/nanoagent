"""Trace 追踪：把每一步 LLM 调用、工具调用记录为 JSONL 文件。

参考 LangSmith/Langfuse 的思想但做到最简：一行一个 JSON 事件，
既方便人眼查看，也方便后续接可视化面板。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional


# 并行工具会从多个工作线程调用 Tracer.log，同一文件必须以单次 write 原子落盘，
# 否则 Windows 下追加写可能交错出半行 JSON（被 load_trace 静默丢弃）。
_WRITE_LOCK = threading.Lock()


class Tracer:
    """事件追踪器。path 为 None 或 enabled=False 时为 no-op。"""

    def __init__(self, path: str | Path | None = None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self.run_id: Optional[str] = None

    def start_run(self, agent_name: str) -> str:
        """开始一次 run，返回本次 run 的 id。"""
        self.run_id = uuid.uuid4().hex[:12]
        self.log(
            "run_start",
            run_id=self.run_id,
            agent=agent_name,
        )
        return self.run_id

    def end_run(self, status: str = "ok", **extra) -> None:
        if self.run_id is not None:
            self.log("run_end", run_id=self.run_id, status=status, **extra)
        self.run_id = None

    def log(self, event_type: str, **data) -> None:
        """追加一条事件；未启用时不做任何事，写文件失败也不影响主流程。

        处于一次 run 期间时自动附带 run_id，便于事后按 run 关联事件。
        """
        if not self.enabled:
            return
        # 键名必须与 observability 的读取方一致（event.get("event")）；
        # 历史上这里写的是 "type"，导致 trace_summary/trace_report/OTel 统计恒为 0。
        record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event_type}
        if self.run_id is not None:
            record["run_id"] = self.run_id
        record.update(data)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except OSError:
            pass
