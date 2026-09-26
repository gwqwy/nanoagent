"""HTTP 服务入口（可选依赖 fastapi/uvicorn）。

启动：
    python -m nanoagent.server                  # 默认 127.0.0.1:8000
    uvicorn nanoagent.server:app --port 8000

接口：
    POST /chat                      {"session_id": "...", "message": "..."}
    POST /sessions/{session_id}/clear
    GET  /sessions
    GET  /health
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional

from pydantic import BaseModel

from .agent import Agent
from .config import load_dotenv, settings
from .llm import AsyncLLM, LLM
from .memory import Memory
from .tracing import Tracer

SESSIONS_FILE = ".nanoagent/sessions.json"


class ChatRequest(BaseModel):
    """POST /chat 请求体。"""

    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    """POST /chat 响应体。"""

    reply: str
    tool_calls: List[dict]
    iterations: int
    output: Optional[dict] = None  # response_model 启用时的结构化输出


def build_agent() -> Agent:
    cfg = settings()
    llm = AsyncLLM(model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"])
    return Agent(
        name="nanoagent-server",
        instructions="你是 nanoagent HTTP 服务背后的 AI 助手，用中文简洁回答。",
        llm=llm,
        memory=Memory(persist_path=SESSIONS_FILE),
        tracer=Tracer(path="traces/server-trace.jsonl"),
    )


def create_app(agent: Agent | None = None):
    """应用工厂，测试时可注入 mock agent。"""
    from fastapi import FastAPI, HTTPException

    agent = agent or build_agent()

    app = FastAPI(title="nanoagent", version=_version())

    @app.post("/chat")
    async def chat(req: ChatRequest) -> ChatResponse:
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="message 不能为空")
        try:
            if hasattr(agent.llm, "achat"):  # 异步 agent 直接跑
                result = await agent.arun(req.message, session_id=req.session_id)
            else:  # 注入的同步 agent 丢线程池，避免阻塞事件循环
                result = await asyncio.to_thread(agent.run, req.message, session_id=req.session_id)
        except Exception as exc:  # noqa: BLE001 —— 上游模型错误统一转 502
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
        return ChatResponse(
            reply=result.content,
            tool_calls=result.tool_calls,
            iterations=result.iterations,
            output=None if result.output is None else result.output.model_dump(),
        )

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """SSE 流式端点。事件格式：
            data: {"type": "delta", "text": "..."}
            data: {"type": "tool_call", "name": "...", "arguments": {...}, "result": "..."}
            data: {"type": "done", "reply": "...", "tool_calls": [...], "iterations": N}
        同步 llm 时退化为单条 done 事件（在线程池中执行）。
        """
        from fastapi.responses import StreamingResponse

        if not req.message.strip():
            raise HTTPException(status_code=400, detail="message 不能为空")

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        async def event_source():
            try:
                if hasattr(agent.llm, "achat_stream"):
                    events = []
                    async for ev in agent.arun_stream(req.message, session_id=req.session_id):
                        events.append(ev)
                        if ev["type"] == "delta":
                            yield sse({"type": "delta", "text": ev["text"]})
                        elif ev["type"] == "tool_call":
                            yield sse({"type": "tool_call", "name": ev["name"],
                                       "arguments": ev["arguments"], "result": ev["result"]})
                    done = events[-1] if events and events[-1]["type"] == "done" else None
                    if done:
                        result = done["result"]
                        yield sse({"type": "done", "reply": result.content,
                                   "tool_calls": result.tool_calls,
                                   "iterations": result.iterations})
                else:
                    result = await asyncio.to_thread(agent.run, req.message, session_id=req.session_id)
                    yield sse({"type": "done", "reply": result.content,
                               "tool_calls": result.tool_calls, "iterations": result.iterations})
            except Exception as exc:  # noqa: BLE001
                yield sse({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/clear")
    def clear_session(session_id: str) -> Dict[str, str]:
        agent.memory.clear(session_id)
        return {"status": "cleared", "session_id": session_id}

    @app.get("/sessions")
    def sessions() -> Dict[str, List[str]]:
        return {"sessions": agent.memory.sessions()}

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model": agent.llm.model}

    return app


def _version() -> str:
    from . import __version__

    return __version__


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("nanoagent.server:app", host="127.0.0.1", port=8000)
