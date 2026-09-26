"""HTTP 服务入口（可选依赖 fastapi/uvicorn）。

启动：
    python -m nanoagent.server                  # 默认 127.0.0.1:8000
    uvicorn nanoagent.server:app --port 8000

接口：
    POST /chat                      {"session_id": "...", "message": "..."}
    POST /sessions/{session_id}/clear
    GET  /sessions
    GET  /health

鉴权：设置环境变量 NANOAGENT_API_TOKEN 后，除 /health 外的接口都要求
`Authorization: Bearer <token>`。未设置时默认只监听 127.0.0.1（本机使用）；
若要对外暴露（host=0.0.0.0）则**必须**设置该 token，否则拒绝启动。
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from typing import Dict, List, Optional

from pydantic import BaseModel

from .agent import Agent
from .config import load_dotenv, settings
from .llm import AsyncLLM, LLM
from .memory import Memory
from .tracing import Tracer

SESSIONS_FILE = ".nanoagent/sessions.json"
API_TOKEN_ENV = "NANOAGENT_API_TOKEN"


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


def _api_token() -> str:
    """服务端 token（空串表示未启用鉴权，仅限本机监听）。"""
    load_dotenv()
    return (os.environ.get(API_TOKEN_ENV) or "").strip()


def create_app(agent: Agent | None = None):
    """应用工厂，测试时可注入 mock agent。"""
    from fastapi import Depends, FastAPI, Header, HTTPException

    agent = agent or build_agent()

    app = FastAPI(title="nanoagent", version=_version())

    def require_token(authorization: Optional[str] = Header(default=None)) -> None:
        """启用鉴权时校验 Bearer token；未配置 token 时直接放行（本机模式）。"""
        token = _api_token()
        if not token:
            return
        presented = (authorization or "").strip()
        if not hmac.compare_digest(presented, f"Bearer {token}"):
            raise HTTPException(
                status_code=401,
                detail=f"缺少或无效的 Authorization 头（应为 Bearer <{API_TOKEN_ENV}>）",
            )

    @app.post("/chat")
    async def chat(req: ChatRequest, _: None = Depends(require_token)) -> ChatResponse:
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
    async def chat_stream(req: ChatRequest, _: None = Depends(require_token)):
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
    def clear_session(session_id: str, _: None = Depends(require_token)) -> Dict[str, str]:
        agent.memory.clear(session_id)
        return {"status": "cleared", "session_id": session_id}

    @app.get("/sessions")
    def sessions(_: None = Depends(require_token)) -> Dict[str, List[str]]:
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
    import sys

    import uvicorn

    host = os.environ.get("NANOAGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("NANOAGENT_PORT", "8000"))
    if host not in ("127.0.0.1", "localhost", "::1") and not _api_token():
        print(
            f"拒绝启动：监听 {host} 会把接口暴露到本机之外，"
            f"请先设置环境变量 {API_TOKEN_ENV}（Bearer token）再启动。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    uvicorn.run("nanoagent.server:app", host=host, port=port)
