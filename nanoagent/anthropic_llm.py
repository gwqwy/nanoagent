"""Anthropic 协议的 LLM 客户端适配器（Messages API）。

让 nanoagent 能直连只说 Anthropic 协议的服务：Claude 官方 API、
DeepSeek 的 /anthropic 兼容端点等。与 OpenAI 版 LLM 平级，
同样收敛到 LLMResponse 这个稳定结构，Agent 侧零改动。

    from nanoagent import AnthropicLLM
    llm = AnthropicLLM(base_url="https://api.deepseek.com/anthropic", model="deepseek-flash")
    agent = Agent(llm=llm)                     # chat / 工具 / 流式全部可用

协议差异（相对 OpenAI Chat Completions）：
    - system 是请求顶层的独立字段，不在 messages 里
    - max_tokens 是必填参数
    - 工具 schema 字段是 input_schema（OpenAI 是 parameters）
    - assistant 的工具调用是 tool_use content block；工具结果以 tool_result
      block 放在下一条 user 消息里（本适配器自动归并连续的 role=tool 消息）
    - 响应的 content 是 block 数组；流式是 SSE 事件流
    - 多模态图片 block 为 {"type": "image", "source": {...}}（自动从
      OpenAI 的 image_url part 转换）

不提供 embeddings（Anthropic 协议没有该接口）。HTTP 用 httpx 直连
（openai 依赖已自带 httpx），测试可用 transport 注入 MockTransport。
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

from .config import settings
from .llm import (
    LLMResponse,
    StreamResult,
    ToolCall,
    _accumulate,
    _with_retry,
    parse_tool_arguments,
    parse_tool_arguments_ex,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 600.0
DEFAULT_CONTEXT_WINDOW = 200000   # Claude 系列默认上下文窗口（token）
MAX_RETRIES = 2
RETRY_BACKOFF = 0.5


# ======================================================================
# 协议转换（纯函数，单独可测）
# ======================================================================
def _convert_image_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI 的 image_url part → Anthropic 的 image block。"""
    url = (part.get("image_url") or {}).get("url", "")
    match = re.match(r"^data:([^;,]+);base64,(.*)$", url, re.DOTALL)
    if match:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_user_content(content: Any) -> Any:
    """user 消息 content：字符串原样；多模态数组逐块转换。"""
    if not isinstance(content, list):
        return content
    blocks: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            blocks.append(_convert_image_part(part))
    return blocks or ""


def to_anthropic_messages(messages: List[dict]) -> Tuple[str, List[dict]]:
    """OpenAI 风格消息 → (system 字符串, Anthropic messages 数组)。

    - system 消息抽取合并为顶层 system 字段
    - role=tool 归并为一条 user 消息里的 tool_result blocks
        （Anthropic 要求工具结果在 user 消息中，且 tool_use 之后必须紧跟）
    - assistant.tool_calls → assistant 消息里的 tool_use blocks
    """
    system_parts: List[str] = []
    out: List[dict] = []
    pending_results: List[Dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue

        if role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            })
            continue

        flush_results()

        if role == "assistant" and message.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in message["tool_calls"]:
                function = tc.get("function", {})
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": function.get("name", ""),
                    "input": parse_tool_arguments(function.get("arguments")),
                })
            out.append({"role": "assistant", "content": blocks})
            continue

        out.append({"role": role, "content": _convert_user_content(content)})

    flush_results()
    return "\n".join(system_parts), out


def to_anthropic_tools(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    """OpenAI 工具 schema → Anthropic 工具 schema（parameters → input_schema）。"""
    if not tools:
        return None
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


def from_anthropic_response(data: Dict[str, Any]) -> LLMResponse:
    """Anthropic 响应 → 统一 LLMResponse。"""
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(ToolCall(id=block.get("id", ""), name=block.get("name", ""),
                                       arguments=block.get("input") or {}))
    usage_raw = data.get("usage") or {}
    usage = {
        "prompt_tokens": usage_raw.get("input_tokens", 0),
        "completion_tokens": usage_raw.get("output_tokens", 0),
    }
    return LLMResponse(content="".join(text_parts), tool_calls=tool_calls, usage=usage)


# ======================================================================
# SSE 流解析
# ======================================================================
def parse_sse_event(event: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
    """处理一个 SSE data 事件，返回要产出的文本增量（无则 None）。

    state 在整个流生命周期内复用，键：
        blocks: index -> {"kind": "text"|"tool_use", "text": str, "id","name","args"}
        usage:  {"prompt_tokens", "completion_tokens"}
    """
    etype = event.get("type")
    if etype == "message_start":
        usage = (event.get("message") or {}).get("usage") or {}
        state.setdefault("usage", {})["prompt_tokens"] = usage.get("input_tokens", 0)
    elif etype == "content_block_start":
        block = event.get("content_block") or {}
        index = event.get("index", 0)
        if block.get("type") == "tool_use":
            state["blocks"][index] = {
                "kind": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""),
                "args": "", "text": "",
            }
        else:
            state["blocks"][index] = {"kind": "text", "text": "", "id": "", "name": "", "args": ""}
    elif etype == "content_block_delta":
        index = event.get("index", 0)
        entry = state["blocks"].setdefault(index, {"kind": "text", "text": "", "id": "", "name": "", "args": ""})
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            entry["text"] += delta.get("text", "")
            return delta.get("text", "")
        if delta.get("type") == "input_json_delta":
            entry["args"] += delta.get("partial_json", "")
    elif etype == "message_delta":
        usage = event.get("usage") or {}
        if usage:
            state.setdefault("usage", {})["completion_tokens"] = usage.get("output_tokens", 0)
    elif etype == "error":
        # 审计 N-10f：中途 error 事件此前被静默吞掉，残缺流被当正常响应收尾。
        # 记入 state，由 finalize_stream 抛出，调用方才能感知失败。
        err = event.get("error") or {}
        state["error"] = f"{err.get('type', 'api_error')}: {err.get('message', '未知错误')}"
    return None


def finalize_stream(state: Dict[str, Any]) -> LLMResponse:
    """流结束：把累积的 blocks/usage 组装成完整 LLMResponse。"""
    # 审计 N-10f：中途收到 error 事件时显式失败，而不是把残缺流当正常响应
    if state.get("error"):
        raise RuntimeError(f"Anthropic 流式响应中途出错: {state['error']}")
    content = "".join(
        entry["text"] for _index, entry in sorted(state["blocks"].items())
        if entry["kind"] == "text"
    )
    tool_calls = []
    for _index, entry in sorted(state["blocks"].items()):
        if entry["kind"] != "tool_use":
            continue
        args, arg_err = parse_tool_arguments_ex(entry["args"])
        tool_calls.append(ToolCall(id=entry["id"], name=entry["name"],
                                   arguments=args, arguments_error=arg_err))
    return LLMResponse(content=content, tool_calls=tool_calls, usage=state.get("usage", {}))


def build_payload(
    model: str,
    temperature: float,
    max_tokens: int,
    messages: list,
    tools: list | None,
    stream: bool,
) -> dict:
    """组装 Anthropic Messages API 请求体（同步/异步客户端共用）。"""
    system, converted = to_anthropic_messages(messages)
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": converted,
    }
    if system:
        payload["system"] = system
    converted_tools = to_anthropic_tools(tools)
    if converted_tools:
        payload["tools"] = converted_tools
    if stream:
        payload["stream"] = True
    return payload


# ======================================================================
# 客户端
# ======================================================================
def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


class AnthropicLLM:
    """Anthropic Messages API 客户端（同步），接口与 OpenAI 版 LLM 对齐。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,  # Anthropic 协议必填
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        transport: Optional[httpx.BaseTransport] = None,  # 测试注入 MockTransport
    ):
        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = (base_url or cfg["base_url"] or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or cfg["api_key"] or "missing-api-key"
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 与 OpenAI 版 LLM 对齐的公开属性：桌面端 status()/context_usage() 依赖它们，
        # 此前本类没有这两个属性 → 用量与上下文占用恒为空（缺陷审计 H-09）。
        self.context_window = context_window
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._client = httpx.Client(
            timeout=REQUEST_TIMEOUT, transport=transport,
            headers=_build_headers(self.api_key),
        )

    # ------------------------------------------------------------------
    def _build_payload(self, messages: list, tools: list | None, stream: bool) -> dict:
        return build_payload(self.model, self.temperature, self.max_tokens, messages, tools, stream)

    def _post(self, payload: dict) -> Dict[str, Any]:
        # 与 llm.py 一致：对 429 / 5xx / 连接类错误做指数退避重试
        response = _with_retry(
            lambda: self._client.post(f"{self.base_url}/v1/messages", json=payload),
            MAX_RETRIES, RETRY_BACKOFF,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Anthropic API {response.status_code}: {response.text[:500]}")
        return response.json()

    def chat(self, messages: list, tools: list | None = None) -> LLMResponse:
        """非流式对话，返回统一的 LLMResponse。"""
        result = from_anthropic_response(self._post(self._build_payload(messages, tools, stream=False)))
        _accumulate(self.total_usage, result.usage)
        return result

    def chat_stream(self, messages: list, tools: list | None = None) -> StreamResult:
        """流式对话：迭代得到文本增量，迭代结束后从 result.response 取完整响应。"""
        result = StreamResult()
        return result.bind(self._consume_stream(self._build_payload(messages, tools, stream=True), result))

    def _consume_stream(self, payload: dict, result: StreamResult) -> Iterator[str]:
        state: Dict[str, Any] = {"blocks": {}, "usage": {}}
        with self._client.stream("POST", f"{self.base_url}/v1/messages", json=payload) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Anthropic API {response.status_code}: {body[:500]}")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())
                text = parse_sse_event(event, state)
                if text:
                    yield text
        result.response = finalize_stream(state)
        _accumulate(self.total_usage, result.response.usage)

    def embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError("Anthropic 协议没有 embeddings 接口，RAG 请改用 OpenAI 兼容客户端")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """关闭底层 HTTP 连接池（此前从不关闭 → 连接泄漏）。"""
        self._client.close()

    def __enter__(self) -> "AnthropicLLM":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class AsyncAnthropicLLM:
    """Anthropic Messages API 客户端（异步），接口与 AsyncLLM 对齐。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = (base_url or cfg["base_url"] or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or cfg["api_key"] or "missing-api-key"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, transport=transport,
            headers=_build_headers(self.api_key),
        )

    async def achat(self, messages: list, tools: list | None = None) -> LLMResponse:
        payload = self._build_payload(messages, tools, stream=False)
        response = await self._aretry(
            lambda: self._client.post(f"{self.base_url}/v1/messages", json=payload)
        )
        if response.status_code != 200:
            raise RuntimeError(f"Anthropic API {response.status_code}: {response.text[:500]}")
        result = from_anthropic_response(response.json())
        _accumulate(self.total_usage, result.usage)
        return result

    async def _aretry(self, fn):
        """异步版重试：与同步路径同样的可重试判定与指数退避。"""
        import asyncio

        from .llm import _should_retry

        last: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt >= MAX_RETRIES or not _should_retry(exc):
                    raise
                await asyncio.sleep(RETRY_BACKOFF * (2 ** attempt))
        raise last  # pragma: no cover —— 循环内必然 return 或 raise

    async def achat_stream(self, messages: list, tools: list | None = None):
        from .llm import AsyncStreamResult

        result = AsyncStreamResult()
        return result.bind(self._aconsume_stream(self._build_payload(messages, tools, stream=True), result))

    async def _aconsume_stream(self, payload: dict, result):
        state: Dict[str, Any] = {"blocks": {}, "usage": {}}
        async with self._client.stream("POST", f"{self.base_url}/v1/messages", json=payload) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Anthropic API {response.status_code}: {body[:500]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())
                text = parse_sse_event(event, state)
                if text:
                    yield text
        result.response = finalize_stream(state)
        _accumulate(self.total_usage, result.response.usage)

    def _build_payload(self, messages: list, tools: list | None, stream: bool) -> dict:
        return build_payload(self.model, self.temperature, self.max_tokens, messages, tools, stream)

    async def aembeddings(self, texts: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError("Anthropic 协议没有 embeddings 接口，RAG 请改用 OpenAI 兼容客户端")

    async def aclose(self) -> None:
        """关闭底层异步 HTTP 连接池（此前从不关闭 → 连接泄漏）。"""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncAnthropicLLM":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()
