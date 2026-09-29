"""OpenAI 兼容的 LLM 客户端封装。

参考 Agno 的做法：把模型调用收敛到一个薄封装里，
框架其余部分只依赖 LLMResponse 这个稳定的数据结构，
换模型/换服务商只改这里的 base_url。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .config import settings

# 框架级重试：对限流(429)/连接失败/5xx 指数退避；SDK 自带的重试之外再兜一层
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _should_retry(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return type(exc).__name__ in ("APIConnectionError", "APITimeoutError")


def _with_retry(fn, max_retries: int, backoff: float):
    """执行 fn()；对可重试异常指数退避后重试，耗尽后抛出最后一个异常。"""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not _should_retry(exc):
                raise
            time.sleep(backoff * (2 ** attempt))


def parse_tool_arguments(raw: str | None) -> dict:
    """把模型返回的 JSON 字符串参数解析为 dict，容错空串/坏 JSON。"""
    if not raw:
        return {}
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


@dataclass
class ToolCall:
    """一次工具调用请求（由模型发起）。"""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """统一的模型响应结构，框架内部所有模块只认它。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    reasoning: str = ""  # 思考过程（reasoning_content，模型支持时非空）

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class StreamResult:
    """流式结果：可 for 循环拿文本增量，迭代结束后 .response 是完整 LLMResponse。"""

    def __init__(self) -> None:
        self.response: Optional[LLMResponse] = None
        self._generator: Optional[Iterator[str]] = None

    def bind(self, generator: Iterator[str]) -> "StreamResult":
        self._generator = generator
        return self

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._generator is None:
            raise RuntimeError("StreamResult 未绑定生成器")
        return next(self._generator)

    def close(self) -> None:
        """提前终止：关闭底层生成器（未读完的 HTTP 流随之释放）。

        bind 的对象未必是真生成器（测试常绑普通迭代器），无 close 时跳过。
        """
        gen, self._generator = self._generator, None
        if gen is None:
            return
        close = getattr(gen, "close", None)
        if callable(close):
            close()


async def _aretry(fn, max_retries: int, backoff: float):
    """_with_retry 的异步版。"""
    import asyncio

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not _should_retry(exc):
                raise
            await asyncio.sleep(backoff * (2 ** attempt))


# 审计 N-10e：多个 agent 线程共享同一 LLM 实例时，total 的读-改-写必须互斥
# （实测 2 线程 × 20 万次累加曾丢失 4 万+ 次更新）
_ACCUMULATE_LOCK = threading.Lock()


def _accumulate(total: Dict[str, int], usage: Dict[str, int]) -> None:
    with _ACCUMULATE_LOCK:
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            total[key] = total.get(key, 0) + (usage.get(key, 0) or 0)


def _usage_dict(u: Any) -> Dict[str, int]:
    """SDK 的 usage 对象 → 统一用量字典（含缓存命中字段，防御式提取）。

    - DeepSeek 约定：``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
    - OpenAI 约定：``prompt_tokens_details.cached_tokens``
    未知字段经 model_extra 也能被 getattr 读到；两套约定都归一到 ``cached_tokens``。
    """
    if not u:
        return {}
    usage: Dict[str, int] = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
    }
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    if hit is None:
        details = getattr(u, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", None) if details else None
    if hit:
        usage["cached_tokens"] = int(hit)
    return usage


class LLM:
    """OpenAI 兼容接口客户端，支持非流式与流式两种调用。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        reasoning_effort: str | None = None,
        context_window: int = 1_000_000,
    ):
        from openai import OpenAI  # 延迟导入，纯逻辑单测不需要安装 openai

        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = base_url or cfg["base_url"] or "https://api.openai.com/v1"
        self.api_key = api_key or cfg["api_key"] or "missing-api-key"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.reasoning_effort = reasoning_effort  # low/medium/high；None 不发送
        self.context_window = context_window      # 上下文窗口估计值（token）
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # ------------------------------------------------------------------
    def _build_kwargs(self, messages: list, tools: list | None, stream: bool) -> dict:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream"] = True
            # 审计 N-06：流式默认不含 usage 帧，total_usage 对流式恒为零。
            # OpenAI 官方 API / DeepSeek / vLLM 均支持；个别兼容端点若返回 400，
            # 需在子类里去掉该参数。
            kwargs["stream_options"] = {"include_usage": True}
        if self.reasoning_effort:
            if self.reasoning_effort == "off":
                # 显式关闭思考：Qwen/DashScope/vLLM 等常见 OpenAI 兼容端点的约定。
                # 只留空不下发时，这些端点默认开思考——「思考模式关了还在深度思考」的根源。
                kwargs["extra_body"] = {"enable_thinking": False}
            else:
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return kwargs

    def chat(self, messages: list, tools: list | None = None) -> LLMResponse:
        """非流式对话，返回统一的 LLMResponse（限流/连接失败自动退避重试）。"""
        kwargs = self._build_kwargs(messages, tools, stream=False)
        resp = _with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            self.max_retries, self.retry_backoff,
        )
        message = resp.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=parse_tool_arguments(tc.function.arguments),
            )
            for tc in (message.tool_calls or [])
        ]
        usage = _usage_dict(resp.usage)
        _accumulate(self.total_usage, usage)
        return LLMResponse(
            content=message.content or "", tool_calls=tool_calls, usage=usage,
            reasoning=getattr(message, "reasoning_content", "") or "",
        )

    def chat_stream(self, messages: list, tools: list | None = None) -> StreamResult:
        """流式对话：迭代得到文本增量，迭代结束后从 result.response 取完整响应。"""
        kwargs = self._build_kwargs(messages, tools, stream=True)
        stream = _with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            self.max_retries, self.retry_backoff,
        )
        result = StreamResult()
        return result.bind(self._consume_stream(stream, result))

    def _consume_stream(self, stream, result: StreamResult) -> Iterator[str]:
        """把 SDK 的流事件转成文本增量；结束时把完整响应写回 result。"""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls_acc: Dict[int, dict] = {}
        usage: Dict[str, int] = {}
        for event in stream:
            # usage 帧有两种形态都要接住：独立的无 choices 帧（OpenAI 官方），
            # 以及挂在最后一个带 choices 帧上的 usage（部分网关如此——此前只认
            # 前者，导致这些端点的 total_usage 恒为 0、用量统计全是 0）
            event_usage = getattr(event, "usage", None)
            if event_usage and (getattr(event_usage, "prompt_tokens", 0)
                                or getattr(event_usage, "completion_tokens", 0)):
                usage = _usage_dict(event_usage)
            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            if delta is None:
                continue
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            if delta.content:
                content_parts.append(delta.content)
                yield delta.content
            for tc in delta.tool_calls or []:
                index = tc.index if tc.index is not None else len(tool_calls_acc)
                acc = tool_calls_acc.setdefault(index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    acc["id"] += tc.id
                if tc.function and tc.function.name:
                    acc["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    acc["args"] += tc.function.arguments
        tool_calls = [
            ToolCall(id=acc["id"], name=acc["name"], arguments=parse_tool_arguments(acc["args"]))
            for acc in tool_calls_acc.values()
        ]
        _accumulate(self.total_usage, usage)
        result.response = LLMResponse(
            content="".join(content_parts), tool_calls=tool_calls, usage=usage,
            reasoning="".join(reasoning_parts),
        )

    def embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """调用 OpenAI 兼容 /embeddings 接口，返回按输入顺序排列的向量列表。"""
        resp = self._client.embeddings.create(model=model, input=texts)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


class AsyncStreamResult:
    """异步流式结果：`async for` 拿文本增量，结束后 .response 是完整 LLMResponse。"""

    def __init__(self) -> None:
        self.response: Optional[LLMResponse] = None
        self._agen: Any = None

    def bind(self, agen) -> "AsyncStreamResult":
        self._agen = agen
        return self

    def __aiter__(self) -> "AsyncStreamResult":
        return self

    async def __anext__(self) -> str:
        if self._agen is None:
            raise RuntimeError("AsyncStreamResult 未绑定生成器")
        return await self._agen.__anext__()

    async def aclose(self) -> None:
        """提前终止：关闭底层异步生成器（未读完的 HTTP 流随之释放）。"""
        agen, self._agen = self._agen, None
        if agen is not None:
            aclose = getattr(agen, "aclose", None)
            if callable(aclose):
                await aclose()


class AsyncLLM:
    """LLM 的异步版本，方法与 LLM 一一对应（a 前缀），双轨并存。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        reasoning_effort: str | None = None,
        context_window: int = 1_000_000,
    ):
        from openai import AsyncOpenAI  # 延迟导入，纯逻辑单测不需要安装 openai

        cfg = settings()
        self.model = model or cfg["model"]
        self.base_url = base_url or cfg["base_url"] or "https://api.openai.com/v1"
        self.api_key = api_key or cfg["api_key"] or "missing-api-key"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.reasoning_effort = reasoning_effort  # low/medium/high；None 不发送
        self.context_window = context_window      # 上下文窗口估计值（token）
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    # ------------------------------------------------------------------
    def _build_kwargs(self, messages: list, tools: list | None, stream: bool) -> dict:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream"] = True
            # 审计 N-06：流式默认不含 usage 帧，total_usage 对流式恒为零。
            # OpenAI 官方 API / DeepSeek / vLLM 均支持；个别兼容端点若返回 400，
            # 需在子类里去掉该参数。
            kwargs["stream_options"] = {"include_usage": True}
        if self.reasoning_effort:
            if self.reasoning_effort == "off":
                # 显式关闭思考：同同步类（见上）
                kwargs["extra_body"] = {"enable_thinking": False}
            else:
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return kwargs

    async def achat(self, messages: list, tools: list | None = None) -> LLMResponse:
        """异步非流式对话，返回统一的 LLMResponse（限流/连接失败自动退避重试）。"""
        kwargs = self._build_kwargs(messages, tools, stream=False)
        resp = await _aretry(
            lambda: self._client.chat.completions.create(**kwargs),
            self.max_retries, self.retry_backoff,
        )
        message = resp.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=parse_tool_arguments(tc.function.arguments),
            )
            for tc in (message.tool_calls or [])
        ]
        usage = _usage_dict(resp.usage)
        _accumulate(self.total_usage, usage)
        return LLMResponse(
            content=message.content or "", tool_calls=tool_calls, usage=usage,
            reasoning=getattr(message, "reasoning_content", "") or "",
        )

    async def achat_stream(self, messages: list, tools: list | None = None) -> AsyncStreamResult:
        """异步流式对话：async for 得到文本增量，结束后从 result.response 取完整响应。"""
        kwargs = self._build_kwargs(messages, tools, stream=True)
        stream = await _aretry(
            lambda: self._client.chat.completions.create(**kwargs),
            self.max_retries, self.retry_backoff,
        )
        result = AsyncStreamResult()
        return result.bind(self._consume_stream(stream, result))

    async def _consume_stream(self, stream, result: AsyncStreamResult):
        """把 SDK 的异步流事件转成文本增量；结束时把完整响应写回 result。"""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls_acc: Dict[int, dict] = {}
        usage: Dict[str, int] = {}
        async for event in stream:
            # 同步版同款：两种 usage 帧形态都要接住（见同步版注释）
            event_usage = getattr(event, "usage", None)
            if event_usage and (getattr(event_usage, "prompt_tokens", 0)
                                or getattr(event_usage, "completion_tokens", 0)):
                usage = _usage_dict(event_usage)
            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            if delta is None:
                continue
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            if delta.content:
                content_parts.append(delta.content)
                yield delta.content
            for tc in delta.tool_calls or []:
                index = tc.index if tc.index is not None else len(tool_calls_acc)
                acc = tool_calls_acc.setdefault(index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    acc["id"] += tc.id
                if tc.function and tc.function.name:
                    acc["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    acc["args"] += tc.function.arguments
        tool_calls = [
            ToolCall(id=acc["id"], name=acc["name"], arguments=parse_tool_arguments(acc["args"]))
            for acc in tool_calls_acc.values()
        ]
        _accumulate(self.total_usage, usage)
        result.response = LLMResponse(
            content="".join(content_parts), tool_calls=tool_calls, usage=usage,
            reasoning="".join(reasoning_parts),
        )

    async def aembeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """异步 /embeddings 调用，返回按输入顺序排列的向量列表。"""
        resp = await self._client.embeddings.create(model=model, input=texts)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
