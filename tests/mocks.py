"""测试共享的 Mock LLM 与工具样本。

MockLLM 按 pre 预设的脚本依次返回 LLMResponse，
并记录每次收到的 messages，便于断言 agent loop 的行为。
"""

from __future__ import annotations

import json
from typing import List

from nanoagent.llm import LLMResponse, ToolCall


class MockLLM:
    """脚本化 LLM：chat/chat_stream/achat/achat_stream 依次吐出预设响应。"""

    model = "mock-model"

    def __init__(self, responses: List[LLMResponse]):
        self.responses = list(responses)
        self.calls: List[dict] = []

    def _next(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.responses:
            raise AssertionError("MockLLM 的预设响应已用尽")
        return self.responses.pop(0)

    def chat(self, messages, tools=None) -> LLMResponse:
        return self._next(messages, tools)

    def chat_stream(self, messages, tools=None):
        response = self._next(messages, tools)
        return _MockStream(response)

    async def achat(self, messages, tools=None) -> LLMResponse:
        return self._next(messages, tools)

    async def achat_stream(self, messages, tools=None):
        response = self._next(messages, tools)
        return AsyncMockStream(response)


class AsyncMockStream:
    """模拟 AsyncStreamResult：逐字吐出 content，结束时携带完整响应。"""

    def __init__(self, response: LLMResponse):
        self.response = response
        self._text = response.content

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._text:
            raise StopAsyncIteration
        char, self._text = self._text[0], self._text[1:]
        return char


class _MockStream:
    """模拟 StreamResult：逐字吐出 content，结束时携带完整响应。"""

    def __init__(self, response: LLMResponse):
        self.response = response
        self._text = response.content

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if not self._text:
            raise StopIteration
        char, self._text = self._text[0], self._text[1:]
        return char


def text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content)


def tool_response(call_id: str, name: str, arguments: dict, content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def assistant_tool_message(response: LLMResponse) -> dict:
    """把带 tool_calls 的响应转成应回填给模型的 assistant 消息（用于断言）。"""
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in response.tool_calls
        ],
    }
