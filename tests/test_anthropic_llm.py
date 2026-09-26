"""anthropic_llm.py 测试：协议转换、SSE 解析、MockTransport 端到端（含 Agent 集成）。"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from nanoagent.agent import Agent
from nanoagent.anthropic_llm import (
    AnthropicLLM,
    AsyncAnthropicLLM,
    finalize_stream,
    from_anthropic_response,
    parse_sse_event,
    to_anthropic_messages,
    to_anthropic_tools,
)
from nanoagent.llm import LLMResponse

from tests.mocks import text_response  # noqa: F401  保持与其它测试一致（占位）


class MessageConversionTests(unittest.TestCase):
    def test_system_hoisted_to_top_level(self):
        messages = [
            {"role": "system", "content": "规则A"},
            {"role": "user", "content": "你好"},
        ]
        system, out = to_anthropic_messages(messages)
        self.assertEqual(system, "规则A")
        self.assertEqual(out, [{"role": "user", "content": "你好"}])

    def test_tool_result_becomes_user_tool_result_block(self):
        messages = [
            {"role": "user", "content": "算一下"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "add", "arguments": "{\"a\": 1, \"b\": 2}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "3"},
        ]
        system, out = to_anthropic_messages(messages)
        self.assertEqual(out[1]["role"], "assistant")
        self.assertEqual(out[1]["content"], [
            {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1, "b": 2}}
        ])
        self.assertEqual(out[2]["role"], "user")
        self.assertEqual(out[2]["content"], [
            {"type": "tool_result", "tool_use_id": "t1", "content": "3"}
        ])

    def test_consecutive_tool_results_merge_into_one_user_message(self):
        messages = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                            {"id": "t2", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "r1"},
            {"role": "tool", "tool_call_id": "t2", "content": "r2"},
        ]
        _system, out = to_anthropic_messages(messages)
        self.assertEqual(len(out), 2)  # assistant + 一条合并的 user
        self.assertEqual(len(out[1]["content"]), 2)

    def test_multimodal_parts_converted(self):
        data_url = "data:image/png;base64,QUJD"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "看图"},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}},
        ]}]
        _system, out = to_anthropic_messages(messages)
        blocks = out[0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "看图"})
        self.assertEqual(blocks[1], {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "QUJD"}})
        self.assertEqual(blocks[2], {"type": "image", "source": {"type": "url", "url": "https://x.com/a.png"}})

    def test_tools_schema_conversion(self):
        tools = [{"type": "function", "function": {
            "name": "add", "description": "相加", "parameters": {"type": "object", "properties": {}}}}]
        converted = to_anthropic_tools(tools)
        self.assertEqual(converted, [{
            "name": "add", "description": "相加",
            "input_schema": {"type": "object", "properties": {}},
        }])
        self.assertIsNone(to_anthropic_tools(None))


class ResponseParsingTests(unittest.TestCase):
    def test_text_only_response(self):
        response = from_anthropic_response({
            "content": [{"type": "text", "text": "你好"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        self.assertEqual(response.content, "你好")
        self.assertFalse(response.has_tool_calls)
        self.assertEqual(response.usage, {"prompt_tokens": 10, "completion_tokens": 5})

    def test_tool_use_response(self):
        response = from_anthropic_response({
            "content": [
                {"type": "text", "text": "我来算"},
                {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1, "b": 2}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 8},
        })
        self.assertEqual(response.content, "我来算")
        self.assertEqual(response.tool_calls[0].name, "add")
        self.assertEqual(response.tool_calls[0].arguments, {"a": 1, "b": 2})


class SSEParseTests(unittest.TestCase):
    def _stream_state(self):
        return {"blocks": {}, "usage": {}}

    def test_full_stream_assembly(self):
        state = self._stream_state()
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你好"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "呀"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "t1", "name": "add"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "1}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ]
        yielded = [text for event in events if (text := parse_sse_event(event, state))]
        self.assertEqual("".join(yielded), "你好呀")

        response = finalize_stream(state)
        self.assertEqual(response.content, "你好呀")
        self.assertEqual(response.tool_calls[0].name, "add")
        self.assertEqual(response.tool_calls[0].arguments, {"a": 1})
        self.assertEqual(response.usage, {"prompt_tokens": 11, "completion_tokens": 9})


def _anthropic_body(content_blocks, usage=(10, 5)):
    return {
        "content": content_blocks,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


class MockTransportTests(unittest.TestCase):
    def test_chat_via_mock_transport(self):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            # 请求体应符合 Anthropic 协议
            self.assertIn("max_tokens", payload)
            self.assertEqual(payload["system"], "助手")
            self.assertNotIn("system", payload["messages"][0])
            self.assertEqual(request.headers["x-api-key"], "test-key")
            return httpx.Response(200, json=_anthropic_body([{"type": "text", "text": "收到"}], (7, 3)))

        llm = AnthropicLLM(
            model="test-model", base_url="https://mock.anthropic", api_key="test-key",
            transport=httpx.MockTransport(handler),
        )
        response = llm.chat([{"role": "system", "content": "助手"}, {"role": "user", "content": "你好"}])
        self.assertEqual(response.content, "收到")
        self.assertEqual(response.usage["prompt_tokens"], 7)

    def test_chat_error_raises(self):
        llm = AnthropicLLM(
            model="m", base_url="https://mock.anthropic", api_key="k",
            transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "bad"})),
        )
        with self.assertRaises(RuntimeError):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_stream_via_mock_transport(self):
        sse_lines = "\n".join([
            'event: message_start',
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 4}}}',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "流式"}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "回答"}}',
            'data: {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 2}}',
            'data: {"type": "message_stop"}',
        ]) + "\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_lines.encode("utf-8"))

        llm = AnthropicLLM(
            model="m", base_url="https://mock.anthropic", api_key="k",
            transport=httpx.MockTransport(handler),
        )
        stream = llm.chat_stream([{"role": "user", "content": "hi"}])
        chunks = list(stream)
        self.assertEqual("".join(chunks), "流式回答")
        self.assertEqual(stream.response.content, "流式回答")
        self.assertEqual(stream.response.usage, {"prompt_tokens": 4, "completion_tokens": 2})

    def test_agent_end_to_end_with_tools(self):
        """Agent + AnthropicLLM 全链路：模型发起 tool_use → 执行 → 最终回答。"""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=_anthropic_body([
                    {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 2, "b": 3}},
                ]))
            payload = json.loads(request.content)
            # 第二次请求应带上 tool_result
            user_blocks = payload["messages"][-1]["content"]
            self.assertEqual(user_blocks[0]["type"], "tool_result")
            return httpx.Response(200, json=_anthropic_body([{"type": "text", "text": "和是 5"}]))

        from nanoagent import tool

        @tool
        def add(a: int, b: int) -> int:
            """相加。"""
            return a + b

        llm = AnthropicLLM(
            model="m", base_url="https://mock.anthropic", api_key="k",
            transport=httpx.MockTransport(handler),
        )
        result = Agent(llm=llm, tools=[add]).run("算 2+3")
        self.assertEqual(result.content, "和是 5")
        self.assertEqual(result.tool_calls[0]["result"], "5")


class AsyncMockTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_achat(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_anthropic_body([{"type": "text", "text": "异步收到"}]))

        llm = AsyncAnthropicLLM(
            model="m", base_url="https://mock.anthropic", api_key="k",
            transport=httpx.MockTransport(handler),
        )
        response = await llm.achat([{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "异步收到")
        await llm._client.aclose()

    async def test_aconsume_stream(self):
        sse_lines = "\n".join([
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "异步流"}}',
            'data: {"type": "message_stop"}',
        ]) + "\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_lines.encode("utf-8"))

        llm = AsyncAnthropicLLM(
            model="m", base_url="https://mock.anthropic", api_key="k",
            transport=httpx.MockTransport(handler),
        )
        stream = await llm.achat_stream([{"role": "user", "content": "hi"}])
        chunks = []
        async for text in stream:
            chunks.append(text)
        self.assertEqual("".join(chunks), "异步流")
        self.assertEqual(stream.response.content, "异步流")
        await llm._client.aclose()


if __name__ == "__main__":
    unittest.main()
