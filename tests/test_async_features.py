"""Phase 1 对齐特性测试：异步 agent loop、结构化输出、Workflow checkpoint、MCP。

MCP 集成测试会真实拉起子进程服务器（tests/mcp_echo_server.py），
mcp SDK 未安装时自动跳过。
"""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pydantic
from pydantic import BaseModel

from nanoagent.agent import Agent, OutputValidationError, extract_json
from nanoagent.llm import LLMResponse, ToolCall
from nanoagent.memory import Memory
from nanoagent.multi import Pipeline, Team, Workflow
from nanoagent.tools import tool
from tests.mocks import MockLLM, text_response, tool_response

try:
    from nanoagent.mcp import MCPServer, mcp_schema_to_nanoagent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False


# ---------------------------------------------------------------- 工具样本
@tool
def sync_double(x: int) -> int:
    """同步翻倍。

    Args:
        x: 整数
    """
    time.sleep(0.1)
    return x * 2


@tool
async def async_square(x: int) -> int:
    """异步平方。

    Args:
        x: 整数
    """
    import asyncio

    await asyncio.sleep(0.1)
    return x * x


def make_async_agent(responses, tools=None, **kwargs):
    options = dict(
        name="async-tester",
        instructions="测试",
        llm=MockLLM(responses),
        tools=tools or [],
        memory=Memory(),
        tracer=None,
    )
    options.update(kwargs)
    return Agent(**options)


# ---------------------------------------------------------------- 异步 loop
class AsyncAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_arun_direct_answer(self):
        agent = make_async_agent([text_response("异步回答")])
        result = await agent.arun("hi")
        self.assertEqual(result.content, "异步回答")
        self.assertEqual(result.iterations, 1)

    async def test_arun_with_mixed_sync_async_tools(self):
        agent = make_async_agent(
            [
                tool_response(
                    "m1",
                    "multi",
                    {},
                ),
                text_response("全部完成"),
            ],
            tools=[sync_double, async_square],
        )
        agent.llm.responses[0].tool_calls = [
            ToolCall(id="t1", name="sync_double", arguments={"x": 3}),
            ToolCall(id="t2", name="async_square", arguments={"x": 4}),
        ]
        result = await agent.arun("并行调用两种工具")
        self.assertEqual(result.tool_calls[0]["result"], "6")   # 同步工具经线程池
        self.assertEqual(result.tool_calls[1]["result"], "16")  # 异步工具直接 await
        tool_msgs = [m for m in agent.llm.calls[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["t1", "t2"])

    async def test_arun_parallel_faster_than_serial(self):
        agent = make_async_agent([text_response("ok")], tools=[sync_double])
        calls = [ToolCall(id=f"t{i}", name="sync_double", arguments={"x": i}) for i in range(3)]
        started = time.perf_counter()
        results = await agent._aexecute_tool_calls(calls)
        elapsed = time.perf_counter() - started
        self.assertEqual([r["result"] for r in results], ["0", "2", "4"])
        self.assertLess(elapsed, 0.25, "三个 0.1s 的同步工具应在线程池中并发")

    async def test_arun_stream_events(self):
        agent = make_async_agent([text_response("流式OK")])
        events = [e async for e in agent.arun_stream("hi")]
        deltas = [e for e in events if e["type"] == "delta"]
        self.assertEqual("".join(e["text"] for e in deltas), "流式OK")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["result"].content, "流式OK")

    async def test_arun_requires_async_llm(self):
        agent = Agent(name="x", llm=object(), memory=Memory(), tracer=None)
        with self.assertRaises(TypeError):
            await agent.arun("hi")

    async def test_multi_arun(self):
        a = Agent(name="a", instructions="", llm=MockLLM([text_response("A")]), memory=Memory(), tracer=None)
        b = Agent(name="b", instructions="", llm=MockLLM([text_response("B")]), memory=Memory(), tracer=None)
        self.assertEqual(await Pipeline([a, b]).arun("go"), "B")

        leader = Agent(name="l", instructions="", llm=MockLLM([text_response("L")]), memory=Memory(), tracer=None)
        member = Agent(name="m", instructions="", llm=MockLLM([text_response("M")]), memory=Memory(), tracer=None)
        self.assertEqual(await Team(leader, [member]).arun("go"), "L")


# ---------------------------------------------------------------- 结构化输出
class Movie(BaseModel):
    title: str
    year: int


class StructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_json_answer(self):
        agent = make_async_agent([text_response('{"title": "流浪地球", "year": 2019}')],
                                 response_model=Movie)
        result = await agent.arun("推荐一部电影")
        self.assertEqual(result.output.title, "流浪地球")
        self.assertEqual(result.output.year, 2019)

    async def test_markdown_fence_answer(self):
        agent = make_async_agent([text_response('```json\n{"title": "AI", "year": 2026}\n```')],
                                 response_model=Movie)
        result = await agent.arun("hi")
        self.assertEqual(result.output.year, 2026)

    async def test_retry_after_invalid_json(self):
        agent = make_async_agent(
            [
                text_response("这不是 JSON"),                      # 第一次失败
                text_response('{"title": "Retry", "year": 2020}'),  # 重试成功
            ],
            response_model=Movie,
        )
        result = await agent.arun("hi")
        self.assertEqual(result.output.title, "Retry")
        # 重试的追问里应包含上次错误与 schema
        retry_prompt = agent.llm.calls[1]["messages"][-1]["content"]
        self.assertIn("JSON Schema", retry_prompt)

    async def test_retry_exhausted_raises(self):
        agent = make_async_agent(
            [text_response("坏"), text_response("还是坏")],
            response_model=Movie,
        )
        with self.assertRaises(OutputValidationError):
            await agent.arun("hi")

    async def test_sync_run_structured(self):
        agent = Agent(
            name="s",
            instructions="",
            llm=MockLLM([text_response('{"title": "Sync", "year": 1}')]),
            memory=Memory(),
            tracer=None,
            response_model=Movie,
        )
        result = agent.run("hi")
        self.assertEqual(result.output.title, "Sync")

    async def test_invalid_response_model_rejected(self):
        with self.assertRaises(TypeError):
            make_async_agent([], response_model=dict)

    def test_extract_json_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('答案是 {"a": 1} 请查收'), {"a": 1})
        with self.assertRaises(ValueError):
            extract_json("完全没有 JSON")


# ---------------------------------------------------------------- checkpoint
class CheckpointTests(unittest.IsolatedAsyncioTestCase):
    def _workflow(self, reviews, checkpoint_path, **kwargs):
        rounds = len(reviews)
        planner = Agent(name="pl", instructions="", llm=MockLLM([text_response("计划")] * rounds), memory=Memory(), tracer=None)
        coder = Agent(name="co", instructions="", llm=MockLLM([text_response(f"代码{i}") for i in range(rounds)]), memory=Memory(), tracer=None)
        reviewer = Agent(name="re", instructions="", llm=MockLLM([text_response(r) for r in reviews]), memory=Memory(), tracer=None)
        summarizer = Agent(name="su", instructions="", llm=MockLLM([text_response("汇总")] * rounds), memory=Memory(), tracer=None)
        return Workflow(planner, coder, reviewer, summarizer, checkpoint_path=checkpoint_path, **kwargs)

    async def test_checkpoint_written_each_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.json"
            workflow = self._workflow(["FAIL 一", "PASS"], checkpoint_path=path)
            await workflow.arun("任务A")
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(state["done"])
            self.assertTrue(state["passed"])
            self.assertEqual(state["rounds_done"], 2)

    async def test_resume_continues_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.json"
            # 手工构造"第一轮已完成但未通过"的断点
            path.write_text(json.dumps({
                "task": "任务B",
                "session_id": "wf-manual",
                "max_rounds": 3,
                "pass_marker": "PASS",
                "plan": "已有计划",
                "rounds_done": 1,
                "code": "代码1",
                "review": "FAIL 缺测试",
                "passed": False,
                "done": False,
            }, ensure_ascii=False), encoding="utf-8")

            # 恢复后：planner 不应再被调用，coder 只应收到第二轮
            planner = Agent(name="pl", instructions="", llm=MockLLM([text_response("不应被调用")]), memory=Memory(), tracer=None)
            coder = Agent(name="co", instructions="", llm=MockLLM([text_response("代码2")]), memory=Memory(), tracer=None)
            reviewer = Agent(name="re", instructions="", llm=MockLLM([text_response("PASS")]), memory=Memory(), tracer=None)
            summarizer = Agent(name="su", instructions="", llm=MockLLM([text_response("最终汇总")]), memory=Memory(), tracer=None)

            result = await Workflow.aresume(path, planner, coder, reviewer, summarizer)
            self.assertTrue(result.passed)
            self.assertEqual(result.rounds, 2)
            self.assertEqual(result.code, "代码2")
            self.assertEqual(len(planner.llm.calls), 0, "恢复后不应重新做计划")
            second_prompt = coder.llm.calls[0]["messages"][-1]["content"]
            self.assertIn("缺测试", second_prompt)

            # checkpoint 应更新为完成态
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(state["done"])

    async def test_sync_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.json"
            path.write_text(json.dumps({
                "task": "任务C", "session_id": "s", "max_rounds": 2, "pass_marker": "PASS",
                "plan": "P", "rounds_done": 0, "code": "", "review": "", "passed": False, "done": False,
            }, ensure_ascii=False), encoding="utf-8")
            planner = Agent(name="pl", instructions="", llm=MockLLM([text_response("x")]), memory=Memory(), tracer=None)
            coder = Agent(name="co", instructions="", llm=MockLLM([text_response("C1")]), memory=Memory(), tracer=None)
            reviewer = Agent(name="re", instructions="", llm=MockLLM([text_response("PASS")]), memory=Memory(), tracer=None)
            summarizer = Agent(name="su", instructions="", llm=MockLLM([text_response("S")]), memory=Memory(), tracer=None)
            result = Workflow.resume(path, planner, coder, reviewer, summarizer)
            self.assertTrue(result.passed)
            self.assertEqual(result.rounds, 1)

    async def test_resume_missing_file_raises(self):
        agents = [Agent(name=f"a{i}", instructions="", llm=MockLLM([text_response("x")]), memory=Memory(), tracer=None) for i in range(4)]
        with self.assertRaises(FileNotFoundError):
            await Workflow.aresume("不存在.json", *agents)


# ---------------------------------------------------------------- MCP
class McpAdapterTests(unittest.TestCase):
    @unittest.skipUnless(HAS_MCP, "需要安装 mcp")
    def test_schema_adapter(self):
        class FakeTool:
            name = "web_search"
            description = "搜索网页"
            inputSchema = {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "关键词"}},
                "required": ["query"],
            }

        tool = mcp_schema_to_nanoagent(FakeTool())
        self.assertEqual(tool.name, "web_search")
        self.assertEqual(tool.parameters["properties"]["query"]["type"], "string")
        self.assertEqual(tool.parameters["required"], ["query"])
        self.assertTrue(tool.is_async)  # 转发函数是协程

    @unittest.skipUnless(HAS_MCP, "需要安装 mcp")
    def test_schema_adapter_missing_fields(self):
        class FakeTool:
            name = "noargs"
            description = None
            inputSchema = None

        tool = mcp_schema_to_nanoagent(FakeTool())
        self.assertEqual(tool.description, "MCP 工具 noargs")
        self.assertEqual(tool.parameters, {"type": "object", "properties": {}, "required": []})


@unittest.skipUnless(HAS_MCP, "需要安装 mcp")
class McpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """真实拉起子进程 MCP 服务器的端到端测试。"""

    SERVER = str(Path(__file__).resolve().parent / "mcp_echo_server.py")

    async def test_end_to_end_with_agent(self):
        server = await MCPServer.connect_stdio(sys.executable, [self.SERVER])
        try:
            tools = await server.tools()
            self.assertEqual(sorted(t.name for t in tools), ["add", "echo"])
            echo_tool = next(t for t in tools if t.name == "echo")
            self.assertEqual(echo_tool.parameters["properties"]["text"]["type"], "string")

            agent = make_async_agent(
                [tool_response("mc-1", "echo", {"text": "你好"}), text_response("MCP 调用成功")],
                tools=tools,
            )
            result = await agent.arun("调一下 echo")
            self.assertEqual(result.tool_calls[0]["result"], "echo: 你好")
            self.assertEqual(result.content, "MCP 调用成功")
        finally:
            await server.disconnect()


if __name__ == "__main__":
    unittest.main()
