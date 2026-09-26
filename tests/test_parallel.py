"""并行工具执行测试：耗时对比、顺序保持、异常隔离、参数校验。"""

import threading
import time
import unittest

from nanoagent.agent import Agent
from nanoagent.llm import ToolCall
from nanoagent.memory import Memory
from nanoagent.tools import tool
from tests.mocks import MockLLM, text_response, tool_response

_lock = threading.Lock()
_calls = []


@tool
def slow_probe(tag: str) -> str:
    """慢探针工具，记录调用顺序。

    Args:
        tag: 标记
    """
    time.sleep(0.2)
    with _lock:
        _calls.append(tag)
    return f"done-{tag}"


@tool
def boom(tag: str) -> str:
    """必炸工具。

    Args:
        tag: 标记
    """
    raise RuntimeError(f"炸-{tag}")


def make_agent(tools, responses, **overrides):
    options = dict(
        name="p",
        instructions="",
        llm=MockLLM(responses),
        tools=tools,
        memory=Memory(),
        tracer=None,
    )
    options.update(overrides)
    return Agent(**options)


class ParallelExecutionTests(unittest.TestCase):
    def test_parallel_faster_than_serial(self):
        global _calls
        _calls = []
        agent = make_agent([slow_probe, boom], [text_response("ok")])
        calls = [
            ToolCall(id="t1", name="slow_probe", arguments={"tag": "a"}),
            ToolCall(id="t2", name="slow_probe", arguments={"tag": "b"}),
            ToolCall(id="t3", name="slow_probe", arguments={"tag": "c"}),
        ]
        started = time.perf_counter()
        results = agent._execute_tool_calls(calls)
        elapsed = time.perf_counter() - started
        self.assertEqual([r["result"] for r in results], ["done-a", "done-b", "done-c"])
        self.assertLess(elapsed, 0.5, "三个 0.2s 的工具应并行完成（串行需 0.6s）")

    def test_serial_when_disabled(self):
        agent = make_agent([slow_probe], [text_response("ok")], parallel_tools=False)
        calls = [
            ToolCall(id="t1", name="slow_probe", arguments={"tag": "a"}),
            ToolCall(id="t2", name="slow_probe", arguments={"tag": "b"}),
        ]
        started = time.perf_counter()
        agent._execute_tool_calls(calls)
        elapsed = time.perf_counter() - started
        self.assertGreaterEqual(elapsed, 0.4, "关闭并行后应串行（2 × 0.2s）")

    def test_order_preserved_with_error_isolation(self):
        agent = make_agent([slow_probe, boom], [text_response("ok")])
        calls = [
            ToolCall(id="t1", name="boom", arguments={"tag": "x"}),
            ToolCall(id="t2", name="slow_probe", arguments={"tag": "y"}),
        ]
        results = agent._execute_tool_calls(calls)
        self.assertEqual(len(results), 2)
        self.assertIn("炸-x", results[0]["result"])  # 异常转字符串，不影响其他工具
        self.assertEqual(results[1]["result"], "done-y")

    def test_invalid_max_workers_rejected(self):
        with self.assertRaises(ValueError):
            make_agent([], [], max_workers=0)

    def test_full_loop_with_parallel_tools(self):
        agent = make_agent(
            [slow_probe],
            [tool_response("m", "multi", {}), text_response("最终回答")],
        )
        # 直接在 MockLLM 响应里塞两个 tool_calls
        agent.llm.responses[0].tool_calls = [
            ToolCall(id="p1", name="slow_probe", arguments={"tag": "a"}),
            ToolCall(id="p2", name="slow_probe", arguments={"tag": "b"}),
        ]
        result = agent.run("触发并行")
        # 回填给第二轮 LLM 的消息应包含两条 tool 消息，且顺序与 tool_calls 一致
        tool_msgs = [m for m in agent.llm.calls[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["p1", "p2"])
        self.assertEqual(result.content, "最终回答")


if __name__ == "__main__":
    unittest.main()
