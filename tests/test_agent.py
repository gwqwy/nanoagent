"""agent loop 测试：工具调用回填、迭代上限、记忆写入、流式事件。"""

import unittest

from nanoagent.agent import Agent
from nanoagent.memory import Memory
from nanoagent.tools import tool
from tests.mocks import MockLLM, text_response, tool_response


@tool
def get_weather(city: str) -> str:
    """查询城市天气。

    Args:
        city: 城市名
    """
    return f"{city} 晴 25℃"


def make_agent(responses, **overrides):
    options = dict(
        name="tester",
        instructions="测试用",
        llm=MockLLM(responses),
        tools=[get_weather],
        memory=Memory(),
        tracer=None,
    )
    options.update(overrides)
    return Agent(**options)


class AgentLoopTests(unittest.TestCase):
    def test_direct_answer_without_tools(self):
        agent = make_agent([text_response("你好！")])
        result = agent.run("打个招呼", session_id="s1")
        self.assertEqual(result.content, "你好！")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls, [])

    def test_tool_call_then_final_answer(self):
        agent = make_agent(
            [
                tool_response("call-1", "get_weather", {"city": "北京"}),
                text_response("北京今天晴，25 度。"),
            ]
        )
        result = agent.run("北京天气怎么样", session_id="s1")
        self.assertEqual(result.content, "北京今天晴，25 度。")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["name"], "get_weather")
        self.assertEqual(result.tool_calls[0]["result"], "北京 晴 25℃")

        # 第二轮 LLM 调用应包含 assistant(tool_calls) 与 tool 结果消息
        second_call = agent.llm.calls[1]["messages"]
        roles = [m["role"] for m in second_call]
        self.assertIn("tool", roles)
        tool_msg = next(m for m in second_call if m["role"] == "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call-1")
        self.assertIn("晴", tool_msg["content"])

    def test_unknown_tool_returns_error_to_model(self):
        agent = make_agent(
            [
                tool_response("call-x", "not_exist", {}),
                text_response("好的，我换个说法。"),
            ]
        )
        result = agent.run("触发未知工具")
        self.assertEqual(result.content, "好的，我换个说法。")
        tool_msg = next(m for m in agent.llm.calls[1]["messages"] if m["role"] == "tool")
        self.assertIn("未注册的工具", tool_msg["content"])

    def test_max_iterations_stops_loop(self):
        agent = make_agent(
            [tool_response(f"c{i}", "get_weather", {"city": "北京"}) for i in range(3)],
            max_iterations=3,
        )
        result = agent.run("循环调用")
        self.assertEqual(result.iterations, 3)
        self.assertEqual(len(agent.llm.responses), 0)

    def test_memory_records_user_and_final(self):
        memory = Memory()
        agent = make_agent([text_response("回答一")], memory=memory)
        agent.run("问题一", session_id="m1")
        history = memory.history("m1")
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"], "问题一")
        self.assertEqual(history[1]["content"], "回答一")

    def test_history_passed_to_llm(self):
        memory = Memory()
        memory.add("h1", "user", "旧问题")
        memory.add("h1", "assistant", "旧回答")
        agent = make_agent([text_response("ok")], memory=memory)
        agent.run("新问题", session_id="h1")
        messages = agent.llm.calls[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"], "旧问题")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "新问题")


class AgentStreamTests(unittest.TestCase):
    def test_stream_events(self):
        agent = make_agent([text_response("流式回答")])
        events = list(agent.run_stream("流式测试", session_id="st"))
        deltas = [e for e in events if e["type"] == "delta"]
        done = [e for e in events if e["type"] == "done"]
        self.assertEqual("".join(e["text"] for e in deltas), "流式回答")
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["result"].content, "流式回答")

    def test_stream_with_tool_call(self):
        agent = make_agent(
            [
                tool_response("sc-1", "get_weather", {"city": "上海"}),
                text_response("上海晴。"),
            ]
        )
        events = list(agent.run_stream("上海天气"))
        tool_events = [e for e in events if e["type"] == "tool_call"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["result"], "上海 晴 25℃")
        self.assertEqual(events[-1]["result"].content, "上海晴。")


if __name__ == "__main__":
    unittest.main()
