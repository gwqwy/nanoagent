"""agent loop 测试：工具调用回填、迭代上限、记忆写入、流式事件。"""

import unittest

from nanoagent.agent import Agent
from nanoagent.llm import LLMResponse, ToolCall
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

    def test_memory_tool_traces_opt_in(self):
        """N-22：默认不存工具轨迹；开启后把工具调用以可读文本写入记忆。"""
        memory = Memory()
        responses = [
            tool_response("c1", "get_weather", {"city": "上海"}),
            text_response("上海晴。"),
        ]
        agent = make_agent(responses, memory=memory)  # 默认 memory_tool_traces=False
        agent.run("上海天气", session_id="t0")
        self.assertEqual([m["role"] for m in memory.history("t0")], ["user", "assistant"])

        memory2 = Memory()
        agent2 = make_agent(
            [
                tool_response("c1", "get_weather", {"city": "上海"}),
                text_response("上海晴。"),
            ],
            memory=memory2,
            memory_tool_traces=True,
        )
        agent2.run("上海天气", session_id="t1")
        history = memory2.history("t1")
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "assistant"])
        self.assertIn("[工具调用记录]", history[1]["content"])
        self.assertIn("get_weather", history[1]["content"])
        self.assertIn("上海 晴 25℃", history[1]["content"])
        self.assertEqual(history[2]["content"], "上海晴。")


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

    def test_stream_result_carries_usage_and_reasoning(self):
        # N-19 回归：run_stream 的 AgentResult 需与 run() 一致，聚合 usage / reasoning
        agent = make_agent(
            [
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "北京"})],
                    usage={"prompt_tokens": 10, "completion_tokens": 4},
                    reasoning="先查天气",
                ),
                LLMResponse(
                    content="北京晴。",
                    usage={"prompt_tokens": 20, "completion_tokens": 6},
                    reasoning="汇总结果",
                ),
            ]
        )
        events = list(agent.run_stream("北京天气"))
        result = events[-1]["result"]
        self.assertEqual(result.usage["prompt_tokens"], 30)
        self.assertEqual(result.usage["completion_tokens"], 10)
        self.assertEqual(result.reasoning, "先查天气\n\n汇总结果")


class StreamStopTests(unittest.TestCase):
    """should_stop 中断钩子：流式回复可被调用方请求停止（停止按钮的框架底座）。"""

    def test_stop_mid_stream(self):
        # "流式回答" 逐字吐出；第 2 个 delta 后请求停止 → 只收到 1 个 delta，无 done
        agent = make_agent([text_response("流式回答")])
        calls = {"n": 0}

        def should_stop():
            calls["n"] += 1
            return calls["n"] > 2   # 循环开始 1 次 + 第 1 个 delta 前 1 次 = 2 次放行

        events = list(agent.run_stream("测试", session_id="stop",
                                       should_stop=should_stop))
        self.assertEqual([e["type"] for e in events], ["delta"])
        self.assertEqual(events[0]["text"], "流")
        # 不写记忆：中断的回合不落盘（与消费方提前 break 的语义一致）
        self.assertEqual(agent.memory.history("stop"), [])

    def test_stop_immediately_yields_nothing(self):
        agent = make_agent([text_response("不会出现")])
        events = list(agent.run_stream("测试", should_stop=lambda: True))
        self.assertEqual(events, [])
        self.assertEqual(agent.memory.history("default"), [])

    def test_stop_false_runs_to_completion(self):
        agent = make_agent([text_response("完整回答")])
        events = list(agent.run_stream("测试", session_id="ok",
                                       should_stop=lambda: False))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["result"].content, "完整回答")
        self.assertEqual(len(agent.memory.history("ok")), 2)   # 正常落盘

    def test_async_stream_stop(self):
        import asyncio

        async def main():
            agent = make_agent([text_response("异步回答")])
            # MockLLM 自带 achat/achat_stream（满足 _require_async_llm 协议），
            # 这里只验证 should_stop 在异步路径同样生效。
            calls = {"n": 0}

            def should_stop():
                calls["n"] += 1
                return calls["n"] > 2

            events = []
            async for event in agent.arun_stream("测试", session_id="astop",
                                                 should_stop=should_stop):
                events.append(event)
            return events

        events = asyncio.run(main())
        self.assertEqual([e["type"] for e in events], ["delta"])


class ToolGateTests(unittest.TestCase):
    """tool_gate 审批中间层：每次工具执行前过门，拦截信息回填给模型（fail-closed）。"""

    def test_gate_allows_and_blocks(self):
        agent = make_agent(
            [
                tool_response("g1", "get_weather", {"city": "北京"}),
                text_response("好的，不查了。"),
            ],
            tool_gate=lambda name, args: "北京不让查" if args.get("city") == "北京" else None,
        )
        result = agent.run("查北京天气")
        blocked = [t for t in result.tool_calls if "拒绝" in t["result"]]
        self.assertEqual(len(blocked), 1)
        self.assertIn("北京不让查", blocked[0]["result"])
        # 同一 agent 换放行的城市：门不拦截，工具真实执行
        agent.llm.responses.extend(
            [tool_response("g2", "get_weather", {"city": "上海"}), text_response("上海晴。")]
        )
        result2 = agent.run("查上海天气")
        allowed = [t for t in result2.tool_calls if "晴" in t["result"]]
        self.assertEqual(len(allowed), 1)

    def test_gate_exception_fails_closed(self):
        def boom(name, args):
            raise RuntimeError("审批服务挂了")

        agent = make_agent(
            [tool_response("g1", "get_weather", {"city": "广州"}), text_response("收到。")],
            tool_gate=boom,
        )
        result = agent.run("天气")
        self.assertIn("审批回调异常", result.tool_calls[0]["result"])

    def test_no_gate_runs_normally(self):
        agent = make_agent(
            [tool_response("g1", "get_weather", {"city": "深圳"}), text_response("深圳晴。")],
            tool_gate=None,
        )
        result = agent.run("天气")
        self.assertEqual(result.tool_calls[0]["result"], "深圳 晴 25℃")


class Batch44NanoTests(unittest.TestCase):
    """第四十四批：非流式取消 / 工具超时 / 输出限幅 / 坏参数自修。"""

    def test_run_should_stop(self):
        agent = make_agent([text_response("不会出现")])
        result = agent.run("测试", session_id="ns", should_stop=lambda: True)
        self.assertTrue(result.stopped)
        self.assertEqual(result.content, "")
        self.assertEqual(agent.memory.history("ns"), [])   # 不落盘

    def test_run_should_stop_false_completes(self):
        agent = make_agent([text_response("完整回答")])
        result = agent.run("测试", session_id="ns2", should_stop=lambda: False)
        self.assertFalse(result.stopped)
        self.assertEqual(result.content, "完整回答")

    def test_tool_timeout_sync(self):
        import time as _time

        @tool
        def slow_tool() -> str:
            """故意慢。"""
            _time.sleep(1.2)
            return "终于完成"

        agent = make_agent(
            [tool_response("t1", "slow_tool", {}), text_response("收到")],
            tools=[get_weather, slow_tool],
            tool_timeout=0.2,
        )
        result = agent.run("跑")
        self.assertIn("超过 0.2 秒", result.tool_calls[0]["result"])

    def test_tool_output_limit(self):
        agent = make_agent(
            [tool_response("t1", "get_weather", {"city": "北京"}), text_response("好")],
            tool_output_limit=4,
        )
        result = agent.run("天气")
        self.assertIn("输出已截断", result.tool_calls[0]["result"])
        self.assertTrue(result.tool_calls[0]["result"].startswith("北京 晴"))

    def test_bad_arguments_json_self_correct(self):
        from nanoagent.llm import LLMResponse, ToolCall

        broken = LLMResponse(content="", tool_calls=[
            ToolCall(id="b1", name="get_weather", arguments={},
                     arguments_error="参数不是合法 JSON（ Expecting value，位置 0）")])
        agent = make_agent([broken, text_response("明白，重试")])
        result = agent.run("天气")
        # 工具未被真实执行；根因回填给模型
        self.assertIn("参数无法解析", result.tool_calls[0]["result"])
        self.assertIn("请修正参数", result.tool_calls[0]["result"])


if __name__ == "__main__":
    unittest.main()
