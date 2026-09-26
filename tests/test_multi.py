"""多 agent 编排测试：agent-as-tool 委托、Pipeline、Team。"""

import unittest

from nanoagent.agent import Agent
from nanoagent.memory import Memory
from nanoagent.multi import Pipeline, Team, agent_as_tool
from tests.mocks import MockLLM, text_response


def make_agent(name, reply):
    return Agent(
        name=name,
        instructions="测试",
        llm=MockLLM([text_response(reply)]),
        memory=Memory(),
        tracer=None,
    )


class AgentAsToolTests(unittest.TestCase):
    def test_delegate_returns_sub_agent_reply(self):
        writer = make_agent("writer", "写好了")
        tool_fn = agent_as_tool(writer)
        registered = tool_fn._tool
        self.assertEqual(registered.name, "writer")
        self.assertEqual(registered.invoke({"task": "写一首诗"}), "写好了")
        # 委托调用应写进子 agent 自己的会话
        self.assertTrue(writer.memory.sessions())


class PipelineTests(unittest.TestCase):
    def test_output_flows_through_agents(self):
        a = make_agent("a", "A的输出")
        b = make_agent("b", "B的输出")
        pipe = Pipeline([a, b])
        self.assertEqual(pipe.run("开始"), "B的输出")
        # 第二个 agent 收到的输入应是第一个的输出
        self.assertEqual(b.llm.calls[0]["messages"][-1]["content"], "A的输出")

    def test_empty_pipeline_rejected(self):
        with self.assertRaises(ValueError):
            Pipeline([])


class TeamTests(unittest.TestCase):
    def test_member_registered_as_leader_tool(self):
        leader = make_agent("leader", "完成")
        member = make_agent("researcher", "调研结果")
        team = Team(leader, [member])
        self.assertIn("researcher", team.leader.tools.names())
        self.assertEqual(team.run("做个调研"), "完成")

        # leader 的 LLM 应看到成员工具的 schema
        schemas = leader.llm.calls[0]["tools"]
        self.assertIn("researcher", [s["function"]["name"] for s in schemas])

    def test_empty_members_rejected(self):
        with self.assertRaises(ValueError):
            Team(make_agent("l", "x"), [])


if __name__ == "__main__":
    unittest.main()


class RoundtableTests(unittest.TestCase):
    """圆桌讨论：多角色轮流发言、互相可见、可选主持人总结。"""

    def test_rounds_and_transcript_order(self):
        from nanoagent.multi import Roundtable

        # 每个 agent 两轮发言 → MockLLM 需要两个回复
        a = Agent(name="alice", instructions="乐观派",
                  llm=MockLLM([text_response("观点A1"), text_response("回应：同意B")]),
                  memory=Memory(), tracer=None)
        b = Agent(name="bob", instructions="谨慎派",
                  llm=MockLLM([text_response("观点B1"), text_response("补充B2")]),
                  memory=Memory(), tracer=None)
        table = Roundtable([a, b], rounds=2)
        result = table.run("是否发布新版本")
        self.assertEqual(len(result.transcript), 4)
        self.assertEqual([t["agent"] for t in result.transcript],
                         ["alice", "bob", "alice", "bob"])
        self.assertEqual([t["round"] for t in result.transcript], [1, 1, 2, 2])
        # 后发言者能看到此前全部发言（含第 2 轮 alice 能看到 bob 的第 1 轮）
        self.assertIn("观点B1", a.llm.calls[-1]["messages"][-1]["content"])
        self.assertEqual(result.summary, "")

    def test_moderator_summarizes(self):
        from nanoagent.multi import Roundtable

        a = make_agent("alice", "我认为可以发")
        mod = make_agent("chief", "结论：发")
        table = Roundtable([a], rounds=1, moderator=mod)
        result = table.run("发布决策")
        self.assertEqual(result.summary, "结论：发")
        # 主持人收到了全部发言记录
        self.assertIn("我认为可以发", mod.llm.calls[0]["messages"][-1]["content"])

    def test_validation(self):
        from nanoagent.multi import Roundtable

        a = make_agent("alice", "x")
        with self.assertRaises(ValueError):
            Roundtable([])
        with self.assertRaises(ValueError):
            Roundtable([a], rounds=0)

    def test_no_pollution_of_agent_sessions(self):
        from nanoagent.multi import Roundtable

        a = make_agent("alice", "ok")
        Roundtable([a], rounds=1).run("议题")
        self.assertEqual(a.memory.sessions(), [])  # save=False，不污染会话
