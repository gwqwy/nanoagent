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
