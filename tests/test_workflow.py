"""Workflow 工作流测试：反馈回路、PASS 判定、轮次上限、会话隔离。"""

import unittest

from nanoagent.agent import Agent
from nanoagent.memory import Memory
from nanoagent.multi import Workflow, WorkflowResult
from tests.mocks import MockLLM, text_response


def make_agent(name, responses):
    return Agent(
        name=name,
        instructions="测试",
        llm=MockLLM([text_response(r) for r in responses]),
        memory=Memory(),
        tracer=None,
    )


def make_workflow(reviews, coder_replies=None, **kwargs):
    """reviews: 每轮审查回答（决定回路次数）；coder_replies: 每轮代码输出。"""
    rounds = len(reviews)
    coder_replies = coder_replies or [f"代码v{i + 1}" for i in range(rounds)]
    planner = make_agent("planner", ["计划内容"] * rounds)
    coder = make_agent("coder", coder_replies)
    reviewer = make_agent("reviewer", reviews)
    summarizer = make_agent("summarizer", ["交付说明"] * rounds)
    workflow = Workflow(planner, coder, reviewer, summarizer, **kwargs)
    return workflow, planner, coder, reviewer, summarizer


class WorkflowTests(unittest.TestCase):
    def test_pass_on_first_round(self):
        workflow, planner, coder, reviewer, summarizer = make_workflow(
            ["代码符合计划\nPASS"]
        )
        result = workflow.run("做个功能")
        self.assertTrue(result.passed)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.summary, "交付说明")
        # 各角色确实只跑了自己该跑的次数
        self.assertEqual(len(planner.llm.responses), 0)
        self.assertEqual(len(coder.llm.responses), 0)
        self.assertEqual(len(reviewer.llm.responses), 0)
        self.assertEqual(len(summarizer.llm.responses), 0)

    def test_fail_then_pass_feedback_loop(self):
        workflow, planner, coder, reviewer, summarizer = make_workflow(
            ["FAIL\n- 缺少错误处理", "已修复\nPASS"],
            coder_replies=["代码v1", "代码v2"],
        )
        result = workflow.run("做个功能")
        self.assertTrue(result.passed)
        self.assertEqual(result.rounds, 2)
        # 第二轮程序员收到的输入应包含第一轮的审查意见
        second_prompt = coder.llm.calls[1]["messages"][-1]["content"]
        self.assertIn("缺少错误处理", second_prompt)
        # 汇总员应看到最终代码与审查结论
        summary_prompt = summarizer.llm.calls[0]["messages"][-1]["content"]
        self.assertIn("代码v2", summary_prompt)
        self.assertIn("共 2 轮", summary_prompt)

    def test_max_rounds_exhausted(self):
        workflow, _, _, _, _ = make_workflow(
            ["FAIL 问题一", "FAIL 问题二", "FAIL 问题三"],
            max_rounds=3,
        )
        result = workflow.run("做个功能")
        self.assertFalse(result.passed)
        self.assertEqual(result.rounds, 3)
        self.assertEqual(result.summary, "交付说明")  # 达到上限也要汇总收尾

    def test_result_shape(self):
        workflow, _, _, _, _ = make_workflow(["PASS"])
        result = workflow.run("任务X")
        self.assertIsInstance(result, WorkflowResult)
        self.assertEqual(result.task, "任务X")
        self.assertEqual(result.plan, "计划内容")
        self.assertEqual(result.code, "代码v1")
        for field in ("task", "plan", "code", "review", "summary", "rounds", "passed"):
            self.assertTrue(hasattr(result, field))

    def test_session_isolated_between_runs(self):
        workflow, planner, _, _, _ = make_workflow(["PASS", "PASS"])
        workflow.run("任务一")
        workflow.run("任务二")
        # 每次工作流用独立会话，两次运行互不串话
        self.assertEqual(len(planner.memory.sessions()), 2)
        sessions = planner.memory.sessions()
        self.assertNotEqual(planner.memory.history(sessions[0]), [])
        self.assertNotEqual(planner.memory.history(sessions[1]), [])


class PassDetectionTests(unittest.TestCase):
    def setUp(self):
        workflow, *_ = make_workflow(["PASS"])
        self.workflow = workflow

    def test_pass_on_last_line(self):
        self.assertTrue(self.workflow._is_pass("意见若干\nPASS"))

    def test_fail_on_last_line(self):
        self.assertFalse(self.workflow._is_pass("有一点问题\nFAIL"))

    def test_fallback_pass_without_fail(self):
        self.assertTrue(self.workflow._is_pass("整体通过，符合 PASS 要求"))

    def test_fallback_fail_keyword(self):
        # 整体同时出现 PASS 和 FAIL 且末行无标记 → 判为不通过（保守）
        self.assertFalse(self.workflow._is_pass("PASS 但存在 FAIL 风险"))

    def test_empty_review_fails(self):
        self.assertFalse(self.workflow._is_pass("   \n  "))


class WorkflowValidationTests(unittest.TestCase):
    def test_rejects_non_agent_roles(self):
        with self.assertRaises(TypeError):
            Workflow("not-agent", "x", "y", "z")

    def test_rejects_invalid_max_rounds(self):
        agents = [make_agent("a", ["r"]) for _ in range(4)]
        with self.assertRaises(ValueError):
            Workflow(*agents, max_rounds=0)


if __name__ == "__main__":
    unittest.main()
