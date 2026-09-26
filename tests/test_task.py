"""task.py 测试：规划解析、逐项推进、checkpoint 续跑、异常隔离、异步版。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.agent import Agent
from nanoagent.task import TaskPlanningError, TaskRunner, TodoItem

from tests.mocks import MockLLM, text_response, tool_response


def plan_response(*titles: str):
    return text_response(json.dumps(
        [{"title": t, "detail": f"{t}的要求"} for t in titles], ensure_ascii=False
    ))


def make_runner(responses, checkpoint=None, on_progress=None, tools=None):
    agent = Agent(llm=MockLLM(responses), tools=tools)
    return TaskRunner(agent, checkpoint=checkpoint, on_progress=on_progress)


class PlanningTests(unittest.TestCase):
    def test_plan_and_drive_end_to_end(self):
        runner = make_runner([
            plan_response("写代码", "跑测试"),
            text_response("代码写好了"),
            text_response("测试全绿"),
            text_response("汇总：目标完成"),
        ])
        result = runner.run("做个功能")
        self.assertEqual([i.title for i in result.items], ["写代码", "跑测试"])
        self.assertTrue(result.all_done)
        self.assertEqual(result.done, 2)
        self.assertEqual(result.summary, "汇总：目标完成")

    def test_plan_wrapped_in_dict_is_tolerated(self):
        runner = make_runner([
            text_response('{"tasks": [{"title": "唯一的任务"}]}'),
            text_response("结果"),
            text_response("汇总"),
        ])
        result = runner.run("目标")
        self.assertEqual(len(result.items), 1)

    def test_plan_retry_on_bad_json(self):
        llm = MockLLM([
            text_response("这不是 JSON"),
            plan_response("任务甲"),
            text_response("干完了"),
            text_response("汇总"),
        ])
        runner = TaskRunner(Agent(llm=llm))
        result = runner.run("目标")
        self.assertEqual(result.items[0].title, "任务甲")
        # 第一次是坏回答，第二次才是正确规划：共 4 次调用
        self.assertEqual(len(llm.calls), 4)

    def test_plan_final_failure_raises(self):
        runner = make_runner([text_response("坏"), text_response("还是坏")])
        with self.assertRaises(TaskPlanningError):
            runner.run("目标")

    def test_max_tasks_caps_plan(self):
        runner = make_runner([
            plan_response("一", "二", "三"),
            text_response("r1"), text_response("r2"), text_response("r3"),
            text_response("汇总"),
        ])
        runner.max_tasks = 2
        result = runner.run("目标")
        self.assertEqual(len(result.items), 2)


class ExecutionTests(unittest.TestCase):
    def test_items_executed_sequentially_with_context(self):
        llm = MockLLM([
            plan_response("第一步", "第二步"),
            text_response("第一步的结果"),
            text_response("第二步的结果"),
            text_response("全部完成"),
        ])
        runner = TaskRunner(Agent(llm=llm))
        result = runner.run("目标")

        # 执行第 2 项时，提示词里带着第 1 项的结果（连续性）
        second_prompt = llm.calls[2]["messages"][-1]["content"]
        self.assertIn("第一步的结果", second_prompt)
        self.assertIn("第二步", second_prompt)
        self.assertEqual(result.items[1].result, "第二步的结果")

    def test_run_item_exception_marked_skipped(self):
        runner = make_runner([
            plan_response("坏项", "好项"),
            text_response("好项结果"),
            text_response("汇总"),
        ])
        original = runner._run_item

        def flaky(goal, items, item, session_id):
            if item.id == 1:
                raise RuntimeError("单项炸了")
            return original(goal, items, item, session_id)

        runner._run_item = flaky
        result = runner.run("目标")
        self.assertEqual(result.items[0].status, "skipped")
        self.assertIn("RuntimeError", result.items[0].result)
        self.assertEqual(result.items[1].status, "done")
        self.assertTrue(result.all_done)

    def test_on_progress_callback_sequence(self):
        seen = []
        runner = make_runner(
            [
                plan_response("甲", "乙"),
                text_response("r1"),
                text_response("r2"),
                text_response("汇总"),
            ],
            on_progress=seen.append,
        )
        runner.run("目标")
        statuses = [(item.id, item.status) for item in seen]
        self.assertEqual(statuses, [(1, "in_progress"), (1, "done"), (2, "in_progress"), (2, "done")])

    def test_tools_available_during_execution(self):
        llm = MockLLM([
            plan_response("用工具做事"),
            tool_response("t1", "calc", {"a": 2, "b": 3}, content=""),
            text_response("工具算完了"),
            text_response("汇总"),
        ])

        from nanoagent import tool

        @tool
        def calc(a: int, b: int) -> int:
            """相加。"""
            return a + b

        runner = TaskRunner(Agent(llm=llm, tools=[calc]))
        result = runner.run("目标")
        self.assertEqual(result.items[0].status, "done")
        self.assertEqual(result.items[0].result, "工具算完了")


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_persisted_after_plan_and_each_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "task.json")
            runner = make_runner([
                plan_response("甲", "乙"),
                text_response("r1"),
                text_response("r2"),
                text_response("汇总"),
            ], checkpoint=path)
            runner.run("目标")

            # 固定单文件后端现在用信封把多个逻辑 key 分开存放，经后端接口读取
            from nanoagent.checkpoints import JsonFileBackend

            state = JsonFileBackend(path).load("task")
            self.assertEqual(state["goal"], "目标")
            self.assertEqual(len(state["items"]), 2)
            self.assertEqual([i["status"] for i in state["items"]], ["done", "done"])

    def test_resume_continues_without_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "task.json")
            # 第一次：规划 + 完成第 1 项后"中断"（手动构造只完成一项的状态）
            runner = make_runner([
                plan_response("甲", "乙"),
                text_response("r1"),
            ], checkpoint=path)
            items = runner._plan("目标")
            items[0].status = "done"
            items[0].result = "甲的结果"
            runner._persist("目标", items, "task")

            # 续跑：新 runner 只需要第 2 项 + 汇总两次调用，规划不会被重跑
            llm2 = MockLLM([text_response("乙的结果"), text_response("最终汇总")])
            runner2 = TaskRunner(Agent(llm=llm2))
            result = TaskRunner.resume(path, runner2.agent)

            first_call = llm2.calls[0]["messages"][-1]["content"]
            self.assertIn("乙", first_call)
            self.assertNotIn("请只输出 JSON", first_call)  # 没有再规划
            self.assertEqual(result.items[0].result, "甲的结果")  # 已完成项原样保留
            self.assertEqual(result.items[1].result, "乙的结果")
            self.assertEqual(result.summary, "最终汇总")

    def test_resume_missing_file_raises(self):
        agent = Agent(llm=MockLLM([]))
        with self.assertRaises(FileNotFoundError):
            TaskRunner.resume("不存在的.json", agent)


class AsyncTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_arun_end_to_end(self):
        agent = Agent(llm=MockLLM([
            plan_response("异步甲", "异步乙"),
            text_response("a1"),
            text_response("a2"),
            text_response("异步汇总"),
        ]))
        result = await TaskRunner(agent).arun("异步目标")
        self.assertEqual(result.items[0].result, "a1")
        self.assertEqual(result.summary, "异步汇总")

    async def test_arun_plan_failure_raises(self):
        agent = Agent(llm=MockLLM([text_response("坏"), text_response("坏")]))
        with self.assertRaises(TaskPlanningError):
            await TaskRunner(agent).arun("目标")

    async def test_aresume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "task.json")
            agent = Agent(llm=MockLLM([
                plan_response("甲", "乙"),
                text_response("a1"),
            ]))
            runner = TaskRunner(agent, checkpoint=path)
            items = runner._plan("目标")
            items[0].status = "done"
            items[0].result = "甲完成"
            runner._persist("目标", items, "task")

            agent2 = Agent(llm=MockLLM([text_response("乙完成"), text_response("异步最终汇总")]))
            result = await TaskRunner.aresume(path, agent2)
            self.assertEqual(result.items[1].result, "乙完成")
            self.assertEqual(result.summary, "异步最终汇总")


if __name__ == "__main__":
    unittest.main()
