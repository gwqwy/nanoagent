"""批 1-4 新能力测试：evals、checkpoint 后端、TaskRunner 重规划、SSE 端点。"""

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
from nanoagent.checkpoints import (
    InMemoryBackend,
    JsonFileBackend,
    SQLiteBackend,
    resolve_checkpoint,
)
from nanoagent.evals import EvalRunner
from nanoagent.task import TaskRunner
from nanoagent.llm import LLMResponse

from tests.mocks import MockLLM, text_response, tool_response


def plan_response(*titles):
    return text_response(json.dumps(
        [{"title": t} for t in titles], ensure_ascii=False))


class EvalsTests(unittest.TestCase):
    def _agent(self, answer):
        return Agent(llm=MockLLM([text_response(answer)]))

    def test_contains_scoring(self):
        runner = EvalRunner()
        runner.add_case(expect_contains=["25", "北京"])
        report = runner.run(self._agent("北京晴，25 度"))
        self.assertEqual(report.pass_rate, 1.0)

        runner2 = EvalRunner()
        runner2.add_case(expect_contains=["降雨"])
        report2 = runner2.run(self._agent("北京晴"))
        self.assertEqual(report2.pass_rate, 0.0)
        self.assertIn("缺少关键词", report2.cases[0].reason)

    def test_regex_scoring(self):
        runner = EvalRunner()
        runner.add_case(expect_regex=r"\d+")
        report = runner.run(self._agent("答案是 42"))
        self.assertTrue(report.cases[0].passed)

    def test_judge_scoring(self):
        judge = MockLLM([text_response('{"score": 85, "reason": "覆盖完整"}')])
        runner = EvalRunner(judge_llm=judge)
        runner.add_case(judge_rubric="摘要是否完整")
        report = runner.run(self._agent("摘要内容"))
        self.assertTrue(report.cases[0].passed)
        self.assertEqual(report.cases[0].score, 85)

    def test_judge_unparseable_scores_zero(self):
        judge = MockLLM([text_response("不合格输出")])
        runner = EvalRunner(judge_llm=judge)
        runner.add_case(judge_rubric="任意")
        report = runner.run(self._agent("答案"))
        self.assertFalse(report.cases[0].passed)
        self.assertIn("无法解析", report.cases[0].reason)

    def test_case_error_does_not_abort_batch(self):
        runner = EvalRunner()
        runner.add_case(expect_contains=["x"])  # 第二个用例会因 MockLLM 响应用尽而报错
        runner.add_case(expect_contains=["ok"])
        agent = Agent(llm=MockLLM([text_response("没有关键词"), text_response("有 ok")]))
        report = runner.run(agent)
        self.assertEqual(len(report.cases), 2)
        self.assertFalse(report.cases[0].passed)  # 缺关键词判 0 分
        self.assertTrue(report.cases[1].passed)

    def test_case_tools_temporarily_registered(self):
        from nanoagent import tool

        @tool
        def calc(a: int, b: int) -> int:
            """相加。"""
            return a + b

        llm = MockLLM([tool_response("t1", "calc", {"a": 1, "b": 2}), text_response("结果 3")])
        agent = Agent(llm=llm)
        runner = EvalRunner()
        runner.add_case(expect_contains=["3"], tools=[calc])
        report = runner.run(agent)
        self.assertTrue(report.cases[0].passed)
        self.assertNotIn("calc", agent.tools.names())  # 用例结束后恢复

    def test_report_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            runner = EvalRunner()
            runner.add_case(expect_contains=["hi"])
            report = runner.run(self._agent("say hi"))
            saved = report.save(path)
            data = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(len(data["cases"]), 1)


class CheckpointBackendTests(unittest.TestCase):
    def test_json_file_roundtrip_and_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = JsonFileBackend(Path(tmp))
            backend.save("a", {"x": 1})
            backend.save("b", {"y": [1, 2]})
            self.assertEqual(backend.load("a"), {"x": 1})
            self.assertEqual(backend.keys(), ["a", "b"])
            backend.delete("a")
            self.assertIsNone(backend.load("a"))

    def test_fixed_json_file_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = JsonFileBackend(Path(tmp) / "task.json")
            backend.save("task", {"goal": "g"})
            self.assertEqual(backend.load("task"), {"goal": "g"})
            self.assertEqual(backend.keys(), ["task"])

    def test_sqlite_backend_roundtrip_and_cross_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.db"
            backend = SQLiteBackend(path)
            backend.save("task", {"goal": "g", "items": [1, 2]})
            # 另一个实例（模拟另一个进程）读同一文件
            self.assertEqual(SQLiteBackend(path).load("task"), {"goal": "g", "items": [1, 2]})
            backend.save("task", {"goal": "g2"})  # upsert
            self.assertEqual(backend.load("task"), {"goal": "g2"})
            backend.delete("task")
            self.assertIsNone(backend.load("task"))

    def test_in_memory_backend(self):
        backend = InMemoryBackend()
        state = {"items": [{"id": 1}]}
        backend.save("task", state)
        state["items"][0]["id"] = 999  # 改外部对象不影响已存档内容
        self.assertEqual(backend.load("task"), {"items": [{"id": 1}]})
        self.assertEqual(backend.keys(), ["task"])

    def test_resolve_checkpoint_dispatch(self):
        # 用临时目录，避免 SQLiteBackend 构造时在仓库 cwd 落下一个 x.db 残留
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsInstance(resolve_checkpoint(str(Path(tmp) / "x.json")), JsonFileBackend)
            self.assertIsInstance(resolve_checkpoint(str(Path(tmp) / "x.db")), SQLiteBackend)
        backend = InMemoryBackend()
        self.assertIs(resolve_checkpoint(backend), backend)


class CheckpointWiringTests(unittest.TestCase):
    def test_taskrunner_with_sqlite_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = SQLiteBackend(Path(tmp) / "runs.db")
            runner = TaskRunner(Agent(llm=MockLLM([
                plan_response("甲", "乙"), text_response("r1"),
            ])), checkpoint=backend)
            items = runner._plan("目标")
            items[0].status = "done"
            items[0].result = "甲完成"
            runner._persist("目标", items, "task")

            agent2 = Agent(llm=MockLLM([text_response("乙完成"), text_response("汇总")]))
            result = TaskRunner.resume(backend, agent2)
            self.assertEqual(result.items[1].result, "乙完成")

    def test_taskrunner_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            TaskRunner.resume(InMemoryBackend(), Agent(llm=MockLLM([])))

    def test_taskrunner_with_fixed_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            runner = TaskRunner(Agent(llm=MockLLM([
                plan_response("甲"), text_response("r1"), text_response("汇总"),
            ])), checkpoint=path)
            result = runner.run("目标")
            self.assertTrue(result.all_done)
            from nanoagent.checkpoints import JsonFileBackend

            self.assertEqual(JsonFileBackend(path).load("task")["goal"], "目标")


class ReplanningTests(unittest.TestCase):
    def test_add_marker_appends_new_task(self):
        llm = MockLLM([
            plan_response("主任务"),
            text_response("主任务完成\n[ADD] 追加任务"),
            text_response("追加任务完成"),
            text_response("汇总"),
        ])
        result = TaskRunner(Agent(llm=llm)).run("目标")
        self.assertEqual([i.title for i in result.items], ["主任务", "追加任务"])
        self.assertEqual(result.items[1].result, "追加任务完成")  # 新任务被继续执行
        self.assertTrue(result.all_done)

    def test_add_dedupes_and_respects_cap(self):
        from nanoagent.task import TodoItem

        adds = TaskRunner._harvest_adds(
            "[ADD] 甲\n[ADD] 甲\n[ADD] 乙\n普通行", [TodoItem(id=1, title="旧")], max_tasks=3)
        self.assertEqual([a.title for a in adds], ["甲", "乙"])

        items = [TodoItem(id=i, title=f"t{i}") for i in range(1, 3)]  # 已有 2 项，上限 3
        adds = TaskRunner._harvest_adds("[ADD] a\n[ADD] b", items, max_tasks=3)
        self.assertEqual(len(adds), 1)  # 只能再加 1 项

    def test_add_lines_stripped_from_result(self):
        llm = MockLLM([
            plan_response("主任务"),
            text_response("完成说明\n[ADD] 新任务"),
            text_response("新任务结果"),
            text_response("汇总"),
        ])
        result = TaskRunner(Agent(llm=llm)).run("目标")
        self.assertNotIn("[ADD]", result.items[0].result)


class SSEEndpointTests(unittest.TestCase):
    def test_stream_endpoint_with_mock_agent(self):
        from fastapi.testclient import TestClient

        from nanoagent.server import create_app

        agent = Agent(llm=MockLLM([text_response("流式回答")]))
        client = TestClient(create_app(agent))
        response = client.post("/chat/stream", json={"session_id": "u1", "message": "你好"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        events = [json.loads(line[len("data: "):]) for line in response.text.splitlines()
                  if line.startswith("data: ")]
        deltas = [e for e in events if e["type"] == "delta"]
        done = [e for e in events if e["type"] == "done"]
        self.assertEqual("".join(d["text"] for d in deltas), "流式回答")
        self.assertEqual(done[0]["reply"], "流式回答")

    def test_stream_endpoint_empty_message_400(self):
        from fastapi.testclient import TestClient

        from nanoagent.server import create_app

        client = TestClient(create_app(Agent(llm=MockLLM([]))))
        self.assertEqual(client.post("/chat/stream", json={"message": "  "}).status_code, 400)


class LLMRetryTests(unittest.TestCase):
    def test_with_retry_backoff_and_exhaustion(self):
        from nanoagent.llm import _with_retry

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            exc = RuntimeError("x")
            exc.status_code = 429
            raise exc

        # 全部失败：重试 2 次后抛出
        with self.assertRaises(RuntimeError):
            _with_retry(flaky, max_retries=2, backoff=0)
        self.assertEqual(calls["n"], 3)

        # 第三次成功
        calls["n"] = 0

        def recovers():
            calls["n"] += 1
            if calls["n"] < 3:
                exc = RuntimeError("x")
                exc.status_code = 500
                raise exc
            return "ok"

        self.assertEqual(_with_retry(recovers, max_retries=2, backoff=0), "ok")

    def test_non_retryable_raises_immediately(self):
        from nanoagent.llm import _with_retry

        calls = {"n": 0}

        def bad_request():
            calls["n"] += 1
            exc = RuntimeError("x")
            exc.status_code = 400
            raise exc

        with self.assertRaises(RuntimeError):
            _with_retry(bad_request, max_retries=3, backoff=0)
        self.assertEqual(calls["n"], 1)  # 400 不重试


if __name__ == "__main__":
    unittest.main()
