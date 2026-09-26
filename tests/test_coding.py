"""coding.py 测试：路径监狱、文件工具、搜索、确认门、Agent 集成。"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.agent import Agent
from nanoagent.coding import CODING_INSTRUCTIONS, CodingWorkspace, SandboxViolation

from tests.mocks import MockLLM, text_response, tool_response


def make_ws(**kwargs) -> CodingWorkspace:
    tmp = tempfile.TemporaryDirectory()
    ws = CodingWorkspace(Path(tmp.name), **kwargs)
    ws._tmp = tmp  # 挂在对象上便于 tearDown
    return ws


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_ws(auto_approve=True)
        self.addCleanup(self.ws._tmp.cleanup)

    def test_relative_path_resolves_inside(self):
        path = self.ws._safe_path("src/a.py")
        self.assertTrue(str(path).startswith(str(self.ws.root)))

    def test_dotdot_escape_rejected(self):
        with self.assertRaises(SandboxViolation):
            self.ws._safe_path("../outside.txt")

    def test_absolute_outside_rejected(self):
        with self.assertRaises(SandboxViolation):
            self.ws._safe_path(str(Path(self.ws._tmp.name).parent / "elsewhere.txt"))

    def test_inner_dotdot_allowed_if_lands_inside(self):
        # sub/../a.py 最终落回工作区内，合法（与示例 09 行为一致）
        path = self.ws._safe_path("sub/../a.py")
        self.assertTrue(str(path).startswith(str(self.ws.root)))


class FileToolTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_ws(auto_approve=True)
        self.addCleanup(self.ws._tmp.cleanup)

    def test_write_then_read_with_line_numbers(self):
        self.ws.write_file("src/a.py", "第一行\n第二行\n第三行")
        output = self.ws.read_file("src/a.py")
        self.assertIn("1: 第一行", output)
        self.assertIn("3: 第三行", output)

    def test_read_range(self):
        self.ws.write_file("a.txt", "l1\nl2\nl3\nl4")
        output = self.ws.read_file("a.txt", start_line=2, end_line=3)
        self.assertIn("2: l2", output)
        self.assertIn("3: l3", output)
        self.assertNotIn("l1", output)

    def test_read_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.ws.read_file("nope.txt")

    def test_edit_unique_match(self):
        self.ws.write_file("a.py", "def foo():\n    return 1\n")
        result = self.ws.edit_file("a.py", "return 1", "return 42")
        self.assertIn("已编辑", result)
        self.assertIn("return 42", self.ws.read_file("a.py"))

    def test_edit_not_found(self):
        self.ws.write_file("a.txt", "hello")
        self.assertIn("找不到", self.ws.edit_file("a.txt", "world", "!"))

    def test_edit_multiple_requires_replace_all(self):
        self.ws.write_file("a.txt", "x\nx\nx")
        self.assertIn("命中 3 处", self.ws.edit_file("a.txt", "x", "y"))
        self.assertIn("已编辑", self.ws.edit_file("a.txt", "x", "y", replace_all=True))
        self.assertEqual(self.ws.read_file("a.txt"), "    1: y\n    2: y\n    3: y")

    def test_file_diff_preview(self):
        self.ws.write_file("a.txt", "old")
        diff = self.ws.file_diff("a.txt", "new")
        self.assertIn("-old", diff)
        self.assertIn("+new", diff)

    def test_delete_with_auto_approve(self):
        self.ws.write_file("gone.txt", "bye")
        self.assertIn("已删除", self.ws.delete_file("gone.txt"))
        with self.assertRaises(FileNotFoundError):
            self.ws.read_file("gone.txt")


class BrowseSearchTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_ws(auto_approve=True)
        self.addCleanup(self.ws._tmp.cleanup)
        self.ws.write_file("src/app.py", "def hello():\n    pass\n")
        self.ws.write_file("src/cache.tmp", "junk")
        self.ws.write_file("__pycache__/x.py", "compiled")

    def test_list_files_skips_ignores(self):
        listing = self.ws.list_files("**/*")
        self.assertIn("src/app.py", listing)
        self.assertNotIn("__pycache__", listing)

    def test_list_files_glob(self):
        listing = self.ws.list_files("**/*.py")
        self.assertIn("src/app.py", listing)
        self.assertNotIn("cache.tmp", listing)

    def test_search_code(self):
        hits = self.ws.search_code(r"def \w+", "**/*.py")
        self.assertIn("src/app.py:1:", hits)


class ApprovalGateTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_ws()  # 默认：不 auto_approve、无 confirm
        self.addCleanup(self.ws._tmp.cleanup)
        self.ws.write_file("victim.txt", "data")

    def test_run_command_blocked_by_default(self):
        result = self.ws.run_command("echo hi")
        self.assertIn("默认被拒绝", result)
        self.assertIn("auto_approve", result)

    def test_delete_blocked_by_default(self):
        self.assertIn("默认被拒绝", self.ws.delete_file("victim.txt"))
        self.assertTrue((Path(self.ws.root) / "victim.txt").exists())  # 未被删

    def test_confirm_callback_allows_and_denies(self):
        approved = CodingWorkspace(self.ws.root, confirm=lambda action: "echo" in action)
        self.assertIn("exit 0", approved.run_command("echo hi"))
        self.assertIn("已被人工确认拒绝", approved.run_command("rm -rf /"))

        denies_all = CodingWorkspace(self.ws.root, confirm=lambda action: False)
        self.assertIn("拒绝", denies_all.delete_file("victim.txt"))

    def test_auto_approve_skips_gate(self):
        ws = CodingWorkspace(self.ws.root, auto_approve=True)
        self.assertTrue(str(ws.run_command("echo hi")).startswith("[exit 0]") or "hi" in ws.run_command("echo hi"))

    def test_command_timeout_reported(self):
        ws = CodingWorkspace(self.ws.root, auto_approve=True)
        self.assertIn("超时", ws.run_command("python -c \"import time; time.sleep(5)\"", timeout=1))

    def test_command_cwd_is_workspace(self):
        ws = CodingWorkspace(self.ws.root, auto_approve=True)
        self.ws.write_file("marker.txt", "here")
        output = ws.run_command("dir /b" if os.name == "nt" else "ls")
        self.assertIn("marker.txt", output)


class GitAndProcessTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_ws(auto_approve=True)
        self.addCleanup(self.ws._tmp.cleanup)

    def test_git_not_a_repo(self):
        result = self.ws.git_status()
        self.assertIn("失败", result)

    def test_git_commit_flow(self):
        import subprocess

        subprocess.run(["git", "init"], cwd=str(self.ws.root), capture_output=True)
        subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=str(self.ws.root), capture_output=True)
        subprocess.run(["git", "config", "user.name", "tester"], cwd=str(self.ws.root), capture_output=True)
        self.ws.write_file("hello.py", "print('hi')\n")

        diff = self.ws.git_diff()
        self.assertIn("hello.py", diff)

        blocked_ws = make_ws()  # 无确认门：commit 默认拒绝
        self.addCleanup(blocked_ws._tmp.cleanup)
        self.assertIn("默认被拒绝", blocked_ws.git_commit("x"))

        result = self.ws.git_commit("add hello")
        self.assertNotIn("失败", result)
        self.assertEqual(self.ws.git_status().count("hello.py"), 0)  # 已干净

    def test_background_process_lifecycle(self):
        ws = self.ws
        result = ws.start_process('python -u -c "import time; print(\'booted\', flush=True); time.sleep(10)"')
        self.assertIn("已启动", result)
        pid = int(result.split("#")[1].split()[0])
        try:
            for _ in range(20):  # 最多等 2 秒等输出落盘
                time.sleep(0.1)
                if "booted" in ws.read_process(pid):
                    break
            output = ws.read_process(pid)
            self.assertIn("运行中", output)
            self.assertIn("booted", output)
        finally:
            self.assertIn("已终止", ws.stop_process(pid))

    def test_start_process_blocked_by_default(self):
        ws = make_ws()  # 无确认门
        self.addCleanup(ws._tmp.cleanup)
        self.assertIn("默认被拒绝", ws.start_process("python -c pass"))

    def test_read_process_unknown_id(self):
        self.assertIn("不存在", self.ws.read_process(999))

    def test_spawn_subagent_requires_factory(self):
        self.assertIn("agent_factory", self.ws.spawn_subagent("随便做点啥"))

    def test_spawn_subagent_with_factory(self):
        from nanoagent import Agent
        from nanoagent.tools import make_tool

        calls = []

        def factory(instructions: str):
            calls.append(instructions)

            def echo_task(task: str) -> str:
                return f"子agent完成: {task}"

            echo = make_tool(echo_task, name="echo_task")
            return Agent(llm=MockLLM([tool_response("t1", "echo_task", {"task": "子任务A"}),
                                      text_response("子agent完成: 子任务A")]),
                         tools=[echo])

        ws = CodingWorkspace(self.ws.root, agent_factory=factory)
        result = ws.spawn_subagent("子任务A", instructions="帮手")
        self.assertIn("子任务A", result)
        self.assertEqual(calls, ["帮手"])

    def test_spawn_subagent_counter_independent_from_pid(self):
        """N-17：子代理会话编号与后台进程 pid 各用独立计数器，互不影响。"""
        from nanoagent import Agent

        seen = []

        def factory(instructions: str):
            def _run(task, session_id=None, **kw):
                seen.append(session_id)
                return type("R", (), {"content": "ok"})()

            agent = Agent(llm=MockLLM([]))
            agent.run = _run  # type: ignore[assignment]
            return agent

        ws = CodingWorkspace(self.ws.root, agent_factory=factory, auto_approve=True)
        # 先起一个后台进程，消耗一个 pid
        started = ws.start_process("echo hi")
        self.assertIn("#1", started)
        self.assertEqual(ws._next_pid, 2)

        ws.spawn_subagent("任务A")
        ws.spawn_subagent("任务B")
        # 子代理编号从 1 开始，不受已用掉的 pid 影响
        self.assertEqual(seen, ["subagent-1", "subagent-2"])
        self.assertEqual(ws._next_pid, 2)  # pid 计数器未被 spawn 改写
        ws.stop_process(1)  # 清理后台进程


class AgentIntegrationTests(unittest.TestCase):
    def test_agent_writes_and_reads_via_tools(self):
        ws = make_ws(auto_approve=True)
        self.addCleanup(ws._tmp.cleanup)
        llm = MockLLM([
            tool_response("t1", "write_file", {"path": "hello.py", "content": "print('hi')"}),
            tool_response("t2", "read_file", {"path": "hello.py"}),
            text_response("写好了"),
        ])
        agent = Agent(llm=llm, tools=ws.as_tools(), instructions=CODING_INSTRUCTIONS)
        result = agent.run("写一个 hello.py")
        self.assertEqual(result.tool_calls[0]["result"], "新建 hello.py（1 行）")
        self.assertIn("1: print('hi')", result.tool_calls[1]["result"])
        self.assertTrue((Path(ws.root) / "hello.py").is_file())

    def test_tools_have_clean_names(self):
        ws = make_ws(auto_approve=True)
        self.addCleanup(ws._tmp.cleanup)
        names = {t.name for t in ws.as_tools()}
        self.assertEqual(names, {
            "read_file", "write_file", "edit_file", "file_diff",
            "delete_file", "list_files", "search_code", "run_command",
            "git_status", "git_diff", "git_commit",
            "start_process", "read_process", "stop_process", "spawn_subagent",
        })

    def test_allowed_tools_whitelist(self):
        ws = make_ws(auto_approve=True, allowed_tools={"read_file", "write_file"})
        self.addCleanup(ws._tmp.cleanup)
        names = {t.name for t in ws.as_tools()}
        self.assertEqual(names, {"read_file", "write_file"})


if __name__ == "__main__":
    unittest.main()
