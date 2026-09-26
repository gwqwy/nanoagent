"""bootstrap.py / MCPManager 测试：配置持久化、一键装配、MCP 端到端。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.bootstrap import bootstrap_agent, bootstrap_agent_sync
from nanoagent.mcp import MCPManager

from tests.mocks import MockLLM, text_response

try:
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

PLUGIN_PY = '''
from nanoagent import tool

@tool
def plugin_hello(who: str) -> str:
    """插件工具：打招呼。"""
    return f"hello {who}"

def register(ctx):
    ctx.tools.register(plugin_hello)
'''

UNKNOWN_JUNK = "这不是任何已知插件格式"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class MCPManagerConfigTests(unittest.TestCase):
    def test_add_remove_list_persist_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            manager = MCPManager(path)
            self.assertEqual(manager.list(), {})

            manager.add("echo", "python", ["srv.py"], {"KEY": "v"})
            self.assertTrue(path.is_file())
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["mcpServers"]["echo"]["command"], "python")
            self.assertEqual(on_disk["mcpServers"]["echo"]["env"], {"KEY": "v"})

            reloaded = MCPManager(path)  # 持久化后可重新加载
            self.assertEqual(list(reloaded.list()), ["echo"])

            reloaded.remove("echo")
            self.assertEqual(MCPManager(path).list(), {})

    def test_add_validation_and_remove_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = MCPManager(Path(tmp) / "mcp.json")
            with self.assertRaises(ValueError):
                manager.add("", "python")
            with self.assertRaises(KeyError):
                manager.remove("不存在的")

    def test_load_invalid_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            path.write_text('{"servers": {}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                MCPManager(path)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build_config(self):
        _write(self.root / ".nanoagent" / "plugins" / "hello" / "plugin.py", PLUGIN_PY)
        _write(self.root / ".nanoagent" / "skills" / "wiki" / "SKILL.md",
               "---\nname: wiki\ndescription: 查百科\n---\nwiki 手册正文")
        _write(self.root / ".nanoagent" / "plugins" / "junk" / "readme.txt", UNKNOWN_JUNK)

    def test_bootstrap_end_to_end(self):
        self._build_config()
        llm = MockLLM([text_response("就绪")])
        agent, report = bootstrap_agent_sync(
            self.root / ".nanoagent", llm=llm, instructions="测试助手"
        )
        # 插件工具 + use_skill 都在 agent 上
        self.assertIn("plugin_hello", agent.tools.names())
        self.assertIn("use_skill", agent.tools.names())
        self.assertIn("wiki 手册正文", agent.skills.load("wiki").body)
        # 报告：坏插件进 failed，不拖垮整体；技能被索引
        self.assertEqual(report.skills, ["wiki"])
        self.assertEqual(report.tools, ["plugin_hello", "use_skill"])
        self.assertEqual(len(report.failed), 1)
        self.assertEqual(report.failed[0]["name"], "junk")
        self.assertIn("测试助手", agent.instructions)
        self.assertIn("- wiki: 查百科", agent.instructions)
        self.assertIn("plugin hello[nanoagent:ACTIVE]", report.summary())

    def test_bootstrap_empty_dir(self):
        llm = MockLLM([text_response("ok")])
        agent, report = bootstrap_agent_sync(self.root / ".nanoagent", llm=llm)
        self.assertEqual(report.plugins, [])
        self.assertEqual(report.tools, [])
        self.assertEqual(agent.run("在吗").content, "ok")

    def test_bootstrap_invalid_mcp_config_recorded_as_failed(self):
        self._build_config()
        _write(self.root / ".nanoagent" / "mcp.json", "不是 JSON")
        llm = MockLLM([text_response("ok")])
        _, report = bootstrap_agent_sync(self.root / ".nanoagent", llm=llm)
        # mcp.json 解析失败不应阻断装配
        self.assertTrue(report.failed)
        self.assertIn("plugin_hello", report.tools)


@unittest.skipUnless(HAS_MCP, "需要安装 mcp")
class BootstrapMcpTests(unittest.IsolatedAsyncioTestCase):
    SERVER = str(Path(__file__).resolve().parent / "mcp_echo_server.py")

    async def test_bootstrap_connects_mcp_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".nanoagent"
            _write(root / "mcp.json", json.dumps({
                "mcpServers": {"echo": {"command": sys.executable, "args": [self.SERVER]}}
            }))
            llm = MockLLM([text_response("ok")])
            agent, report = await bootstrap_agent(root, llm=llm)
            try:
                self.assertEqual(report.mcp_servers, ["echo"])
                self.assertIn("echo", agent.tools.names())
            finally:
                await report.aclose()

    async def test_mcp_manager_connect_all_and_disconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            manager = MCPManager(path)
            manager.add("echo", sys.executable, [self.SERVER])
            tools = await manager.connect_all()
            try:
                self.assertEqual(sorted(t.name for t in tools), ["add", "echo"])
            finally:
                await manager.disconnect_all()


if __name__ == "__main__":
    unittest.main()
