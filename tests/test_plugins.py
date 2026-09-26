"""plugins.py 测试：mini-Cordis 内核、格式探测与适配器、PluginManager 生命周期。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.plugins import (
    ACTIVE,
    DISPOSED,
    FAILED,
    EventBus,
    PluginContext,
    PluginManager,
    UnknownFormat,
    register_adapter,
)
from nanoagent.tools import Tool, tool

PLUGIN_PY_REGISTER = '''
from nanoagent import tool

@tool
def hello(who: str) -> str:
    """打招呼。

    Args:
        who: 对象
    """
    return f"hello {who}"

def register(ctx):
    ctx.tools.register(hello)
    ctx.effect(lambda: (lambda: None))  # 空 disposer，验证 effect 通道可用
    ctx.provide("greeting", "hi")
'''

PLUGIN_PY_TOOLS = '''
from nanoagent import tool

@tool
def adder(a: int, b: int) -> int:
    """求和。"""
    return a + b

TOOLS = [adder]
'''


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class EventBusTests(unittest.TestCase):
    def test_emit_calls_handlers_and_dispose_unsubscribes(self):
        bus = EventBus()
        seen = []
        dispose = bus.on("e", lambda **kw: seen.append(kw))
        bus.emit("e", name="x")
        dispose()
        bus.emit("e", name="y")
        self.assertEqual(seen, [{"name": "x"}])

    def test_handler_error_does_not_block_others(self):
        bus = EventBus()
        seen = []
        bus.on("e", lambda **kw: 1 / 0)
        bus.on("e", lambda **kw: seen.append(kw.get("name")))
        bus.emit("e", name="ok")
        self.assertEqual(seen, ["ok"])


class PluginContextTests(unittest.TestCase):
    def test_provide_and_dispose(self):
        ctx = PluginContext("p")
        dispose = ctx.provide("svc", 123)
        self.assertEqual(ctx.services["svc"], 123)
        dispose()
        self.assertNotIn("svc", ctx.services)

    def test_effect_disposers_run_lifo(self):
        ctx = PluginContext("p")
        order = []
        ctx.effect(lambda: (order.append("a"), lambda: order.append("dispose-a"))[1])
        ctx.effect(lambda: (order.append("b"), lambda: order.append("dispose-b"))[1])
        ctx.effect(lambda: None)  # 不返回 disposer 也是合法 effect
        ctx.deactivate()
        self.assertEqual(order, ["a", "b", "dispose-b", "dispose-a"])

    def test_deactivate_idempotent_and_swallows_errors(self):
        ctx = PluginContext("p")
        calls = []

        def bad_disposer():
            calls.append("bad")
            raise RuntimeError("boom")

        ctx.effect(lambda: bad_disposer)
        ctx.effect(lambda: (lambda: calls.append("good")))
        ctx.deactivate()
        ctx.deactivate()  # 再跑一遍不应报错也不应重复执行
        self.assertEqual(calls, ["good", "bad"])

    def test_on_uses_shared_bus(self):
        bus = EventBus()
        ctx = PluginContext("p", bus=bus)
        seen = []
        ctx.on("ping", lambda **kw: seen.append(kw.get("n")))
        bus.emit("ping", n=1)
        self.assertEqual(seen, [1])

    def test_facade_registration(self):
        ctx = PluginContext("p")

        @tool
        def sample(x: int) -> int:
            """平方。"""
            return x * x

        def plain(y: int) -> int:
            return y

        ctx.tools.register(sample)
        ctx.tools.register(plain)  # 普通函数自动包装
        ctx.skills.add_dir("/a")
        ctx.skills.add_dir("/a")  # 去重
        ctx.mcp.add_server("echo", {"command": "python"})
        self.assertEqual([t.name for t in ctx.tools_list], ["sample", "plain"])
        self.assertEqual(ctx.skill_dirs, ["/a"])
        self.assertEqual(ctx.mcp_servers, {"echo": {"command": "python"}})


class PluginManagerFormatTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manager = PluginManager()

    def tearDown(self):
        self._tmp.cleanup()

    # -- 各格式探测 -------------------------------------------------------
    def test_detect_nanoagent_with_register(self):
        path = _write(self.root / "plugins" / "hello" / "plugin.py", PLUGIN_PY_REGISTER)
        _write(self.root / "plugins" / "hello" / "plugin.json", '{"name": "hello"}')
        manager = PluginManager()
        manager.scan(path.parent.parent)
        plugin = manager.plugins["hello"]
        self.assertEqual(plugin.format, "nanoagent")

    def test_activate_nanoagent_register_and_effect(self):
        _write(self.root / "plugins" / "hello" / "plugin.py", PLUGIN_PY_REGISTER)
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        plugin = manager.activate("hello")
        self.assertEqual(plugin.state, ACTIVE)
        self.assertEqual(plugin.provided["tools"], ["hello"])
        self.assertIn("greeting", plugin.ctx.services)

        tools = manager.collect_tools()
        self.assertEqual(tools[0].invoke({"who": "世界"}), "hello 世界")

        # 卸载走 LIFO disposer：PLUGIN_PY_REGISTER 里的 effect 只是记录，不报错即可
        manager.deactivate("hello")
        self.assertEqual(plugin.state, DISPOSED)
        self.assertEqual(manager.collect_tools(), [])

    def test_activate_nanoagent_tools_list(self):
        _write(self.root / "plugins" / "hi" / "plugin.py", PLUGIN_PY_TOOLS)
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        plugin = manager.activate("hi")
        self.assertEqual(plugin.provided["tools"], ["adder"])

    def test_detect_and_load_dsh_package(self):
        base = self.root / "plugins" / "dsh-wiki"
        _write(base / "package.json", json.dumps({
            "name": "@deepseek-ai/dsh-wiki",
            "dsh": {"bundle": "dsh-tools"},
            "keywords": ["dsh-plugin"],
        }))
        _write(base / "skills" / "wiki" / "SKILL.md",
               "---\nname: wiki\ndescription: 查百科\n---\nwiki 正文")
        _write(base / ".mcp.json", json.dumps({
            "mcpServers": {"wiki-mcp": {"command": "python", "args": ["srv.py"]}}
        }))
        _write(base / "dist" / "index.js", "// cordis 插件代码体")
        _write(base / "cordis.patch.yml", "plugins:\n  - id: wiki")

        manager = PluginManager()
        manager.scan(self.root / "plugins")
        plugin = manager.activate("dsh-wiki")
        self.assertEqual(plugin.format, "dsh")
        self.assertEqual(plugin.state, ACTIVE)
        self.assertEqual(manager.skills.names(), ["wiki"])
        self.assertEqual(plugin.provided["mcp"], ["wiki-mcp"])
        self.assertTrue(any(note.startswith("ts-code") for note in plugin.skipped))
        self.assertEqual(manager.collect_mcp_servers()["wiki-mcp"]["command"], "python")

    def test_detect_mcp_only_package(self):
        base = self.root / "plugins" / "just-mcp"
        _write(base / "mcp.json", json.dumps({
            "mcpServers": {"svc": {"command": "node", "args": ["x.js"]}}
        }))
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        plugin = manager.activate("just-mcp")
        self.assertEqual(plugin.format, "mcp-only")
        self.assertEqual(plugin.provided["mcp"], ["svc"])

    def test_detect_agent_skills_package(self):
        base = self.root / "plugins" / "skill-pack"
        _write(base / "skills" / "alpha" / "SKILL.md",
               "---\nname: alpha\ndescription: A\n---\n正文")
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        plugin = manager.activate("skill-pack")
        self.assertEqual(plugin.format, "agent-skills")
        self.assertIn("alpha", manager.skills.names())

    def test_unknown_format_fails_without_blocking_others(self):
        (self.root / "plugins" / "junk").mkdir(parents=True)
        (self.root / "plugins" / "junk" / "readme.txt").write_text("?", encoding="utf-8")
        _write(self.root / "plugins" / "hi" / "plugin.py", PLUGIN_PY_TOOLS)
        manager = PluginManager()
        discovered = manager.scan(self.root / "plugins")
        self.assertEqual(len(discovered), 2)
        junk = manager.plugins["junk"]
        self.assertEqual(junk.state, FAILED)
        self.assertIn("register_adapter", junk.error)
        # 坏插件不拖垮整体
        self.assertEqual(manager.activate("hi").state, ACTIVE)
        with self.assertRaises(KeyError):
            manager.activate("不存在的插件")

    def test_duplicate_names_get_suffix(self):
        _write(self.root / "plugins" / "same" / "plugin.py", PLUGIN_PY_TOOLS)
        _write(self.root / "plugins2" / "same" / "plugin.py", PLUGIN_PY_TOOLS)
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        manager.scan(self.root / "plugins2")
        self.assertIn("same", manager.plugins)
        self.assertIn("same-2", manager.plugins)

    def test_broken_plugin_py_recorded_as_failed(self):
        _write(self.root / "plugins" / "bad" / "plugin.py", "raise RuntimeError('装不上')")
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        plugin = manager.activate("bad")
        self.assertEqual(plugin.state, FAILED)
        self.assertIn("RuntimeError", plugin.error)
        self.assertEqual(manager.collect_tools(), [])

    def test_custom_adapter_via_register_adapter(self):
        def detect(path: Path) -> bool:
            return (path / "marker.txt").is_file()

        def load(plugin, ctx) -> None:
            plugin.manifest = {"custom": True}
            ctx.tools.register(
                Tool(name="custom_tool", description="自定义", parameters={
                    "type": "object", "properties": {}, "required": []
                }, func=lambda: "custom")
            )

        register_adapter("test-custom", detect, load)
        try:
            _write(self.root / "plugins" / "custom" / "marker.txt", "1")
            manager = PluginManager()
            manager.scan(self.root / "plugins")
            plugin = manager.activate("custom")
            self.assertEqual(plugin.format, "test-custom")
            self.assertEqual(plugin.provided["tools"], ["custom_tool"])
        finally:
            # 移除自定义适配器，避免影响其它测试
            from nanoagent import plugins as plugins_mod
            plugins_mod._ADAPTERS.pop("test-custom", None)

    def test_activate_twice_returns_same_plugin(self):
        _write(self.root / "plugins" / "hi" / "plugin.py", PLUGIN_PY_TOOLS)
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        first = manager.activate("hi")
        second = manager.activate("hi")
        self.assertIs(first, second)
        self.assertEqual(first.provided["tools"], ["adder"])  # 不重复装载

    def test_list_and_detect_format_directly(self):
        _write(self.root / "plugins" / "hi" / "plugin.py", PLUGIN_PY_TOOLS)
        manager = PluginManager()
        manager.scan(self.root / "plugins")
        listing = manager.list()
        self.assertEqual(listing[0]["name"], "hi")
        self.assertEqual(listing[0]["format"], "nanoagent")
        self.assertEqual(manager.detect_format(self.root / "plugins" / "hi"), "nanoagent")
        with self.assertRaises(UnknownFormat):
            manager.detect_format(self.root / "plugins2" if (self.root / "plugins2").is_dir() else self.root)


if __name__ == "__main__":
    unittest.main()
