"""manage.py 测试：技能移除需同时支持目录 / 单文件 / .md 三种形态（N-20）。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from nanoagent import manage


def _ns(config: str, name: str) -> argparse.Namespace:
    return argparse.Namespace(config=config, name=name)


class RemoveSkillTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = Path(self._tmp.name)
        self.skills = self.config / "skills"
        self.skills.mkdir(parents=True)

    def test_remove_directory_skill(self):
        target = self.skills / "wiki"
        target.mkdir()
        (target / "SKILL.md").write_text("---\ndescription: x\n---\n正文", encoding="utf-8")
        manage.cmd_remove_skill(_ns(str(self.config), "wiki"))
        self.assertFalse(target.exists())

    def test_remove_md_file_skill(self):
        target = self.skills / "quick.md"
        target.write_text("---\ndescription: y\n---\n正文", encoding="utf-8")
        manage.cmd_remove_skill(_ns(str(self.config), "quick"))
        self.assertFalse(target.exists())

    def test_remove_plain_file_skill(self):
        target = self.skills / "loose"
        target.write_text("正文", encoding="utf-8")
        manage.cmd_remove_skill(_ns(str(self.config), "loose"))
        self.assertFalse(target.exists())

    def test_remove_missing_skill_raises(self):
        with self.assertRaises(SystemExit):
            manage.cmd_remove_skill(_ns(str(self.config), "不存在"))


if __name__ == "__main__":
    unittest.main()


class RemovePluginTests(unittest.TestCase):
    """审计 N-10a：remove-plugin 的 name 不得越出 plugins 目录。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = Path(self._tmp.name)
        self.plugins = self.config / "plugins"
        self.plugins.mkdir(parents=True)
        victim = self.config / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("宝贵数据", encoding="utf-8")
        self.victim = victim

    def test_removes_legitimate_plugin(self):
        plugin = self.plugins / "demo"
        plugin.mkdir()
        (plugin / "x.txt").write_text("x", encoding="utf-8")
        rc = manage.cmd_remove_plugin(_ns(str(self.config), "demo"))
        self.assertEqual(rc, 0)
        self.assertFalse(plugin.exists())

    def test_rejects_traversal_name(self):
        with self.assertRaises(SystemExit):
            manage.cmd_remove_plugin(_ns(str(self.config), "../victim"))
        self.assertTrue(self.victim.exists())  # plugins 外目录完好

    def test_rejects_dotdot_and_root(self):
        with self.assertRaises(SystemExit):
            manage.cmd_remove_plugin(_ns(str(self.config), ".."))
        with self.assertRaises(SystemExit):
            manage.cmd_remove_plugin(_ns(str(self.config), "."))
        self.assertTrue(self.plugins.exists())

    def test_rejects_missing_plugin(self):
        with self.assertRaises(SystemExit):
            manage.cmd_remove_plugin(_ns(str(self.config), "no-such"))
