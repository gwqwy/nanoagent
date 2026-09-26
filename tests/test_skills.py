"""skills.py 测试：frontmatter 解析 / 注册表 / 渐进式披露 / enable_skills。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.agent import Agent
from nanoagent.skills import Skill, SkillRegistry, parse_frontmatter

from tests.mocks import MockLLM, text_response, tool_response


class FrontmatterTests(unittest.TestCase):
    def test_parse_with_frontmatter(self):
        meta, body = parse_frontmatter(
            '---\nname: wiki\ndescription: "查百科"\n---\n\n# 步骤\n先搜索'
        )
        self.assertEqual(meta, {"name": "wiki", "description": "查百科"})
        self.assertEqual(body, "# 步骤\n先搜索")

    def test_parse_without_frontmatter(self):
        meta, body = parse_frontmatter("正文而已")
        self.assertEqual(meta, {})
        self.assertEqual(body, "正文而已")

    def test_parse_unclosed_frontmatter_treated_as_plain_text(self):
        meta, body = parse_frontmatter("---\nname: wiki\n没有收尾")
        self.assertEqual(meta, {})
        self.assertIn("name: wiki", body)

    def test_parse_single_quotes_and_colon_value(self):
        meta, _ = parse_frontmatter("---\ndescription: '工具: 说明'\n---\n正文")
        self.assertEqual(meta["description"], "工具: 说明")


class SkillRegistryTests(unittest.TestCase):
    def _make_skill_dir(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "skills" / "wiki").mkdir(parents=True)
        (root / "skills" / "wiki" / "SKILL.md").write_text(
            "---\nname: wiki\ndescription: 查询百科\n---\n正文A", encoding="utf-8"
        )
        (root / "skills" / "deploy").mkdir(parents=True)
        (root / "skills" / "deploy" / "SKILL.md").write_text(
            "---\ndescription: 部署应用\n---\n正文B", encoding="utf-8"
        )
        return root

    def test_add_and_summary(self):
        registry = SkillRegistry()
        registry.add(Skill(name="a", description="技能A", body="..."))
        registry.add(Skill(name="b", description="技能B", body="..."))
        self.assertEqual(registry.names(), ["a", "b"])
        self.assertEqual(registry.list_summary(), "- a: 技能A\n- b: 技能B")

    def test_add_file_name_fallback_to_directory(self):
        registry = SkillRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "mydir" / "SKILL.md"
            skill_file.parent.mkdir()
            skill_file.write_text("---\ndescription: 描述\n---\n正文", encoding="utf-8")
            skill = registry.add_file(skill_file)
        self.assertEqual(skill.name, "mydir")
        self.assertEqual(skill.description, "描述")

    def test_add_dir_recursive_and_missing_silent(self):
        registry = SkillRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_skill_dir(tmp)
            loaded = registry.add_dir(root / "skills")
            self.assertEqual(len(loaded), 2)
            # 名字缺省时回退目录名
            self.assertEqual(sorted(registry.names()), ["deploy", "wiki"])
        self.assertEqual(registry.add_dir("不存在目录"), [])

    def test_load_unknown_lists_available(self):
        registry = SkillRegistry()
        registry.add(Skill(name="a", description="", body="x"))
        with self.assertRaises(KeyError) as ctx:
            registry.load("nope")
        self.assertIn("a", str(ctx.exception))

    def test_load_returns_full_skill(self):
        registry = SkillRegistry()
        registry.add(Skill(name="a", description="d", body="完整正文"))
        self.assertEqual(registry.load("a").body, "完整正文")


class EnableSkillsTests(unittest.TestCase):
    def test_enable_skills_registers_tool_and_index(self):
        registry = SkillRegistry()
        registry.add(Skill(name="wiki", description="查百科", body="wiki 正文"))
        agent = Agent(llm=MockLLM([text_response("ok")]), instructions="我是助手")
        agent.enable_skills(registry)

        self.assertIn("use_skill", agent.tools.names())
        self.assertIn("- wiki: 查百科", agent.instructions)
        # 重复调用幂等：索引不重复追加
        agent.enable_skills(registry)
        self.assertEqual(agent.instructions.count("# 可用技能"), 1)

    def test_agent_loop_uses_skill_tool(self):
        registry = SkillRegistry()
        registry.add(Skill(name="wiki", description="查百科", body="wiki 的完整操作手册"))
        llm = MockLLM(
            [tool_response("t1", "use_skill", {"name": "wiki"}), text_response("已学会")]
        )
        agent = Agent(llm=llm)
        agent.enable_skills(registry)
        result = agent.run("教我 wiki")
        self.assertEqual(result.tool_calls[0]["result"], "wiki 的完整操作手册")
        self.assertEqual(result.content, "已学会")

    def test_use_skill_unknown_returns_error_string(self):
        registry = SkillRegistry()
        registry.add(Skill(name="a", description="", body="x"))
        tool = registry.use_skill_tool()
        tool_obj = tool._tool
        self.assertIn("可用技能: a", tool_obj.invoke({"name": "missing"}))

    def test_use_skill_tool_cached_idempotent(self):
        registry = SkillRegistry()
        self.assertIs(registry.use_skill_tool(), registry.use_skill_tool())


if __name__ == "__main__":
    unittest.main()
