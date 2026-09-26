"""工具系统测试：schema 生成、docstring 解析、注册表执行、工具名净化。"""

import unittest

from nanoagent.llm import ToolCall
from nanoagent.tools import ToolRegistry, make_tool, sanitize_tool_name, tool
from typing import List, Optional


@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。

    Args:
        a: 第一个加数
        b: 第二个加数
    """
    return a + b


@tool(name="greet", description="打招呼")
def greet(name: str = "世界") -> str:
    return f"你好, {name}"


def no_doc(x: float) -> float:
    return x * 2


class ToolNameSanitizeTests(unittest.TestCase):
    """OpenAI 兼容服务要求工具名匹配 ^[a-zA-Z0-9_-]+$（DeepSeek 400 实测）。"""

    def test_ascii_name_untouched(self):
        self.assertEqual(sanitize_tool_name("get_weather"), "get_weather")
        # 点号非法：替换后追加哈希，仍符合 OpenAI 工具名约束
        cleaned = sanitize_tool_name("tool-2.v1")
        self.assertRegex(cleaned, r"^[a-zA-Z0-9_-]+$")
        self.assertTrue(cleaned.startswith("tool-2_v1"))

    def test_non_ascii_gets_sanitized_with_hash(self):
        name = sanitize_tool_name("研究员")
        self.assertRegex(name, r"^[a-zA-Z0-9_-]+$")
        self.assertNotEqual(name, sanitize_tool_name("撰稿人"))  # 不同中文名不冲突
        self.assertEqual(sanitize_tool_name("研究员"), sanitize_tool_name("研究员"))  # 确定性

    def test_make_tool_applies_sanitization(self):
        def 中文工具(x: int) -> int:
            return x

        registered = make_tool(中文工具)
        self.assertRegex(registered.name, r"^[a-zA-Z0-9_-]+$")

    def test_registry_no_collision_between_chinese_names(self):
        registry = ToolRegistry()
        registry.register(make_tool(lambda: 1, name="研究员"))
        registry.register(make_tool(lambda: 2, name="撰稿人"))
        self.assertEqual(len(registry.names()), 2)


class ToolSchemaTests(unittest.TestCase):
    def test_schema_from_annotations(self):
        registered = make_tool(add)
        self.assertEqual(registered.name, "add")
        self.assertEqual(registered.parameters["type"], "object")
        self.assertEqual(registered.parameters["properties"]["a"]["type"], "integer")
        self.assertEqual(registered.parameters["properties"]["a"]["description"], "第一个加数")
        self.assertEqual(registered.parameters["properties"]["b"]["description"], "第二个加数")
        self.assertEqual(sorted(registered.parameters["required"]), ["a", "b"])

    def test_description_from_docstring(self):
        registered = make_tool(add)
        self.assertTrue(registered.description.startswith("计算两个整数之和"))

    def test_optional_and_container_types(self):
        def sample(tags: List[str], limit: Optional[int] = None, flags: dict = None):
            return True

        schema = make_tool(sample).parameters
        self.assertEqual(schema["properties"]["tags"]["type"], "array")
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")
        self.assertEqual(schema["properties"]["limit"]["type"], "integer")
        self.assertEqual(schema["properties"]["flags"]["type"], "object")
        self.assertNotIn("limit", schema["required"])

    def test_decorator_with_name_override(self):
        self.assertEqual(make_tool(greet).name, "greet")
        self.assertEqual(make_tool(greet).description, "打招呼")

    def test_missing_docstring_falls_back(self):
        registered = make_tool(no_doc)
        self.assertEqual(registered.description, "工具 no_doc")
        self.assertEqual(registered.parameters["properties"]["x"]["type"], "number")


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_execute(self):
        registry = ToolRegistry()
        registry.register(add)
        result = registry.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}))
        self.assertEqual(result, "5")

    def test_execute_returns_error_for_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute(ToolCall(id="1", name="missing", arguments={}))
        self.assertIn("未注册的工具", result)

    def test_exception_becomes_string(self):
        @tool
        def boom():
            """故意抛错。"""
            raise ValueError("炸了")

        registry = ToolRegistry()
        registry.register(boom)
        result = registry.execute(ToolCall(id="1", name="boom", arguments={}))
        self.assertIn("ValueError", result)
        self.assertIn("炸了", result)

    def test_non_string_result_json_serialized(self):
        @tool
        def info() -> dict:
            """返回字典。"""
            return {"city": "北京"}

        registry = ToolRegistry()
        registry.register(info)
        result = registry.execute(ToolCall(id="1", name="info", arguments={}))
        self.assertIn("北京", result)

    def test_schemas_openai_format(self):
        registry = ToolRegistry()
        registry.register(add)
        registry.register(greet)
        schemas = registry.schemas()
        self.assertEqual(len(schemas), 2)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("name", schema["function"])

    def test_register_plain_function(self):
        registry = ToolRegistry()

        def plain(x: int) -> int:
            return x

        registry.register(plain)
        self.assertEqual(registry.names(), ["plain"])


if __name__ == "__main__":
    unittest.main()
