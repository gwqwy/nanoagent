"""示例 09 文件工具的安全测试：路径越界防护必须可靠。

直接用 importlib 加载示例模块（__main__ 块有保护不会执行），
并把 WORKSPACE 替换到临时目录，避免污染示例目录。
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "09_workflow_with_tools.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("example_09", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkspaceJailTests(unittest.TestCase):
    def setUp(self):
        self.module = load_example_module()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.module.WORKSPACE = Path(self._tmp.name).resolve()

    def test_write_read_roundtrip(self):
        result = self.module.write_file._tool.invoke({"path": "fib.py", "content": "def fib(n): ..."})
        self.assertIn("已写入", result)
        content = self.module.read_file._tool.invoke({"path": "fib.py"})
        self.assertEqual(content, "def fib(n): ...")

    def test_subdirectory_write(self):
        self.module.write_file._tool.invoke({"path": "pkg/mod.py", "content": "x = 1"})
        files = self.module.list_files._tool.invoke({})
        self.assertIn("pkg/mod.py", files)

    def test_read_missing_file_lists_existing(self):
        self.module.write_file._tool.invoke({"path": "a.py", "content": "x"})
        result = self.module.read_file._tool.invoke({"path": "nope.py"})
        self.assertIn("文件不存在", result)
        self.assertIn("a.py", result)

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.module.write_file._tool.invoke({"path": "../outside.py", "content": "x"})
        with self.assertRaises(ValueError):
            self.module.read_file._tool.invoke({"path": "sub/../../outside.py"})

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            self.module.write_file._tool.invoke(
                {"path": str(Path(self._tmp.name) / "x.py"), "content": "x"}
            )

    def test_nested_traversal_stays_in_jail(self):
        # sub/.. 解析后回到工作区根，应允许
        self.module.write_file._tool.invoke({"path": "sub/../ok.py", "content": "x"})
        self.assertIn("ok.py", self.module.list_files._tool.invoke({}))


if __name__ == "__main__":
    unittest.main()
