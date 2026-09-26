"""摘要记忆测试：触发时机、摘要注入、滚动压缩、持久化。

压缩时机（max_messages=6, keep_recent=2）：第 7 条 add 触发第一次压缩，
保留 m5/m6；此后再涨回 7 条时触发第二次，第二次 prompt 应包含旧摘要。
"""

import tempfile
import unittest
from pathlib import Path

from nanoagent.llm import LLMResponse
from nanoagent.memory import SummaryMemory
from tests.mocks import MockLLM, text_response


class _CounterLLM:
    """记录摘要请求、返回递增编号摘要文本的假 LLM。"""

    model = "mock-model"

    def __init__(self):
        self.prompts = []

    def chat(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        return LLMResponse(content=f"摘要#{len(self.prompts)}")


def make_memory(counter=None, responses=None, **kwargs):
    kwargs.setdefault("max_messages", 6)
    kwargs.setdefault("keep_recent", 2)
    llm = counter or MockLLM([text_response(r) for r in (responses or [])])
    return SummaryMemory(llm=llm, **kwargs)


class SummaryMemoryTests(unittest.TestCase):
    def test_no_compression_below_threshold(self):
        counter = _CounterLLM()
        memory = SummaryMemory(max_messages=6, keep_recent=2, llm=counter)
        for i in range(6):
            memory.add("s", "user", f"m{i}")
        self.assertEqual(counter.prompts, [])
        self.assertEqual(memory.summary("s"), "")
        self.assertEqual([m["content"] for m in memory.history("s")], [f"m{i}" for i in range(6)])

    def test_compress_on_overflow(self):
        counter = _CounterLLM()
        memory = SummaryMemory(max_messages=6, keep_recent=2, llm=counter)
        for i in range(7):
            memory.add("s", "user", f"m{i}")
        self.assertEqual(len(counter.prompts), 1)
        self.assertIn("m0", counter.prompts[0])  # 被裁的旧消息进了摘要请求
        self.assertNotIn("m6", counter.prompts[0])  # 最近消息不应进摘要请求
        self.assertEqual(memory.summary("s"), "摘要#1")
        self.assertEqual([m["content"] for m in memory.history("s")[1:]], ["m5", "m6"])

    def test_history_injects_summary_first(self):
        memory = make_memory(responses=["摘要甲"])
        for i in range(7):
            memory.add("s", "user", f"m{i}")
        history = memory.history("s")
        self.assertEqual(history[0]["role"], "system")
        self.assertIn("摘要甲", history[0]["content"])
        self.assertEqual([m["content"] for m in history[1:]], ["m5", "m6"])

    def test_rolling_summary_includes_previous(self):
        counter = _CounterLLM()
        memory = SummaryMemory(max_messages=6, keep_recent=2, llm=counter)
        for i in range(7):
            memory.add("s", "user", f"m{i}")
        for i in range(7, 14):
            memory.add("s", "user", f"m{i}")
        self.assertEqual(len(counter.prompts), 2)
        self.assertIn("已有摘要", counter.prompts[1])
        self.assertIn("摘要#1", counter.prompts[1])  # 旧摘要进入第二次压缩请求
        self.assertEqual(memory.summary("s"), "摘要#2")
        # 第二次压缩保留 [m10, m11]，之后又追加了 m12/m13
        self.assertEqual([m["content"] for m in memory.history("s")[1:]], ["m10", "m11", "m12", "m13"])

    def test_persist_roundtrip_with_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            memory = make_memory(responses=["持久化摘要"], persist_path=path)
            for i in range(7):
                memory.add("s", "user", f"m{i}")
            memory.save()

            restored = SummaryMemory(
                llm=_CounterLLM(), max_messages=6, keep_recent=2, persist_path=path
            )
            self.assertEqual(restored.summary("s"), "持久化摘要")
            self.assertEqual(restored.history("s"), memory.history("s"))

    def test_load_legacy_format(self):
        # 兼容旧 Memory 保存的纯 sessions 格式
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text('{"s1": [{"role": "user", "content": "hi"}]}', encoding="utf-8")
            memory = SummaryMemory(llm=_CounterLLM(), max_messages=6, keep_recent=2, persist_path=path)
            self.assertEqual(memory.history("s1"), [{"role": "user", "content": "hi"}])
            self.assertEqual(memory.summary("s1"), "")

    def test_clear_removes_summary(self):
        memory = make_memory(responses=["待清除"])
        for i in range(7):
            memory.add("s", "user", f"m{i}")
        memory.clear("s")
        self.assertEqual(memory.summary("s"), "")
        self.assertEqual(memory.history("s"), [])

    def test_invalid_keep_recent(self):
        with self.assertRaises(ValueError):
            SummaryMemory(max_messages=6, keep_recent=6, llm=_CounterLLM())
        with self.assertRaises(ValueError):
            SummaryMemory(max_messages=6, keep_recent=0, llm=_CounterLLM())


if __name__ == "__main__":
    unittest.main()
