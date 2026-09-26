"""token 化上下文管理测试：估算器、Memory 双维限窗、SummaryMemory 自动压缩触发。"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.memory import Memory, SummaryMemory, estimate_tokens, message_tokens


class EstimateTests(unittest.TestCase):
    def test_ascii_about_four_chars_per_token(self):
        self.assertEqual(estimate_tokens("abcdefgh"), 2)  # 8 个 ASCII 字符
        self.assertEqual(estimate_tokens(""), 0)

    def test_cjk_about_one_token_per_char(self):
        self.assertEqual(estimate_tokens("你好世界"), 4)

    def test_mixed(self):
        tokens = estimate_tokens("abc你好")
        self.assertEqual(tokens, 1 + 2)  # abc≈1 + 两个汉字

    def test_multimodal_message_tokens(self):
        parts = [
            {"type": "text", "text": "你好"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
        ]
        self.assertEqual(message_tokens(parts), 2 + 1024)
        self.assertEqual(message_tokens("你好"), 2)


class TokenWindowTests(unittest.TestCase):
    def test_evicts_by_token_budget(self):
        memory = Memory(max_tokens=30)
        for i in range(10):
            memory.add("s", "user", "这句话大约八个字" + str(i))  # 每条约 10 token
        self.assertLessEqual(memory.tokens("s"), 30)
        self.assertLess(len(memory.history("s")), 10)  # 丢过旧消息

    def test_message_count_limit_still_works(self):
        memory = Memory(max_messages=3)
        for i in range(6):
            memory.add("s", "user", f"msg{i}")
        self.assertEqual(len(memory.history("s")), 3)
        self.assertEqual(memory.history("s")[0]["content"], "msg3")

    def test_combined_limits(self):
        memory = Memory(max_messages=10, max_tokens=25)
        for i in range(10):
            memory.add("s", "user", "十个字的测试消息" * 2)  # 每条约 18 token
        self.assertLessEqual(len(memory.history("s")), 10)
        self.assertLessEqual(memory.tokens("s"), 25)

    def test_oversized_single_message_also_evicted(self):
        memory = Memory(max_tokens=5)
        memory.add("s", "user", "这一条消息远超五个token的上限")
        self.assertEqual(memory.history("s"), [])  # 单条也超限，只能清空

    def test_validation(self):
        with self.assertRaises(ValueError):
            Memory(max_messages=0)
        with self.assertRaises(ValueError):
            Memory(max_tokens=0)


class SummaryAutoCompactTests(unittest.TestCase):
    def test_token_overflow_triggers_compression(self):
        class CountingLLM:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, tools=None):
                self.calls += 1
                from nanoagent.llm import LLMResponse

                return LLMResponse(content="摘要内容")

        llm = CountingLLM()
        # 每条约 22 token；max_tokens=40，写入后超限即自动触发压缩（Claude Code 式 compaction）
        memory = SummaryMemory(max_messages=50, keep_recent=2, max_tokens=40, llm=llm)
        memory.add("s", "user", "这是第一条相当长的消息用于撑高token数量估计值")
        memory.add("s", "user", "这是第二条相当长的消息用于撑高token数量估计值")
        memory.add("s", "user", "这是第三条相当长的消息用于撑高token数量估计值")
        self.assertGreaterEqual(llm.calls, 1)  # token 超限自动触发压缩
        self.assertEqual(memory.summary("s"), "摘要内容")
        # 压缩后原始消息只剩 keep_recent 条以内；history() 额外多一条注入的摘要
        history = memory.history("s")
        self.assertEqual(history[0]["role"], "system")
        self.assertIn("摘要", history[0]["content"])
        self.assertLessEqual(len(history) - 1, 2)
        # 关键不变量：token 上限必须真正生效（含摘要自身占用），keep_recent 只是软约束
        from nanoagent.memory import estimate_tokens

        used = memory.tokens("s") + estimate_tokens(memory.summary("s"))
        self.assertLessEqual(used, 40, f"token 上限失效：实际占用 {used}")

    def test_no_compression_when_within_budget(self):
        class NoCallLLM:
            def chat(self, messages, tools=None):
                raise AssertionError("不应触发压缩")

        memory = SummaryMemory(max_messages=50, keep_recent=2, max_tokens=10_000, llm=NoCallLLM())
        memory.add("s", "user", "短消息")
        self.assertEqual(memory.summary("s"), "")


if __name__ == "__main__":
    unittest.main()
