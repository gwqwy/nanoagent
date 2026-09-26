"""token 化上下文管理测试：估算器、Memory 双维限窗、SummaryMemory 自动压缩触发。

⚠️ 估算器的取值**依赖运行环境**：装了 tiktoken 走 cl100k_base 精确计数，没装走零依赖启发式，
两者对同一段文本给出的数字并不相同（例："abcdefgh" 精确=1 / 启发式=2）。
CI 用 `pip install -e .[server,faiss,mcp,tiktoken]` 装了 tiktoken，开发机默认没装，
所以**任何断言具体 token 数的用例都必须先用 `forced_heuristic()` 或 tiktoken 真值锁定路径**，
否则会出现"本地绿、CI 红"。
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent import memory as _memory
from nanoagent.memory import Memory, SummaryMemory, _heuristic_estimate, estimate_tokens, message_tokens

try:  # 是否装了 tiktoken（CI 装了，开发机通常没有）
    import tiktoken as _tiktoken  # noqa: F401

    _HAS_TIKTOKEN = True
except ImportError:
    _tiktoken = None
    _HAS_TIKTOKEN = False


@contextlib.contextmanager
def forced_heuristic():
    """临时把编码器缓存置为不可用，强制 `estimate_tokens` 走零依赖启发式。

    不用它的话，同一个用例在"装了 tiktoken"和"没装"的机器上会得到不同结果。
    """
    saved = _memory._TIKTOKEN_ENCODER
    _memory._TIKTOKEN_ENCODER = False  # False = 不可用
    try:
        yield
    finally:
        _memory._TIKTOKEN_ENCODER = saved


class HeuristicEstimateTests(unittest.TestCase):
    """零依赖启发式算法本身——不依赖 tiktoken 装没装，取值恒定。"""

    def test_ascii_about_four_chars_per_token(self):
        self.assertEqual(_heuristic_estimate("abcdefgh"), 2)  # 8 个 ASCII 字符
        self.assertEqual(_heuristic_estimate("a" * 400), 100)
        self.assertEqual(_heuristic_estimate(""), 0)

    def test_cjk_about_one_token_per_char(self):
        self.assertEqual(_heuristic_estimate("你好世界"), 4)

    def test_mixed(self):
        self.assertEqual(_heuristic_estimate("abc你好"), 1 + 2)  # abc≈1 + 两个汉字


class EstimateDispatchTests(unittest.TestCase):
    """`estimate_tokens` 是调度器：能用 tiktoken 就精确，否则回退启发式。"""

    def test_falls_back_to_heuristic_without_tiktoken(self):
        with forced_heuristic():
            self.assertEqual(estimate_tokens("abcdefgh"), 2)
            self.assertEqual(estimate_tokens("你好世界"), 4)
            self.assertEqual(estimate_tokens(""), 0)

    @unittest.skipUnless(_HAS_TIKTOKEN, "未安装 tiktoken，跳过精确计数路径")
    def test_prefers_tiktoken_when_available(self):
        """装了 tiktoken 就必须真的走它——不能静默退化成启发式。

        断言与真实编码结果逐个相等即可：这几个串上两者取值不同，相等本身就证明了走的是精确路径。
        """
        try:
            encoder = _tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 —— 离线时取不到 BPE 词表
            self.skipTest(f"cl100k_base 不可用（可能是离线）：{exc}")
        for text in ("abcdefgh", "你好世界", "abc你好", "The quick brown fox."):
            with self.subTest(text=text):
                self.assertEqual(estimate_tokens(text), len(encoder.encode(text, disallowed_special=())))

    def test_multimodal_message_tokens(self):
        parts = [
            {"type": "text", "text": "你好"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
        ]
        with forced_heuristic():
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
