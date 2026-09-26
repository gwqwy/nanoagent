"""审计修复回归：N-03 / N-06 / N-10e / N-10f。"""
import asyncio
import unittest

from nanoagent.anthropic_llm import finalize_stream, parse_sse_event
from nanoagent.llm import _accumulate


class AccumulateThreadSafetyTests(unittest.TestCase):
    """N-10e：共享 LLM 的 usage 计数在多线程下不得丢失更新。"""

    def test_concurrent_accumulate_no_lost_updates(self):
        total = {"prompt_tokens": 0, "completion_tokens": 0}
        n_threads, n_iters = 4, 20000

        def worker():
            for _ in range(n_iters):
                _accumulate(total, {"prompt_tokens": 1, "completion_tokens": 2})

        import threading
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(total["prompt_tokens"], n_threads * n_iters)
        self.assertEqual(total["completion_tokens"], n_threads * n_iters * 2)


class AnthropicStreamErrorTests(unittest.TestCase):
    """N-10f：流式中途 error 事件必须显式失败，不得伪装成功。"""

    def test_error_event_raises_on_finalize(self):
        state: dict = {"blocks": {}}
        self.assertEqual(parse_sse_event(
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "部分回答"}}, state), "部分回答")
        parse_sse_event({"type": "error", "error": {"type": "overloaded_error",
                                                    "message": "Overloaded"}}, state)
        with self.assertRaises(RuntimeError) as ctx:
            finalize_stream(state)
        self.assertIn("Overloaded", str(ctx.exception))


class McpConnectRollbackTests(unittest.TestCase):
    """N-03：握手后段失败必须回滚 transport；disconnect 逐层容错。"""

    def test_handshake_failure_closes_transport(self):
        from nanoagent.mcp import MCPServer

        class FakeTransport:
            def __init__(self):
                self.entered = False
                self.exited = False

            async def __aenter__(self):
                self.entered = True
                return object(), object()

            async def __aexit__(self, *exc):
                self.exited = True

        class FlakySession:
            async def __aenter__(self):
                raise RuntimeError("handshake boom")

            async def __aexit__(self, *exc):
                return False

        import nanoagent.mcp as m

        async def scenario():
            server = MCPServer()
            transport = FakeTransport()
            real_client_session = m.ClientSession if hasattr(m, "ClientSession") else None
            # _finish_connect 内部 from mcp import ClientSession —— 打补丁替换
            import mcp as mcp_pkg

            orig = mcp_pkg.ClientSession

            def broken(*a, **k):
                return FlakySession()

            mcp_pkg.ClientSession = broken
            try:
                with self.assertRaises(RuntimeError):
                    await server._finish_connect(transport)
            finally:
                mcp_pkg.ClientSession = orig
            return transport

        transport = asyncio.run(scenario())
        self.assertTrue(transport.entered)
        self.assertTrue(transport.exited)  # 握手失败 → transport 已回滚

    def test_disconnect_tolerates_layer_failure(self):
        from nanoagent.mcp import MCPServer

        class Boom:
            async def __aexit__(self, *exc):
                raise RuntimeError("session close boom")

        class Quiet:
            def __init__(self):
                self.exited = False

            async def __aexit__(self, *exc):
                self.exited = True

        async def scenario():
            server = MCPServer()
            transport = Quiet()
            server._session_cm = Boom()
            server._transport_cm = transport
            with self.assertRaises(RuntimeError):
                await server.disconnect()
            return transport

        transport = asyncio.run(scenario())
        # 一层失败，另一层仍被关闭（finally 保证状态复位与后续清理）
        self.assertTrue(transport.exited)


if __name__ == "__main__":
    unittest.main()
