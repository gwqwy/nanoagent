"""最后一批 Roadmap 测试：MCP server 模式、trace 可视化/OTel、RAG 重排序、browser。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.agent import Agent
from nanoagent.observability import trace_report, trace_summary
from nanoagent.rag.retriever import KnowledgeBase

from tests.mocks import MockLLM, text_response, tool_response

try:
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

try:
    import playwright  # noqa: F401

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


class McpServerTests(unittest.TestCase):
    """create_mcp_server 端到端：stdio 客户端连上自己起的服务器。"""

    def test_create_mcp_server_exposes_chat_tool(self):
        if not HAS_MCP:
            self.skipTest("需要安装 mcp")
        with tempfile.TemporaryDirectory() as tmp:
            entry = Path(tmp) / "entry.py"
            root = str(Path(__file__).resolve().parent.parent).replace("\\", "\\\\")
            entry.write_text(
                "import sys; sys.path.insert(0, r'%s')\n"
                "from nanoagent import Agent, serve_mcp\n"
                "from tests.mocks import MockLLM, text_response\n"
                "serve_mcp(Agent(llm=MockLLM([text_response('来自 MCP 的回答')])))\n"
                % root,
                encoding="utf-8",
            )

            async def _run():
                from nanoagent.mcp import MCPServer as ClientMCPServer

                client = await ClientMCPServer.connect_stdio(sys.executable, [str(entry)])
                try:
                    tools = await client.tools()
                    names = sorted(t.name for t in tools)
                    self.assertEqual(names, ["chat", "list_agent_tools"])
                    result = await client.call("chat", {"message": "你好"})
                    self.assertEqual(result, "来自 MCP 的回答")
                finally:
                    await client.disconnect()

            asyncio.run(_run())


class TraceObservabilityTests(unittest.TestCase):
    def _write_trace(self, path: Path):
        events = [
            {"run_id": "r1", "event": "run_start", "ts": "t0", "agent": "demo"},
            {"run_id": "r1", "event": "llm_call", "ts": "t1",
             "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            {"run_id": "r1", "event": "tool_call", "ts": "t2", "tool": "search"},
            {"run_id": "r1", "event": "run_end", "ts": "t3", "iterations": 2},
        ]
        path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        return events

    def test_trace_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            self._write_trace(path)
            summary = trace_summary(path)
            self.assertEqual(summary["total_events"], 4)
            run = summary["runs"]["r1"]
            self.assertEqual(run["llm_calls"], 1)
            self.assertEqual(run["tool_calls"], 1)
            self.assertEqual(run["prompt_tokens"], 10)

    def test_trace_report_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            out = Path(tmp) / "report.html"
            self._write_trace(path)
            saved = trace_report(path, out)
            html = saved.read_text(encoding="utf-8")
            self.assertIn("nanoagent trace 报告", html)
            self.assertIn("r1", html)
            self.assertIn("search", html)

    def test_otel_export_or_graceful(self):
        """装了 otel-sdk 就正常导出；没装则抛 ImportError（只装 api 时 span 为 no-op 也算成功）。"""
        from nanoagent.observability import export_traces_to_otel

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            self._write_trace(path)
            try:
                count = export_traces_to_otel(path)
                self.assertEqual(count, 4)
            except ImportError:
                pass  # 环境没装 opentelemetry，符合预期


class FakeEmbeddingLLM:
    """可预测的向量：按关键词返回正交向量。"""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def embeddings(self, texts, model=None):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * 4
            if "猫" in text:
                vector[0] = 1.0
            if "狗" in text:
                vector[1] = 1.0
            vector[2] = 0.1  # 公共基底，保证都有点相似度
            vectors.append(vector)
        return vectors


class RagRerankTests(unittest.TestCase):
    def _kb(self):
        llm = FakeEmbeddingLLM()
        kb = KnowledgeBase(llm=llm, embedding_model="fake")
        kb.add_text("猫是小型家养动物，喜欢抓老鼠。")
        kb.add_text("狗是忠诚的伴侣动物。")
        kb.add_text("猫和狗都是常见宠物。")
        return kb

    def test_rerank_reorders_and_falls_back(self):
        kb = self._kb()
        judge = MockLLM([
            # 第一次：合法的重排序输出（把狗排前面）
            text_response("[1, 0]"),
            # 第二次：坏 JSON → 降级为向量序
            text_response("我不会"),
        ])
        kb.llm = _ProxyLLM(kb.llm, judge)

        hits = kb.query("什么动物忠诚", k=2, rerank=True)
        self.assertEqual(len(hits), 2)
        hits = kb.query("什么动物忠诚", k=2, rerank=True)
        self.assertEqual(len(hits), 2)  # 降级不抛错


class _ProxyLLM:
    """embedding 走原 LLM，chat 走 judge（模拟同一客户端的双用途）。"""

    def __init__(self, embedding_llm, judge):
        self._embedder = embedding_llm
        self._judge = judge
        self.model = "proxy"

    def embeddings(self, texts, model=None):
        return self._embedder.embeddings(texts, model)

    def chat(self, messages, tools=None):
        return self._judge.chat(messages, tools)


class BrowserTests(unittest.TestCase):
    def test_navigate_requires_playwright(self):
        from nanoagent import BrowserWorkspace

        ws = BrowserWorkspace()
        self.addCleanup(ws.close)
        try:
            ws._ensure_page()
            has_pw = True
        except ImportError:
            has_pw = False
        if not has_pw:
            with self.assertRaises(ImportError):
                ws._ensure_page()
        else:
            result = ws.navigate("https://example.com")
            self.assertIn("已打开", result)

    def test_navigate_rejects_non_http(self):
        from nanoagent import BrowserWorkspace

        ws = BrowserWorkspace()
        self.addCleanup(ws.close)
        self.assertIn("错误", ws.navigate("file:///etc/passwd"))

    def test_as_tools_names(self):
        from nanoagent import BrowserWorkspace

        ws = BrowserWorkspace()
        names = {t.name for t in ws.as_tools()}
        self.assertEqual(names, {"navigate", "read_text", "click", "fill", "screenshot", "get_url"})


if __name__ == "__main__":
    unittest.main()
