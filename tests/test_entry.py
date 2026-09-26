"""入口冒烟测试：CLI 参数解析、HTTP 应用路由（注入 mock agent，不发真实请求）。"""

import unittest

from nanoagent.agent import Agent
from nanoagent.cli import parse_args
from nanoagent.memory import Memory
from tests.mocks import MockLLM, text_response

try:
    from fastapi.testclient import TestClient

    from nanoagent.server import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class CliArgsTests(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertIsNone(args.model)
        self.assertEqual(args.session, "cli")
        self.assertFalse(args.no_trace)

    def test_overrides(self):
        args = parse_args(["--model", "m1", "--base-url", "http://x/v1", "--session", "s2"])
        self.assertEqual(args.model, "m1")
        self.assertEqual(args.base_url, "http://x/v1")
        self.assertEqual(args.session, "s2")


@unittest.skipUnless(HAS_FASTAPI, "需要安装 server extra: pip install nanoagent[server]")
class ServerAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(
            create_app(
                Agent(
                    name="mock-server",
                    instructions="",
                    llm=MockLLM([text_response("服务端回答")]),
                    memory=Memory(),
                    tracer=None,
                )
            )
        )

    def test_chat(self):
        resp = self.client.post("/chat", json={"session_id": "t", "message": "hi"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["reply"], "服务端回答")
        self.assertEqual(body["iterations"], 1)

    def test_empty_message_rejected(self):
        resp = self.client.post("/chat", json={"session_id": "t", "message": "  "})
        self.assertEqual(resp.status_code, 400)

    def test_health_and_sessions(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.client.post("/chat", json={"session_id": "abc", "message": "x"})
        self.assertIn("abc", self.client.get("/sessions").json()["sessions"])
        self.assertEqual(
            self.client.post("/sessions/abc/clear").json()["status"], "cleared"
        )


if __name__ == "__main__":
    unittest.main()
