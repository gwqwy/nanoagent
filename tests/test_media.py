"""media.py 多模态测试：image_part 构造、Agent images 参数、记忆行为。"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoagent.agent import Agent
from nanoagent.media import build_user_content, image_part, make_png

from tests.mocks import MockLLM, text_response

# 最小合法 PNG 头（魔数 + IHDR），足以触发嗅探
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 8


class ImagePartTests(unittest.TestCase):
    def test_url_passthrough(self):
        part = image_part("https://example.com/a.jpg")
        self.assertEqual(part, {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}})

    def test_data_url_passthrough(self):
        part = image_part("data:image/png;base64,AAAA")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_file_becomes_base64_with_sniffed_mime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no-ext"
            path.write_bytes(PNG_BYTES)
            part = image_part(path)
        url = part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        payload = url.split(",", 1)[1]
        # base64 解码后应还原出 PNG 魔数
        self.assertTrue(base64.b64decode(payload).startswith(b"\x89PNG"))

    def test_suffix_used_when_magic_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.webp"
            path.write_bytes(b"RIFF????WEBPVP8 ")  # 魔数不完整，扩展名兜底
            part = image_part(str(path))
        self.assertTrue(part["image_url"]["url"].startswith("data:image/webp;base64,"))

    def test_bytes_with_magic_sniffing(self):
        part = image_part(JPEG_BYTES)
        self.assertTrue(part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_unknown_bytes_rejected(self):
        with self.assertRaises(ValueError):
            image_part(b"\x00\x01\x02\x03")

    def test_detail_parameter(self):
        part = image_part("https://example.com/a.jpg", detail="low")
        self.assertEqual(part["image_url"]["detail"], "low")

    def test_dict_passthrough(self):
        custom = {"type": "file", "file_id": "file-api-123"}
        self.assertEqual(image_part(custom), custom)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            image_part(12345)


class BuildUserContentTests(unittest.TestCase):
    def test_no_images_keeps_plain_string(self):
        self.assertEqual(build_user_content("你好", None), "你好")
        self.assertEqual(build_user_content("你好", []), "你好")

    def test_with_images_builds_parts(self):
        content = build_user_content("看图", ["https://example.com/a.png"])
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "看图"})
        self.assertEqual(content[1]["type"], "image_url")

    def test_empty_text_only_images(self):
        content = build_user_content("", ["https://example.com/a.png"])
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "image_url")


class AgentMultimodalTests(unittest.TestCase):
    def test_run_sends_multimodal_message(self):
        llm = MockLLM([text_response("红色")])
        agent = Agent(llm=llm)
        result = agent.run("什么颜色", images=[PNG_BYTES])
        self.assertEqual(result.content, "红色")

        user_msg = llm.calls[0]["messages"][-1]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(user_msg["content"][0]["text"], "什么颜色")
        self.assertEqual(user_msg["content"][1]["type"], "image_url")

    def test_run_without_images_keeps_string(self):
        llm = MockLLM([text_response("好")])
        Agent(llm=llm).run("你好")
        self.assertEqual(llm.calls[0]["messages"][-1]["content"], "你好")

    def test_memory_stores_text_only(self):
        llm = MockLLM([text_response("看到方图"), text_response("汇总")])
        agent = Agent(llm=llm)
        agent.run("描述图片", images=[PNG_BYTES])
        agent.run("还有呢", save=True)

        # 第二轮的 history 里 user 消息仍是纯文本（图片不进记忆）
        history = agent.memory.history("default")
        user_messages = [m for m in history if m["role"] == "user"]
        self.assertEqual(user_messages[0]["content"], "描述图片")
        self.assertNotIsInstance(user_messages[0]["content"], list)

    def test_image_detail_forwarded(self):
        llm = MockLLM([text_response("好")])
        Agent(llm=llm).run("看", images=["https://x.com/a.png"], image_detail="low")
        part = llm.calls[0]["messages"][-1]["content"][1]
        self.assertEqual(part["image_url"]["detail"], "low")

    def test_stream_with_images(self):
        llm = MockLLM([text_response("流式看图")])
        agent = Agent(llm=llm)
        chunks = [ev["text"] for ev in agent.run_stream("看", images=[PNG_BYTES]) if ev["type"] == "delta"]
        self.assertEqual("".join(chunks), "流式看图")
        self.assertIsInstance(llm.calls[0]["messages"][-1]["content"], list)


class AgentMultimodalAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_arun_with_images(self):
        llm = MockLLM([text_response("异步看图")])
        agent = Agent(llm=llm)
        result = await agent.arun("什么形状", images=[PNG_BYTES])
        self.assertEqual(result.content, "异步看图")
        user_msg = llm.calls[0]["messages"][-1]
        self.assertEqual(user_msg["content"][0]["text"], "什么形状")
        self.assertEqual(user_msg["content"][1]["type"], "image_url")

    async def test_arun_stream_with_images(self):
        llm = MockLLM([text_response("异步流")])
        agent = Agent(llm=llm)
        chunks = []
        async for ev in agent.arun_stream("看", images=[PNG_BYTES]):
            if ev["type"] == "delta":
                chunks.append(ev["text"])
        self.assertEqual("".join(chunks), "异步流")


class MakePngTests(unittest.TestCase):
    def test_generated_png_is_valid_and_sniffable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_png(4, 3, (255, 0, 0), Path(tmp) / "red.png")
            data = path.read_bytes()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        part = image_part(data)
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
