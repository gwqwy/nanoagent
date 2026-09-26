"""记忆与追踪测试：滑动窗口裁剪、持久化、Tracer JSONL。"""

import json
import tempfile
import unittest
from pathlib import Path

from nanoagent.memory import Memory
from nanoagent.tracing import Tracer


class MemoryTests(unittest.TestCase):
    def test_history_is_copy_without_ts(self):
        memory = Memory()
        memory.add("s", "user", "hi")
        history = memory.history("s")
        history.append({"role": "assistant", "content": "篡改"})
        self.assertEqual(len(memory.history("s")), 1)
        self.assertNotIn("ts", memory.history("s")[0])

    def test_window_trims_oldest(self):
        memory = Memory(max_messages=4)
        for i in range(6):
            memory.add("s", "user", f"msg{i}")
        contents = [m["content"] for m in memory.history("s")]
        self.assertEqual(contents, ["msg2", "msg3", "msg4", "msg5"])

    def test_persist_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.json"
            memory = Memory(persist_path=path)
            memory.add("a", "user", "问题")
            memory.add("a", "assistant", "回答")
            memory.save()

            restored = Memory(persist_path=path)
            self.assertEqual(restored.history("a"), memory.history("a"))

    def test_load_corrupt_file_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("不是JSON", encoding="utf-8")
            memory = Memory(persist_path=path)
            self.assertEqual(memory.sessions(), [])

    def test_clear_session(self):
        memory = Memory()
        memory.add("x", "user", "hi")
        memory.clear("x")
        self.assertEqual(memory.history("x"), [])


class TracerTests(unittest.TestCase):
    def test_disabled_tracer_is_noop(self):
        tracer = Tracer(enabled=False, path="whatever.jsonl")
        tracer.log("event", x=1)
        tracer.start_run("a")
        self.assertFalse(Path("whatever.jsonl").exists())

    def test_events_written_as_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            tracer = Tracer(path=path)
            tracer.start_run("tester")
            tracer.log("tool_call", tool="demo", arguments={"a": 1})
            tracer.end_run(status="ok")

            lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            # 键名必须是 "event"：observability.trace_summary 按 event.get("event") 读取，
            # 两边不一致会让可观测性统计恒为 0（缺陷审计 N-02）。
            self.assertNotIn("type", lines[0])
            kinds = [l["event"] for l in lines]
            self.assertEqual(kinds, ["run_start", "tool_call", "run_end"])
            self.assertEqual(lines[1]["tool"], "demo")
            # run 内所有事件都应携带同一个 run_id
            run_ids = {l.get("run_id") for l in lines}
            self.assertEqual(len(run_ids), 1)
            self.assertIsNotNone(next(iter(run_ids)))

    def test_none_path_disables(self):
        tracer = Tracer(path=None)
        self.assertFalse(tracer.enabled)

    def test_concurrent_log_lines_are_not_interleaved(self):
        # N-15 回归：并行工具会从多个工作线程调用 Tracer.log，
        # 同一文件必须以单次 write 原子落盘，任何一行都应是完整合法 JSON。
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            tracer = Tracer(path=path)
            tracer.start_run("tester")

            def worker(tid: int) -> None:
                for i in range(200):
                    tracer.log("evt", tid=tid, i=i, payload="x" * 200)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1 + 8 * 200)  # run_start + 8 线程 × 200 条
            for line in lines:
                json.loads(line)  # 任一行解析失败即说明写入被交错


if __name__ == "__main__":
    unittest.main()
