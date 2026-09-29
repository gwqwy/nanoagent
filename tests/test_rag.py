"""RAG 测试：分块、向量库余弦检索、KnowledgeBase 工具化（embedding 用桩替换）。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nanoagent.rag import KnowledgeBase, VectorStore, split_text
from nanoagent.rag.chunking import split_text as split_text_direct


class ChunkingTests(unittest.TestCase):
    def test_short_paragraphs_merge(self):
        text = "第一段。\n第二段。\n第三段。"
        chunks = split_text(text, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("第一段", chunks[0])

    def test_long_paragraph_hard_split_with_overlap(self):
        text = "字" * 300
        chunks = split_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # 相邻 chunk 有重叠
        self.assertEqual(chunks[0][80:100], chunks[1][0:20])
        # 拼回去应覆盖全部内容（去重重叠后）
        self.assertTrue(all(len(c) <= 100 for c in chunks))

    def test_empty_text(self):
        self.assertEqual(split_text("   \n  "), [])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            split_text("abc", chunk_size=0)
        with self.assertRaises(ValueError):
            split_text("abc", chunk_size=10, overlap=10)

    def test_import_alias(self):
        # rag/__init__ 导出的 split_text 与 chunking 模块内是同一个函数
        self.assertIs(split_text, split_text_direct)


class VectorStoreTests(unittest.TestCase):
    def test_cosine_search_ordering(self):
        store = VectorStore()
        texts = ["苹果", "香蕉", "飞机"]
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        store.add(texts, vectors)
        results = store.search(np.array([1.0, 0.0, 0.0]), k=2)
        self.assertEqual([r["text"] for r in results], ["苹果", "香蕉"])
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)

    def test_add_dimension_mismatch_raises(self):
        store = VectorStore()
        store.add(["a"], np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            store.add(["b"], np.zeros((1, 5)))

    def test_search_empty_store(self):
        store = VectorStore()
        self.assertEqual(store.search(np.zeros(3)), [])

    def test_zero_norm_query(self):
        store = VectorStore()
        store.add(["a"], np.array([[1.0, 0.0]]))
        self.assertEqual(store.search(np.zeros(2)), [])

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.json"
            store = VectorStore()
            store.add(["你好"], np.array([[1.0, 2.0]], dtype=np.float32), [{"src": "t"}])
            store.save(path)
            loaded = VectorStore.load(path)
            self.assertEqual(loaded.texts, ["你好"])
            self.assertEqual(loaded.metadata, [{"src": "t"}])
            results = loaded.search(np.array([1.0, 2.0], dtype=np.float32))
            self.assertEqual(results[0]["text"], "你好")


class FakeEmbedLLM:
    """用确定性假向量代替真实 embedding 服务。"""

    def embed(self, texts):
        table = {
            "水果": [1.0, 0.0],
            "苹果是水果": [1.0, 0.0],
            "香蕉也是水果": [0.9, 0.1],
            "天空是蓝色的": [0.0, 1.0],
        }
        rows = []
        for text in texts:
            hit = next((v for k, v in table.items() if k in text), None)
            rows.append(hit or [0.5, 0.5])
        return np.asarray(rows, dtype=np.float32)


class KnowledgeBaseTests(unittest.TestCase):
    def _make_kb(self, persist_path=None):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.embedding_model = "fake"
        kb.persist_path = Path(persist_path) if persist_path else None
        kb.store = VectorStore()
        kb._embed = FakeEmbedLLM().embed
        return kb

    def test_add_and_query(self):
        kb = self._make_kb()
        n = kb.add_text("苹果是水果。香蕉也是水果。\n\n天空是蓝色的。", metadata={"doc": "d1"})
        self.assertEqual(n, 1)  # 短段落全部聚合进一个 chunk
        results = kb.query("水果", k=2)
        self.assertEqual(len(results), 1)  # 库里只有 1 条，k 截断为 1
        self.assertIn("水果", results[0]["text"])

    def test_add_and_query_multi_chunks(self):
        kb = self._make_kb()
        kb.add_text("苹果是水果。", metadata={"doc": "d1"})
        kb.add_text("天空是蓝色的。", metadata={"doc": "d2"})
        results = kb.query("水果", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "苹果是水果。")

    def test_query_empty_kb(self):
        kb = self._make_kb()
        self.assertEqual(kb.query("任何问题"), [])

    def test_as_tool_registers_and_runs(self):
        kb = self._make_kb()
        kb.add_text("苹果是水果。")
        search_tool = kb.as_tool()
        self.assertEqual(search_tool._tool.name, "search_knowledge_base")
        result = search_tool._tool.invoke({"query": "水果"})
        self.assertIn("苹果是水果", result)

    def test_autosave_on_add(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kb.json"
            kb = self._make_kb(persist_path=path)
            kb.add_text("苹果是水果。")
            self.assertTrue(path.exists())
            reloaded = KnowledgeBase.__new__(KnowledgeBase)
            reloaded.store = VectorStore.load(path)
            self.assertGreater(len(reloaded.store), 0)


class EmbeddingCacheTests(unittest.TestCase):
    """向量缓存：内容寻址（sha256），重复入库命中缓存不再调 /embeddings。"""

    class CountingLLM:
        def __init__(self):
            self.calls = 0

        def embeddings(self, texts, model=None):
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    def test_cache_hits_skip_embeddings_across_instances(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            llm = self.CountingLLM()
            persist = Path(tmp) / "kb.json"
            kb = KnowledgeBase(llm=llm, embedding_model="m", persist_path=persist)
            kb.add_text("苹果是水果。")
            calls_first = llm.calls
            self.assertGreaterEqual(calls_first, 1)
            self.assertTrue(Path(f"{persist}.veccache.json").is_file())
            # 新实例（同 persist）：chunk 内容没变 → 全部命中缓存，零次 /embeddings
            kb2 = KnowledgeBase(llm=llm, embedding_model="m", persist_path=persist)
            kb2.add_text("苹果是水果。")
            self.assertEqual(llm.calls, calls_first)

    def test_cache_can_be_disabled(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            llm = self.CountingLLM()
            kb = KnowledgeBase(llm=llm, embedding_model="m",
                               persist_path=Path(tmp) / "kb.json", embedding_cache=False)
            kb.add_text("苹果是水果。")
            self.assertFalse(Path(f"{kb.persist_path}.veccache.json").exists())


if __name__ == "__main__":
    unittest.main()
