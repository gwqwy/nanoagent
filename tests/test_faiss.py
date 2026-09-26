"""faiss 后端测试：与 numpy 后端同一套用例 + 双后端结果一致性对比。

faiss 未安装时自动跳过（CI 或他人环境可能没装可选依赖）。
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nanoagent.rag import FaissVectorStore, VectorStore, create_store

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

TEXTS = ["苹果", "香蕉", "飞机"]
VECTORS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@unittest.skipUnless(HAS_FAISS, "需要安装 faiss-cpu: pip install nanoagent[faiss]")
class FaissStoreTests(unittest.TestCase):
    def test_cosine_search_ordering(self):
        store = FaissVectorStore()
        store.add(TEXTS, VECTORS)
        results = store.search(np.array([1.0, 0.0, 0.0]), k=2)
        self.assertEqual([r["text"] for r in results], ["苹果", "香蕉"])
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)

    def test_add_dimension_mismatch_raises(self):
        store = FaissVectorStore()
        store.add(["a"], np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            store.add(["b"], np.zeros((1, 5)))

    def test_add_shape_validation(self):
        store = FaissVectorStore()
        with self.assertRaises(ValueError):
            store.add(["a", "b"], np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            store.add(["a"], np.zeros((1, 3)), metadata=[{}, {}])

    def test_search_empty_store(self):
        store = FaissVectorStore()
        self.assertEqual(store.search(np.zeros(3)), [])

    def test_zero_norm_query(self):
        store = FaissVectorStore()
        store.add(["a"], np.array([[1.0, 0.0]]))
        self.assertEqual(store.search(np.zeros(2)), [])

    def test_query_dimension_mismatch(self):
        store = FaissVectorStore()
        store.add(["a"], np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            store.search(np.zeros(5))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.faiss"
            store = FaissVectorStore()
            store.add(["你好"], np.array([[1.0, 2.0]], dtype=np.float32), [{"src": "t"}])
            store.save(path)

            loaded = FaissVectorStore.load(path)
            self.assertEqual(loaded.texts, ["你好"])
            self.assertEqual(loaded.metadata, [{"src": "t"}])
            results = loaded.search(np.array([1.0, 2.0], dtype=np.float32))
            self.assertEqual(results[0]["text"], "你好")

    def test_save_load_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.faiss"
            FaissVectorStore().save(path)
            loaded = FaissVectorStore.load(path)
            self.assertEqual(len(loaded), 0)
            self.assertEqual(loaded.search(np.zeros(3)), [])

    def test_load_missing_meta_raises(self):
        with self.assertRaises(FileNotFoundError):
            FaissVectorStore.load(Path("不存在的文件.faiss"))


@unittest.skipUnless(HAS_FAISS, "需要安装 faiss-cpu")
class BackendConsistencyTests(unittest.TestCase):
    """两个后端接口一致、检索结果一致。"""

    def test_factory_creates_backends(self):
        self.assertIsInstance(create_store("numpy"), VectorStore)
        self.assertIsInstance(create_store("faiss"), FaissVectorStore)
        with self.assertRaises(ValueError):
            create_store("annoy")

    def test_same_results_as_numpy_backend(self):
        numpy_store = VectorStore()
        faiss_store = FaissVectorStore()
        numpy_store.add(TEXTS, VECTORS, [{"i": i} for i in range(3)])
        faiss_store.add(TEXTS, VECTORS, [{"i": i} for i in range(3)])

        query = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        numpy_results = numpy_store.search(query, k=3)
        faiss_results = faiss_store.search(query, k=3)

        self.assertEqual([r["text"] for r in numpy_results], [r["text"] for r in faiss_results])
        for n, f in zip(numpy_results, faiss_results):
            self.assertAlmostEqual(n["score"], f["score"], places=4)
            self.assertEqual(n["metadata"], f["metadata"])

    def test_kb_accepts_backend_param(self):
        from nanoagent.rag import KnowledgeBase

        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.embedding_model = "fake"
        kb.persist_path = None
        kb.backend = "faiss"
        kb.store = create_store("faiss")
        self.assertIsInstance(kb.store, FaissVectorStore)


if __name__ == "__main__":
    unittest.main()
