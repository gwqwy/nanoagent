"""Checkpoint 后端：把 Workflow / TaskRunner / Memory 的持久化从"本地 JSON 文件"
抽象成可插拔协议。内置三种后端：
    JsonFileBackend   现有行为（默认），单进程够用
    SQLiteBackend     stdlib sqlite3，跨进程共享、无新增依赖
    InMemoryBackend   测试/临时场景
Postgres/Redis 等按同一协议实现 save/load/delete/list 即可接入。

用法（以 TaskRunner 为例，Workflow 同理）：
    backend = SQLiteBackend("runs.db")
    runner = TaskRunner(agent, checkpoint=backend)      # 或 checkpoint="task.json" 等价 JSON 文件
    TaskRunner.resume(checkpoint=backend, key="task", agent=agent)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class CheckpointBackend(Protocol):
    """持久化后端协议：key 是逻辑名（如 "task"、"workflow-demo"）。"""

    def save(self, key: str, state: Dict[str, Any]) -> None: ...
    def load(self, key: str) -> Optional[Dict[str, Any]]: ...
    def delete(self, key: str) -> None: ...
    def keys(self) -> List[str]: ...


class InMemoryBackend:
    """进程内字典后端，测试用。"""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def save(self, key: str, state: Dict[str, Any]) -> None:
        self._data[key] = json.loads(json.dumps(state, ensure_ascii=False))  # 深拷贝

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self) -> List[str]:
        return list(self._data)


class JsonFileBackend:
    """JSON 文件后端。path 指向目录（每个 key 一个文件）或 .json 文件（固定单文件）。"""

    def __init__(self, path: str | Path = ".nanoagent/checkpoints"):
        self.path = Path(path)
        self.fixed_file = self.path.suffix == ".json"
        if self.fixed_file:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.path.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if self.fixed_file:
            return self.path
        safe = key.replace("/", "_").replace("\\", "_")
        return self.path / f"{safe}.json"

    def save(self, key: str, state: Dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def keys(self) -> List[str]:
        if self.fixed_file:
            return [self.path.stem] if self.path.is_file() else []
        return sorted(p.stem for p in self.path.glob("*.json"))


class SQLiteBackend:
    """单文件 SQLite 后端：跨进程可读，stdlib 实现。并发写用 WAL 模式。"""

    def __init__(self, path: str | Path = ".nanoagent/checkpoints.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                "key TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        """每次操作用独立连接并确保关闭（Windows 下文件锁会阻塞删除）。"""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, key: str, state: Dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO checkpoints (key, state) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET state=excluded.state, updated_at=CURRENT_TIMESTAMP",
                (key, json.dumps(state, ensure_ascii=False)),
            )

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT state FROM checkpoints WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, key: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM checkpoints WHERE key = ?", (key,))

    def keys(self) -> List[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT key FROM checkpoints ORDER BY key").fetchall()
        return [r[0] for r in rows]


class PostgresBackend:
    """Postgres 后端（需 pip install nanoagent[postgres]，即 psycopg[binary]）。

    与 SQLiteBackend 相同的表结构与协议，适合多机/容器场景。
    """

    def __init__(self, dsn: str, table: str = "checkpoints"):
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Postgres checkpoint 后端需要先安装: pip install nanoagent[postgres]"
            ) from exc
        import psycopg

        self._psycopg = psycopg
        self.dsn = dsn
        self.table = table
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "key TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TIMESTAMPTZ DEFAULT now())"
            )

    def _connect(self):
        return self._psycopg.connect(self.dsn)

    def save(self, key: str, state: Dict[str, Any]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (key, state) VALUES (%s, %s) "
                f"ON CONFLICT (key) DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
                (key, json.dumps(state, ensure_ascii=False)),
            )

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT state FROM {self.table} WHERE key = %s", (key,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, key: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE key = %s", (key,))

    def keys(self) -> List[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT key FROM {self.table} ORDER BY key")
            return [r[0] for r in cur.fetchall()]


def resolve_checkpoint(spec: str | Path | CheckpointBackend) -> CheckpointBackend:
    """把用户传入的 checkpoint 参数统一成后端对象：字符串/路径 → JSON/SQLite 文件后端。"""
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if path.suffix in (".db", ".sqlite"):
            return SQLiteBackend(path)
        return JsonFileBackend(path)
    return spec
