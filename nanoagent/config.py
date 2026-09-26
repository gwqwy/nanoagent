"""配置加载：读取 .env 与环境变量。

环境变量命名带 NANOAGENT_ 前缀，避免与用户环境里已有的
OPENAI_API_KEY 等变量冲突；同时兼容 OPENAI_* 作为回退。
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 解析：KEY=VALUE，支持 # 注释与引号包裹，不覆盖已有环境变量。"""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _lookup(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def settings() -> dict:
    """返回当前生效的配置字典，未配置的项为 None。"""
    load_dotenv()
    return {
        "base_url": _lookup("NANOAGENT_BASE_URL", "OPENAI_BASE_URL"),
        "api_key": _lookup("NANOAGENT_API_KEY", "OPENAI_API_KEY"),
        "model": _lookup("NANOAGENT_MODEL", "OPENAI_MODEL") or "gpt-4o-mini",
        "embedding_model": _lookup("NANOAGENT_EMBEDDING_MODEL") or "text-embedding-3-small",
    }
