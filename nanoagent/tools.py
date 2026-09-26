"""工具定义与注册。

设计参考 OpenAI Agents SDK：用 @tool 装饰器把普通 Python 函数
变成 agent 可调用的工具，JSON Schema 从类型注解和 docstring 自动生成，
使用者不需要手写 schema。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .llm import ToolCall

_TYPE_MAP: dict = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_PRIMITIVES = (str, int, float, bool)

# OpenAI 兼容服务要求工具名匹配 ^[a-zA-Z0-9_-]+$（DeepSeek 会直接 400）
_TOOL_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """把工具名净化成 OpenAI 兼容格式；有改动时追加短哈希防冲突。

    例如中文名"研究员"→"研究员"不含法字符 → "tool_a1b2c3"，同名的不同工具不会互相覆盖。
    """
    cleaned = _TOOL_NAME_PATTERN.sub("_", name)
    if cleaned == name:
        return name
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    return f"{cleaned.strip('_') or 'tool'}_{digest}"


def _annotation_to_schema(ann: Any) -> dict:
    """把 Python 类型注解转成 JSON Schema 片段。"""
    if ann is inspect.Parameter.empty:
        return {"type": "string"}
    if ann is type(None):
        return {"type": "null"}

    origin = typing.get_origin(ann)
    # Optional[str] 与 str | None 统一取非 None 分支
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        return _annotation_to_schema(args[0]) if args else {"type": "string"}
    if origin in (list, typing.List):
        item_args = typing.get_args(ann)
        return {"type": "array", "items": _annotation_to_schema(item_args[0]) if item_args else {}}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if isinstance(ann, type):
        if issubclass(ann, dict):
            return {"type": "object"}
        if issubclass(ann, (list, tuple, set)):
            return {"type": "array"}
        if ann in _PRIMITIVES:
            return {"type": _TYPE_MAP[ann]}
    return {"type": "string"}


def _parse_docstring(fn: Callable) -> tuple[str, Dict[str, str]]:
    """解析 Google 风格 docstring，返回 (函数描述, {参数名: 参数说明})。"""
    doc = inspect.getdoc(fn)
    if not doc:
        return "", {}
    description_lines: List[str] = []
    param_docs: Dict[str, str] = {}
    in_args = False
    current_param: Optional[str] = None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().rstrip(":") in ("args", "arguments", "parameters", "参数"):
            in_args = True
            continue
        if stripped.lower().rstrip(":") in ("returns", "return", "raises", "example", "示例"):
            in_args = False
            current_param = None
            continue
        if in_args:
            head, sep, tail = stripped.partition(":")
            if sep and head and " " not in head.strip():
                current_param = head.strip()
                param_docs[current_param] = tail.strip()
            elif current_param and stripped:
                param_docs[current_param] += " " + stripped
        else:
            description_lines.append(line)
    description = "\n".join(description_lines).strip()
    return description, param_docs


@dataclass
class Tool:
    """一个可被 agent 调用的工具。func 可以是普通函数或协程函数。"""

    name: str
    description: str
    parameters: dict
    func: Callable

    def to_openai(self) -> dict:
        """转成 OpenAI tools 参数格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    def invoke(self, arguments: dict) -> Any:
        return self.func(**arguments)

    async def arun(self, arguments: dict) -> Any:
        """异步调用：协程函数直接 await，普通函数丢进线程池，二者可混用。"""
        if self.is_async:
            return await self.func(**arguments)
        return await asyncio.to_thread(self.func, **arguments)


def make_tool(fn: Callable, name: str | None = None, description: str | None = None) -> Tool:
    """从函数生成 Tool：函数名做工具名，docstring 做描述，注解生成参数 schema。

    幂等：函数已经被 @tool 包装过时直接返回既有 Tool（除非显式传了覆盖参数）。
    """
    existing = getattr(fn, "_tool", None)
    if existing is not None and name is None and description is None:
        return existing
    doc_description, param_docs = _parse_docstring(fn)
    properties: Dict[str, dict] = {}
    required: List[str] = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param.name] = _annotation_to_schema(param.annotation)
        if param.name in param_docs:
            properties[param.name]["description"] = param_docs[param.name]
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    parameters = {"type": "object", "properties": properties, "required": required}
    return Tool(
        name=sanitize_tool_name(name or fn.__name__),
        description=description or doc_description or f"工具 {fn.__name__}",
        parameters=parameters,
        func=fn,
    )


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """@tool 装饰器，兼容 @tool 和 @tool(name=..., description=...) 两种写法。"""

    def wrapper(f: Callable) -> Callable:
        f._tool = make_tool(f, name=name, description=description)
        return f

    return wrapper(fn) if callable(fn) else wrapper


class ToolRegistry:
    """工具注册表：集中管理本 agent 可用的所有工具。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, source: Callable | Tool) -> Tool:
        """注册工具，来源可以是 @tool 函数、Tool 实例或普通函数（自动包装）。"""
        if isinstance(source, Tool):
            registered = source
        elif callable(source) and hasattr(source, "_tool"):
            registered = source._tool
        elif callable(source):
            registered = make_tool(source)
        else:
            raise TypeError(f"无法注册工具，不支持的类型: {type(source)!r}")
        self._tools[registered.name] = registered
        return registered

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[dict]:
        """返回全部工具的 OpenAI 格式 schema。"""
        return [t.to_openai() for t in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> str:
        """执行一次工具调用，结果统一转成字符串；异常也转为字符串让模型自行纠正。"""
        registered = self._tools.get(tool_call.name)
        if registered is None:
            return f"错误：未注册的工具 '{tool_call.name}'，可用工具: {', '.join(self._tools) or '无'}"
        try:
            result = registered.invoke(tool_call.arguments)
        except Exception as exc:  # noqa: BLE001 —— 工具报错要回传给模型而不是终止循环
            return f"工具 '{tool_call.name}' 执行出错: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    async def aexecute(self, tool_call: ToolCall) -> str:
        """execute 的异步版：协程工具直接 await，普通工具进线程池。"""
        registered = self._tools.get(tool_call.name)
        if registered is None:
            return f"错误：未注册的工具 '{tool_call.name}'，可用工具: {', '.join(self._tools) or '无'}"
        try:
            result = await registered.arun(tool_call.arguments)
        except Exception as exc:  # noqa: BLE001
            return f"工具 '{tool_call.name}' 执行出错: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def merge(self, other: "ToolRegistry") -> None:
        """合并另一个注册表，同名工具会被覆盖。"""
        for registered in other._tools.values():
            self._tools[registered.name] = registered
