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
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .llm import ToolCall

_TYPE_MAP: dict = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# 字符串注解里的基础类型名 → JSON Schema type
_STRING_TYPE_MAP: dict = {
    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
    "list": "array", "List": "array", "Sequence": "array", "Iterable": "array",
    "set": "array", "Set": "array", "tuple": "array", "Tuple": "array",
    "frozenset": "array", "FrozenSet": "array",
    "dict": "object", "Dict": "object", "Mapping": "object", "MutableMapping": "object",
    "DefaultDict": "object",
    "Any": "string", "object": "string",
    "None": "null", "NoneType": "null",
}

_ARRAY_PREFIXES = ("list[", "List[", "Sequence[", "Iterable[", "set[", "Set[",
                   "tuple[", "Tuple[", "frozenset[", "FrozenSet[")
_OBJECT_PREFIXES = ("dict[", "Dict[", "Mapping[", "MutableMapping[", "DefaultDict[")

_NONE_NAMES = ("None", "NoneType")

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


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """按 sep 切分，但忽略括号内部的 sep（用于 `list[dict[str, int]]` 这类嵌套）。"""
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in "[(":
            depth += 1
            buf.append(char)
        elif char in "])":
            depth -= 1
            buf.append(char)
        elif depth == 0 and text.startswith(sep, i):
            parts.append("".join(buf).strip())
            buf = []
            i += len(sep)
            continue
        else:
            buf.append(char)
        i += 1
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _literal_schema(values: List[Any]) -> dict:
    """Literal / Enum 的候选值 → JSON Schema（带 enum 约束）。"""
    if not values:
        return {"type": "string"}
    if all(isinstance(v, bool) for v in values):
        return {"type": "boolean", "enum": values}
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return {"type": "integer", "enum": values}
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return {"type": "number", "enum": values}
    return {"type": "string", "enum": [str(v) for v in values]}


def _string_annotation_to_schema(text: str, depth: int = 0) -> dict:
    """解析**字符串形式**的类型注解。

    使用 `from __future__ import annotations` 的模块里，注解在运行时就是字符串
    （"int" / "list[str]" / "str | None" / "Literal['a','b']"），
    必须在无法拿到真实类型对象时也能正确映射，否则整仓工具的参数类型都会退化成 string。
    """
    if depth > 8:
        return {"type": "string"}
    raw = text.strip().strip("'\"")
    if not raw:
        return {"type": "string"}
    compact = raw.replace(" ", "")

    if compact.startswith("Optional[") and compact.endswith("]"):
        return _string_annotation_to_schema(compact[len("Optional["):-1], depth + 1)
    if compact.startswith("Union[") and compact.endswith("]"):
        parts = [p for p in _split_top_level(compact[len("Union["):-1]) if p not in _NONE_NAMES]
        return _string_annotation_to_schema(parts[0], depth + 1) if parts else {"type": "null"}
    if "|" in compact:
        parts = [p for p in _split_top_level(compact, "|") if p not in _NONE_NAMES]
        return _string_annotation_to_schema(parts[0], depth + 1) if parts else {"type": "null"}
    if compact.startswith("Literal[") and compact.endswith("]"):
        return _literal_schema([p.strip().strip("'\"") for p in _split_top_level(compact[len("Literal["):-1])])
    for prefix in _ARRAY_PREFIXES:
        if compact.startswith(prefix) and compact.endswith("]"):
            inner = _split_top_level(compact[len(prefix):-1])
            schema: dict = {"type": "array"}
            if inner and inner[0] not in ("...", "Ellipsis"):
                schema["items"] = _string_annotation_to_schema(inner[0], depth + 1)
            return schema
    for prefix in _OBJECT_PREFIXES:
        if compact.startswith(prefix) and compact.endswith("]"):
            return {"type": "object"}
    mapped = _STRING_TYPE_MAP.get(compact.split("[")[0])
    return {"type": mapped} if mapped else {"type": "string"}


def _annotation_to_schema(ann: Any, depth: int = 0) -> dict:
    """把 Python 类型注解转成 JSON Schema 片段（兼容字符串注解）。"""
    if depth > 8:
        return {"type": "string"}
    if ann is inspect.Parameter.empty:
        return {"type": "string"}
    if ann is None or ann is type(None):
        return {"type": "null"}

    # 字符串注解（future import / 手写字符串注解）
    if isinstance(ann, str):
        return _string_annotation_to_schema(ann, depth)

    # Literal[...]
    if typing.get_origin(ann) is typing.Literal:
        return _literal_schema(list(typing.get_args(ann)))

    origin = typing.get_origin(ann)
    # Optional[str] 与 str | None 统一取非 None 分支
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        return _annotation_to_schema(args[0], depth + 1) if args else {"type": "null"}
    if origin in (list, typing.List, tuple, typing.Tuple, set, typing.Set,
                  frozenset, typing.FrozenSet) or origin in (
                  getattr(typing, "Sequence", None), getattr(typing, "Iterable", None)):
        item_args = typing.get_args(ann)
        schema: dict = {"type": "array"}
        if item_args and item_args[0] is not Ellipsis:
            schema["items"] = _annotation_to_schema(item_args[0], depth + 1)
        return schema
    if origin in (dict, typing.Dict) or origin in (
            getattr(typing, "Mapping", None), getattr(typing, "MutableMapping", None)):
        return {"type": "object"}

    if isinstance(ann, type):
        if ann in _TYPE_MAP:                      # 精确匹配优先（bool 是 int 子类，须先命中）
            return {"type": _TYPE_MAP[ann]}
        if issubclass(ann, Enum):
            return _literal_schema([member.value for member in ann])
        if issubclass(ann, dict):
            return {"type": "object"}
        if issubclass(ann, (list, tuple, set, frozenset)):
            return {"type": "array"}
    if ann is Any:
        return {"type": "string"}
    return {"type": "string"}


def _resolve_annotations(fn: Callable) -> Dict[str, Any]:
    """解析函数的真实参数类型。

    优先用 `typing.get_type_hints`（能正确展开字符串注解、Optional、嵌套泛型）；
    遇到无法解析的自定义类型（如插件内联定义的类）时静默回退到原始注解
    —— `_annotation_to_schema` 自身也能处理字符串注解，因此回退不会退化。
    """
    try:
        return dict(typing.get_type_hints(fn))
    except Exception:
        return {}



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
    """一个可被 agent 调用的工具。func 可以是普通函数或协程函数。

    thread_safe=False 表示该工具**只能在创建它的线程里调用**（如 Playwright 同步 API、
    绑定到事件循环的 MCP 会话）；Agent 的并行工具执行会因此退化为串行，避免跨线程崩溃。
    """

    name: str
    description: str
    parameters: dict
    func: Callable
    thread_safe: bool = True

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
        """异步调用：协程函数直接 await，普通函数丢进线程池，二者可混用。

        thread_safe=False 的工具（如 Playwright 同步 API）必须留在当前线程执行，
        否则丢进线程池会触发跨线程使用错误。
        """
        if self.is_async:
            return await self.func(**arguments)
        if not self.thread_safe:
            return self.func(**arguments)
        return await asyncio.to_thread(self.func, **arguments)


def make_tool(
    fn: Callable,
    name: str | None = None,
    description: str | None = None,
    *,
    thread_safe: bool = True,
) -> Tool:
    """从函数生成 Tool：函数名做工具名，docstring 做描述，注解生成参数 schema。

    幂等：函数已经被 @tool 包装过时直接返回既有 Tool（除非显式传了覆盖参数）。
    """
    existing = getattr(fn, "_tool", None)
    if existing is not None and name is None and description is None:
        return existing
    doc_description, param_docs = _parse_docstring(fn)
    hints = _resolve_annotations(fn)
    properties: Dict[str, dict] = {}
    required: List[str] = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        # hints 优先（真实类型对象）；退化到 param.annotation（可能是字符串）
        properties[param.name] = _annotation_to_schema(hints.get(param.name, param.annotation))
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
        thread_safe=thread_safe,
    )


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """@tool 装饰器，兼容 @tool 和 @tool(name=..., description=...) 两种写法。"""

    def wrapper(f: Callable) -> Callable:
        f._tool = make_tool(f, name=name, description=description)
        return f

    return wrapper(fn) if callable(fn) else wrapper


def _invoke_sync(registered: Tool, arguments: dict) -> Any:
    """同步路径调用工具；异步工具显式转成同步执行，绝不静默返回 coroutine。

    旧行为：直接 `func(**args)` 得到一个 coroutine 对象，再被 json.dumps/isinstance(str)
    变成 "<coroutine object ...>" 这种垃圾文本回填给模型，同时抛
    "coroutine was never awaited" 警告。现在要么真的跑出结果，要么给出可操作的错误。
    """
    if not registered.is_async:
        return registered.invoke(arguments)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 当前线程没有运行中的事件循环（同步 run / 线程池 worker）→ 安全地起一个跑完
        return asyncio.run(registered.arun(arguments))
    raise TypeError(
        f"工具 '{registered.name}' 是异步工具，不能在已运行的事件循环中通过同步 execute() 调用；"
        "请改用 Agent.arun() / arun_stream() 或 ToolRegistry.aexecute()。"
    )


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
            result = _invoke_sync(registered, tool_call.arguments)
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
