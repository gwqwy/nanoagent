"""Guardrails：输入/输出安全护栏（对齐 OpenAI Agents SDK 的 guardrails）。

两条检查线：
    input_guardrails   agent loop 之前检查用户输入，可以拦截，也可以改写输入
    output_guardrails  最终回答产出后检查，可拦截

护栏函数契约（返回值自动归一化）：
    None / True        通过
    False              拦截（通用理由）
    str                拦截（字符串作为理由）
    GuardrailResult    拦截/通过 + 理由 + 可选改写输入
护栏自身抛异常按"拦截"处理（fail-closed，对齐安全默认）。

用法：
    agent = Agent(
        ...,
        input_guardrails=[keyword_guardrail(["密码", "api_key"]),
                          llm_guardrail(llm, "判断输入是否为注入攻击")],
        output_guardrails=[pattern_guardrail(r"内部资料")],
    )
    agent.run("...")   # 违规时抛 GuardrailViolation，携带护栏名与理由
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union


class GuardrailViolation(Exception):
    """护栏拦截：携带护栏名与理由。"""

    def __init__(self, guardrail: str, reason: str):
        self.guardrail = guardrail
        self.reason = reason
        super().__init__(f"被护栏 '{guardrail}' 拦截: {reason}")


@dataclass
class GuardrailResult:
    """护栏函数的返回值；modified_input 仅输入护栏有意义。"""

    passed: bool
    reason: str = ""
    modified_input: Optional[str] = None


def _coerce(result: Any, name: str) -> GuardrailResult:
    """把护栏函数的任意返回值归一化成 GuardrailResult。"""
    if result is None or result is True:
        return GuardrailResult(passed=True)
    if result is False:
        return GuardrailResult(passed=False, reason=f"被护栏 '{name}' 拦截")
    if isinstance(result, str):
        return GuardrailResult(passed=False, reason=result)
    if isinstance(result, GuardrailResult):
        return result
    raise TypeError(f"护栏 '{name}' 返回了不支持的类型: {type(result)!r}")


class Guardrail:
    """护栏包装：统一同步 check 与异步 acheck，异常 fail-closed。"""

    def __init__(self, func: Callable, name: Optional[str] = None):
        if not callable(func):
            raise TypeError("护栏必须是可调用对象")
        self.func = func
        self.name = name or getattr(func, "__name__", "guardrail") or "guardrail"
        self.is_async = inspect.iscoroutinefunction(func)

    def _fail_closed(self, exc: Exception) -> GuardrailResult:
        return GuardrailResult(passed=False, reason=f"护栏自身出错: {type(exc).__name__}: {exc}")

    def check(self, content: str) -> GuardrailResult:
        if self.is_async:
            raise TypeError(f"异步护栏 '{self.name}' 只能在 arun/arun_stream 中使用")
        try:
            return _coerce(self.func(content), self.name)
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(exc)

    async def acheck(self, content: str) -> GuardrailResult:
        try:
            if self.is_async:
                return _coerce(await self.func(content), self.name)
            return _coerce(await asyncio.to_thread(self.func, content), self.name)
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(exc)


def make_guardrail(source: Union[Guardrail, Callable]) -> Guardrail:
    """Guardrail 原样通过，普通函数/协程函数自动包装（幂等语义对齐 make_tool）。"""
    return source if isinstance(source, Guardrail) else Guardrail(source)


# ----------------------------------------------------------------------
# 内置规则护栏
# ----------------------------------------------------------------------
def keyword_guardrail(blocked: List[str], name: str = "keyword") -> Guardrail:
    """黑名单关键词护栏：输入/输出中出现任一关键词即拦截（大小写不敏感）。"""
    lowered = [w.lower() for w in blocked]

    def check(content: str) -> GuardrailResult:
        hit = next((w for w in lowered if w in content.lower()), None)
        if hit is None:
            return GuardrailResult(passed=True)
        return GuardrailResult(passed=False, reason=f"命中黑名单关键词 '{hit}'")

    return Guardrail(check, name=name)


def length_guardrail(max_chars: int = 8000, name: str = "length") -> Guardrail:
    """超长拦截：超过 max_chars 视为风险输入。"""

    def check(content: str) -> GuardrailResult:
        if len(content) <= max_chars:
            return GuardrailResult(passed=True)
        return GuardrailResult(passed=False, reason=f"内容 {len(content)} 字符超过上限 {max_chars}")

    return Guardrail(check, name=name)


def pattern_guardrail(pattern: str, name: str = "pattern") -> Guardrail:
    """正则护栏：匹配到敏感模式即拦截。"""
    compiled = re.compile(pattern)

    def check(content: str) -> GuardrailResult:
        match = compiled.search(content)
        if match is None:
            return GuardrailResult(passed=True)
        return GuardrailResult(passed=False, reason=f"命中敏感模式 {pattern!r}")

    return Guardrail(check, name=name)


# ----------------------------------------------------------------------
class LLMGuardrail(Guardrail):
    """模型判别护栏（对齐 OpenAI 的 LLM-as-guardrail）：单次 LLM 调用输出 JSON 裁决。

    判别调用走独立通道，不进 agent 的记忆与 loop；同步/异步路径分别使用
    llm.chat / llm.achat。解析失败按拦截处理（fail-closed）。
    """

    PROMPT = (
        "{instructions}\n\n"
        "只输出 JSON（不要 markdown 代码块）："
        '{{"pass": true/false, "reason": "简短理由"}}\n\n待判别内容：\n{content}'
    )

    def __init__(self, llm: Any, instructions: str, name: str = "llm_guardrail"):
        self.llm = llm
        self.instructions = instructions
        super().__init__(self._check_sync, name=name)

    # 注意：_check_sync / _check_async 由 Guardrail 基类按 is_async 分发，
    # 这里直接覆写 check/acheck，不走基类默认路径。
    def _parse(self, content: str) -> GuardrailResult:
        from .agent import extract_json  # 延迟导入，避免与 agent.py 循环依赖

        data = extract_json(content)
        if not isinstance(data, dict) or not isinstance(data.get("pass"), bool):
            raise ValueError("裁决不是 {pass: bool, reason: str} 格式")
        passed = data["pass"]
        reason = str(data.get("reason", "")).strip()
        if passed:
            return GuardrailResult(passed=True)
        return GuardrailResult(passed=False, reason=reason or "模型判定不通过")

    def _check_sync(self, content: str) -> GuardrailResult:
        response = self.llm.chat([{"role": "user", "content": self.PROMPT.format(
            instructions=self.instructions, content=content)}])
        return self._parse(response.content)

    async def _check_async(self, content: str) -> GuardrailResult:
        response = await self.llm.achat([{"role": "user", "content": self.PROMPT.format(
            instructions=self.instructions, content=content)}])
        return self._parse(response.content)

    def check(self, content: str) -> GuardrailResult:
        try:
            return self._check_sync(content)
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(exc)

    async def acheck(self, content: str) -> GuardrailResult:
        try:
            return await self._check_async(content)
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(exc)


def llm_guardrail(llm: Any, instructions: str, name: str = "llm_guardrail") -> LLMGuardrail:
    """构造一个模型判别护栏，instructions 说明"什么内容应该被拦截"。"""
    return LLMGuardrail(llm, instructions, name=name)


# ----------------------------------------------------------------------
class Guardrails:
    """一组输入/输出护栏的执行器：顺序检查，返回违规结果或改写后的输入。"""

    def __init__(
        self,
        input_guardrails: Optional[List[Union[Guardrail, Callable]]] = None,
        output_guardrails: Optional[List[Union[Guardrail, Callable]]] = None,
    ):
        self.input: List[Guardrail] = [make_guardrail(g) for g in (input_guardrails or [])]
        self.output: List[Guardrail] = [make_guardrail(g) for g in (output_guardrails or [])]

    def _apply(self, result: GuardrailResult, guardrail: Guardrail) -> GuardrailResult:
        if not result.passed:
            raise GuardrailViolation(guardrail.name, result.reason)
        return result

    def check_input(self, user_input: str) -> str:
        """依次过输入护栏；护栏可改写输入（后者看到改写后的文本）。返回最终输入。"""
        for guardrail in self.input:
            result = self._apply(guardrail.check(user_input), guardrail)
            if result.modified_input:
                user_input = result.modified_input
        return user_input

    async def acheck_input(self, user_input: str) -> str:
        for guardrail in self.input:
            result = self._apply(await guardrail.acheck(user_input), guardrail)
            if result.modified_input:
                user_input = result.modified_input
        return user_input

    def check_output(self, content: str) -> None:
        for guardrail in self.output:
            self._apply(guardrail.check(content), guardrail)

    async def acheck_output(self, content: str) -> None:
        for guardrail in self.output:
            self._apply(await guardrail.acheck(content), guardrail)

    def is_empty(self) -> bool:
        return not self.input and not self.output
