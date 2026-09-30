"""核心 Agent 类与 agent loop。

loop 逻辑（与所有主流框架同构）：
    1. 组装 messages = [system] + 历史记忆 + 本次输入
    2. 调用 LLM，若返回 tool_calls 则逐个执行、把结果回填给模型
    3. 重复直到模型给出最终回答，或达到 max_iterations 防死循环上限
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

from pydantic import BaseModel

from .guardrails import GuardrailViolation, Guardrails
from .llm import LLM, LLMResponse, ToolCall
from .media import build_user_content
from .memory import Memory
from .tools import ToolRegistry, tool as tool_decorator
from .tracing import Tracer


class OutputValidationError(Exception):
    """结构化输出在重试后仍无法通过 response_model 校验。"""


def extract_json(text: str) -> Any:
    """从模型回答中提取 JSON：支持裸 JSON、``` 围栏、首尾大括号截取。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("回答中不包含可解析的 JSON")


@dataclass
class AgentResult:
    """一次 run 的最终产物。"""

    content: str
    tool_calls: List[dict] = field(default_factory=list)  # [{name, arguments, result}]
    iterations: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    output: Any = None
    reasoning: str = ""  # 思考过程（多轮工具调用时按顺序拼接）
    stopped: bool = False  # should_stop 中止时为 True（部分轮次已执行，内容不完整）  # response_model 校验通过后的结构化对象（未启用时为 None）


class Agent:
    """一个具备工具调用与记忆能力的 agent。"""

    def __init__(
        self,
        name: str = "assistant",
        instructions: str = "",
        model: str | None = None,
        llm: LLM | None = None,
        tools: Optional[List[Callable]] = None,
        memory: Memory | None = None,
        max_iterations: int = 10,
        tracer: Tracer | None = None,
        parallel_tools: bool = True,
        max_workers: int = 4,
        response_model: type[BaseModel] | None = None,
        input_guardrails: Optional[List[Any]] = None,
        output_guardrails: Optional[List[Any]] = None,
        memory_tool_traces: bool = False,
        tool_gate: Optional[Callable[[str, dict], Any]] = None,
        tool_timeout: Optional[float] = None,
        tool_output_limit: Optional[int] = None,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations 必须 >= 1")
        if max_workers < 1:
            raise ValueError("max_workers 必须 >= 1")
        if response_model is not None and not (
            isinstance(response_model, type) and issubclass(response_model, BaseModel)
        ):
            raise TypeError("response_model 必须是 pydantic.BaseModel 的子类")
        self.name = name
        self.instructions = instructions or f"You are {name}, a helpful AI agent."
        self.llm = llm or LLM(model=model)
        self.tools = ToolRegistry()
        for item in tools or []:
            self.tools.register(item)
        self.memory = memory or Memory()
        self.max_iterations = max_iterations
        self.tracer = tracer or Tracer(enabled=False)
        # 并行工具要求工具函数线程安全；注册表本身只读，无并发问题
        self.parallel_tools = parallel_tools
        self.max_workers = max_workers
        self.response_model = response_model
        self.guardrails = Guardrails(input_guardrails, output_guardrails)
        # N-22：默认只把 user/最终回答写进记忆（保持既有语义）。开启后额外把本轮
        # 工具调用与结果以可读文本追加进记忆，多轮对话中模型能看到上一轮的工具交互。
        self.memory_tool_traces = memory_tool_traces
        self.skills = None  # enable_skills 后为 SkillRegistry
        self._skills_marker = "# 可用技能"
        # 工具审批中间层：每次工具执行前调用 tool_gate(name, arguments)——
        # 返回 None/True 放行；False 或字符串则拦截，字符串作为「拒绝原因」
        # 回填给模型（工具结果以 tool role 返回，模型可据此换路或向用户说明）。
        # 这是框架级 human-in-the-loop：宿主（CLI/桌面/库使用者）各接各的确认 UI。
        self.tool_gate = tool_gate
        # 工具执行超时（秒）：None=不限。超时返回错误给模型（工具可能仍在后台
        # 跑完——同步路径用线程等待实现，无法强杀；这是框架的已知边界）。
        self.tool_timeout = tool_timeout
        # 工具输出限幅（字符）：None=不限。超限截断并附提示，防任何插件撑爆上下文。
        self.tool_output_limit = tool_output_limit

    def _apply_tool_gate(self, name: str, arguments: dict) -> str | None:
        """过工具门：放行返回 None，拦截返回给模型看的错误文本（fail-closed）。"""
        if self.tool_gate is None:
            return None
        try:
            verdict = self.tool_gate(name, arguments)
        except Exception as exc:  # noqa: BLE001 —— 审批回调本身出错按拒绝处理
            return f"错误：工具 {name} 未获审批通过（审批回调异常: {type(exc).__name__}: {exc}）"
        if verdict is None or verdict is True:
            return None
        if verdict is False:
            return f"错误：工具 {name} 被审批层拒绝"
        return f"错误：工具 {name} 被审批层拒绝：{verdict}"

    def _limit_output(self, result: str) -> str:
        """工具输出限幅：超长截断并提示（模型可分次读取或缩小请求范围）。"""
        text = str(result or "")
        if self.tool_output_limit is None or len(text) <= self.tool_output_limit:
            return text
        return (text[: self.tool_output_limit]
                + f"\n…（输出已截断：原长 {len(text)} 字符，仅显示前 {self.tool_output_limit}。"
                  "请缩小请求范围或分段获取。）")

    # ------------------------------------------------------------------
    def _record_turn(
        self, session_id: str, user_input: str, final_content: str, tool_log: List[dict]
    ) -> None:
        """把一轮对话写入记忆（N-22：可选附带工具调用轨迹）。"""
        self.memory.add(session_id, "user", user_input)
        if self.memory_tool_traces and tool_log:
            self.memory.add(session_id, "assistant", self._format_tool_trace(tool_log))
        self.memory.add(session_id, "assistant", final_content)

    @staticmethod
    def _format_tool_trace(tool_log: List[dict]) -> str:
        lines = ["[工具调用记录]"]
        for record in tool_log:
            args = json.dumps(record.get("arguments", {}), ensure_ascii=False)
            lines.append(f"- {record.get('name')}({args}) => {record.get('result')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _guard_input(self, user_input: str) -> str:
        """过输入护栏（可改写输入）；拦截时记录 trace 并抛 GuardrailViolation。"""
        if self.guardrails.is_empty():
            return user_input
        try:
            return self.guardrails.check_input(user_input)
        except GuardrailViolation as exc:
            self.tracer.log("guardrail_blocked", kind="input", guardrail=exc.guardrail, reason=exc.reason)
            raise

    def _guard_output(self, content: str) -> None:
        if self.guardrails.is_empty() or not self.guardrails.output:
            return
        try:
            self.guardrails.check_output(content)
        except GuardrailViolation as exc:
            self.tracer.log("guardrail_blocked", kind="output", guardrail=exc.guardrail, reason=exc.reason)
            raise

    async def _aguard_input(self, user_input: str) -> str:
        if self.guardrails.is_empty():
            return user_input
        try:
            return await self.guardrails.acheck_input(user_input)
        except GuardrailViolation as exc:
            self.tracer.log("guardrail_blocked", kind="input", guardrail=exc.guardrail, reason=exc.reason)
            raise

    async def _aguard_output(self, content: str) -> None:
        if self.guardrails.is_empty() or not self.guardrails.output:
            return
        try:
            await self.guardrails.acheck_output(content)
        except GuardrailViolation as exc:
            self.tracer.log("guardrail_blocked", kind="output", guardrail=exc.guardrail, reason=exc.reason)
            raise

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        user_input: str,
        session_id: str,
        images: Optional[List[Any]] = None,
        image_detail: Optional[str] = None,
    ) -> List[dict]:
        """组装本轮消息；images 非空时 user 消息升级为多模态 content 数组。"""
        messages: List[dict] = [{"role": "system", "content": self.instructions}]
        messages.extend(self.memory.history(session_id))
        messages.append({
            "role": "user",
            "content": build_user_content(user_input, images, detail=image_detail),
        })
        return messages

    @staticmethod
    def _tool_call_payload(tool_calls: List[ToolCall]) -> List[dict]:
        """把 ToolCall 列表转成要回填给模型的 assistant 消息格式。"""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in tool_calls
        ]

    def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[dict]:
        """执行一轮工具调用，返回与 tool_calls 等长的结果列表（顺序一致）。

        列表元素形如 {name, arguments, result}；并/串行两条路径共用此方法。
        只要本轮出现线程不安全工具（Playwright 同步 API、绑定事件循环的 MCP 会话等），
        就整体退化为串行 —— 否则它们会在其它线程里崩溃。
        """
        if len(tool_calls) == 1 or not self.parallel_tools:
            return [self._execute_one(tc) for tc in tool_calls]
        if any(not self._tool_thread_safe(tc.name) for tc in tool_calls):
            return [self._execute_one(tc) for tc in tool_calls]

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tool_calls))) as pool:
            futures = [pool.submit(self._execute_one, tc) for tc in tool_calls]
            return [future.result() for future in futures]  # 按提交顺序取回，保证确定性

    def _tool_thread_safe(self, name: str) -> bool:
        """查询工具是否可跨线程调用；未注册的工具按安全处理（会由 execute 报错）。"""
        tool = self.tools.get(name)
        return True if tool is None else bool(getattr(tool, "thread_safe", True))

    def _run_with_timeout(self, fn):
        """带超时执行同步工具（tool_timeout=None 时直接调用）。"""
        import concurrent.futures

        if self.tool_timeout is None:
            return fn()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            try:
                return future.result(timeout=self.tool_timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"工具执行超过 {self.tool_timeout} 秒（可能卡死；结果不可信，请换路或重试）"
                ) from None

    def _execute_one(self, tool_call: ToolCall) -> dict:
        started = time.perf_counter()
        blocked = self._apply_tool_gate(tool_call.name, tool_call.arguments)
        if blocked is not None:
            result = blocked            # 审批层拦截：不执行工具，原因回填模型
        elif tool_call.arguments_error:
            result = (f"错误：工具 {tool_call.name} 的参数无法解析——{tool_call.arguments_error}。"
                      "请修正参数后重新调用。")   # 坏 JSON 自我修正：根因回填而非静默空参
        else:
            try:
                result = self._run_with_timeout(
                    lambda: self.tools.execute(tool_call))
            except TimeoutError as exc:
                result = f"错误：{exc}"
        result = self._limit_output(result)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.tracer.log(
            "tool_call",
            tool=tool_call.name,
            arguments=tool_call.arguments,
            result=result,
            elapsed_ms=elapsed_ms,
        )
        return {"name": tool_call.name, "arguments": tool_call.arguments, "result": result}

    async def _aexecute_tool_calls(self, tool_calls: List[ToolCall]) -> List[dict]:
        """异步执行一轮工具调用：asyncio.gather 并发，结果顺序与 tool_calls 一致。"""
        return list(await asyncio.gather(*(self._aexecute_one(tc) for tc in tool_calls)))

    async def _aexecute_one(self, tool_call: ToolCall) -> dict:
        started = time.perf_counter()
        blocked = self._apply_tool_gate(tool_call.name, tool_call.arguments)
        if blocked is not None:
            result = blocked
        elif tool_call.arguments_error:
            result = (f"错误：工具 {tool_call.name} 的参数无法解析——{tool_call.arguments_error}。"
                      "请修正参数后重新调用。")
        else:
            try:
                if self.tool_timeout is not None:
                    result = await asyncio.wait_for(
                        self.tools.aexecute(tool_call), timeout=self.tool_timeout)
                else:
                    result = await self.tools.aexecute(tool_call)
            except (asyncio.TimeoutError, TimeoutError):
                result = f"错误：工具执行超过 {self.tool_timeout} 秒（可能卡死；结果不可信，请换路或重试）"
        result = self._limit_output(result)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.tracer.log(
            "tool_call",
            tool=tool_call.name,
            arguments=tool_call.arguments,
            result=result,
            elapsed_ms=elapsed_ms,
        )
        return {"name": tool_call.name, "arguments": tool_call.arguments, "result": result}

    # ------------------------------------------------------------------
    def _require_async_llm(self) -> None:
        if not hasattr(self.llm, "achat"):
            raise TypeError(
                f"arun/arun_stream 需要 llm 支持 achat（如 AsyncLLM），当前是 {type(self.llm).__name__}"
            )

    def _structure_prompt(self, content: str, error: str | None) -> List[dict]:
        """构造"转 JSON"的追问消息，供同步/异步结构化输出共用。"""
        schema = json.dumps(self.response_model.model_json_schema(), ensure_ascii=False)
        prompt = (
            f"请把下面的内容转换为符合以下 JSON Schema 的 JSON，"
            f"只输出 JSON 本身，不要 markdown 代码块：\n{schema}\n\n内容：\n{content}"
        )
        if error:
            prompt += f"\n\n上次转换出错（必须修正）：{error}"
        return [{"role": "user", "content": prompt}]

    def _try_parse_output(self, content: str) -> Any:
        """解析并校验结构化输出，失败抛异常（由调用方决定是否重试）。"""
        return self.response_model.model_validate(extract_json(content))

    # ------------------------------------------------------------------
    def run(
        self,
        user_input: str,
        session_id: str = "default",
        save: bool = True,
        images: Optional[List[Any]] = None,
        image_detail: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """执行 agent loop，返回最终回答。

        images: 图片源列表（http(s) URL / 本地路径 / bytes / 已构造的 part dict），
        传入后 user 消息升级为多模态 content 数组；记忆只持久化文本部分。
        should_stop: 可选中止钩子——每轮循环开始前检查，命中即停（返回的
        AgentResult.stopped=True，内容为空、不写记忆、不做结构化解析；
        流式路径用 run_stream 的同名参数可保留部分内容）。
        """
        self.tracer.start_run(self.name)
        user_input = self._guard_input(user_input)
        messages = self._build_messages(user_input, session_id, images, image_detail)
        tool_log: List[dict] = []
        usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        final_content = ""
        reasoning_parts: List[str] = []
        iterations = 0
        output = None

        try:
            for i in range(self.max_iterations):
                if should_stop is not None and should_stop():
                    # 非流式中止：返回 stopped 标记（内容为空、不写记忆、不解析）
                    return AgentResult(content="", tool_calls=tool_log,
                                       iterations=iterations, usage=usage_total,
                                       reasoning="\n\n".join(reasoning_parts), stopped=True)
                iterations = i + 1
                response = self.llm.chat(messages, tools=self.tools.schemas() or None)
                for key in usage_total:
                    usage_total[key] += response.usage.get(key, 0)
                if getattr(response, "reasoning", ""):
                    reasoning_parts.append(response.reasoning)
                self.tracer.log(
                    "llm_call",
                    iteration=iterations,
                    content=response.content,
                    reasoning=getattr(response, "reasoning", ""),
                    tool_calls=[(tc.name, tc.arguments) for tc in response.tool_calls],
                    usage=response.usage,
                )

                if not response.has_tool_calls:
                    final_content = response.content
                    messages.append({"role": "assistant", "content": final_content})
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": self._tool_call_payload(response.tool_calls),
                    }
                )
                for tc, record in zip(response.tool_calls, self._execute_tool_calls(response.tool_calls)):
                    tool_log.append(record)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": record["result"]})
            else:
                # 达到迭代上限仍未给出最终回答：最后一轮通常仍是 tool_calls，
                # content 常为空串——空串写进记忆、再进结构化解析都是垃圾，留明确标记
                final_content = response.content or (
                    f"（未产出最终回答：已达 max_iterations={self.max_iterations} 上限，"
                    "最后一轮仍在请求工具调用）")
                messages.append({"role": "assistant", "content": final_content})
            self._guard_output(final_content)

            # 结构化输出放在 try 内，保证 structure_retry/failed 事件仍带 run_id
            # （与 arun 一致；原先在 finally 之后写，run_id 已置 None 而丢失归属）
            if self.response_model is not None:
                error = None
                for attempt in range(2):
                    try:
                        output = self._try_parse_output(final_content)
                        break
                    except Exception as exc:  # noqa: BLE001 —— 解析/校验失败回填模型重试一次
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt == 0:
                            self.tracer.log("structure_retry", error=error)
                            messages.extend(self._structure_prompt(final_content, error))
                            retry = self.llm.chat(messages)
                            final_content = retry.content
                            for key in usage_total:
                                usage_total[key] += retry.usage.get(key, 0)
                            messages.append({"role": "assistant", "content": final_content})
                if output is None:
                    self.tracer.log("structure_failed", error=error)
                    raise OutputValidationError(f"结构化输出在重试后仍失败，最后一次错误: {error}")
        finally:
            self.tracer.end_run(iterations=iterations)

        if save:
            self._record_turn(session_id, user_input, final_content, tool_log)
        return AgentResult(
            content=final_content,
            tool_calls=tool_log,
            iterations=iterations,
            usage=usage_total,
            output=output,
            reasoning="\n\n".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    def run_stream(
        self,
        user_input: str,
        session_id: str = "default",
        save: bool = True,
        images: Optional[List[Any]] = None,
        image_detail: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[dict, None, None]:
        """流式版 agent loop，产出事件字典：

        {"type": "delta", "text": ...}       —— 模型文本增量
        {"type": "tool_call", name, arguments, result} —— 一次工具调用完成
        {"type": "done", result: AgentResult} —— 结束，携带完整结果

        should_stop: 可选中止钩子（返回 True 即请求停止）。在每轮开始与每个
        delta 之间检查；命中后关闭底层流并直接返回——不再产出 done、不写
        记忆、不跑输出护栏（部分生成的内容按调用方要求丢弃）。
        """
        self.tracer.start_run(self.name)
        user_input = self._guard_input(user_input)
        messages = self._build_messages(user_input, session_id, images, image_detail)
        tool_log: List[dict] = []
        usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        final_content = ""
        reasoning_parts: List[str] = []
        iterations = 0
        # 审计 N-05：输出护栏开启时事件先缓冲、护栏通过后再统一放行——
        # 否则 delta 已流出，事后拦截只能拦 done 与记忆，拦不回已下发的内容。
        defer_output = not self.guardrails.is_empty() and bool(self.guardrails.output)
        deferred: List[dict] = []
        stopped = False

        try:
            for i in range(self.max_iterations):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                iterations = i + 1
                stream = self.llm.chat_stream(messages, tools=self.tools.schemas() or None)
                parts: List[str] = []
                for text in stream:
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                    parts.append(text)
                    event = {"type": "delta", "text": text}
                    if defer_output:
                        deferred.append(event)
                    else:
                        yield event
                if stopped:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                    break
                response = stream.response or LLMResponse(content="".join(parts))
                for key in usage_total:
                    usage_total[key] += response.usage.get(key, 0)
                if getattr(response, "reasoning", ""):
                    reasoning_parts.append(response.reasoning)
                self.tracer.log(
                    "llm_call",
                    iteration=iterations,
                    content=response.content,
                    reasoning=getattr(response, "reasoning", ""),
                    tool_calls=[(tc.name, tc.arguments) for tc in response.tool_calls],
                    usage=response.usage,
                )

                if not response.has_tool_calls:
                    final_content = response.content
                    messages.append({"role": "assistant", "content": final_content})
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": self._tool_call_payload(response.tool_calls),
                    }
                )
                for tc, record in zip(response.tool_calls, self._execute_tool_calls(response.tool_calls)):
                    tool_log.append(record)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": record["result"]})
                    event = {
                        "type": "tool_call",
                        "name": record["name"],
                        "arguments": record["arguments"],
                        "result": record["result"],
                    }
                    if defer_output:
                        deferred.append(event)
                    else:
                        yield event
            else:
                final_content = response.content or (
                    f"（未产出最终回答：已达 max_iterations={self.max_iterations} 上限，"
                    "最后一轮仍在请求工具调用）")
                messages.append({"role": "assistant", "content": final_content})
            if not stopped:
                self._guard_output(final_content)
                if defer_output:
                    for event in deferred:
                        yield event
        finally:
            self.tracer.end_run(iterations=iterations)

        if stopped:
            return
        if save:
            self._record_turn(session_id, user_input, final_content, tool_log)
        result = AgentResult(
            content=final_content,
            tool_calls=tool_log,
            iterations=iterations,
            usage=usage_total,
            reasoning="\n\n".join(reasoning_parts),
        )
        yield {"type": "done", "result": result}

    # ------------------------------------------------------------------
    async def arun(
        self,
        user_input: str,
        session_id: str = "default",
        save: bool = True,
        images: Optional[List[Any]] = None,
        image_detail: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """run 的异步版：需要 llm 为 AsyncLLM（或任何提供 achat 的客户端）。

        工具并发改用 asyncio.gather（协程工具直接 await，普通工具进线程池），
        支持高并发场景下单线程承载大量会话。images 用法见 run()。
        """
        self._require_async_llm()
        self.tracer.start_run(self.name)
        user_input = await self._aguard_input(user_input)
        messages = self._build_messages(user_input, session_id, images, image_detail)
        tool_log: List[dict] = []
        usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        final_content = ""
        reasoning_parts: List[str] = []
        iterations = 0
        output = None

        try:
            for i in range(self.max_iterations):
                if should_stop is not None and should_stop():
                    return AgentResult(content="", tool_calls=tool_log,
                                       iterations=iterations, usage=usage_total,
                                       reasoning="\n\n".join(reasoning_parts), stopped=True)
                iterations = i + 1
                response = await self.llm.achat(messages, tools=self.tools.schemas() or None)
                for key in usage_total:
                    usage_total[key] += response.usage.get(key, 0)
                if getattr(response, "reasoning", ""):
                    reasoning_parts.append(response.reasoning)
                self.tracer.log(
                    "llm_call",
                    iteration=iterations,
                    content=response.content,
                    reasoning=getattr(response, "reasoning", ""),
                    tool_calls=[(tc.name, tc.arguments) for tc in response.tool_calls],
                    usage=response.usage,
                )

                if not response.has_tool_calls:
                    final_content = response.content
                    messages.append({"role": "assistant", "content": final_content})
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": self._tool_call_payload(response.tool_calls),
                    }
                )
                for tc, record in zip(
                    response.tool_calls, await self._aexecute_tool_calls(response.tool_calls)
                ):
                    tool_log.append(record)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": record["result"]})
            else:
                final_content = response.content or (
                    f"（未产出最终回答：已达 max_iterations={self.max_iterations} 上限，"
                    "最后一轮仍在请求工具调用）")
                messages.append({"role": "assistant", "content": final_content})

            await self._aguard_output(final_content)
            if self.response_model is not None:
                error = None
                for attempt in range(2):
                    try:
                        output = self._try_parse_output(final_content)
                        break
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt == 0:
                            self.tracer.log("structure_retry", error=error)
                            messages.extend(self._structure_prompt(final_content, error))
                            retry = await self.llm.achat(messages)
                            final_content = retry.content
                            for key in usage_total:
                                usage_total[key] += retry.usage.get(key, 0)
                            messages.append({"role": "assistant", "content": final_content})
                if output is None:
                    self.tracer.log("structure_failed", error=error)
                    raise OutputValidationError(f"结构化输出在重试后仍失败，最后一次错误: {error}")
        finally:
            self.tracer.end_run(iterations=iterations)

        if save:
            self._record_turn(session_id, user_input, final_content, tool_log)
        return AgentResult(
            content=final_content,
            tool_calls=tool_log,
            iterations=iterations,
            usage=usage_total,
            output=output,
            reasoning="\n\n".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    async def arun_stream(
        self,
        user_input: str,
        session_id: str = "default",
        save: bool = True,
        images: Optional[List[Any]] = None,
        image_detail: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[dict, None, None]:
        """run_stream 的异步版，事件格式与同步版一致；should_stop 语义亦相同。"""
        self._require_async_llm()
        self.tracer.start_run(self.name)
        user_input = await self._aguard_input(user_input)
        messages = self._build_messages(user_input, session_id, images, image_detail)
        tool_log: List[dict] = []
        usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        final_content = ""
        reasoning_parts: List[str] = []
        iterations = 0
        output = None
        # 审计 N-05：与 run_stream 相同——输出护栏开启时先缓冲事件再放行
        defer_output = not self.guardrails.is_empty() and bool(self.guardrails.output)
        deferred: List[dict] = []
        stopped = False

        try:
            for i in range(self.max_iterations):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                iterations = i + 1
                stream = await self.llm.achat_stream(messages, tools=self.tools.schemas() or None)
                parts: List[str] = []
                async for text in stream:
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                    parts.append(text)
                    event = {"type": "delta", "text": text}
                    if defer_output:
                        deferred.append(event)
                    else:
                        yield event
                if stopped:
                    aclose = getattr(stream, "aclose", None)
                    if callable(aclose):
                        await aclose()
                    break
                response = stream.response or LLMResponse(content="".join(parts))
                for key in usage_total:
                    usage_total[key] += response.usage.get(key, 0)
                if getattr(response, "reasoning", ""):
                    reasoning_parts.append(response.reasoning)
                self.tracer.log(
                    "llm_call",
                    iteration=iterations,
                    content=response.content,
                    reasoning=getattr(response, "reasoning", ""),
                    tool_calls=[(tc.name, tc.arguments) for tc in response.tool_calls],
                    usage=response.usage,
                )

                if not response.has_tool_calls:
                    final_content = response.content
                    messages.append({"role": "assistant", "content": final_content})
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": self._tool_call_payload(response.tool_calls),
                    }
                )
                for tc, record in zip(
                    response.tool_calls, await self._aexecute_tool_calls(response.tool_calls)
                ):
                    tool_log.append(record)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": record["result"]})
                    event = {
                        "type": "tool_call",
                        "name": record["name"],
                        "arguments": record["arguments"],
                        "result": record["result"],
                    }
                    if defer_output:
                        deferred.append(event)
                    else:
                        yield event
            else:
                final_content = response.content or (
                    f"（未产出最终回答：已达 max_iterations={self.max_iterations} 上限，"
                    "最后一轮仍在请求工具调用）")
                messages.append({"role": "assistant", "content": final_content})

            if not stopped:
                await self._aguard_output(final_content)
                if defer_output:
                    for event in deferred:
                        yield event
                if self.response_model is not None:
                    error = None
                    for attempt in range(2):
                        try:
                            output = self._try_parse_output(final_content)
                            break
                        except Exception as exc:  # noqa: BLE001
                            error = f"{type(exc).__name__}: {exc}"
                            if attempt == 0:
                                self.tracer.log("structure_retry", error=error)
                                messages.extend(self._structure_prompt(final_content, error))
                                retry = await self.llm.achat(messages)
                                final_content = retry.content
                                for key in usage_total:
                                    usage_total[key] += retry.usage.get(key, 0)
                                messages.append({"role": "assistant", "content": final_content})
                    if output is None:
                        self.tracer.log("structure_failed", error=error)
                        raise OutputValidationError(f"结构化输出在重试后仍失败，最后一次错误: {error}")
        finally:
            self.tracer.end_run(iterations=iterations)

        if stopped:
            return
        if save:
            self._record_turn(session_id, user_input, final_content, tool_log)
        yield {
            "type": "done",
            "result": AgentResult(
                content=final_content,
                tool_calls=tool_log,
                iterations=iterations,
                usage=usage_total,
                output=output,
                reasoning="\n\n".join(reasoning_parts),
            ),
        }

    # ------------------------------------------------------------------
    def enable_skills(self, registry) -> "Agent":
        """接入技能注册表（对齐 DeepSeek Harness / Claude Code 的 SKILL.md 渐进式披露）。

        注册 use_skill 工具，并把技能索引（名字+描述）追加进 system prompt；
        模型按需用 use_skill 加载完整正文。重复调用幂等。
        """
        self.skills = registry
        self.tools.register(registry.use_skill_tool())
        summary = registry.list_summary()
        if summary and self._skills_marker not in self.instructions:
            self.instructions = (
                f"{self.instructions}\n\n{self._skills_marker}\n{summary}"
            ).strip()
        return self

    def as_tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable:
        """把本 agent 包装成一个工具（agent-as-tool），供上级 agent 委托任务。"""
        tool_name = name or self.name

        @tool_decorator(
            name=tool_name,
            description=description
            or f"把任务委托给子代理 '{self.name}' 处理，返回它的最终回答。适合需要专项能力完成的子任务。",
        )
        def delegate(task: str) -> str:
            return self.run(task, session_id=f"delegate-{tool_name}").content

        return delegate
