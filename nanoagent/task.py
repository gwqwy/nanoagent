"""TaskRunner：自主任务分解与逐项推进（连续完成任务的第 3 层）。

与 multi.py 的 Workflow 区别：Workflow 的阶段（计划/开发/审查）是预先编排死的，
适合固定 SOP；TaskRunner 的步骤清单由模型在运行时自己拆解，适合开放式目标。

流程：
    1. 规划：把目标交给 agent 拆解成 JSON 任务清单（解析失败自动重试一次）
    2. 推进：逐项取任务 → agent 自主用工具完成 → 结果写回上下文 → 下一项；
       单项执行异常记为 skipped，不阻断整体
    3. 收尾：全部结束后让 agent 汇总产出
    4. 全程 checkpoint：规划完成、每项完成后落盘，进程中断后
       TaskRunner.resume() 接着上次进度继续，已完成项不会重跑

用法：
    runner = TaskRunner(agent, checkpoint="task.json")
    result = runner.run("给我的项目补全测试并保证全部通过")
    print(result.summary)          # 最终汇总
    for item in result.items:      # 每个子任务的状态与结果
        print(item.status, item.title)

    # 中断后续跑（重建 runner 后）：
    result = TaskRunner.resume("task.json", agent)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, extract_json

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"


class TaskPlanningError(Exception):
    """任务清单在重试后仍无法从模型回答中解析。"""


@dataclass
class TodoItem:
    """清单里的一项子任务。"""

    id: int
    title: str
    detail: str = ""
    status: str = STATUS_PENDING
    result: str = ""


@dataclass
class TaskResult:
    """一次连续任务的最终产物。"""

    goal: str
    items: List[TodoItem] = field(default_factory=list)
    summary: str = ""

    @property
    def done(self) -> int:
        return sum(1 for item in self.items if item.status == STATUS_DONE)

    @property
    def all_done(self) -> bool:
        return bool(self.items) and all(
            item.status in (STATUS_DONE, STATUS_SKIPPED) for item in self.items
        )

    def summary_lines(self) -> List[str]:
        lines = [f"目标: {self.goal}（{self.done}/{len(self.items)} 完成）"]
        for item in self.items:
            mark = {STATUS_DONE: "✓", STATUS_SKIPPED: "✗", STATUS_IN_PROGRESS: "…"}.get(
                item.status, "·"
            )
            lines.append(f"{mark} {item.title}" + (f" —— {item.result}" if item.result else ""))
        return lines


_PLANNING_PROMPT = (
    "你是任务规划器。把下面的目标拆解成一系列可独立执行、按顺序完成的子任务，"
    "不超过 {max_tasks} 个。\n"
    "只输出 JSON 数组，不要 markdown 代码块，格式：\n"
    '[{{"title": "子任务标题", "detail": "具体要求与验收标准"}}]\n\n目标：{goal}'
)

ADD_MARKER = "[ADD]"  # 执行结果中以此开头的行会被解析为新任务（执行中重规划）

_EXECUTION_PROMPT = (
    "总目标：{goal}\n\n"
    "任务清单（共 {total} 项，当前第 {index} 项）：\n{checklist}\n\n"
    "{history}现在执行第 {index} 项：{title}\n{detail}\n\n"
    "可以调用工具完成它；完成后只输出这一项的结果说明（做了什么、产物在哪、结论）。\n"
    "若执行中发现必须新增子任务才能达成目标，在回答末尾另起一行写：{marker} 新任务标题。"
)

_SUMMARY_PROMPT = (
    "总目标：{goal}\n\n各子任务的结果：\n{results}\n\n"
    "请面向目标给出最终汇总：整体完成情况、关键产物与结论。"
)


def _render_checklist(items: List[TodoItem]) -> str:
    mark = {STATUS_DONE: "[完成]", STATUS_SKIPPED: "[跳过]", STATUS_IN_PROGRESS: "[进行中]"}
    return "\n".join(
        f"{mark.get(item.status, '[待办]')} {item.id}. {item.title}"
        + (f"：{item.detail}" if item.detail else "")
        for item in items
    )


class TaskRunner:
    """驱动一个 Agent 连续完成一个大目标。执行器复用 Agent 本身（含工具/记忆）。"""

    def __init__(
        self,
        agent: Agent,
        max_tasks: int = 10,
        checkpoint: str | Path | Any | None = None,
        on_progress: Optional[Callable[[TodoItem], None]] = None,
    ):
        """checkpoint 可以是文件路径（等价 JSON 文件后端）、SQLiteBackend 等任意后端。

        后端里的逻辑 key 默认 "task"（可用 resume(checkpoint, agent, key=...) 改）。
        """
        if max_tasks < 1:
            raise ValueError("max_tasks 必须 >= 1")
        from .checkpoints import resolve_checkpoint

        self.agent = agent
        self.max_tasks = max_tasks
        self.checkpoint = checkpoint
        self._backend = resolve_checkpoint(checkpoint) if checkpoint else None
        self._checkpoint_key = "task"
        self.on_progress = on_progress

    # ------------------------------------------------------------------
    def _persist(self, goal: str, items: List[TodoItem], session_id: str) -> None:
        if not self._backend:
            return
        state = {
            "goal": goal,
            "session_id": session_id,
            "max_tasks": self.max_tasks,
            "items": [asdict(item) for item in items],
        }
        try:
            self._backend.save(self._checkpoint_key, state)
        except OSError:
            pass

    @staticmethod
    def _load_state(checkpoint: str | Path | Any, key: str = "task") -> Dict[str, Any]:
        from .checkpoints import resolve_checkpoint

        state = resolve_checkpoint(checkpoint).load(key)
        if state is None:
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
        if not isinstance(state, dict) or "goal" not in state or "items" not in state:
            raise ValueError(f"checkpoint 内容无效: {checkpoint}")
        return state

    def _notify(self, item: TodoItem) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(replace(item))  # 快照，观察者不受后续状态变化影响
            except Exception:  # noqa: BLE001 —— 回调出错不阻断任务推进
                pass

    # ------------------------------------------------------------------
    def _plan_messages(self, goal: str) -> List[dict]:
        return [{"role": "user", "content": _PLANNING_PROMPT.format(goal=goal, max_tasks=self.max_tasks)}]

    def _parse_plan(self, content: str) -> List[TodoItem]:
        data = extract_json(content)
        if isinstance(data, dict):  # 容错：模型包了一层 {"tasks": [...]}
            data = data.get("tasks") or data.get("items")
        if not isinstance(data, list) or not data:
            raise ValueError("任务清单必须是至少含一项的数组")
        items: List[TodoItem] = []
        for i, entry in enumerate(data[: self.max_tasks], start=1):
            if isinstance(entry, str):
                entry = {"title": entry}
            if not isinstance(entry, dict) or not str(entry.get("title", "")).strip():
                raise ValueError(f"第 {i} 项缺少 title")
            items.append(
                TodoItem(
                    id=i,
                    title=str(entry["title"]).strip(),
                    detail=str(entry.get("detail", "")).strip(),
                )
            )
        return items

    def _plan(self, goal: str) -> List[TodoItem]:
        """调模型拆解目标；解析失败把错误回填重试一次，仍失败抛 TaskPlanningError。"""
        messages = self._plan_messages(goal)
        error: Optional[str] = None
        for attempt in range(2):
            response = self.agent.llm.chat(messages)
            content = response.content
            messages.append({"role": "assistant", "content": content})
            try:
                return self._parse_plan(content)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    messages.append(
                        {"role": "user", "content": f"解析出错（必须修正）：{error}\n请重新只输出 JSON 数组。"}
                    )
        raise TaskPlanningError(f"任务清单解析失败：{error}")

    # ------------------------------------------------------------------
    def _history_text(self, items: List[TodoItem]) -> str:
        finished = [item for item in items if item.result]
        if not finished:
            return ""
        return "已完成任务的结果：\n" + "\n".join(
            f"{item.id}. {item.title}：{item.result}" for item in finished
        ) + "\n\n"

    @staticmethod
    def _harvest_adds(result_text: str, items: List[TodoItem], max_tasks: int) -> List[TodoItem]:
        """从执行结果里收割 [ADD] 行，作为新任务追加到清单尾部（执行中重规划）。"""
        added: List[TodoItem] = []
        next_id = (items[-1].id if items else 0) + 1
        for line in result_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(ADD_MARKER):
                continue
            title = stripped[len(ADD_MARKER):].strip(" ：: ")
            if not title or any(item.title == title for item in items + added):
                continue
            if len(items) + len(added) >= max_tasks:
                break
            added.append(TodoItem(id=next_id, title=title))
            next_id += 1
        return added

    def _run_item(self, goal: str, items: List[TodoItem], item: TodoItem, session_id: str) -> str:
        prompt = _EXECUTION_PROMPT.format(
            goal=goal,
            total=len(items),
            index=item.id,
            checklist=_render_checklist(items),
            history=self._history_text(items),
            title=item.title,
            detail=item.detail,
            marker=ADD_MARKER,
        )
        return self.agent.run(prompt, session_id=session_id).content

    def _summarize(self, goal: str, items: List[TodoItem], session_id: str) -> str:
        results = "\n".join(
            f"{item.id}. {item.title}（{item.status}）：{item.result or '无结果'}" for item in items
        )
        return self.agent.run(_SUMMARY_PROMPT.format(goal=goal, results=results), session_id=session_id).content

    # ------------------------------------------------------------------
    def _drive(self, goal: str, items: List[TodoItem], session_id: str) -> TaskResult:
        """推进清单：执行全部未完成项（含执行中新增的），再汇总。"""
        for item in items:
            if item.status in (STATUS_DONE, STATUS_SKIPPED):
                continue
            item.status = STATUS_IN_PROGRESS
            self._notify(item)
            try:
                raw = self._run_item(goal, items, item, session_id)
                # 执行中重规划：收割 [ADD] 新任务（追加进 items，本轮循环会继续执行）
                new_items = self._harvest_adds(raw, items, self.max_tasks)
                cleaned = "\n".join(
                    line for line in raw.splitlines() if not line.strip().startswith(ADD_MARKER)
                ).strip()
                item.result = cleaned
                items.extend(new_items)
                item.status = STATUS_DONE
            except Exception as exc:  # noqa: BLE001 —— 单项失败不阻断整体
                item.status = STATUS_SKIPPED
                item.result = f"{type(exc).__name__}: {exc}"
            self._notify(item)
            self._persist(goal, items, session_id)

        summary = self._summarize(goal, items, session_id)
        self._persist(goal, items, session_id)
        return TaskResult(goal=goal, items=items, summary=summary)

    def run(self, goal: str, session_id: str = "task") -> TaskResult:
        items = self._plan(goal)
        self._persist(goal, items, session_id)
        return self._drive(goal, items, session_id)

    @classmethod
    def resume(cls, checkpoint: str | Path | Any, agent: Agent,
               key: str = "task", on_progress: Optional[Callable[[TodoItem], None]] = None) -> TaskResult:
        """从 checkpoint 恢复：不重新规划，直接从第一个未完成项继续。"""
        state = cls._load_state(checkpoint, key)
        items = [TodoItem(**entry) for entry in state["items"]]
        runner = cls(
            agent,
            max_tasks=state.get("max_tasks", 10),
            checkpoint=checkpoint,
            on_progress=on_progress,
        )
        runner._checkpoint_key = key
        return runner._drive(state["goal"], items, state.get("session_id", "task"))

    # ------------------------------------------------------------------
    async def _arun_item(self, goal: str, items: List[TodoItem], item: TodoItem, session_id: str) -> str:
        prompt = _EXECUTION_PROMPT.format(
            goal=goal,
            total=len(items),
            index=item.id,
            checklist=_render_checklist(items),
            history=self._history_text(items),
            title=item.title,
            detail=item.detail,
            marker=ADD_MARKER,
        )
        return (await self.agent.arun(prompt, session_id=session_id)).content

    async def _asummarize(self, goal: str, items: List[TodoItem], session_id: str) -> str:
        results = "\n".join(
            f"{item.id}. {item.title}（{item.status}）：{item.result or '无结果'}" for item in items
        )
        return (await self.agent.arun(_SUMMARY_PROMPT.format(goal=goal, results=results), session_id=session_id)).content

    async def _adrive(self, goal: str, items: List[TodoItem], session_id: str) -> TaskResult:
        for item in items:
            if item.status in (STATUS_DONE, STATUS_SKIPPED):
                continue
            item.status = STATUS_IN_PROGRESS
            self._notify(item)
            try:
                raw = await self._arun_item(goal, items, item, session_id)
                new_items = self._harvest_adds(raw, items, self.max_tasks)
                cleaned = "\n".join(
                    line for line in raw.splitlines() if not line.strip().startswith(ADD_MARKER)
                ).strip()
                item.result = cleaned
                items.extend(new_items)
                item.status = STATUS_DONE
            except Exception as exc:  # noqa: BLE001
                item.status = STATUS_SKIPPED
                item.result = f"{type(exc).__name__}: {exc}"
            self._notify(item)
            self._persist(goal, items, session_id)
        summary = await self._asummarize(goal, items, session_id)
        self._persist(goal, items, session_id)
        return TaskResult(goal=goal, items=items, summary=summary)

    async def arun(self, goal: str, session_id: str = "task") -> TaskResult:
        """异步版：agent 必须支持 achat（如 AsyncLLM）。"""
        messages = self._plan_messages(goal)
        error: Optional[str] = None
        for attempt in range(2):
            response = await self.agent.llm.achat(messages)
            content = response.content
            messages.append({"role": "assistant", "content": content})
            try:
                items = self._parse_plan(content)
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    messages.append(
                        {"role": "user", "content": f"解析出错（必须修正）：{error}\n请重新只输出 JSON 数组。"}
                    )
        else:
            raise TaskPlanningError(f"任务清单解析失败：{error}")
        self._persist(goal, items, session_id)
        return await self._adrive(goal, items, session_id)

    @classmethod
    async def aresume(cls, checkpoint: str | Path | Any, agent: Agent,
                      key: str = "task", on_progress: Optional[Callable[[TodoItem], None]] = None) -> TaskResult:
        """resume 的异步版。"""
        state = cls._load_state(checkpoint, key)
        items = [TodoItem(**entry) for entry in state["items"]]
        runner = cls(
            agent,
            max_tasks=state.get("max_tasks", 10),
            checkpoint=checkpoint,
            on_progress=on_progress,
        )
        runner._checkpoint_key = key
        return await runner._adrive(state["goal"], items, state.get("session_id", "task"))
