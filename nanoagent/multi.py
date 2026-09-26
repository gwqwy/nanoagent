"""多 agent 编排。

三种模式（参考 CrewAI 的角色分工与 OpenAI Agents SDK 的 handoff）：
- agent-as-tool 委托：把子 agent 注册成父 agent 的工具，由父 agent 决定何时调用
- Pipeline 流水线：固定顺序，前一个 agent 的输出作为下一个的输入
- Workflow 工作流：计划 → 开发 ↔ 审查（反馈回路）→ 汇总，每个角色只看到
  自己需要的"产物包"，审查不过自动打回重做
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .agent import Agent
from .tools import tool as tool_decorator


def agent_as_tool(agent: Agent, name: str | None = None, description: str | None = None):
    """把一个 agent 包装成工具函数。Agent.as_tool 的独立函数版本。"""
    return agent.as_tool(name=name, description=description)


class Pipeline:
    """顺序流水线：task 依次流过每个 agent，最后的输出作为整体输出。"""

    def __init__(self, agents: Sequence[Agent]):
        if not agents:
            raise ValueError("Pipeline 至少需要一个 agent")
        self.agents = list(agents)

    def run(self, task: str) -> str:
        data = task
        for agent in self.agents:
            data = agent.run(data).content
        return data

    async def arun(self, task: str) -> str:
        """Pipeline 的异步版。"""
        data = task
        for agent in self.agents:
            data = (await agent.arun(data)).content
        return data


class Team:
    """由 leader 统筹、members 提供专项能力的协作团队。

    members 被注册为 leader 的工具，leader 根据任务自主决定委托给谁。
    """

    def __init__(self, leader: Agent, members: Sequence[Agent]):
        if not members:
            raise ValueError("Team 至少需要一个 member")
        self.leader = leader
        self.members: List[Agent] = list(members)
        for member in self.members:
            self.leader.tools.register(
                agent_as_tool(
                    member,
                    description=f"子代理 '{member.name}'。把需要该能力的任务委托给它，"
                    f"在 description 中补充它的职责说明会更有效。",
                )
            )

    def run(self, task: str) -> str:
        return self.leader.run(task).content

    async def arun(self, task: str) -> str:
        """Team 的异步版。"""
        return (await self.leader.arun(task)).content


@dataclass
class WorkflowResult:
    """一次工作流的全部产物。"""

    task: str
    plan: str
    code: str
    review: str
    summary: str
    rounds: int
    passed: bool


def _load_checkpoint(checkpoint) -> dict:
    """从后端读取工作流 checkpoint（字符串/路径 → JSON 文件后端，向后兼容）。"""
    from .checkpoints import resolve_checkpoint

    backend = resolve_checkpoint(checkpoint)
    state = backend.load("workflow")
    if state is None:
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    if not isinstance(state, dict) or "task" not in state or "plan" not in state:
        raise ValueError(f"checkpoint 内容无效: {checkpoint}")
    return state


class Workflow:
    """计划 → 开发 ↔ 审查 → 汇总 的固定协作工作流（含反馈回路）。

    设计要点：
    - 职责边界由各 agent 的 instructions 划定，Workflow 只负责传"产物包"
    - 审查员不过关时把审查意见回填给程序员，最多重试 max_rounds 轮
    - pass 判定看审查回答的最后一行是否含 pass_marker（默认 PASS），
      末行无标记时兜底：整体含 PASS 且不含 FAIL 也算通过
    - 所有 agent 共用本次 run 的 session_id（默认随机生成），互不串话；
      四个 agent 也可以带自己的工具（如程序员带文件读写工具，见示例 09）
    """

    FAIL_MARKER = "FAIL"

    def __init__(
        self,
        planner: Agent,
        coder: Agent,
        reviewer: Agent,
        summarizer: Agent,
        max_rounds: int = 3,
        pass_marker: str = "PASS",
        checkpoint_path: str | Path | None = None,
    ):
        from .checkpoints import resolve_checkpoint
        missing = [
            role
            for role, agent in (
                ("planner", planner),
                ("coder", coder),
                ("reviewer", reviewer),
                ("summarizer", summarizer),
            )
            if not isinstance(agent, Agent)
        ]
        if missing:
            raise TypeError(f"以下角色不是 Agent 实例: {', '.join(missing)}")
        if max_rounds < 1:
            raise ValueError("max_rounds 必须 >= 1")
        self.planner = planner
        self.coder = coder
        self.reviewer = reviewer
        self.summarizer = summarizer
        self.max_rounds = max_rounds
        self.pass_marker = pass_marker
        self.checkpoint_path = checkpoint_path  # 兼容保留；实际读写走后端
        self._checkpoint = resolve_checkpoint(checkpoint_path) if checkpoint_path else None

    # ------------------------------------------------------------------
    def _is_pass(self, review: str) -> bool:
        """PASS 判定：末行 FAIL 优先（保守），其次末行 PASS，最后兜底整体判断。"""
        lines = [line.strip() for line in review.strip().splitlines() if line.strip()]
        if not lines:
            return False
        last = lines[-1]
        if self.FAIL_MARKER in last:
            return False
        if self.pass_marker in last:
            return True
        return self.pass_marker in review and self.FAIL_MARKER not in review

    # ------------------------------------------------------------------
    def _persist(self, state: dict) -> None:
        """checkpoint 落盘；未启用或写失败不打断工作流。"""
        if not self._checkpoint:
            return
        try:
            self._checkpoint.save("workflow", state)
        except OSError:
            pass

    def _state(self, task, session_id, plan, rounds_done, code, review, passed, done=False) -> dict:
        return {
            "task": task,
            "session_id": session_id,
            "max_rounds": self.max_rounds,
            "pass_marker": self.pass_marker,
            "plan": plan,
            "rounds_done": rounds_done,
            "code": code,
            "review": review,
            "passed": passed,
            "done": done,
        }

    def run(self, task: str, session_id: str | None = None, _resume: dict | None = None) -> WorkflowResult:
        """执行工作流。_resume 供 resume() 从 checkpoint 恢复，正常调用不要传。"""
        if _resume is not None:
            state = _resume
            task = state["task"]
            session_id = session_id or state["session_id"]
            plan, code, review = state["plan"], state["code"], state["review"]
            passed, start_round = state["passed"], state["rounds_done"] + 1
        else:
            session_id = session_id or f"wf-{uuid.uuid4().hex[:8]}"
            plan = self.planner.run(
                f"任务：{task}\n\n请输出开发计划：目标、实现步骤、涉及的文件。不要写代码。",
                session_id=session_id,
            ).content
            code = review = ""
            passed = False
            start_round = 1
            self._persist(self._state(task, session_id, plan, 0, "", "", False))

        issues = review if review else "无（首次开发）"
        rounds = start_round - 1
        for round_no in range(start_round, self.max_rounds + 1):
            rounds = round_no
            code = self.coder.run(
                f"任务：{task}\n\n开发计划：\n{plan}\n\n需修正的审查意见：{issues}",
                session_id=session_id,
            ).content
            review = self.reviewer.run(
                f"第 {round_no} 轮审查。\n\n开发计划：\n{plan}\n\n代码：\n{code}\n\n"
                f"对照计划逐项审查。最后一行输出 {self.pass_marker} 或 {self.FAIL_MARKER} 与问题清单。",
                session_id=session_id,
            ).content
            passed = self._is_pass(review)
            self._persist(self._state(task, session_id, plan, round_no, code, review, passed))
            if passed:
                break
            issues = review  # 反馈回路：审查意见回填给下一轮程序员

        summary = self.summarizer.run(
            f"任务：{task}\n\n开发计划：\n{plan}\n\n代码：\n{code}\n\n"
            f"审查结论（共 {rounds} 轮）：\n{review}\n\n请汇总成最终交付说明。",
            session_id=session_id,
        ).content
        self._persist(self._state(task, session_id, plan, rounds, code, review, passed, done=True))

        return WorkflowResult(
            task=task, plan=plan, code=code, review=review, summary=summary,
            rounds=rounds, passed=passed,
        )

    async def arun(self, task: str, session_id: str | None = None, _resume: dict | None = None) -> WorkflowResult:
        """run 的异步版（含同样的 checkpoint 逻辑）。"""
        if _resume is not None:
            state = _resume
            task = state["task"]
            session_id = session_id or state["session_id"]
            plan, code, review = state["plan"], state["code"], state["review"]
            passed, start_round = state["passed"], state["rounds_done"] + 1
        else:
            session_id = session_id or f"wf-{uuid.uuid4().hex[:8]}"
            plan = (await self.planner.arun(
                f"任务：{task}\n\n请输出开发计划：目标、实现步骤、涉及的文件。不要写代码。",
                session_id=session_id,
            )).content
            code = review = ""
            passed = False
            start_round = 1
            self._persist(self._state(task, session_id, plan, 0, "", "", False))

        issues = review if review else "无（首次开发）"
        rounds = start_round - 1
        for round_no in range(start_round, self.max_rounds + 1):
            rounds = round_no
            code = (await self.coder.arun(
                f"任务：{task}\n\n开发计划：\n{plan}\n\n需修正的审查意见：{issues}",
                session_id=session_id,
            )).content
            review = (await self.reviewer.arun(
                f"第 {round_no} 轮审查。\n\n开发计划：\n{plan}\n\n代码：\n{code}\n\n"
                f"对照计划逐项审查。最后一行输出 {self.pass_marker} 或 {self.FAIL_MARKER} 与问题清单。",
                session_id=session_id,
            )).content
            passed = self._is_pass(review)
            self._persist(self._state(task, session_id, plan, round_no, code, review, passed))
            if passed:
                break
            issues = review

        summary = (await self.summarizer.arun(
            f"任务：{task}\n\n开发计划：\n{plan}\n\n代码：\n{code}\n\n"
            f"审查结论（共 {rounds} 轮）：\n{review}\n\n请汇总成最终交付说明。",
            session_id=session_id,
        )).content
        self._persist(self._state(task, session_id, plan, rounds, code, review, passed, done=True))

        return WorkflowResult(
            task=task, plan=plan, code=code, review=review, summary=summary,
            rounds=rounds, passed=passed,
        )

    # ------------------------------------------------------------------
    @classmethod
    def resume(
        cls, checkpoint: str | Path | Any, planner: Agent, coder: Agent,
        reviewer: Agent, summarizer: Agent,
    ) -> WorkflowResult:
        """从 checkpoint 恢复工作流，只执行剩余轮次（同步版）。

        checkpoint 可以是文件路径（等价 JSON 文件后端）或任意 CheckpointBackend。
        """
        state = _load_checkpoint(checkpoint)
        workflow = cls(
            planner, coder, reviewer, summarizer,
            max_rounds=state.get("max_rounds", 3),
            pass_marker=state.get("pass_marker", "PASS"),
            checkpoint_path=checkpoint,
        )
        return workflow.run(state["task"], _resume=state)

    @classmethod
    async def aresume(
        cls, checkpoint: str | Path | Any, planner: Agent, coder: Agent,
        reviewer: Agent, summarizer: Agent,
    ) -> WorkflowResult:
        """resume 的异步版。"""
        state = _load_checkpoint(checkpoint)
        workflow = cls(
            planner, coder, reviewer, summarizer,
            max_rounds=state.get("max_rounds", 3),
            pass_marker=state.get("pass_marker", "PASS"),
            checkpoint_path=checkpoint,
        )
        return await workflow.arun(state["task"], _resume=state)
