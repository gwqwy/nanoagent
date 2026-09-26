"""Evals：agent 任务的批量评估与回归基准。

衡量的是"agent 把任务做得多好"（与 tests/ 的机制测试互补）。三种判分器：
    contains  期望关键词全部出现在回答中（确定性，零成本）
    regex     期望正则命中
    judge     LLM-as-judge：把回答交给判别模型按评分说明打 0-100 分

用法：
    from nanoagent import EvalRunner, LLM

    runner = EvalRunner()
    runner.add_case(question="北京天气？", expect_contains=["25"], tools=[get_weather])
    runner.add_case(question="总结这段话", judge_rubric="摘要是否覆盖全部要点且不超50字")
    report = runner.run(agent)
    print(report.summary_text())     # 通过率 / 各用例得分
    report.save("evals/report.json") # 存档，用于回归对比
"""

from __future__ import annotations

import json
import re as _re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional


class EvalResult:
    """一次判分的结果。score: 0-100；passed: score >= threshold。"""

    def __init__(self, score: float, passed: bool, reason: str = ""):
        self.score = score
        self.passed = passed
        self.reason = reason

    def to_dict(self) -> dict:
        return {"score": self.score, "passed": self.passed, "reason": self.reason}


@dataclass
class EvalCase:
    """一个评估用例：question 是输入，expect_contains/regex/judge_rubric 三选一（可组合）。"""

    name: str
    question: str = ""
    expect_contains: Optional[List[str]] = None
    expect_regex: Optional[str] = None
    judge_rubric: Optional[str] = None          # 提供则启用 LLM-as-judge
    judge_threshold: float = 70.0               # judge 模式的及格线
    tools: Optional[List[Any]] = None           # 本用例专属工具（临时注册）
    session_id: str = ""                        # 留空 = 自动用 "eval-<用例名>"，互不污染


@dataclass
class CaseReport:
    case_name: str
    passed: bool
    score: float
    answer: str
    reason: str = ""
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "case": self.case_name, "passed": self.passed, "score": self.score,
            "answer": self.answer[:500], "reason": self.reason,
            "error": self.error, "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class EvalReport:
    cases: List[CaseReport] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    @property
    def total_elapsed_ms(self) -> int:
        return sum(c.elapsed_ms for c in self.cases)

    def summary_text(self) -> str:
        lines = [f"通过率: {self.pass_rate:.0%}（{sum(1 for c in self.cases if c.passed)}/{len(self.cases)}），"
                 f"总耗时 {self.total_elapsed_ms}ms"]
        for case in self.cases:
            mark = "✓" if case.passed else "✗"
            line = f"  {mark} {case.case_name}  score={case.score:g}"
            if case.error:
                line += f"  error: {case.error}"
            elif case.reason:
                line += f"  ({case.reason})"
            lines.append(line)
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "pass_rate": self.pass_rate,
            "cases": [c.to_dict() for c in self.cases],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class EvalRunner:
    """跑一组评估用例：确定性判分本地完成，judge 判分走独立 LLM 调用。"""

    JUDGE_PROMPT = (
        "你是评估员。根据评分标准给下面的回答打 0-100 分。\n"
        "只输出 JSON（不要 markdown 代码块）：{{\"score\": 数字, \"reason\": \"简短理由\"}}\n\n"
        "评分标准：{rubric}\n\n问题：{question}\n\n回答：{answer}"
    )

    def __init__(self, judge_llm: Any = None, cases: Optional[List[EvalCase]] = None):
        self.judge_llm = judge_llm
        self.cases: List[EvalCase] = list(cases or [])

    def add_case(self, name: str | None = None, **kwargs: Any) -> EvalCase:
        """添加用例；name 缺省时自动编号（case-1、case-2…）。"""
        if name is None:
            name = f"case-{len(self.cases) + 1}"
        case = EvalCase(name=name, **kwargs)
        self.cases.append(case)
        return case

    # ------------------------------------------------------------------
    def _score_contains(self, case: EvalCase, answer: str) -> EvalResult:
        missing = [kw for kw in (case.expect_contains or []) if kw not in answer]
        if missing:
            return EvalResult(0, False, f"缺少关键词: {', '.join(missing)}")
        return EvalResult(100, True, f"命中全部 {len(case.expect_contains)} 个关键词")

    def _score_regex(self, case: EvalCase, answer: str) -> EvalResult:
        if _re.search(case.expect_regex or "", answer):
            return EvalResult(100, True, f"命中正则 {case.expect_regex!r}")
        return EvalResult(0, False, f"未命中正则 {case.expect_regex!r}")

    def _score_judge(self, case: EvalCase, answer: str) -> EvalResult:
        from .agent import extract_json

        if self.judge_llm is None:
            raise ValueError("judge 用例需要 EvalRunner(judge_llm=...)")
        prompt = self.JUDGE_PROMPT.format(rubric=case.judge_rubric, question=case.question, answer=answer)
        response = self.judge_llm.chat([{"role": "user", "content": prompt}])
        try:
            data = extract_json(response.content)
            score = float(data["score"])
            reason = str(data.get("reason", ""))
        except Exception as exc:  # noqa: BLE001 —— 判分失败按 0 分计（保守）
            return EvalResult(0, False, f"judge 输出无法解析: {type(exc).__name__}: {exc}")
        score = max(0.0, min(100.0, score))
        return EvalResult(score, score >= case.judge_threshold, reason)

    def score_case(self, case: EvalCase, answer: str) -> EvalResult:
        """按用例配置判分；多个判分器同时配置时取最低分（全部通过才算过）。"""
        results: List[EvalResult] = []
        if case.expect_contains:
            results.append(self._score_contains(case, answer))
        if case.expect_regex:
            results.append(self._score_regex(case, answer))
        if case.judge_rubric:
            results.append(self._score_judge(case, answer))
        if not results:
            raise ValueError(f"用例 '{case.name}' 没有配置任何判分器")
        worst = min(results, key=lambda r: r.score)
        reasons = "；".join(r.reason for r in results if r.reason)
        return EvalResult(worst.score, all(r.passed for r in results), reasons)

    # ------------------------------------------------------------------
    def run_case(self, agent: Any, case: EvalCase) -> CaseReport:
        """跑单个用例：每例独立会话；用例自带工具时临时注册。

        会话隔离：session_id 留空时按用例名生成（"eval-<用例名>"）并在开跑前清空，
        否则所有用例共用同一会话，前一个用例的对话历史会泄漏进下一个（评测结果不可信）。
        显式指定 session_id 的用例尊重用户选择，不做清空。
        """
        started = time.perf_counter()
        session_id = case.session_id or f"eval-{case.name}"
        if not case.session_id:
            memory = getattr(agent, "memory", None)
            if memory is not None and hasattr(memory, "clear"):
                memory.clear(session_id)   # 同用例重跑也要从干净状态开始
        try:
            from .tools import ToolRegistry

            original_tools = agent.tools
            if case.tools:
                merged = ToolRegistry()
                for name in original_tools.names():
                    merged.register(original_tools.get(name))
                for item in case.tools:
                    merged.register(item)
                agent.tools = merged
            try:
                result = agent.run(case.question, session_id=session_id)
            finally:
                agent.tools = original_tools

            scored = self.score_case(case, result.content)
            return CaseReport(
                case_name=case.name, passed=scored.passed, score=scored.score,
                answer=result.content, reason=scored.reason,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 —— 用例报错计 0 分不中断整批
            return CaseReport(
                case_name=case.name, passed=False, score=0, answer="",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )

    def run(self, agent: Any) -> EvalReport:
        """顺序跑全部用例，返回汇总报告。"""
        return EvalReport(cases=[self.run_case(agent, case) for case in self.cases])
