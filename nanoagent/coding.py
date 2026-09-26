"""编程工具集：把 nanoagent 变成编程 agent 的开箱即用组件。

CodingWorkspace 提供一套文件/shell 工具，带三层安全设计：
    1. 路径监狱：所有文件操作限制在工作区根目录内（绝对路径与 .. 逃逸都拒绝）
    2. 确认门：run_command / delete_file 默认拒绝执行（fail-closed），
       需要 auto_approve=True 或提供 confirm 回调由人工放行
    3. 输出限幅：命令输出与文件读取超长截断，避免撑爆模型上下文

用法：
    from nanoagent import Agent, CodingWorkspace

    ws = CodingWorkspace("./workspace", auto_approve=True)
    agent = Agent(name="程序员", tools=ws.as_tools(),
                  instructions="你是程序员，先读代码再改，改完必须跑测试。")
    agent.run("在 utils.py 里实现 slugify 并写好测试")

工具清单（与 Claude Code 等编程 agent 对齐的最小集）：
    read_file / write_file / edit_file / list_files / search_code /
    run_command / delete_file / file_diff
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .tools import Tool, make_tool

DEFAULT_IGNORES = {"__pycache__", ".git", "node_modules", ".venv", ".nanoagent"}
MAX_OUTPUT_CHARS = 4000
MAX_FILE_CHARS = 20000


class SandboxViolation(Exception):
    """试图访问工作区之外的路径。"""


class CodingWorkspace:
    """一个受限的工作区：agent 的全部文件操作都发生在 root 之内。"""

    def __init__(
        self,
        root: str | Path = "./workspace",
        *,
        confirm: Optional[Callable[[str], bool]] = None,
        auto_approve: bool = False,
        ignores: Optional[set] = None,
        allowed_tools: Optional[set] = None,
        agent_factory: Optional[Callable[[str], Any]] = None,
    ):
        """allowed_tools: 工具名白名单（None 表示全部可用），实现按工具粒度的权限控制；
        agent_factory: 接收 instructions 返回带本工作区工具的 Agent，spawn_subagent 依赖它。
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.confirm = confirm
        self.auto_approve = auto_approve
        self.ignores = set(ignores) if ignores is not None else set(DEFAULT_IGNORES)
        self.allowed_tools = allowed_tools
        self.agent_factory = agent_factory
        self._processes: Dict[int, Any] = {}
        self._process_logs: Dict[int, Path] = {}
        self._process_files: Dict[int, Any] = {}
        self._next_pid = 1
        self._next_subagent_id = 1  # N-17：子代理会话编号与后台进程 pid 分开计数

    # ------------------------------------------------------------------
    def _safe_path(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """把相对路径解析到工作区内；逃逸（绝对路径出界 / .. 跳出）一律拒绝。

        工作区内含 .. 的相对路径（如 sub/../a.py）是允许的，最终位置在界内即可。
        """
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise SandboxViolation(f"路径越出工作区: {path}（工作区: {self.root}）")
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        return resolved

    def _display(self, path: Path) -> str:
        """给模型看的路径：工作区相对路径。"""
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def _glob_paths(self, pattern: str) -> List[Path]:
        """受约束的 glob：模式与命中项都不得越出工作区。

        安全说明：``Path.glob`` 会把 ``..`` 当作可展开分量，因此 ``self.root.glob("../*")``
        能列出工作区外的文件 —— 仅靠 ``_safe_path`` 保护单文件读写是不够的。
        这里先拒绝模式本身的逃逸，再对每个命中项做一次 ``_safe_path`` 兜底。
        """
        raw = str(pattern or "").strip()
        if not raw:
            raise SandboxViolation("glob 模式不能为空")
        normalized = raw.replace("\\", "/")
        if Path(raw).is_absolute() or normalized.startswith("/"):
            raise SandboxViolation(f"glob 模式不允许绝对路径: {pattern}")
        if ".." in normalized.split("/"):
            raise SandboxViolation(f"glob 模式不允许包含 '..': {pattern}")
        safe_paths: List[Path] = []
        for path in sorted(self.root.glob(raw)):
            try:
                safe_paths.append(self._safe_path(path))
            except SandboxViolation:
                continue
        return safe_paths

    def _require_approval(self, action: str) -> Optional[str]:
        """危险操作的确认门。返回 None 表示放行，返回字符串表示拒绝（含理由）。"""
        if self.auto_approve:
            return None
        if self.confirm is not None:
            if self.confirm(action):
                return None
            return f"操作已被人工确认拒绝: {action}"
        return (
            f"危险操作默认被拒绝（fail-closed）: {action}\n"
            "如需放行：创建 CodingWorkspace 时设置 auto_approve=True，"
            "或提供 confirm 回调做人工确认。"
        )

    @staticmethod
    def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...（输出过长，已截断，共 {len(text)} 字符）"

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        """宽容地把模型传来的参数转成 int。

        模型即使拿到正确的 schema 也可能把数字写成字符串（"10"）；
        旧实现在这种输入下会直接 `TypeError: unsupported operand type(s) for -`。
        """
        if isinstance(value, bool) or value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """宽容地解析布尔参数（"true"/"1"/"yes" 等字符串形式）。"""
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
        return default

    # ------------------------------------------------------------------
    # 文件工具
    # ------------------------------------------------------------------
    def read_file(self, path: str, start_line: int = 0, end_line: int = 0) -> str:
        """读取工作区内文件，带行号；可选行区间（1 起始，0 表示不限）。

        Args:
            path: 工作区相对路径
            start_line: 起始行（含），0 表示从头
            end_line: 结束行（含），0 表示到文件末尾
        """
        resolved = self._safe_path(path, must_exist=True)
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        start_line = self._as_int(start_line, 0)
        end_line = self._as_int(end_line, 0)
        start = max(start_line - 1, 0) if start_line else 0
        end = end_line if end_line else len(lines)
        selected = lines[start:end]
        numbered = [f"{start + i + 1:>5}: {line}" for i, line in enumerate(selected)]
        return self._truncate("\n".join(numbered) or "（空文件）", MAX_FILE_CHARS)

    def write_file(self, path: str, content: str) -> str:
        """写入文件（新建或整体覆盖），自动创建父目录；返回操作说明。

        Args:
            path: 工作区相对路径
            content: 完整文件内容
        """
        resolved = self._safe_path(path)
        existed = resolved.is_file()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        action = "覆盖" if existed else "新建"
        return f"{action} {self._display(resolved)}（{len(content.splitlines())} 行）"

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """精确字符串替换编辑：old_text 必须唯一命中（除非 replace_all=True）。

        Args:
            path: 工作区相对路径
            old_text: 要被替换的原文（必须与文件内容精确匹配）
            new_text: 替换后的新文本
            replace_all: 命中多处时是否全部替换
        """
        resolved = self._safe_path(path, must_exist=True)
        replace_all = self._as_bool(replace_all)
        source = resolved.read_text(encoding="utf-8")
        count = source.count(old_text)
        if count == 0:
            return f"错误：在 {self._display(resolved)} 中找不到要替换的文本"
        if count > 1 and not replace_all:
            return f"错误：{self._display(resolved)} 中命中 {count} 处，需确认唯一或传 replace_all=true"

        updated = source.replace(old_text, new_text) if replace_all else source.replace(old_text, new_text, 1)
        resolved.write_text(updated, encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(
            source.splitlines(), updated.splitlines(),
            fromfile=self._display(resolved), tofile=self._display(resolved), lineterm="",
        ))
        return f"已编辑 {self._display(resolved)}（{count if replace_all else 1} 处）\n{self._truncate(diff)}"

    def file_diff(self, path: str, content: str) -> str:
        """预览 diff：比较给定内容与工作区文件的差异（不写入）。

        Args:
            path: 工作区相对路径
            content: 拟写入的新内容
        """
        resolved = self._safe_path(path)
        old = resolved.read_text(encoding="utf-8").splitlines() if resolved.is_file() else []
        diff = "\n".join(difflib.unified_diff(
            old, content.splitlines(),
            fromfile=f"{self._display(resolved)}（当前）", tofile=f"{self._display(resolved)}（拟写入）", lineterm="",
        ))
        return diff or "无差异"

    def delete_file(self, path: str) -> str:
        """删除工作区内的文件（危险操作，需确认门放行）。

        Args:
            path: 工作区相对路径
        """
        resolved = self._safe_path(path, must_exist=True)
        blocked = self._require_approval(f"delete_file: {self._display(resolved)}")
        if blocked:
            return blocked
        resolved.unlink()
        return f"已删除 {self._display(resolved)}"

    # ------------------------------------------------------------------
    # 浏览与搜索
    # ------------------------------------------------------------------
    def list_files(self, pattern: str = "**/*") -> str:
        """按 glob 列出工作区文件（自动跳过 __pycache__/.git 等目录）。

        Args:
            pattern: glob 模式，如 **/*.py 或 src/*.md
        """
        try:
            paths = self._glob_paths(pattern)
        except SandboxViolation as exc:
            return f"错误：{exc}"
        matches = []
        for path in paths:
            if not path.is_file():
                continue
            if any(part in self.ignores for part in path.parts):
                continue
            matches.append(self._display(path))
        return "\n".join(matches) or "（无匹配文件）"

    def search_code(self, query: str, pattern: str = "**/*") -> str:
        """在工作区内做正则搜索，返回 `文件:行号: 内容` 列表。

        Args:
            query: 正则表达式（Python 语法）
            pattern: 限定文件范围的 glob 模式
        """
        import re as _re

        try:
            regex = _re.compile(query)
        except _re.error as exc:
            return f"错误：正则不合法: {exc}"
        try:
            paths = self._glob_paths(pattern)
        except SandboxViolation as exc:
            return f"错误：{exc}"

        hits: List[str] = []
        for path in paths:
            if not path.is_file() or any(part in self.ignores for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{self._display(path)}:{line_no}: {line.strip()}")
            if len(hits) >= 100:
                break
        return self._truncate("\n".join(hits) or "（无匹配）")

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------
    def run_command(self, command: str, timeout: int = 60) -> str:
        """在工作区根目录执行 shell 命令，返回 stdout+stderr（危险操作，需确认门放行）。

        Args:
            command: 完整命令行，如 python -m pytest -q
            timeout: 超时秒数
        """
        blocked = self._require_approval(f"run_command: {command}")
        if blocked:
            return blocked
        timeout = max(1, self._as_int(timeout, 60))
        try:
            completed = subprocess.run(
                command, shell=True, cwd=str(self.root),
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return f"错误：命令超时（>{timeout}s）已被终止: {command}"
        output = (completed.stdout or "") + (completed.stderr or "")
        return self._truncate(f"[exit {completed.returncode}] {output.strip() or '（无输出）'}")

    # ------------------------------------------------------------------
    # Git 集成
    # ------------------------------------------------------------------
    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=str(self.root), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            return f"git {args[0]} 失败: {output.strip()}"
        return output.strip() or "（无输出）"

    def git_status(self) -> str:
        """查看工作区 git 状态（工作区不是 git 仓库时返回提示）。"""
        return self._git("status", "--short", "--branch")

    def git_diff(self, path: str = "") -> str:
        """查看未提交改动（含已暂存），可选限定单个文件。

        Args:
            path: 工作区相对路径，空表示全部改动
        """
        args = ["diff", "HEAD"]
        if path:
            args.append(self._safe_path(path).relative_to(self.root).as_posix())
        result = self._git(*args)
        # 空仓库还没有 HEAD，回退到普通 diff；未跟踪文件不会出现在 diff 里，补 status 段
        if "unknown revision" in result:
            result = self._git("diff")
        if result == "（无输出）" or "unknown revision" in result:
            status = self._git("status", "--short")
            result = f"{result}\n未跟踪/状态:\n{status}" if status != "（无输出）" else result
        return self._truncate(result)

    def git_commit(self, message: str) -> str:
        """暂存全部改动并提交（危险操作，需确认门放行）。

        Args:
            message: 提交说明
        """
        blocked = self._require_approval(f"git_commit: {message}")
        if blocked:
            return blocked
        add_result = self._git("add", "-A")
        if add_result.startswith("git add 失败"):
            return add_result
        return self._git("commit", "-m", message)

    # ------------------------------------------------------------------
    # 后台进程（长驻服务不阻塞 agent loop）
    # ------------------------------------------------------------------
    def start_process(self, command: str) -> str:
        """后台启动一条命令，输出写入工作区 .processes/<id>.log（需确认门放行）。

        Args:
            command: 完整命令行，如 python -m http.server 8000
        """
        blocked = self._require_approval(f"start_process: {command}")
        if blocked:
            return blocked
        log_dir = self.root / ".processes"
        log_dir.mkdir(exist_ok=True)
        pid = self._next_pid
        self._next_pid += 1
        log_path = log_dir / f"{pid}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, shell=True, cwd=str(self.root),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        self._processes[pid] = process
        self._process_logs[pid] = log_path
        self._process_files[pid] = log_file
        return f"后台进程 #{pid} 已启动: {command}（日志: .processes/{pid}.log）"

    def read_process(self, process_id: int, tail_lines: int = 50) -> str:
        """读取后台进程的输出尾部与运行状态。

        Args:
            process_id: start_process 返回的进程编号
            tail_lines: 最多返回的行数
        """
        process_id = self._as_int(process_id, -1)
        tail_lines = max(1, self._as_int(tail_lines, 50))
        process = self._processes.get(process_id)
        if process is None:
            return f"错误：不存在后台进程 #{process_id}"
        log_path = self._process_logs[process_id]
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []
        status = "运行中" if process.poll() is None else f"已退出（exit {process.returncode}）"
        if process.poll() is not None:
            self._release_process_log(process_id)   # 进程已自然退出：顺手关闭日志句柄
        tail = "\n".join(lines[-tail_lines:])
        return f"进程 #{process_id} [{status}]\n{self._truncate(tail) or '（尚无输出）'}"

    def _release_process_log(self, process_id: int) -> None:
        """关闭并移除后台进程的日志句柄（进程自然退出时也要回收，避免句柄泄漏）。"""
        handle = self._process_files.pop(process_id, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def stop_process(self, process_id: int) -> str:
        """终止一个后台进程（危险操作，需确认门放行）。

        Args:
            process_id: start_process 返回的进程编号
        """
        process_id = self._as_int(process_id, -1)
        process = self._processes.get(process_id)
        if process is None:
            return f"错误：不存在后台进程 #{process_id}"
        blocked = self._require_approval(f"stop_process: #{process_id}")
        if blocked:
            return blocked
        if process.poll() is None:
            if os.name == "nt":
                # shell=True 时 terminate 只杀 shell，用 taskkill /T 杀整棵进程树
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self._release_process_log(process_id)
        return f"后台进程 #{process_id} 已终止"

    # ------------------------------------------------------------------
    # 子 agent（任务中途派生独立上下文的帮手）
    # ------------------------------------------------------------------
    def spawn_subagent(self, task: str, instructions: str = "") -> str:
        """派生一个带独立记忆的子 agent 完成子任务，返回其最终回答（不污染主对话上下文）。

        Args:
            task: 交给子 agent 的任务
            instructions: 子 agent 的角色说明，空则沿用默认
        """
        if self.agent_factory is None:
            return (
                "错误：spawn_subagent 需要 CodingWorkspace(agent_factory=...)，"
                "工厂接收 instructions 返回配置好工具的 Agent。"
            )
        agent = self.agent_factory(instructions)
        # N-17：用独立计数器，避免子代理编号与后台进程 pid 混用造成观感冲突
        session_id = f"subagent-{self._next_subagent_id}"
        self._next_subagent_id += 1
        result = agent.run(task, session_id=session_id)
        return result.content

    # ------------------------------------------------------------------
    def as_tools(self) -> List[Tool]:
        """把工作区能力打包成工具列表，直接传给 Agent(tools=...)。

        allowed_tools 白名单生效：只暴露名单内的工具。
        """
        all_tools = {
            "read_file": self.read_file, "write_file": self.write_file,
            "edit_file": self.edit_file, "file_diff": self.file_diff,
            "delete_file": self.delete_file, "list_files": self.list_files,
            "search_code": self.search_code, "run_command": self.run_command,
            "git_status": self.git_status, "git_diff": self.git_diff,
            "git_commit": self.git_commit, "start_process": self.start_process,
            "read_process": self.read_process, "stop_process": self.stop_process,
            "spawn_subagent": self.spawn_subagent,
        }
        names = self.allowed_tools or set(all_tools)
        return [make_tool(all_tools[name]) for name in all_tools if name in names]


CODING_INSTRUCTIONS = (
    "你是程序员。规则：\n"
    "1. 先 list_files/search_code/read_file 了解现状，再动手；\n"
    "2. 改动用 edit_file 做精确替换，新建用 write_file；\n"
    "3. 写完必须 run_command 跑测试验证，失败则修复后重跑；\n"
    "4. 阶段性成果用 git_commit 提交（先 git_status/git_diff 确认改动范围）；\n"
    "5. 长驻服务用 start_process 后台启动，read_process 查看输出；\n"
    "6. 不确定时用 search_code 找线索，不要凭空猜 API。"
)
