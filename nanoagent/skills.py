"""Agent Skills：SKILL.md 渐进式披露。

SKILL.md 是当前业界通用（DeepSeek Harness / Claude Code 等共用）的技能格式：
YAML frontmatter 里写 name / description，正文是给模型看的操作说明。
渐进式披露分两层，避免把技能全文塞爆上下文：
    第一层  list_summary()   只注入名字与描述，让模型知道"有哪些技能"
    第二层  use_skill 工具    模型按需加载某个技能的完整正文

用法：
    registry = SkillRegistry()
    registry.add_dir("./skills")
    agent.enable_skills(registry)   # 自动注册 use_skill 工具 + 注入技能索引
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .tools import tool as tool_decorator

SKILL_FILE = "SKILL.md"

_FRONTMATTER_LINE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
_BLOCK_SCALAR = re.compile(r"^([>|][+-]?)\s*$")


def _fold_lines(chunk: List[str], folded: bool) -> str:
    """把块标量的行拼成字符串：folded(>) 用空格折叠、空行变段落换行；literal(|) 保留换行。"""
    if not folded:
        return "\n".join(chunk)
    paras: List[List[str]] = [[]]
    for part in chunk:
        if part:
            paras[-1].append(part)
        elif paras[-1]:
            paras.append([])
    return "\n".join(" ".join(p) for p in paras if p)


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """解析 markdown 头部的 `--- ... ---` frontmatter。

    支持 `key: value` 单行格式与 YAML 块标量（`>` 折叠 / `|` 保留，含 +/- chomping）；
    嵌套结构（如 metadata:）整体跳过不展开。值两侧引号会被剥掉，不引入 YAML 依赖；
    没有 frontmatter 或格式不完整时返回 ({}, 全文)。
    """
    text = text.strip()
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() != "---":
            continue
        meta: Dict[str, str] = {}
        body = lines[1:i]
        k = 0
        while k < len(body):
            line = body[k]
            k += 1
            match = _FRONTMATTER_LINE.match(line.strip())
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            indent = len(line) - len(line.lstrip())
            block = _BLOCK_SCALAR.match(value)
            if block or value == "":
                # 块标量：收集缩进更深的行；空值（嵌套结构）：跳过缩进子行
                chunk: List[str] = []
                while k < len(body):
                    nxt = body[k]
                    if nxt.strip() == "":
                        if block:
                            chunk.append("")
                        k += 1
                        continue
                    if len(nxt) - len(nxt.lstrip()) <= indent:
                        break
                    if block:
                        chunk.append(nxt.strip())
                    k += 1
                while chunk and chunk[-1] == "":
                    chunk.pop()
                if block:
                    meta[key] = _fold_lines(chunk, value.startswith(">"))
                # 空值且非块标量 → 嵌套 dict，忽略
            else:
                meta[key] = value.strip("'\"").strip()
        return meta, "\n".join(lines[i + 1 :]).strip()
    return {}, text  # 只有开头 --- 没有收尾，按无 frontmatter 处理


@dataclass
class Skill:
    """一个已加载的技能。body 是 frontmatter 之后的正文。"""

    name: str
    description: str
    body: str
    path: str = ""

    def summary(self) -> str:
        return f"- {self.name}: {self.description}"


class SkillRegistry:
    """技能注册表：从目录收集 SKILL.md，支持索引披露与按名加载。"""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._use_skill_tool: Optional[Callable] = None  # 缓存，保证幂等

    # ------------------------------------------------------------------
    def add(self, skill: Skill) -> Skill:
        """注册一个技能，同名覆盖（后装的插件可定制版本）。"""
        self._skills[skill.name] = skill
        return skill

    def add_file(self, path: str | Path) -> Skill:
        """从单个 SKILL.md 加载。名字取 frontmatter 的 name，缺失时用目录/文件名。"""
        path = Path(path)
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        name = meta.get("name") or (path.parent.name if path.name == SKILL_FILE else path.stem)
        return self.add(
            Skill(
                name=name,
                description=meta.get("description", ""),
                body=body,
                path=str(path),
            )
        )

    def add_dir(self, dir_path: str | Path) -> List[Skill]:
        """递归收集目录下所有 SKILL.md；目录不存在时静默返回空列表。"""
        root = Path(dir_path)
        loaded: List[Skill] = []
        if root.is_dir():
            for path in sorted(root.rglob(SKILL_FILE)):
                loaded.append(self.add_file(path))
        return loaded

    def remove(self, name: str) -> bool:
        """按名移除一个技能（用户禁用技能时用）；不存在返回 False。"""
        return self._skills.pop(name, None) is not None

    # ------------------------------------------------------------------
    def names(self) -> List[str]:
        return list(self._skills)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def list_summary(self) -> str:
        """渐进式披露第一层：全部技能的名字+描述（注入 system prompt 用）。"""
        return "\n".join(skill.summary() for skill in self._skills.values())

    def load(self, name: str) -> Skill:
        """渐进式披露第二层：按名取完整技能；不存在时列出可用项方便模型纠正。"""
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(self._skills) or "无"
            raise KeyError(f"未找到技能 '{name}'，可用技能: {available}")
        return skill

    # ------------------------------------------------------------------
    def use_skill_tool(self) -> Callable:
        """生成（并缓存）use_skill 工具，供 agent 注册调用。"""
        if self._use_skill_tool is None:
            registry = self

            @tool_decorator(
                name="use_skill",
                description="按名称加载一个技能的完整说明。调用前先查看系统提示里的技能索引。",
            )
            def use_skill(name: str) -> str:
                """加载指定技能的完整正文。

                Args:
                    name: 技能名称，见系统提示中的技能索引。
                """
                try:
                    return registry.load(name).body
                except KeyError as exc:
                    return str(exc)

            self._use_skill_tool = use_skill
        return self._use_skill_tool
