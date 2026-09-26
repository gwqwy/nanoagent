"""插件系统：Python 版 mini-Cordis 内核 + 多格式适配器（对齐 DeepSeek Harness）。

DeepSeek Harness（dsh）的核心理念是"一切皆插件"：插件通过 ctx 与宿主交互，
ctx.effect 注册的资源必须返回 disposer，卸载时按 LIFO 回滚；fiber 有
PENDING/ACTIVE/FAILED/DISPOSED 状态机。本模块在 Python 里对齐这套模型：

    PluginContext  对齐 dsh 的 ctx：provide / on / effect + 便捷注册入口
    Plugin         对齐 fiber：name / format / state / provided / disposers
    PluginManager  扫描目录、探测格式、激活/卸载插件、汇总提供物

格式适配层让 nanoagent 能消化多种插件包（"支持所有格式"）：
    nanoagent     plugin.json + plugin.py（原生格式，register(ctx) 或 TOOLS）
    dsh           package.json 带 dsh 声明 / cordis.patch.yml —— 提取
                  其中 Python 能消费的声明层（skills、MCP 配置），
                  TS/JS 代码体无法在 Python 运行时执行，明确记入 skipped
    mcp-only      仅含 mcpServers 配置的包
    agent-skills  通用 SKILL.md / skills/ 目录（Claude Code 等共用格式）
新格式用 register_adapter() 注册即可接入，探测顺序即注册顺序。

用法：
    manager = PluginManager()
    manager.scan(".nanoagent/plugins")
    manager.activate_all()
    agent = Agent(..., tools=manager.collect_tools())
    agent.enable_skills(manager.skills)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .skills import SkillRegistry
from .tools import Tool

# 对齐 dsh 的 fiber 状态机
PENDING = "PENDING"
ACTIVE = "ACTIVE"
FAILED = "FAILED"
DISPOSED = "DISPOSED"

SKIPPED_TS = "ts-code"
SKILL_MD = "SKILL.md"


class UnknownFormat(Exception):
    """目录内容不属于任何已知插件格式。"""

    def __init__(self, path: Path, found: List[str]):
        listing = ", ".join(found) or "（空目录）"
        super().__init__(
            f"无法识别插件格式: {path}（找到: {listing}）。"
            "可用 register_adapter 注册自定义格式适配器。"
        )


# ----------------------------------------------------------------------
class EventBus:
    """极简事件总线：emit 时逐个调用 handler，单个 handler 出错不影响其余。"""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}

    def on(self, event: str, handler: Callable) -> Callable:
        """订阅事件，返回取消订阅的 disposer（对齐 ctx.on 的返回约定）。"""
        self._handlers.setdefault(event, []).append(handler)

        def dispose() -> None:
            handlers = self._handlers.get(event, [])
            if handler in handlers:
                handlers.remove(handler)

        return dispose

    def emit(self, event: str, **payload: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(**payload)
            except Exception:  # noqa: BLE001 —— 观察者出错不阻断宿主流程
                pass


# ----------------------------------------------------------------------
class _ToolsFacade:
    """ctx.tools：对齐 dsh 的 ctx.tools.register。"""

    def __init__(self, ctx: "PluginContext") -> None:
        self._ctx = ctx

    def register(self, source: Callable | Tool) -> Tool:
        if isinstance(source, Tool):
            tool = source
        else:
            from .tools import make_tool

            tool = make_tool(source)
        self._ctx.tools_list.append(tool)
        return tool


class _SkillsFacade:
    """ctx.skills：插件声明自己携带的技能目录。"""

    def __init__(self, ctx: "PluginContext") -> None:
        self._ctx = ctx

    def add_dir(self, path: str | Path) -> None:
        path = str(path)
        if path not in self._ctx.skill_dirs:
            self._ctx.skill_dirs.append(path)


class _McpFacade:
    """ctx.mcp：插件声明自己需要连接的 MCP 服务器（mcpServers 通用格式）。"""

    def __init__(self, ctx: "PluginContext") -> None:
        self._ctx = ctx

    def add_server(self, name: str, config: Dict[str, Any]) -> None:
        self._ctx.mcp_servers[name] = dict(config)


class PluginContext:
    """对齐 dsh 的 ctx：插件与宿主交互的唯一边界。

    effect(fn) 立即执行 fn 并把返回的 disposer 记入账本；deactivate() 按
    LIFO 逐个执行 disposer，单个失败吞掉继续（幂等，对齐 dsh 的回滚语义）。
    """

    def __init__(self, plugin_name: str, bus: Optional[EventBus] = None) -> None:
        self.plugin_name = plugin_name
        self.bus = bus or EventBus()
        self.services: Dict[str, Any] = {}
        self.tools_list: List[Tool] = []
        self.skill_dirs: List[str] = []
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.skipped: List[str] = []
        self._disposers: List[Callable] = []
        # 便捷入口，命名对齐 dsh 的 ctx.tools / ctx.skills / ctx.mcp
        self.tools = _ToolsFacade(self)
        self.skills = _SkillsFacade(self)
        self.mcp = _McpFacade(self)

    # -- 核心三原语 ------------------------------------------------------
    def provide(self, name: str, service: Any) -> Callable:
        """提供一个服务，返回撤销注册的 disposer。"""
        self.services[name] = service

        def dispose() -> None:
            self.services.pop(name, None)

        self._disposers.append(dispose)
        return dispose

    def on(self, event: str, handler: Callable) -> Callable:
        """订阅宿主事件（activate/deactivate/error），返回 disposer。"""
        return self.bus.on(event, handler)

    def effect(self, fn: Callable[[], Optional[Callable]]) -> None:
        """执行 fn 并把其返回的 disposer 记入账本（dsh 的 ctx.effect）。"""
        disposer = fn()
        if disposer is not None:
            self._disposers.append(disposer)

    def _add_disposer(self, disposer: Callable) -> None:
        self._disposers.append(disposer)

    # -- 生命周期 --------------------------------------------------------
    def deactivate(self) -> None:
        """LIFO 回滚全部 disposer；幂等，可安全重复调用。"""
        while self._disposers:
            disposer = self._disposers.pop()
            try:
                disposer()
            except Exception:  # noqa: BLE001 —— 回滚失败不阻断其余清理
                pass


# ----------------------------------------------------------------------
@dataclass
class Plugin:
    """一个被发现的插件记录（对齐 dsh 的 fiber）。"""

    name: str
    path: str
    format: str
    state: str = PENDING
    manifest: Dict[str, Any] = field(default_factory=dict)
    ctx: Optional[PluginContext] = None
    provided: Dict[str, List[str]] = field(default_factory=dict)  # tools/skills/mcp/services
    skipped: List[str] = field(default_factory=list)  # ["ts-code: 2 个 .ts 文件已跳过", ...]
    error: str = ""

    def summary(self) -> dict:
        return {
            "name": self.name,
            "format": self.format,
            "state": self.state,
            "provided": self.provided,
            "skipped": self.skipped,
            "error": self.error,
        }


# ----------------------------------------------------------------------
# 格式适配器：detector(path) -> bool；loader(plugin, ctx) -> None
Adapter = Callable  # (detector, loader) 二元组，用 tuple 存放

_ADAPTERS: Dict[str, tuple] = {}


def register_adapter(name: str, detector: Callable, loader: Callable) -> None:
    """注册自定义格式适配器（探测顺序 = 注册顺序，内置的在前）。"""
    _ADAPTERS[name] = (detector, loader)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _looks_like_dsh(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("dsh"), (dict, str)):
        return True
    keywords = data.get("keywords")
    return isinstance(keywords, list) and "dsh-plugin" in keywords


# ---- 内置适配器 ------------------------------------------------------
def _detect_nanoagent(path: Path) -> bool:
    return (path / "plugin.py").is_file() or (path / "plugin.json").is_file()


def _load_nanoagent(plugin: Plugin, ctx: PluginContext) -> None:
    root = Path(plugin.path)
    plugin.manifest = _read_json(root / "plugin.json") or {"name": plugin.name}
    py_file = root / "plugin.py"
    if py_file.is_file():
        module_name = f"nanoagent_plugin_{plugin.name}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        register_fn = getattr(module, "register", None)
        if callable(register_fn):
            register_fn(ctx)
        for item in getattr(module, "TOOLS", []) or []:
            ctx.tools.register(item)
        if register_fn is None and not ctx.tools_list:
            ctx.skipped.append("no-entry: plugin.py 中既无 register(ctx) 也无 TOOLS")


def _detect_dsh(path: Path) -> bool:
    if (path / "cordis.patch.yml").is_file():
        return True
    data = _read_json(path / "package.json")
    return _looks_like_dsh(data)


def _load_dsh(plugin: Plugin, ctx: PluginContext) -> None:
    """dsh 插件适配：提取 Python 能消费的声明层，TS 代码体记入 skipped。

    dsh 插件的 tools/skills 是 Cordis 服务（进程内 TS 代码），无法在 Python
    运行时执行；这里消费的是包内的静态声明：skills 目录（SKILL.md）、
    .mcp.json / mcp.json 的 mcpServers、package.json 元信息与 patch 配置。
    """
    root = Path(plugin.path)
    plugin.manifest = _read_json(root / "package.json") or {"name": plugin.name}
    patch = root / "cordis.patch.yml"
    if patch.is_file():
        plugin.manifest["cordis_patch"] = patch.read_text(encoding="utf-8")
        ctx.skipped.append("patch: cordis.patch.yml 仅存档，不参与 Python 挂载")

    # skills：根目录 skills/ 下每个子目录一个 SKILL.md，或根目录直接一个
    if (root / "skills").is_dir() or (root / SKILL_MD).is_file():
        ctx.skills.add_dir(root)
    else:
        for child in sorted(root.glob("*/skills")):
            ctx.skills.add_dir(child)

    # mcp：读取包内 .mcp.json / mcp.json 的 mcpServers 声明
    for mcp_file in (".mcp.json", "mcp.json"):
        data = _read_json(root / mcp_file)
        for name, cfg in (data.get("mcpServers") or {}).items():
            ctx.mcp.add_server(name, cfg)

    # 代码体：*.ts / *.js（package.json 等清单除外）无法执行
    code_files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in (".ts", ".js") and p.name not in ("package.json",)
    ]
    if code_files:
        ctx.skipped.append(
            f"{SKIPPED_TS}: {len(code_files)} 个 TS/JS 代码文件无法在 Python 运行时执行，已跳过"
        )


def _detect_mcp_only(path: Path) -> bool:
    for name in (".mcp.json", "mcp.json"):
        data = _read_json(path / name)
        if isinstance(data.get("mcpServers"), dict) and data["mcpServers"]:
            return True
    return False


def _load_mcp_only(plugin: Plugin, ctx: PluginContext) -> None:
    root = Path(plugin.path)
    plugin.manifest = {"name": plugin.name}
    for mcp_file in (".mcp.json", "mcp.json"):
        data = _read_json(root / mcp_file)
        for name, cfg in (data.get("mcpServers") or {}).items():
            ctx.mcp.add_server(name, cfg)


def _detect_agent_skills(path: Path) -> bool:
    if (path / SKILL_MD).is_file() or (path / "skills").is_dir():
        return True
    return any((child / SKILL_MD).is_file() for child in path.iterdir() if child.is_dir())


def _load_agent_skills(plugin: Plugin, ctx: PluginContext) -> None:
    root = Path(plugin.path)
    plugin.manifest = {"name": plugin.name}
    if (root / SKILL_MD).is_file():
        ctx.skills.add_dir(root)
    for child in sorted(root.glob("*")):
        if (child / SKILL_MD).is_file() or (child / "skills" / SKILL_MD).is_file():
            ctx.skills.add_dir(child)
        elif child.is_dir() and child.name == "skills":
            ctx.skills.add_dir(child)


register_adapter("nanoagent", _detect_nanoagent, _load_nanoagent)
register_adapter("dsh", _detect_dsh, _load_dsh)
register_adapter("mcp-only", _detect_mcp_only, _load_mcp_only)
register_adapter("agent-skills", _detect_agent_skills, _load_agent_skills)


# ----------------------------------------------------------------------
class PluginManager:
    """扫描插件目录、按格式激活、汇总提供物（对齐 dsh plugin 的宿主侧）。"""

    def __init__(self, skills: Optional[SkillRegistry] = None) -> None:
        self.skills = skills or SkillRegistry()
        self.plugins: Dict[str, Plugin] = {}
        self.bus = EventBus()

    # -- 发现 ------------------------------------------------------------
    def detect_format(self, path: Path) -> str:
        for name, (detector, _loader) in _ADAPTERS.items():
            if detector(path):
                return name
        found = sorted(p.name for p in path.iterdir()) if path.is_dir() else []
        raise UnknownFormat(path, found)

    def scan(self, plugins_dir: str | Path) -> List[Plugin]:
        """发现目录下的插件（不激活）。同名后到者加序号后缀。"""
        root = Path(plugins_dir)
        discovered: List[Plugin] = []
        if not root.is_dir():
            return discovered
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            final_name = name
            counter = 2
            while final_name in self.plugins:
                final_name = f"{name}-{counter}"
                counter += 1
            try:
                fmt = self.detect_format(path)
            except UnknownFormat as exc:
                self.plugins[final_name] = Plugin(
                    name=final_name, path=str(path), format="unknown",
                    state=FAILED, error=str(exc),
                )
                self.bus.emit("error", name=final_name, error=str(exc))
                discovered.append(self.plugins[final_name])
                continue
            plugin = Plugin(name=final_name, path=str(path), format=fmt)
            self.plugins[final_name] = plugin
            discovered.append(plugin)
        return discovered

    # -- 激活 / 卸载 ------------------------------------------------------
    def activate(self, name: str) -> Plugin:
        """激活一个插件：建 ctx、按格式装载、加载其技能目录。失败不外抛。"""
        plugin = self.plugins.get(name)
        if plugin is None:
            raise KeyError(f"插件 '{name}' 不存在，先 scan() 发现插件")
        if plugin.state == ACTIVE:
            return plugin

        ctx = PluginContext(plugin.name, bus=self.bus)
        _detector, loader = _ADAPTERS.get(plugin.format, (None, None))
        try:
            if loader is None:
                raise UnknownFormat(Path(plugin.path), [])
            loader(plugin, ctx)
            before = set(self.skills.names())
            for skill_dir in ctx.skill_dirs:
                self.skills.add_dir(skill_dir)
            loaded_skills = sorted(set(self.skills.names()) - before)
        except Exception as exc:  # noqa: BLE001 —— 单个插件失败不拖垮整体
            ctx.deactivate()
            plugin.state = FAILED
            plugin.ctx = None
            plugin.error = f"{type(exc).__name__}: {exc}"
            self.bus.emit("error", name=plugin.name, error=plugin.error)
            return plugin

        plugin.ctx = ctx
        plugin.state = ACTIVE
        plugin.provided = {
            "tools": [t.name for t in ctx.tools_list],
            "skills": loaded_skills,
            "mcp": sorted(ctx.mcp_servers),
            "services": sorted(ctx.services),
        }
        plugin.skipped = list(ctx.skipped)
        self.bus.emit("activate", name=plugin.name, format=plugin.format)
        return plugin

    def activate_all(self) -> List[Plugin]:
        return [self.activate(name) for name in list(self.plugins)]

    def deactivate(self, name: str) -> Plugin:
        plugin = self.plugins.get(name)
        if plugin is None:
            raise KeyError(f"插件 '{name}' 不存在")
        if plugin.ctx is not None:
            plugin.ctx.deactivate()
            plugin.ctx = None
        plugin.state = DISPOSED
        plugin.provided = {}
        self.bus.emit("deactivate", name=plugin.name)
        return plugin

    def deactivate_all(self) -> None:
        for name in list(self.plugins):
            if self.plugins[name].state == ACTIVE:
                self.deactivate(name)

    # -- 汇总 ------------------------------------------------------------
    def collect_tools(self) -> List[Tool]:
        """收集所有 ACTIVE 插件提供的工具（顺序 = 激活顺序）。"""
        tools: List[Tool] = []
        for plugin in self.plugins.values():
            if plugin.state == ACTIVE and plugin.ctx is not None:
                tools.extend(plugin.ctx.tools_list)
        return tools

    def collect_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """收集所有 ACTIVE 插件声明的 MCP 服务器配置。"""
        servers: Dict[str, Dict[str, Any]] = {}
        for plugin in self.plugins.values():
            if plugin.state == ACTIVE and plugin.ctx is not None:
                servers.update(plugin.ctx.mcp_servers)
        return servers

    def list(self) -> List[dict]:
        return [plugin.summary() for plugin in self.plugins.values()]
