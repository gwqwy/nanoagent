"""Browser Workspace：浏览器自动化工具（computer use 的浏览器子集）。

基于 Playwright（可选依赖: pip install nanoagent[browser] && playwright install chromium）。
截图工具返回图片 base64，可直接配 Agent 的多模态消息让模型"看"页面。

用法：
    from nanoagent import Agent, BrowserWorkspace

    ws = BrowserWorkspace(headless=True)
    agent = Agent(name="浏览器助手", tools=ws.as_tools(),
                  instructions="你是能操作浏览器的助手。navigate 后用 read_text 了解页面。")
    agent.run("打开 example.com 并总结页面内容")
    ws.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .tools import Tool, make_tool

MAX_TEXT_CHARS = 6000


class BrowserWorkspace:
    """一个受控浏览器会话。所有工具操作同一个页面（懒启动）。"""

    def __init__(self, *, headless: bool = True, allowed_tools: Optional[set] = None):
        self.headless = headless
        self.allowed_tools = allowed_tools
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    def _ensure_page(self) -> Any:
        if self._page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:
                raise ImportError(
                    "浏览器工具需要先安装: pip install nanoagent[browser] && playwright install chromium"
                ) from exc
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()
        return self._page

    # ------------------------------------------------------------------
    def navigate(self, url: str) -> str:
        """打开一个网址（http/https），返回页面标题。

        Args:
            url: 完整 URL
        """
        if not url.startswith(("http://", "https://")):
            return "错误：仅支持 http/https URL"
        page = self._ensure_page()
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        return f"已打开 {url}，标题: {page.title()}"

    def read_text(self) -> str:
        """读取当前页面可见文本（超长截断）。"""
        page = self._ensure_page()
        text = page.inner_text("body")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + f"\n...（共 {len(text)} 字符，已截断）"
        return text

    def click(self, selector: str) -> str:
        """点击页面元素（CSS 选择器或 Playwright text= 语法）。

        Args:
            selector: 如 #submit、a:has-text("下一页")
        """
        page = self._ensure_page()
        page.click(selector, timeout=10000)
        return f"已点击 {selector}"

    def fill(self, selector: str, text: str) -> str:
        """清空并填写输入框。

        Args:
            selector: 输入框选择器，如 input[name="q"]
            text: 要输入的文本
        """
        page = self._ensure_page()
        page.fill(selector, text, timeout=10000)
        return f"已在 {selector} 填入 {len(text)} 字符"

    def screenshot(self, path: str = "screenshot.png") -> str:
        """截取当前页面保存到文件，返回可直接用于 images=[...] 的 data URL。"""
        page = self._ensure_page()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(target), full_page=False)
        from .media import image_part

        data_url = image_part(target)["image_url"]["url"]
        return f"截图已保存: {target}；data URL 前 60 字符: {data_url[:60]}..."

    def get_url(self) -> str:
        """返回当前页面 URL 与标题。"""
        page = self._ensure_page()
        return f"{page.url}  {page.title()}"

    # ------------------------------------------------------------------
    def as_tools(self) -> List[Tool]:
        all_tools = {
            "navigate": self.navigate, "read_text": self.read_text,
            "click": self.click, "fill": self.fill,
            "screenshot": self.screenshot, "get_url": self.get_url,
        }
        names = self.allowed_tools or set(all_tools)
        return [make_tool(all_tools[name]) for name in all_tools if name in names]

    def close(self) -> None:
        """关闭浏览器并释放 Playwright 资源。"""
        if self._browser is not None:
            self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
