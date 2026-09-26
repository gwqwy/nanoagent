"""多模态消息构造：把图片源转成 OpenAI 兼容的 content parts。

对齐 DeepSeek Vision / GPT-4o 等视觉模型的消息格式（user 消息的 content 为数组）：

    {"type": "text", "text": "描述这张图"}
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,... 或 https://..."}}

注意：多数服务商只允许图片出现在 user 消息里（DeepSeek 对 system/assistant 中的
图片直接返回 400）；nanoagent 的记忆只持久化文本，图片是单次请求的临时输入。

用法：
    agent.run("描述这张图", images=["./diagram.png"])
    agent.run("对比两图", images=["https://example.com/a.jpg", Path("b.png")])
"""

from __future__ import annotations

import base64
import copy
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 按文件魔数识别 MIME（服务商按实际内容检测，扩展名只是兜底）
_MAGIC_SIGNATURES = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]
_SUFFIX_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}

ImageSource = Union[str, Path, bytes, Dict[str, Any]]


def _sniff_mime(data: bytes) -> str:
    for magic, mime in _MAGIC_SIGNATURES:
        if data.startswith(magic):
            return mime
    return "application/octet-stream"


def image_part(source: ImageSource, detail: Optional[str] = None) -> Dict[str, Any]:
    """把单个图片源转成 {"type": "image_url", ...} content part。

    source 支持：
        str   http(s) URL        原样直传，由服务商下载
        str   data: URL          原样直传
        str/Path 本地文件        读取并编码成 base64 data URL
        bytes                    编码成 base64 data URL（MIME 自动嗅探）
        dict                     已是合法 content part，原样透传
    detail: "low"/"high"/"auto"，对齐 OpenAI 视觉参数，透传给服务商。
    """
    if isinstance(source, dict):
        # 深拷贝：避免下方写 detail 时污染调用方传入的嵌套 dict
        part = copy.deepcopy(source)
    elif isinstance(source, (str, Path)):
        text = str(source)
        if text.startswith(("http://", "https://", "data:")):
            url = text
        else:
            path = Path(text)
            data = path.read_bytes()
            mime = _SUFFIX_MIME.get(path.suffix.lower()) or _sniff_mime(data)
            url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        part = {"type": "image_url", "image_url": {"url": url}}
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        mime = _sniff_mime(data)
        if mime == "application/octet-stream":
            raise ValueError("无法识别图片格式（支持 JPEG/PNG/GIF/WebP），请传 URL 或带扩展名的文件")
        url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        part = {"type": "image_url", "image_url": {"url": url}}
    else:
        raise TypeError(f"不支持的图片源类型: {type(source)!r}")

    if detail is not None:
        inner = part.get("image_url")
        if not isinstance(inner, dict):
            raise ValueError(
                f"图片 content part 缺少 image_url 字段，无法设置 detail: {part.get('type', part)!r}"
            )
        inner["detail"] = detail
    return part


def build_user_content(
    text: str,
    images: Optional[List[ImageSource]] = None,
    detail: Optional[str] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """构造 user 消息的 content：无图片时保持纯字符串（兼容一切模型），有图时升级为数组。"""
    if not images:
        return text
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.extend(image_part(source, detail=detail) for source in images)
    return parts


# ----------------------------------------------------------------------
def make_png(width: int, height: int, rgb: tuple, path: Union[str, Path]) -> Path:
    """零依赖生成一张纯色 PNG（测试/演示用）。"""
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(tag: bytes, payload: bytes) -> bytes:
        data = tag + payload
        return struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit RGB
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    path = Path(path)
    path.write_bytes(png)
    return path
