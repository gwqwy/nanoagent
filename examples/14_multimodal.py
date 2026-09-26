"""示例 14：多模态 —— 给 agent 发图片（对齐 OpenAI/DeepSeek Vision 消息格式）。

要点：
    - images 参数接受 http(s) URL、本地文件路径、bytes、已构造的 part dict
    - 本地文件/bytes 自动编码为 base64 data URL（按文件魔数识别格式）
    - 传入图片后 user 消息升级为 content 数组：[{"type":"text"},{"type":"image_url"}]
    - 记忆只持久化文本，图片是单次请求的临时输入
    - 需要模型本身支持视觉（DeepSeek 的 deepseek-flash 已原生支持；
      纯文本模型会拒绝 image_url 消息）
"""

from pathlib import Path

from nanoagent import Agent, make_png
from nanoagent.config import settings


def main():
    cfg = settings()
    if not cfg.get("api_key") or cfg["api_key"].startswith("sk-xxxx"):
        raise SystemExit("请先在 .env 配置 NANOAGENT_API_KEY（视觉示例需要真实模型服务）")

    agent = Agent(name="视觉助手", instructions="你是能看图的助手，用中文简洁回答。")

    # 1) 本地文件：零依赖生成一张纯色 PNG，让模型说出颜色
    image_path = make_png(64, 64, (220, 30, 30), Path(".nanoagent/demo_red.png"))
    result = agent.run(
        "这张图片主要是什么颜色？只回答颜色名。",
        images=[image_path],
    )
    print(f"本地图片: {result.content}（共 {result.iterations} 轮）")

    # 2) http(s) URL 直传：服务商负责下载（取消注释即可实测）
    # result = agent.run(
    #     "用一句话描述这张图片的内容。",
    #     images=["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"],
    # )
    # print(f"URL 图片: {result.content}")

    # 3) 多图对比 + detail 参数（low 更快更省）
    blue = make_png(64, 64, (30, 60, 220), Path(".nanoagent/demo_blue.png"))
    result = agent.run(
        "这两张图片的颜色分别是什么？",
        images=[image_path, blue],
        image_detail="low",
    )
    print(f"多图对比: {result.content}")


if __name__ == "__main__":
    main()
