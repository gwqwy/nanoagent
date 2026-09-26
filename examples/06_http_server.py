"""示例 06：把 agent 发布为 HTTP 服务。

启动后即可用任意 HTTP 客户端对话：
    curl -X POST http://127.0.0.1:8000/chat ^
         -H "Content-Type: application/json" ^
         -d "{\"session_id\": \"u1\", \"message\": \"你好\"}"

交互式接口文档: http://127.0.0.1:8000/docs
"""

import uvicorn

from nanoagent.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
