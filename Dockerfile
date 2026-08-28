# duplex-voice 服务端 Docker 部署（App 客户端 + 服务端模式）
#
# 构建:  docker build -t duplex-voice-server .
# 运行:  docker run -d --name duplex-voice \
#          -p 8787:8787 \
#          -v $(pwd)/config.yaml:/app/config.yaml \
#          -v duplex-tts-cache:/app/web/tts_cache \
#          duplex-voice-server
# （公网部署加一层 nginx HTTPS 反代 + 打开 config.server.auth.required）

FROM python:3.11-slim

WORKDIR /app

# 依赖（纯 Python，无系统级编译）
RUN pip install --no-cache-dir fastapi uvicorn websockets httpx httpx_sse numpy

# 代码（不含 config.yaml——key 通过挂载注入，不入镜像）
COPY web/ /app/web/
COPY duplex_voice/ /app/duplex_voice/
COPY start.py /app/start.py

EXPOSE 8787

CMD ["python", "start.py"]
