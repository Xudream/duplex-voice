# duplex-voice 运行指南

Web 版全双工语音助手：浏览器打开页面即可对话（前端 Silero VAD 在浏览器本地推理，
ASR / 语义 VAD / LLM / TTS 走云端 API）。

## 环境要求

| 项 | 要求 |
|----|------|
| 系统 | Windows 10/11 · macOS · Linux（脚本跨平台） |
| Python | 3.10+ |
| 浏览器 | Chrome / Edge（需 Web Audio + getUserMedia，麦克风授权） |
| 网络 | 可访问 DashScope（阿里云百炼）API |
| API Key | DASHSCOPE_API_KEY（sk- 开头，云端 ASR/LLM/TTS 必需） |

## 方式一：脚本启动（推荐）

首次运行自动检查并安装依赖（fastapi / uvicorn / websockets / httpx 等），
自动校验 API key（环境变量 → ~/.zshrc / ~/.bashrc 自动提取）。

```bash
# macOS / Linux
./start.sh

# Windows（cmd 或 PowerShell）
start.bat

# 指定语义 VAD 模式（默认 omni——qwen3.5-omni-flash 模型判断；
# rule = 纯规则，无云端依赖）
./start.sh --vad rule
```

启动成功后终端显示 `http://127.0.0.1:8787`，浏览器打开即可。
停止：Ctrl+C。

跨平台逻辑统一在 `start.py`（Windows/macOS/Linux 均可直接 `python start.py` 运行）。

### 首次运行前准备

```bash
# 1. 安装 Python 3.10+（https://www.python.org/downloads/）
# 2. 配置 API key（二选一）：

#   macOS / Linux：
export DASHSCOPE_API_KEY=sk-你的key        # 临时
echo 'export DASHSCOPE_API_KEY=sk-你的key' >> ~/.zshrc   # 永久

#   Windows（cmd）：
set DASHSCOPE_API_KEY=sk-你的key
#   Windows（PowerShell）：
$env:DASHSCOPE_API_KEY="sk-你的key"
```

不配置 key 也可以跑起来看页面，但语音对话会报"API key 未配置"。

## 方式二：Python IDE 运行（PyCharm / VS Code）

1. 用 IDE 打开项目目录（duplex-voice/）
2. 创建虚拟环境并安装依赖：

   ```bash
   python -m venv .venv
   # macOS/Linux：
   source .venv/bin/activate
   # Windows：
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. 配置运行环境变量（IDE 里二选一）：
   - PyCharm：Run → Edit Configurations → 选 `server.py` → Environment variables 填：
     `DASHSCOPE_API_KEY=sk-你的key;SEMANTIC_VAD=omni`（Windows 分号分隔 / macOS 冒号）
   - VS Code：`.vscode/launch.json`：

     ```json
     {
       "version": "0.2.0",
       "configurations": [{
         "name": "voice server",
         "type": "debugpy",
         "request": "launch",
         "program": "${workspaceFolder}/web/server.py",
         "env": { "DASHSCOPE_API_KEY": "sk-你的key", "SEMANTIC_VAD": "omni" }
       }]
     }
     ```

4. 运行 `web/server.py`（不是 start.py——IDE 直接跑主服务）
5. 浏览器打开 http://127.0.0.1:8787

### IDE 方式跳过依赖检查

IDE 方式直接运行 server.py，**不会自动装依赖**——依赖缺失时手动：
`pip install -r requirements.txt`。

## 运行架构速览

```
浏览器页面 (web/index.html)
  ├─ Silero VAD v6（ONNX WASM 本地推理，vendor/）
  ├─ 录音/播放/判停/切段（前端状态机）
  └─ SSE ↔ web/server.py（FastAPI :8787）
        ├─ ASR：fun-asr-flash（DashScope WS 真流式）
        ├─ 语义 VAD：qwen3.5-omni-flash（SEMANTIC_VAD=omni 时）
        ├─ 快 LLM：本地 Ollama 或云端 qwen-turbo（⚙️ 配置面板切换）
        ├─ 慢 LLM：qwen3.5-27b
        └─ TTS：qwen3-tts（整段/流式双模式）
```

## 常见问题

| 现象 | 处理 |
|------|------|
| 页面"连接中…"变红 | server 没起来——看启动终端输出 |
| 一直"聆听中"不识别 | ① 麦克风授权（浏览器地址栏 🔒→麦克风）② 刷新页面（Cmd/Ctrl+Shift+R）③ 检查前端 console 报错 |
| 说话没反应 | 确认 key 有效（`curl -s http://127.0.0.1:8787/api/health` 返回 `"key":true`） |
| 想换模型/prompt | 页面 ⚙️ 配置面板（模型/prompt/VAD 参数/场景全部可配，保存即热生效） |
| 快 LLM 想用云端 | ⚙️ 配置 → 模型 → 快 LLM：Provider=dashscope，模型=qwen-turbo |
| 端口被占用 | 关掉占用 8787 的进程后重跑脚本 |
