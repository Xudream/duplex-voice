# duplex-voice · 级联全双工语音交互系统（Web 版）

> 浏览器端全双工语音助手：免按键多轮对话、语义 VAD 可打断、快慢融合低感知时延。
> 技术栈：Silero VAD v6（ONNX WASM 前端本地推理）+ fun-asr 流式 ASR + qwen3.5-27b 慢通道 / qwen-turbo·Ollama 快通道 + qwen3-tts（整段/流式双模式）。
> 稳定版本：git tag `voice-web-v1.8.0`（2026-08-24）
> 设计文档：`docs/软件设计.md`（v1.9.0——前端处理/语音理解/全双工/内容生成/系统架构五模块）

## 功能特性

- **全双工**：一次唤醒多轮、免按键、可打断——语义 VAD（VadJudge 可插拔 rule/omni）区分 backchannel 应声/真实打断/环境噪声
- **快慢融合**（FusionStrategy 可插拔）：快通道承接语先出声（~2.2s 响应），慢通道完整回复无缝接播；慢直达模式可选
- **TTS 双模式**：整段（分句并行合成）/ 流式（realtime WS 分句流式——首块 ~0.45s）
- **端点可配**：专属空间 / dashscope.aliyuncs.com 老公共端点（host 下拉框 + ASR 配置全跟随）
- **时延可观测**：全链路分阶段实时显示（有结果即显）——响应=说完→开始播 / TTS=ASR 文本→播
- **配置面板**：⚙️ 5 tab（模型/参数/提示词/前端/VAD）保存即热生效

## 快速开始

### 环境要求

| 项 | 要求 |
|----|------|
| 系统 | Windows 10/11 · macOS · Linux（脚本跨平台） |
| Python | 3.10+ |
| 浏览器 | Chrome / Edge（需 Web Audio + getUserMedia，麦克风授权） |
| 网络 | 可访问阿里云百炼 DashScope API |
| API Key | 百炼 sk- 开头 key（ASR/LLM/TTS 必需） |

### 启动（三选一）

```bash
# macOS / Linux
./start.sh

# Windows（cmd 或 PowerShell）
start.bat

# 通用（依赖检查 → 自动安装 → key 校验 → 启动）
python start.py
```

首次运行自动检查并安装依赖（fastapi / uvicorn / websockets / httpx 等）。启动成功后终端显示 `http://127.0.0.1:8787`，浏览器打开即可。停止：Ctrl+C。

指定语义 VAD 模式（默认 omni——qwen3.5-omni-flash 模型判断；rule = 纯规则，无云端依赖）：

```bash
./start.sh --vad rule
```

跨平台逻辑统一在 `start.py`。

## API Key 配置

**v1.8.0 起：key 读配置文件，不读环境变量**——编辑项目根目录 `config.yaml`：

```json
{
  "model": {
    "dashscope_api_key": "sk-你的key"
  }
}
```

- 文件已入仓（**key 置空模板**）——真实 key 只填本地，不入 git
- 首次用 `start.py` 启动时：config.yaml 为空会自动从 `~/.zshrc` / `~/.bashrc` 提取 `DASHSCOPE_API_KEY` 写入（迁移友好）
- 不配 key 也能看页面，语音对话会报"API key 缺失"

## Python IDE 运行（PyCharm / VS Code）

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

3. 配置 `config.yaml`（API key——见上节；**IDE 方式无需环境变量**）
4. 运行 `web/server.py`（不是 start.py——IDE 直接跑主服务）
5. 浏览器打开 http://127.0.0.1:8787

IDE 方式直接运行 server.py **不会自动装依赖**——依赖缺失时手动 `pip install -r requirements.txt`。

## 项目结构

```
duplex-voice/
├── web/
│   ├── index.html          # 前端单文件（收音/VAD/播放链/时延显示/⚙️ 配置面板）
│   ├── server.py           # FastAPI :8787（ASR/快慢 LLM/TTS/语义 VAD/融合策略/配置热生效）
│   ├── config.json         # 运行期配置（⚙️ 面板读写——已入仓）
│   ├── vendor/             # Silero VAD ONNX + WASM（前端本地推理）
│   └── tts_cache/          # TTS 音频缓存（运行时自动创建）
├── duplex_voice/
│   ├── adapter/            # Provider 适配器：asr（流式/非流式）/ llm（OpenAI 兼容+禁思考）/
│   │                       #   tts（整段）/ tts_stream（realtime WS）/ fusion（融合策略注册表）
│   └── judge/              # VadJudge 语义 VAD 接缝（rule / omni 可插拔）
├── start.py                # 跨平台启动器（依赖检查/自动安装/key 校验）
├── start.sh / start.bat    # 薄封装
├── config.yaml             # API key（空模板入仓——真实 key 只填本地）
├── docs/                   # 软件设计（五模块）/ 语义VAD方案 / 语义VAD-SoulX增训方案
├── tests/                  # pytest（53 passed）
└── requirements.txt        # 纯 ASCII（Windows pip 兼容）
```

## 运行架构速览

```
浏览器页面 (web/index.html)
  ├─ Silero VAD v6（ONNX WASM 本地推理，vendor/）
  ├─ 录音/播放/判停/切段（前端状态机）
  └─ SSE ↔ web/server.py（FastAPI :8787）
        ├─ ASR：fun-asr-flash 流式 WS / qwen3-asr-flash 整段（⚙️ 可配）
        ├─ 语义 VAD：qwen3.5-omni-flash（SEMANTIC_VAD=omni 时，🧠 开关）
        ├─ 快 LLM：本地 Ollama 或云端 qwen-turbo（⚙️ 切换）
        ├─ 慢 LLM：qwen3.5-27b（禁思考）
        └─ TTS：qwen3-tts（整段/流式双模式 🎵）
```

## 配置面板（⚙️）

页面右上角 ⚙️ 打开——5 个 tab 保存即热生效：

| Tab | 内容 |
|-----|------|
| 模型 | Host 端点（下拉框——专属空间/老公共端点，ASR 配置全跟随）、ASR（模型/模式/WS 端点）、快 LLM（Provider/模型）、慢 LLM、TTS（模型/音色）、语义 VAD 模型 |
| 参数 | 时延/播放/并发等运行参数 |
| 提示词 | 快通道/慢通道/直接模式 prompt |
| 前端 | VAD 行为注册表（headset/speaker 场景参数）、前端参数 |
| VAD | 语义 VAD 开关/模式、声学 VAD 阈值 |

## 常见问题

| 现象 | 处理 |
|------|------|
| 页面"连接中…"变红 | server 没起来——看启动终端输出 |
| 一直"聆听中"不识别 | ① 麦克风授权（浏览器地址栏 🔒→麦克风）② 刷新页面（Cmd/Ctrl+Shift+R）③ 检查前端 console 报错 |
| 说话没反应 | 确认 key 有效（`curl -s http://127.0.0.1:8787/api/health` 返回 `"key":true`）；确认 config.yaml 已填 key |
| 换老端点（dashscope.aliyuncs.com）后 ASR 失败 | ⚙️ 模型 tab 切 host 下拉框——ASR 模型自动联动 fun-asr-realtime |
| 想换模型/prompt | 页面 ⚙️ 配置面板（模型/prompt/VAD 参数/场景全部可配，保存即热生效） |
| 快 LLM 想用云端 | ⚙️ 配置 → 模型 → 快 LLM：Provider=dashscope，模型=qwen-turbo |
| 慢回复卡住（老端点） | 确认 server 版本 ≥ v1.8.0（enable_thinking:false 已内置） |
| 端口被占用 | 关掉占用 8787 的进程后重跑脚本 |

## 测试

```bash
python3 -m pytest tests/ -q   # 53 passed
```
