"""duplex-voice — 级联全双工语音交互系统第一版实现。

设计依据: voice-agent-doc/级联全双工语音交互系统-软件设计.md (v2.1)
- 架构: Silero VAD v6 + Qwen3-ASR/LLM/TTS 流式级联
- 两个正交接缝: VAD 判定层(rule/semantic) + 模型适配器层(Provider 注册表)
- 状态机: 四态+子状态 + 定时器 + 打断裁决(双钳制/pre-roll)
"""

__version__ = "0.1.0"
