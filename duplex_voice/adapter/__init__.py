"""模型适配器层（软件设计 §2.8，横切）。

统一 Provider 接口 + 注册表 —— 屏蔽协议/鉴权/编解码/重试/端点，
换模型 = 换配置（asr.provider / llm.provider / tts.provider）。
"""
from .registry import build_providers, get_provider
from .asr import ASRProvider, MockASRProvider, Qwen3ASRProvider
from .llm import LLMProvider, MockLLMProvider, OpenAICompatLLMProvider
from .tts import TTSProvider, MockTTSProvider, Qwen3TTSProvider

__all__ = [
    "build_providers", "get_provider",
    "ASRProvider", "MockASRProvider", "Qwen3ASRProvider",
    "LLMProvider", "MockLLMProvider", "OpenAICompatLLMProvider",
    "TTSProvider", "MockTTSProvider", "Qwen3TTSProvider",
]
