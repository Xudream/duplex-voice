"""Provider 注册表（软件设计 §2.8）。

{capability: {provider_name: factory}} —— 配置选型，快速替换。
"""
from __future__ import annotations

import logging

from ..config import ModelConfig
from .asr import FunASRStreamProvider, MockASRProvider, Qwen3ASRProvider
from .llm import MockFastLLMProvider, MockLLMProvider, OllamaFastProvider, OpenAICompatLLMProvider
from .tts import MockTTSProvider, Qwen3TTSProvider

log = logging.getLogger(__name__)

# 注册表：capability → provider_name → 工厂（惰性实例化，配置注入）
_REGISTRY: dict[str, dict[str, callable]] = {
    "asr": {
        "mock": lambda cfg: MockASRProvider(),
        "qwen3": lambda cfg: Qwen3ASRProvider(
            host=getattr(cfg, "dashscope_host", "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com"),
            api_key=getattr(cfg, "dashscope_api_key", ""),
            model=getattr(cfg, "asr_model", "qwen3-asr-flash")),
        "funasr": lambda cfg: FunASRStreamProvider(
            host=getattr(cfg, "dashscope_host", "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com"),
            api_key=getattr(cfg, "dashscope_api_key", ""),
            model=getattr(cfg, "asr_stream_model", "fun-asr-flash-2026-06-15")),
    },
    "llm_fast": {                        # 快通道：本地小模型（承接语/转述）
        "mock": lambda cfg: MockFastLLMProvider(),
        "ollama": lambda cfg: OllamaFastProvider(
            base_url=cfg.llm_fast_base_url, model=cfg.llm_fast_model),
    },
    "llm_slow": {                        # 慢通道：云端大模型（深度思考）
        "mock": lambda cfg: MockLLMProvider(),
        "qwen3": lambda cfg: OpenAICompatLLMProvider(
            base_url=cfg.llm_slow_base_url, api_key=cfg.llm_slow_api_key,
            model=cfg.llm_slow_model),
    },
    "tts": {
        "mock": lambda cfg: MockTTSProvider(),
        "qwen3": lambda cfg: Qwen3TTSProvider(
            host=getattr(cfg, "dashscope_host", "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com"),
            api_key=getattr(cfg, "dashscope_api_key", ""),
            model=getattr(cfg, "tts_model", "qwen3-tts-instruct-flash"),
            voice=getattr(cfg, "tts_voice", "Cherry")),
    },
}


def get_provider(capability: str, name: str, cfg: ModelConfig):
    """按配置取 Provider 实例。"""
    try:
        return _REGISTRY[capability][name](cfg)
    except KeyError:
        raise ValueError(f"未注册的 {capability} provider: {name}，可用: "
                         f"{list(_REGISTRY.get(capability, {}))}")


def build_providers(cfg: ModelConfig) -> dict[str, object]:
    """按配置装配 Provider（快慢融合双通道）。"""
    return {
        "asr": get_provider("asr", cfg.asr_provider, cfg),
        "llm_fast": get_provider("llm_fast", cfg.llm_fast_provider, cfg),
        "llm_slow": get_provider("llm_slow", cfg.llm_slow_provider, cfg),
        "tts": get_provider("tts", cfg.tts_provider, cfg),
    }
