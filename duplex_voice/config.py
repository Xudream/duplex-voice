"""配置加载（软件设计 §3.3 / §7.1）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    channels: int = 1
    device: str | None = None          # sounddevice 设备名/索引
    aec_enabled: bool = False          # 第一版: 简化 AEC（播放参考帧标记，真 AEC P1）
    ns_level: int = 2
    agc_target_db: float = -20.0


@dataclass(slots=True)
class VadConfig:
    judge: str = "rule"                # "rule" 半双工（缺省）| "semantic" 全双工（插上）
    threshold_segment: float = 0.5
    threshold_interrupt: float = 0.7   # 播放期抬高
    min_silence_ms: int = 500          # 判停静音（可调档 250-1250）
    min_speech_ms: int = 250
    speech_pad_ms: int = 30
    pre_roll_ms: int = 1200            # pre-roll 回滚时长（采集侧 AEC 后）
    window_ms: int = 300               # 窗口多数判定
    window_ratio: float = 0.8
    grace_ms: int = 500                # 播放起始 grace
    noise_floor_ms: int = 450          # 噪声底校准窗
    interrupt_ceiling_db: float = 18.0 # 上限: 安静底 + dB


@dataclass(slots=True)
class FsmConfig:
    wake_window_ms: int = 8000         # 免唤醒窗口
    final_fallback_ms: int = 2000      # ASR final 兜底
    listen_window_ms: int = 1200       # 播放后倾听窗口
    llm_first_token_ms: int = 5000
    llm_total_ms: int = 30000
    slow_wait_ms: int = 12000         # 承接语播完后等慢通道回复（27b 首 token ~5s）
    tts_first_block_ms: int = 3000
    session_total_s: int = 1800        # 30min
    history_max_rounds: int = 20
    history_max_tokens: int = 8000


@dataclass(slots=True)
class ModelConfig:
    asr_provider: str = "funasr"        # funasr(WS真流式) | qwen3(HTTP一次性) | mock
    llm_provider: str = "mock"         # qwen3 | openai | mock
    tts_provider: str = "mock"         # qwen3(dashscope) | mock
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    # DashScope（Qwen3-ASR-Flash / Qwen3-TTS-Flash）—— 实测协议（qwen-asr-api-reference 文档）
    dashscope_api_key: str = ""        # 或环境变量 DASHSCOPE_API_KEY
    dashscope_host: str = "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com"  # 专属业务空间
    asr_model: str = "qwen3-asr-flash"             # compatible-mode chat + input_audio
    tts_model: str = "qwen3-tts-instruct-flash"    # multimodal-generation → output.audio.url
    tts_voice: str = "Cherry"
    # LLM 快慢融合双通道（软件设计 v2.2 §2.5）
    llm_fast_provider: str = "ollama"    # ollama(本地 qwen3.5:4b-mlx) | mock
    llm_fast_base_url: str = "http://127.0.0.1:11434"   # Ollama 原生 /api/chat
    llm_fast_api_key: str = ""
    llm_fast_model: str = "qwen3.5:4b-mlx"
    llm_slow_provider: str = "mock"    # qwen3(云端 qwen3.8-27b) | openai | mock
    llm_slow_base_url: str = "https://llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    llm_slow_api_key: str = ""
    llm_slow_model: str = "qwen3.6-27b"   # 空间无 qwen3.8-27b；27b 档实测可用（qwen3.8-max 为 2.4t 旗舰）
    # 语义 VAD 推理服务（P1；插上 = 全双工）
    semantic_vad_url: str = "ws://127.0.0.1:2335/v1/vad"


@dataclass(slots=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    session_jsonl: str = "sessions.jsonl"
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for section, sub in raw.items():
            if not isinstance(sub, dict) or not hasattr(cfg, section):
                continue
            obj = getattr(cfg, section)
            for k, v in sub.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
        return cfg
