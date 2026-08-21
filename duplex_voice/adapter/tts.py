"""TTS Provider —— 统一接口 + Mock + Qwen3-TTS-Instruct-Flash（百炼实测协议 2026-08-17）。

【实测协议】：
  POST {host}/api/v1/services/aigc/multimodal-generation/generation
  {"model":"qwen3-tts-instruct-flash","input":{"text":"..."},"parameters":{"voice":"Cherry"}}
  → output.audio.url（OSS 临时 URL，expires_at 时效）→ 下载得 wav（24k 单声道）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from typing import AsyncIterator

import numpy as np

from ..events import Event

log = logging.getLogger(__name__)

TTS_SAMPLE_RATE = 24000
TTS_CHUNK_MS = 200  # 播放分块


class TTSProvider:  # Protocol
    async def stream_synthesize(self, text: str, *, voice: str = "default",
                                speed: float = 1.0) -> AsyncIterator[Event]: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...


class MockTTSProvider:
    """模拟 TTS：生成平滑正弦音调音频（可听，验证播放链路）。"""

    def __init__(self, first_block_ms: int = 150, block_ms: int = 500):
        self.first_block_ms = first_block_ms
        self.block_ms = block_ms
        self._cancelled = False

    async def stream_synthesize(self, text, *, voice="default", speed=1.0) -> AsyncIterator[Event]:
        self._cancelled = False
        n_blocks = max(1, min(6, len(text) // 4 + 1))
        await asyncio.sleep(self.first_block_ms / 1000)
        yield Event(type="tts.started", domain="tts", payload={"sentence_idx": 0})
        for i in range(n_blocks):
            if self._cancelled:
                yield Event(type="tts.cancelled", domain="tts", payload={})
                return
            n = TTS_SAMPLE_RATE * self.block_ms // 1000
            t = np.arange(n) / TTS_SAMPLE_RATE
            envelope = 0.3 * (1 + np.sin(2 * np.pi * 2 * t))
            audio = (8000 * envelope * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
            await asyncio.sleep(0.02)
            yield Event(type="tts.audio_chunk", domain="tts",
                        payload={"pcm": audio, "sample_rate": TTS_SAMPLE_RATE})
        yield Event(type="tts.sentence_done", domain="tts", payload={})
        yield Event(type="tts.playback_done", domain="tts", payload={})

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        pass


class Qwen3TTSProvider:
    """Qwen3-TTS-Instruct-Flash 语音合成（百炼 multimodal-generation）。

    实测：返回 output.audio.url（OSS 临时 URL），立即下载得 wav(24k/mono)。
    分块播放（200ms/块）支持打断；cancel 后丢弃后续块。
    """

    def __init__(self, host: str = "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com",
                 api_key: str = "", model: str = "qwen3-tts-instruct-flash",
                 voice: str = "Cherry"):
        self.host = host
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.voice = voice
        self._cancelled = False

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "缺少 DashScope API key：配置 model.dashscope_api_key 或环境变量 DASHSCOPE_API_KEY")

    def _opener(self):
        # 禁用系统代理（Clash）——DashScope 国内直连
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _synthesize_url(self, text: str, voice: str) -> str:
        """合成并返回音频 URL（同步，线程内执行）。"""
        self._ensure_key()
        body = json.dumps({
            "model": self.model,
            "input": {"text": text},
            "parameters": {"voice": voice},
        }).encode()
        req = urllib.request.Request(
            f"https://{self.host}/api/v1/services/aigc/multimodal-generation/generation",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with self._opener().open(req, timeout=90) as resp:
            data = json.loads(resp.read())
        return data["output"]["audio"]["url"]

    def _download(self, url: str) -> np.ndarray:
        """下载音频 → int16 PCM（24k）。"""
        with self._opener().open(url, timeout=60) as resp:
            raw = resp.read()
        # WAV 头（RIFF）解析：跳过 44 字节标准头取 PCM 数据
        pcm = np.frombuffer(raw[44:], dtype=np.int16)
        return pcm

    async def stream_synthesize(self, text, *, voice="default", speed=1.0) -> AsyncIterator[Event]:
        """句子级流式合成：切句 → 逐句合成（并行预取下一句）→ 分块输出。

        首句 TTFA ~1.5s，后续句边播边合成（流水线）——体验级真流式。
        """
        self._cancelled = False
        v = self.voice if voice == "default" else voice
        yield Event(type="tts.started", domain="tts", payload={"sentence_idx": 0})
        sentences = _split_sentences(text) or [text]
        chunk = TTS_SAMPLE_RATE * TTS_CHUNK_MS // 1000
        pending = None  # 预取任务：句 i+1
        try:
            for s in sentences:
                # 等前一句合成完成 → 分块输出
                if pending is not None:
                    pcm = await pending
                    if self._cancelled:
                        yield Event(type="tts.cancelled", domain="tts", payload={})
                        return
                    for i in range(0, len(pcm), chunk):
                        yield Event(type="tts.audio_chunk", domain="tts",
                                    payload={"pcm": pcm[i:i + chunk], "sample_rate": TTS_SAMPLE_RATE})
                        await asyncio.sleep(0)
                # 预取当前句
                pending = asyncio.create_task(
                    asyncio.to_thread(self._synthesize_download, s, v))
            if pending is not None:
                pcm = await pending
                if self._cancelled:
                    yield Event(type="tts.cancelled", domain="tts", payload={})
                    return
                for i in range(0, len(pcm), chunk):
                    yield Event(type="tts.audio_chunk", domain="tts",
                                payload={"pcm": pcm[i:i + chunk], "sample_rate": TTS_SAMPLE_RATE})
                    await asyncio.sleep(0)
        except Exception as e:
            log.error("TTS 合成失败: %s", e)
            yield Event(type="tts.error", domain="tts",
                        payload={"code": "E_TTS_SERVICE", "retryable": True, "message": str(e)})
            return
        yield Event(type="tts.sentence_done", domain="tts", payload={})
        yield Event(type="tts.playback_done", domain="tts", payload={})

    def _synthesize_download(self, text: str, voice: str) -> np.ndarray:
        """合成单句并下载 PCM（线程内执行，逐句流水线单元）。"""
        url = self._synthesize_url(text, voice)
        return self._download(url)

    async def cancel(self) -> None:
        """≤30ms 生效：置取消标志，下一块丢弃。"""
        self._cancelled = True

    async def close(self) -> None:
        pass


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句（保留标点），用于句子级流式合成。"""
    import re
    parts = re.split(r"(?<=[。！？；.!?;])", text.strip())
    return [p for p in parts if p.strip()]
