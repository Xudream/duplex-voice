"""音频播放（T3 线程）。

可打断（≤30ms）：stop() → 播放线程 flush 输出缓冲；
命令经 asyncio.Queue（FSM → player 单向命令）。
"""
from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioPlayer:
    """流式播放：TTS 音频块 → 输出缓冲 → 声卡。支持 stop/flush。"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._cmd: asyncio.Queue = asyncio.Queue()
        self._stream: sd.OutputStream | None = None
        self._buffer: list[np.ndarray] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", blocksize=0,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """打断：立即停止播放并 flush（≤30ms）。"""
        self._stop_evt.set()
        self._buffer.clear()

    def play(self, audio: np.ndarray) -> None:
        """播放音频块（int16 mono，sample_rate 对齐）。"""
        if not self._stream or self._stop_evt.is_set():
            return
        self._buffer.append(audio)
        self._drain()

    def _drain(self) -> None:
        while self._buffer:
            chunk = self._buffer.pop(0)
            try:
                self._stream.write(chunk)
            except Exception as e:  # 声卡忙等：丢块不崩溃
                log.warning("player write error: %s", e)
                break

    def _run(self) -> None:
        # 预留：命令消费（stop/flush）—— 第一版 stop 用事件标志，够用
        while True:
            try:
                cmd = self._cmd.get_nowait()
                if cmd == "flush":
                    self._stop_evt.clear()
                    self._buffer.clear()
            except asyncio.QueueEmpty:
                pass
            self._stop_evt.wait(0.05)
            if self._stop_evt.is_set():
                self._drain()
                self._stop_evt.clear()

    def close(self) -> None:
        self.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
