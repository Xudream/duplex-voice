"""音频采集（T1 线程，RT 优先级意图）。

sounddevice 回调（音频线程）→ 有界帧队列 → FSM 事件循环消费。
第一版：AEC 简化——播放参考帧标记（aec_ref 源），真 AEC 引擎 P1。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioFrame:
    """全系统音频最小单元（30ms）。"""
    pcm: np.ndarray          # 480 samples int16 mono 16k
    ts_ms: int
    source: str              # "mic" | "aec_ref"
    seq: int

    @property
    def rms_db(self) -> float:
        if self.pcm.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(self.pcm.astype(np.float64) ** 2)))
        return 20.0 * np.log10(max(rms, 1e-6)) if rms > 0 else -120.0


class AudioCapture:
    """麦克风采集：sounddevice InputStream → asyncio.Queue（有界 512 帧 ≈15.4s）。

    SPEAK 期队列容量临时提升至 2048（§3.4 帧不丢），由 set_speak_phase() 控制。
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.frame_len = cfg.sample_rate * cfg.frame_ms // 1000
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=512)  # 3.10+ 无需 loop，线程安全
        self._seq = 0
        self._stream: sd.InputStream | None = None
        self._dropped = 0
        self._speak_phase = False

    def set_speak_phase(self, speaking: bool) -> None:
        """SPEAK 期扩队列（打断检测依赖连续帧）。"""
        self._speak_phase = speaking

    def _callback(self, indata: np.ndarray, frames: int, t, status) -> None:  # 音频线程
        if status:
            log.warning("capture status: %s", status)
        frame = indata[:, 0].astype(np.int16).copy() if indata.shape[1] > 1 else indata.copy()
        self._seq += 1
        af = AudioFrame(pcm=frame, ts_ms=int(t.inputBufferAdcTime * 1000), source="mic", seq=self._seq)
        try:
            cap = 2048 if self._speak_phase else 512
            if self._queue.qsize() >= cap:
                self._dropped += 1
                # 满则丢最老帧（§3.4）
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(af)
        except Exception:  # 队列满等异常 → 计数
            self._dropped += 1

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="int16",
            blocksize=self.frame_len,
            device=self.cfg.device,
            callback=self._callback,
        )
        self._stream.start()
        log.info("capture started: %d Hz, frame=%d", self.cfg.sample_rate, self.frame_len)

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self):
        """事件循环消费（T2）。"""
        while True:
            try:
                yield await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # 静音心跳

    @property
    def dropped(self) -> int:
        return self._dropped
