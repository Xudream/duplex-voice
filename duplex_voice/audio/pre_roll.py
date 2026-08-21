"""pre-roll 环形缓冲（软件设计 §3.2，修正后：采集侧 AEC 后）。

打断瞬间用户话语的开头已在麦克风侧累积——回滚的是用户的声音。
双指针环形，容量 = pre_roll_ms / frame_ms 帧。
"""
from __future__ import annotations

import numpy as np


class PreRollBuffer:
    """容量 pre_roll_ms 的 16k PCM 环形缓冲（int16, mono）。"""

    def __init__(self, capacity_ms: int = 1200, frame_ms: int = 30, sample_rate: int = 16000):
        self.capacity = max(1, capacity_ms // frame_ms)          # 帧数（如 40）
        self.frame_len = sample_rate * frame_ms // 1000          # 每帧 samples（480）
        self._buf = np.zeros((self.capacity, self.frame_len), dtype=np.int16)
        self._head = 0
        self._count = 0

    def push(self, frame: np.ndarray) -> None:
        """采集线程每 30ms 推入一帧（AEC 后麦克风帧）。"""
        if frame.shape[0] != self.frame_len:
            raise ValueError(f"帧长必须为 {self.frame_len}，实际 {frame.shape[0]}")
        self._buf[self._head] = frame
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def rollback(self) -> np.ndarray:
        """返回最近最多 capacity 帧（时间正序），供 ASR prefill。"""
        if self._count == 0:
            return np.empty(0, dtype=np.int16)
        start = (self._head - self._count) % self.capacity
        idx = np.arange(start, start + self._count) % self.capacity
        return self._buf[idx].reshape(-1).copy()

    def clear(self) -> None:
        self._count = 0

    @property
    def filled_ms(self) -> int:
        return self._count * 30
