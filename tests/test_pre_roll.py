"""pre-roll 环形缓冲测试（设计 §8.1）。"""
import numpy as np
import pytest

from duplex_voice.audio.pre_roll import PreRollBuffer


def _frame(value: float) -> np.ndarray:
    return np.full(480, value, dtype=np.int16)


def test_full_ring_rollback_time_order():
    buf = PreRollBuffer(capacity_ms=1200, frame_ms=30)
    for i in range(40):            # 填满 40 帧
        buf.push(_frame(i + 1))
    out = buf.rollback()
    assert out.shape[0] == 40 * 480
    # 时间正序：第一帧应为 1
    assert out[0] == 1
    assert out[-1] == 40


def test_ring_wraparound():
    buf = PreRollBuffer(capacity_ms=1200, frame_ms=30)
    for i in range(50):            # 超过容量 10 帧 → 环形覆盖
        buf.push(_frame(i + 1))
    out = buf.rollback()
    assert out.shape[0] == 40 * 480
    assert out[0] == 11            # 最老帧被覆盖
    assert out[-1] == 50


def test_partial_rollback():
    buf = PreRollBuffer(capacity_ms=1200, frame_ms=30)
    buf.push(_frame(7))
    buf.push(_frame(8))
    out = buf.rollback()
    assert out.shape[0] == 2 * 480
    assert out[0] == 7


def test_empty_rollback():
    buf = PreRollBuffer(capacity_ms=1200, frame_ms=30)
    assert buf.rollback().size == 0


def test_clear():
    buf = PreRollBuffer(capacity_ms=1200, frame_ms=30)
    buf.push(_frame(1))
    buf.clear()
    assert buf.rollback().size == 0


def test_filled_ms():
    buf = PreRollBuffer(capacity_ms=1200, frame_ms=30)
    for _ in range(10):
        buf.push(_frame(1))
    assert buf.filled_ms == 300
