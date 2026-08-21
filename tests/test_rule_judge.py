"""RuleVadJudge 测试（设计 §8.1：双钳制/窗口边界/grace）。

测试帧数据：真实语音（tests/assets/test_speech_16k.wav，"打开客厅的灯"）
+ 合成静音——Silero VAD v6 引擎（2026-08-18 集成）只认真实语音分布，
合成正弦/噪声会概率 <0.5 导致误判，测试改用真实语音帧。
"""
import wave
from pathlib import Path

import numpy as np
import pytest

from duplex_voice.config import VadConfig
from duplex_voice.vad import RuleVadJudge

_ASSETS = Path(__file__).resolve().parent / "assets" / "test_speech_16k.wav"
_SPEECH_FRAMES: list[np.ndarray] | None = None


def _load_speech_frames() -> list[np.ndarray]:
    """真实语音 480 帧（16k/30ms）。"""
    global _SPEECH_FRAMES
    if _SPEECH_FRAMES is None:
        with wave.open(str(_ASSETS), "rb") as w:
            pcm = w.readframes(w.getnframes())
        _SPEECH_FRAMES = [np.frombuffer(pcm[i:i + 960], dtype=np.int16)
                          for i in range(0, len(pcm) - 960 + 1, 960)]
    return _SPEECH_FRAMES


def _voice(amp: float = 1.0) -> np.ndarray:
    """真实语音帧（Silero 概率高：取 750ms 处语音中段帧）。"""
    return _load_speech_frames()[25]


def _silence() -> np.ndarray:
    return np.zeros(480, dtype=np.int16)


def _judge(**kw) -> RuleVadJudge:
    cfg = VadConfig(**kw)
    return RuleVadJudge(cfg)


class FakeFrame:
    def __init__(self, pcm, ts_ms=0):
        self.pcm = pcm
        self.ts_ms = ts_ms


def test_speech_segment_basic():
    j = _judge(min_silence_ms=500, min_speech_ms=250)
    # 静音帧（噪声底校准 15 帧 = 450ms）
    for _ in range(15):
        j.feed(FakeFrame(_silence()))
    # 说话 10 帧（300ms）→ speech_start
    events = []
    for i in range(10):
        events += j.feed(FakeFrame(_voice(), ts_ms=1000 + i * 30))
    assert any(e.event_type == "speech_start" for e in events)
    # 静音 20 帧（600ms）→ speech_end
    events = []
    for i in range(20):
        events += j.feed(FakeFrame(_silence(), ts_ms=1400 + i * 30))
    assert any(e.event_type == "speech_end" for e in events)


def test_speech_requires_min_duration():
    """<min_speech_ms 的短音不触发 speech_start。"""
    j = _judge(min_speech_ms=250)
    for _ in range(15):
        j.feed(FakeFrame(_silence()))
    events = []
    for i in range(6):  # 180ms < 250ms
        events += j.feed(FakeFrame(_voice(), ts_ms=1000 + i * 30))
    assert not any(e.event_type == "speech_start" for e in events)


def test_silence_requires_min_duration():
    """<min_silence_ms 的短静音不触发 speech_end。"""
    j = _judge(min_silence_ms=500)
    for _ in range(15):
        j.feed(FakeFrame(_silence()))
    for i in range(10):
        j.feed(FakeFrame(_voice(), ts_ms=1000 + i * 30))
    events = []
    for i in range(10):  # 300ms < 500ms
        events += j.feed(FakeFrame(_silence(), ts_ms=1400 + i * 30))
    assert not any(e.event_type == "speech_end" for e in events)


def test_interrupt_low_energy_no_trigger():
    """低能量（静音）不触发打断。"""
    j = _judge(threshold_interrupt=0.7)
    for _ in range(15):
        j.feed(FakeFrame(_silence()), phase="listen")     # 校准安静底
    j.set_phase("speak")
    j.playback_start(now_ms=0, gap_ms=2000)              # grace 500ms
    events = []
    for i in range(10):
        events += j.feed(FakeFrame(_silence(), ts_ms=600 + i * 30), phase="speak")
    for _ in range(45):
        events += j.feed(FakeFrame(_silence(), ts_ms=1500 + _ * 30), phase="speak")
    assert not any(e.event_type == "takeover_noise" for e in events)  # 未触发


def test_interrupt_grace_blocks_early_frames():
    """播放起始 grace 500ms 内不触发。"""
    j = _judge()
    for _ in range(15):
        j.feed(FakeFrame(_silence()), phase="listen")
    j.set_phase("speak")
    j.playback_start(now_ms=1000, gap_ms=2000)           # grace 到 1500
    events = []
    for i in range(8):                                   # 1000-1240ms 内（grace 中）
        events += j.feed(FakeFrame(_voice(), ts_ms=1000 + i * 30), phase="speak")
    assert events == []                                  # grace 期静默


def test_interrupt_window_majority():
    """窗口多数：300ms 内 ≥80%（8/10）触发。"""
    j = _judge(window_ms=300, window_ratio=0.8)
    for _ in range(15):
        j.feed(FakeFrame(_silence()), phase="listen")
    j.set_phase("speak")
    j.playback_start(now_ms=2000, gap_ms=2000)
    # 10 帧中 9 帧真实语音（含 1250ms 收尾 → 触发）
    events = []
    for i in range(10):
        events += j.feed(FakeFrame(_voice(), ts_ms=2000 + 500 + i * 30), phase="speak")
    # 触发后需要 1250ms 静音才交付事件
    for _ in range(45):
        events += j.feed(FakeFrame(_silence(), ts_ms=3000 + _ * 30), phase="speak")
    assert any(e.event_type == "takeover_noise" for e in events)


def test_phase_switch_resets_window():
    j = _judge()
    j.set_phase("speak")
    j.feed(FakeFrame(_voice(), ts_ms=0), phase="speak")
    j.set_phase("listen")                                # 切回听：窗口清空
    assert len(j._window) == 0
