"""快慢融合测试（软件设计 v2.2 §2.5：承接→转述→增量融合）。"""
import asyncio
from types import SimpleNamespace

import numpy as np

from duplex_voice.adapter.llm import FusionPolicy
from duplex_voice.config import Config
from duplex_voice.events import Event
from duplex_voice.fsm.fsm import DuplexFSM
from duplex_voice.session import SessionManager


class FakeCapture:
    def set_speak_phase(self, v): pass


class FakePlayer:
    def __init__(self): self.stopped = 0
    def stop(self): self.stopped += 1
    def play(self, pcm): pass


class FakeJudge:
    def feed(self, frame, phase=None): return []
    def set_phase(self, p): pass
    def is_voicing(self): return False


class FakeASR:
    async def stream_recognize(self, frames, **kw):
        yield Event(type="asr.final", domain="asr", payload={"text": "x"})
    async def end_segment(self): pass
    async def close(self): pass


class FakePreRoll:
    def push(self, pcm): pass
    def rollback(self): return np.zeros(1, dtype=np.int16)
    def clear(self): pass


class SlowLLM500:
    """慢通道：500ms 首 token（模拟云端深度思考延迟）。"""

    async def stream_chat(self, messages, **kw):
        await asyncio.sleep(0.5)
        yield Event(type="llm.first_token", domain="llm", payload={"channel": "slow"})
        yield Event(type="llm.token", domain="llm", payload={"delta": "好的，", "channel": "slow"})
        yield Event(type="llm.token", domain="llm", payload={"delta": "正在为您打开客厅的灯。", "channel": "slow"})
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": "好的，正在为您打开客厅的灯。", "channel": "slow"})
    async def close(self): pass


class FastFail:
    """快通道故障：抛异常（应被吞，慢通道兜底）。"""

    async def stream_chat(self, messages, **kw):
        raise ConnectionError("本地 4b 服务不可用")
    async def close(self): pass


async def _fast_stream(self, messages, **kw):
    yield Event(type="llm.first_token", domain="llm", payload={"channel": "fast"})
    yield Event(type="llm.complete", domain="llm", payload={"full_text": "好的", "channel": "fast"})


class FastOK:
    async def stream_chat(self, messages, **kw):
        async for e in _fast_stream(self, messages, **kw):
            yield e
    async def close(self): pass


class FakeTTS:
    def __init__(self): self.cancelled = 0
    async def stream_synthesize(self, text, **kw):
        yield Event(type="tts.started", domain="tts", payload={})
        yield Event(type="tts.audio_chunk", domain="tts", payload={"pcm": np.zeros(480, dtype=np.int16)})
        yield Event(type="tts.playback_done", domain="tts", payload={})
    async def cancel(self): self.cancelled += 1
    async def close(self): pass


def _fsm(fast=None, slow=None):
    cfg = Config()
    return DuplexFSM(cfg, FakeCapture(), FakePlayer(), FakeJudge(),
                     FakeASR(), fast or FastFail(), slow or SlowLLM500(), FakeTTS(),
                     SessionManager(":memory:", 20), FakePreRoll())


async def _run_turn(fsm, fast=None):
    """驱动一轮：speech → asr.final → 双通道事件。"""
    await fsm._dispatch(Event(type="speech_start", domain="vad", payload={}))
    await fsm._dispatch(Event(type="speech_end", domain="vad", payload={}))
    await fsm._dispatch(Event(type="asr.final", domain="asr",
                              payload={"text": "帮我写一份周报", "confidence": 0.95}))
    await asyncio.sleep(0.1)
    if fast:
        await fsm._dispatch(Event(type="llm.first_token", domain="llm", payload={"channel": "fast"}))
        await fsm._dispatch(Event(type="llm.complete", domain="llm",
                                  payload={"full_text": "好的", "channel": "fast"}))
    await fsm._dispatch(Event(type="llm.first_token", domain="llm", payload={"channel": "slow"}))
    await fsm._dispatch(Event(type="llm.token", domain="llm", payload={"delta": "好的，", "channel": "slow"}))
    await fsm._dispatch(Event(type="llm.complete", domain="llm",
                              payload={"full_text": "好的，正在为您打开客厅的灯。", "channel": "slow"}))


def test_fast_first_then_slow_merge():
    """快通道承接语先播 → 慢通道完整回复接播（无感切换）。"""
    fsm = _fsm(fast=FastOK())

    async def drive():
        await _run_turn(fsm, fast=True)
        # 快承接语 → SPEAK 播放
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "synthesizing"
        assert fsm._fast_text == "好的"
        # 慢回复完整接管
        assert fsm._slow_text == "好的，正在为您打开客厅的灯。"
        # 承接语播完 → 接播慢回复（_slow_ready 路径）
        await fsm._dispatch(Event(type="tts.playback_done", domain="tts", payload={}))
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "synthesizing"
        await asyncio.sleep(0.05)
        # 慢回复播完 → post_wait → 超时回 LISTEN
        await fsm._dispatch(Event(type="tts.playback_done", domain="tts", payload={}))
        assert fsm.sub.value == "post_wait"
        await fsm._dispatch(Event(type="session.timeout", domain="sess",
                                  payload={"kind": "listen_window"}))
        assert fsm.state.value == "LISTEN"
        # 会话记录 = 慢通道完整回复
        assert fsm.session._turns[-1].assistant_text == "好的，正在为您打开客厅的灯。"

    asyncio.run(drive())


def test_fast_failure_slow_fallback():
    """快通道故障 → 慢通道直出（降级矩阵 7.3）。"""
    fsm = _fsm(fast=FastFail())

    async def drive():
        await _run_turn(fsm, fast=False)
        assert fsm._fast_text == ""
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "synthesizing"
        assert fsm._slow_text == "好的，正在为您打开客厅的灯。"
        await fsm._dispatch(Event(type="tts.playback_done", domain="tts", payload={}))
        assert fsm.sub.value == "post_wait"

    asyncio.run(drive())


def test_slow_complete_after_fast_done():
    """承接语播放完（post_wait）后慢通道才完成 → 直接播慢回复。"""
    fsm = _fsm(fast=FastOK())

    async def drive():
        await _run_turn(fsm, fast=True)
        await fsm._dispatch(Event(type="tts.playback_done", domain="tts", payload={}))
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "synthesizing"
        await fsm._dispatch(Event(type="tts.playback_done", domain="tts", payload={}))
        assert fsm.sub.value == "post_wait"

    asyncio.run(drive())


def test_fusion_policy_units():
    """FusionPolicy 单元：should_speak / build_fast_prompt / transcribe / merge。"""
    assert FusionPolicy.should_speak("帮我写周报") is True
    assert FusionPolicy.should_speak("好") is False          # 极短指令不承接
    prompt = FusionPolicy.build_fast_prompt("打开灯")
    assert prompt[0]["role"] == "system" and "过渡语" in prompt[0]["content"]
    assert FusionPolicy.transcribe("好的", "好的，马上") == "好的，马上"
    assert FusionPolicy.merge("好的，", "正在为您打开") == "好的，正在为您打开"

