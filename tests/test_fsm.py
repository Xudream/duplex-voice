"""FSM 转移表测试（设计 §8.1：全部转移条目事件注入）。"""
import asyncio

import pytest

from duplex_voice.config import Config
from duplex_voice.events import (
    E_ASR_FINAL, E_LLM_COMPLETE, E_LLM_FIRST_TOKEN, E_SPEECH_END, E_SPEECH_START,
    E_TAKEOVER_NOISE, E_TTS_PLAYBACK_DONE, Event,
)
from duplex_voice.fsm import DuplexFSM
from duplex_voice.fsm.fsm import State, Sub
from duplex_voice.session import SessionManager


class FakeCapture:
    def __init__(self):
        self.speak = False

    def set_speak_phase(self, v): self.speak = v


class FakePlayer:
    def __init__(self):
        self.stopped = 0

    def stop(self): self.stopped += 1
    def play(self, pcm): pass


class FakeJudge:
    def __init__(self):
        self.phase = "listen"

    def feed(self, frame, phase=None): return []
    def set_phase(self, p): self.phase = p
    def reset(self): pass


class FakeASR:
    def __init__(self):
        self.ended = 0

    async def stream_recognize(self, frames, **kw):
        yield Event(type="asr.final", domain="asr", payload={"text": "x"})
    async def end_segment(self): self.ended += 1
    async def close(self): pass


class FakeFastLLM:
    """快通道 mock：承接语（首 token 100ms）。"""

    async def stream_chat(self, messages, **kw):
        yield Event(type="llm.first_token", domain="llm", payload={"channel": "fast"})
        yield Event(type="llm.token", domain="llm", payload={"delta": "好的", "channel": "fast"})
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": "好的", "channel": "fast"})
    async def close(self): pass


class FakeLLM:
    async def stream_chat(self, messages, **kw):
        yield Event(type="llm.first_token", domain="llm", payload={"channel": "slow"})
        yield Event(type="llm.token", domain="llm", payload={"delta": "好的", "channel": "slow"})
        yield Event(type="llm.complete", domain="llm", payload={"full_text": "好的", "channel": "slow"})
    async def close(self): pass


class FakeTTS:
    def __init__(self):
        self.cancelled = 0
        self.sent_audio = 0

    async def stream_synthesize(self, text, **kw):
        yield Event(type="tts.started", domain="tts", payload={})
        yield Event(type="tts.audio_chunk", domain="tts", payload={"pcm": __import__("numpy").zeros(480, dtype="int16")})
        yield Event(type="tts.playback_done", domain="tts", payload={})
    async def cancel(self): self.cancelled += 1
    async def close(self): pass


class FakePreRoll:
    def push(self, pcm): pass
    def rollback(self): return __import__("numpy").zeros(480 * 40, dtype="int16")
    def clear(self): pass


def _fsm() -> tuple[DuplexFSM, FakeTTS, FakeASR]:
    cfg = Config()
    fsm = DuplexFSM(cfg, FakeCapture(), FakePlayer(), FakeJudge(),
                    FakeASR(), FakeFastLLM(), FakeLLM(), FakeTTS(),
                    SessionManager(":memory:", 20), FakePreRoll())
    return fsm, fsm.tts, fsm.asr


def test_full_turn_flow():
    """正常对话全流程：speech_start → speech_end → asr.final → LLM → TTS → playback_done → 倾听。"""
    fsm, tts, asr = _fsm()

    async def drive():
        await fsm._dispatch(Event(type=E_SPEECH_START, domain="vad", payload={}))
        assert fsm.state.value == "LISTEN" and fsm.sub.value == "voicing"
        await fsm._dispatch(Event(type=E_SPEECH_END, domain="vad", payload={}))
        assert fsm.sub.value == "finalizing"
        await fsm._dispatch(Event(type=E_ASR_FINAL, domain="asr",
                                  payload={"text": "打开灯", "confidence": 0.9}))
        assert fsm.state.value == "THINK"
        # 快通道承接语（100ms）→ 播放
        await asyncio.sleep(0.05)
        await fsm._dispatch(Event(type="llm.first_token", domain="llm", payload={"channel": "fast"}))
        await fsm._dispatch(Event(type="llm.complete", domain="llm",
                                  payload={"full_text": "好的", "channel": "fast"}))
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "synthesizing"
        await asyncio.sleep(0.05)  # TTS 任务（承接语）
        # 慢通道完成 → 接播
        await fsm._dispatch(Event(type="llm.first_token", domain="llm", payload={"channel": "slow"}))
        await fsm._dispatch(Event(type="llm.complete", domain="llm",
                                  payload={"full_text": "好的，正在为您打开灯", "channel": "slow"}))
        await asyncio.sleep(0.05)  # TTS 任务（慢回复）
        # 承接语播放完成 → 接播慢回复
        await fsm._dispatch(Event(type=E_TTS_PLAYBACK_DONE, domain="tts", payload={}))
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "synthesizing"
        await asyncio.sleep(0.05)
        # 慢回复播放完成 → 倾听窗口
        await fsm._dispatch(Event(type=E_TTS_PLAYBACK_DONE, domain="tts", payload={}))
        assert fsm.state.value == "SPEAK" and fsm.sub.value == "post_wait"
        await fsm._dispatch(Event(type="session.timeout", domain="sess",
                                  payload={"kind": "listen_window"}))
        assert fsm.state.value == "LISTEN" and fsm.sub.value == "idle"
        assert len(fsm.session._turns) >= 1

    asyncio.run(drive())


def test_takeover_interrupt():
    """打断：SPEAK 期 takeover → TTS.cancel + pre-roll + LISTEN/voicing。"""
    fsm, tts, asr = _fsm()
    fsm.state = State.SPEAK
    fsm.sub = Sub.PLAYING

    async def drive():
        await fsm._dispatch(Event(type=E_TAKEOVER_NOISE, domain="vad",
                                  payload={"confidence": 0.85}))
        assert tts.cancelled >= 1
        assert fsm.state.value == "LISTEN" and fsm.sub.value == "voicing"
        assert fsm._interrupt_note is not None

    asyncio.run(drive())


def test_user_cancel_any_state():
    fsm, _, _ = _fsm()
    fsm.state = State.SPEAK
    fsm.sub = Sub.PLAYING

    async def drive():
        await fsm._dispatch(Event(type="user.cancel", domain="sess",
                                  payload={"cause": "user"}))
        assert fsm.state.value == "LISTEN" and fsm.sub.value == "idle"

    asyncio.run(drive())


def test_hi_priority_queue():
    """高优先级事件走独立队列头。"""
    fsm, _, _ = _fsm()
    fsm.on_event(Event(type="user.cancel", domain="sess", payload={}))
    assert fsm._hi_queue.qsize() == 1
    fsm.on_event(Event(type="llm.token", domain="llm", payload={"delta": "x"}))
    assert fsm._queue.qsize() == 1
