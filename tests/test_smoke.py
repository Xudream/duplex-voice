"""冒烟测试：真实装配（build_fsm + mock providers）+ 完整回合 + 自动退出。

不依赖麦克风/stdin——直接驱动事件链，验证 P0b 闭环可跑。
运行: python3 -m pytest tests/test_smoke.py -v
"""
import asyncio
import os

import numpy as np
import pytest

from duplex_voice.config import Config
from duplex_voice.events import Event
from duplex_voice.main import build_fsm


def test_smoke_full_turn(tmp_path):
    """完整回合：speech_start → speech_end → asr.final → LLM → TTS 音频 → 落盘。"""
    cfg = Config()
    cfg.session_jsonl = str(tmp_path / "smoke.jsonl")
    cfg.model.asr_provider = "mock"
    cfg.model.llm_fast_provider = "mock"   # 快慢融合双通道（v2.2）
    cfg.model.llm_slow_provider = "mock"
    cfg.model.tts_provider = "mock"
    fsm = build_fsm(cfg)
    assert fsm.state.value == "LISTEN"

    async def drive():
        # 启动事件循环（后台）
        loop_task = asyncio.create_task(fsm.run())
        await asyncio.sleep(0.1)

        # 模拟 VAD 事件链
        fsm.on_event(Event(type="speech_start", domain="vad", payload={}))
        fsm.on_event(Event(type="speech_end", domain="vad", payload={}))
        fsm.on_event(Event(type="asr.final", domain="asr",
                           payload={"text": "打开客厅的灯", "confidence": 0.96}))
        # 等待 LLM(first_token 200ms) + TTS(150ms) 完成
        await asyncio.sleep(1.2)
        state = fsm.state.value
        sub = fsm.sub.value
        assert state in ("SPEAK", "LISTEN"), f"回合未推进: {state}/{sub}"
        # 慢回复播完（最后一棒）→ 倾听窗口 → 回 LISTEN（快慢融合流程）
        for _ in range(10):
            await asyncio.sleep(0.5)
            if fsm.state.value == "LISTEN":
                break
        assert fsm.state.value == "LISTEN"
        assert len(fsm.session._turns) >= 1
        assert fsm.session._turns[0].user_text == "打开客厅的灯"
        # JSONL 落盘验证
        with open(cfg.session_jsonl, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 1
        assert "打开客厅的灯" in lines[0]
        # 干净退出
        fsm.stop()
        await loop_task

    asyncio.run(drive())


def test_smoke_takeover(tmp_path):
    """打断回合：SPEAK 期 takeover → LISTEN/voicing + 打断注记。"""
    cfg = Config()
    cfg.session_jsonl = str(tmp_path / "smoke2.jsonl")
    cfg.model.llm_fast_provider = "mock"   # 双通道 mock（避免真实 Ollama/云端调用）
    cfg.model.llm_slow_provider = "mock"
    fsm = build_fsm(cfg)

    async def drive():
        loop_task = asyncio.create_task(fsm.run())
        await asyncio.sleep(0.1)
        fsm.on_event(Event(type="speech_start", domain="vad", payload={}))
        fsm.on_event(Event(type="speech_end", domain="vad", payload={}))
        fsm.on_event(Event(type="asr.final", domain="asr",
                           payload={"text": "打开灯", "confidence": 0.9}))
        await asyncio.sleep(0.9)   # LLM 完成 → TTS 开始
        assert fsm.state.value == "SPEAK", f"应在 SPEAK: {fsm.state.value}"
        # 打断（高优先级）
        fsm.on_event(Event(type="takeover_noise", domain="vad",
                           payload={"confidence": 0.85}))
        await asyncio.sleep(0.3)
        assert fsm.state.value == "LISTEN"
        assert fsm.sub.value == "voicing"
        assert fsm._interrupt_note is not None
        assert "interrupted" in fsm._interrupt_note
        fsm.stop()
        await loop_task

    asyncio.run(drive())
