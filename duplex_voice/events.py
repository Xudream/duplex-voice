"""事件信封与事件类型（软件设计 v2.1 §1.4 / §2.4）。

模块→FSM 只经 onEvent()；事件统一结构含 trace；高优先级事件走独立队列头。
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

_seq = itertools.count(1)

# 高优先级事件（打断/取消）—— 直接注入队列头，绕过排队（§1.4）
HI_PRIORITY_EVENTS = frozenset({"takeover_noise", "takeover_intent", "user.cancel"})


@dataclass(slots=True)
class Event:
    """统一事件信封。"""

    type: str
    domain: str
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str = "s_default"
    trace_id: str = "tr_local"
    ts: int = field(default_factory=lambda: int(time.time() * 1000))
    seq: int = field(default_factory=lambda: next(_seq))

    @property
    def is_hi_priority(self) -> bool:
        return self.type in HI_PRIORITY_EVENTS

    def __repr__(self) -> str:  # 紧凑日志
        return f"<{self.type} t={self.ts} seq={self.seq}>"


# 事件类型常量（软件设计 §2.4 事件词汇）
E_SPEECH_START = "speech_start"          # vad 域
E_SPEECH_END = "speech_end"              # vad 域
E_EOT = "eot"                            # vad 域（语义档）
E_TAKEOVER_NOISE = "takeover_noise"      # vad 域（规则档声学打断）
E_TAKEOVER_INTENT = "takeover_intent"    # vad 域（语义档）
E_REJECT = "reject"                      # vad 域（语义档）
E_BACKCHANNEL = "backchannel"            # vad 域（语义档）
E_ASR_PARTIAL = "asr.partial"            # asr 域
E_ASR_FINAL = "asr.final"                # asr 域
E_ASR_ERROR = "asr.error"                # asr 域
E_ASR_TIMEOUT = "asr.timeout"            # asr 域
E_LLM_FIRST_TOKEN = "llm.first_token"    # llm 域
E_LLM_TOKEN = "llm.token"                # llm 域
E_LLM_COMPLETE = "llm.complete"          # llm 域
E_LLM_ERROR = "llm.error"                # llm 域
E_TTS_STARTED = "tts.started"            # tts 域
E_TTS_AUDIO = "tts.audio_chunk"          # tts 域（→ player，不经 FSM）
E_TTS_PLAYBACK_DONE = "tts.playback_done"
E_TTS_CANCELLED = "tts.cancelled"
E_TTS_ERROR = "tts.error"
E_USER_CANCEL = "user.cancel"            # sess 域
E_SESSION_TIMEOUT = "session.timeout"    # sess 域
E_SESSION_EXPIRED = "session.expired"    # sess 域
E_SYSTEM_ERROR = "system.error"          # sys 域
