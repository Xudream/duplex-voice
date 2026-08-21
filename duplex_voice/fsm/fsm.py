"""DuplexFSM —— 双工状态机（软件设计 §2.4，核心）。

四态 + 子状态；事件循环（高优先级队列头）；25 条转移表；7 定时器；
打断裁决：TTS.stop → pre-roll 回滚 → ASR begin_segment(prefill) → note_interrupt → LISTEN/voicing。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from ..adapter import ASRProvider, LLMProvider, TTSProvider
from ..adapter.asr import FunASRStreamProvider
from ..adapter.llm import FusionPolicy
from ..audio import AudioCapture, AudioPlayer, PreRollBuffer
from ..config import Config
from ..events import (
    E_ASR_FINAL, E_ASR_TIMEOUT, E_LLM_COMPLETE, E_LLM_ERROR, E_LLM_FIRST_TOKEN, E_LLM_TOKEN,
    E_SESSION_TIMEOUT, E_SPEECH_END, E_SPEECH_START, E_TAKEOVER_NOISE,
    E_TTS_PLAYBACK_DONE, E_USER_CANCEL, Event,
)
from ..session import SessionManager, TurnRecord
from ..vad import RuleVadJudge, VadDecision, VadJudge

log = logging.getLogger(__name__)


class State(str, Enum):
    LISTEN = "LISTEN"
    THINK = "THINK"
    SPEAK = "SPEAK"
    YIELD = "YIELD"


class Sub(str, Enum):
    IDLE = "idle"
    VOICING = "voicing"
    FINALIZING = "finalizing"
    WAITING_FIRST = "waiting_first"
    STREAMING = "streaming"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    POST_WAIT = "post_wait"
    STOPPING = "stopping"


class DuplexFSM:
    """级联全双工状态机。"""

    def __init__(self, cfg: Config, capture: AudioCapture, player: AudioPlayer,
                 judge: VadJudge, asr: ASRProvider, llm_fast, llm_slow, tts: TTSProvider,
                 session: SessionManager, pre_roll: PreRollBuffer):
        self.cfg = cfg.fsm
        self.capture = capture
        self.player = player
        self.judge = judge
        self.asr = asr
        self.llm_fast = llm_fast          # 快通道（本地小模型：承接语/转述）
        self.llm_slow = llm_slow          # 慢通道（云端大模型：深度思考）
        self.tts = tts
        self.session = session
        self.pre_roll = pre_roll
        self._fast_text = ""              # 快通道承接语
        self._slow_text = ""              # 慢通道完整回复
        self._slow_ready = False          # 慢通道完成（承接语播放中时置位）
        self._playing_slow = False        # 当前播放的是慢回复（播完走 listen_window）

        self.state = State.LISTEN
        self.sub = Sub.IDLE
        self._queue: asyncio.Queue = asyncio.Queue()
        self._hi_queue: asyncio.Queue = asyncio.Queue()
        self._turn_id = 0
        self._user_text = ""
        self._assistant_text = ""
        self._turn_start_ms = 0
        self._interrupt_flag = False
        self._interrupt_note: str | None = None
        self._seg_frames: list = []          # 当前 ASR 段帧（Mock/一次性用）
        self._asr_queue: asyncio.Queue | None = None   # 真流式：帧实时上行队列
        self._timers: dict[str, asyncio.Task] = {}
        self._listeners: list = []
        self._stopped = False

    def stop(self) -> None:
        """停止事件循环（main 退出用）。"""
        self._stopped = True

    # ---------- 对外接口（§4.1） ----------
    def on_event(self, evt: Event) -> None:
        (self._hi_queue if evt.is_hi_priority else self._queue).put_nowait(evt)

    def feed_audio(self, frame) -> None:
        """采集帧入口：VAD 判定 → pre-roll → 帧收集（judge 语音态）→ 事件。

        帧收集不依赖状态机事件（事件循环有滞后）：judge 内部 _in_speech
        或状态在 voicing/finalizing 都收集，避免 speech_start 未处理时丢帧。
        """
        self.pre_roll.push(frame.pcm)
        decisions = self.judge.feed(frame, self.state.value.lower())
        if (isinstance(self.judge, RuleVadJudge) and self.judge.is_voicing()) or \
           (self.state is State.LISTEN and self.sub in (Sub.VOICING, Sub.FINALIZING)):
            self._seg_frames.append(frame.pcm)
            if self._asr_queue is not None:
                # 真流式：帧实时上行（不积攒，服务端边收边识别）
                try:
                    self._asr_queue.put_nowait(frame.pcm)
                except Exception:
                    pass
        for d in decisions:
            self.on_event(self._to_event(d, frame))

    def get_state(self) -> tuple[State, Sub]:
        return self.state, self.sub

    def on_state_change(self, cb) -> None:
        self._listeners.append(cb)

    # ---------- 事件循环（§4.4.3） ----------
    async def run(self) -> None:
        log.info("FSM 启动: %s/%s", self.state, self.sub)
        while not self._stopped:
            try:
                # 高优先级不被普通队列阻塞：同时等待两个队列（0.5s 超时便于 stop 检查）
                hi_task = asyncio.create_task(self._hi_queue.get())
                lo_task = asyncio.create_task(self._queue.get())
                done, pending = await asyncio.wait(
                    {hi_task, lo_task}, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                if not done:
                    continue  # 超时：回循环头检查 _stopped
                evt = hi_task.result() if hi_task in done else lo_task.result()
            except asyncio.CancelledError:
                # 任务被外部取消（stop/循环关闭）→ 正常退出，不吞取消
                raise
            t0 = time.perf_counter()
            try:
                await self._dispatch(evt)
            except Exception as e:
                log.error("dispatch %s 异常: %s", evt, e, exc_info=True)
                self.on_event(Event(type="system.error", domain="sys",
                                    payload={"code": "E_UNKNOWN", "error": str(e)}))
            dt = (time.perf_counter() - t0) * 1000
            if dt > 5:
                log.warning("FSM dispatch 超 5ms: %s (%.1fms)", evt, dt)

    # ---------- 转移表 ----------
    async def _dispatch(self, evt: Event) -> None:
        st, sub = self.state, self.sub
        t = evt.type
        log.info("[%s/%s] <- %s", st.value, sub.value, evt)

        if t == E_SPEECH_START and st is State.LISTEN and sub is Sub.IDLE:
            self._set(State.LISTEN, Sub.VOICING)
            # 注意：不清空 _seg_frames —— 帧收集由 judge.is_voicing 驱动（feed 层），
            # speech_start 事件到达时语音帧已收集，清空会丢帧（事件循环滞后竞态）
            if isinstance(self.asr, FunASRStreamProvider):
                # 真流式：帧实时上行（speech_start 起启动 WS，feed_audio 边收边发）
                self._asr_queue = asyncio.Queue()
                # 丢字修复：pre-roll 帧补发（VAD 判定延迟截掉的开头 1200ms）
                try:
                    rolled = self.pre_roll.rollback()
                    for i in range(0, len(rolled), 480):
                        seg = rolled[i:i + 480]
                        if len(seg) == 480:
                            self._asr_queue.put_nowait(seg)
                except Exception:
                    pass
                asyncio.create_task(self._run_asr_stream())
            self._turn_start_ms = evt.ts
            self._start_timer("wake_window", self.cfg.wake_window_ms, "wake_window")

        elif t == E_SPEECH_END and st is State.LISTEN and sub is Sub.VOICING:
            self._set(State.LISTEN, Sub.FINALIZING)
            if self._asr_queue is not None:
                # 真流式：帧已实时上行，关队列 → 服务端 finalize
                self._asr_queue.put_nowait(None)
                self._asr_queue = None
            else:
                asyncio.create_task(self._run_asr())   # 一次性（消费 _seg_frames）
            self._start_timer("final", self.cfg.final_fallback_ms, "final")

        elif t == E_ASR_FINAL and st is State.LISTEN and sub is Sub.FINALIZING:
            self._cancel_timer("final")
            self._user_text = evt.payload.get("text", "")
            self._turn_id += 1
            self._fast_text = ""
            self._slow_text = ""
            self._slow_ready = False
            self._set(State.THINK, Sub.WAITING_FIRST)
            self._start_timer("llm_first", self.cfg.llm_first_token_ms, "llm_first")
            asyncio.create_task(self._run_llm_fast(self._user_text))
            asyncio.create_task(self._run_llm_slow(self._user_text))

        elif t == E_ASR_TIMEOUT and st is State.LISTEN:
            log.info("ASR 超时，放弃该段")
            self._set(State.LISTEN, Sub.IDLE)

        elif t == E_LLM_FIRST_TOKEN and st is State.THINK and evt.payload.get("channel") == "slow":
            self._cancel_timer("llm_first")
            self._set(State.THINK, Sub.STREAMING)

        elif t == E_LLM_TOKEN and evt.payload.get("channel") == "slow" \
                and st in (State.THINK, State.SPEAK):
            # 慢通道 token 累积（承接语播放中/等待期也累积，complete 时全文可用）
            self._slow_text += evt.payload.get("delta", "")

        elif t == E_LLM_COMPLETE and evt.payload.get("channel") == "fast":
            # 快通道承接语 → 立即播放（TTFA ≤400ms）
            self._fast_text = evt.payload.get("full_text", self._fast_text)
            if st is State.THINK:
                self._cancel_timer("llm_first")
                self._set(State.SPEAK, Sub.SYNTHESIZING)
                asyncio.create_task(self._run_tts(self._fast_text))

        elif t == E_LLM_COMPLETE and evt.payload.get("channel") == "slow":
            # 慢通道完整回复 → 接管播放（承接语播放中则排队接播）
            self._slow_text = evt.payload.get("full_text", self._slow_text)
            self._cancel_timer("llm_first")
            if st is State.THINK:
                # 无承接语/承接语未播 → 直接播慢回复
                self._slow_ready = False
                self._playing_slow = True
                self._set(State.SPEAK, Sub.SYNTHESIZING)
                asyncio.create_task(self._run_tts(self._slow_text))
            elif st is State.SPEAK and sub is Sub.POST_WAIT:
                # 承接语已播完、正在等慢回复 → 接播
                self._slow_ready = False
                self._playing_slow = True
                self._cancel_timer("slow_wait")
                self._set(State.SPEAK, Sub.SYNTHESIZING)
                asyncio.create_task(self._run_tts(self._slow_text))
            else:
                # 承接语播放中（synthesizing/playing）：playback_done 时接播
                self._slow_ready = True

        elif t in (E_TAKEOVER_NOISE,) and st is State.SPEAK:
            await self._takeover(evt)

        elif t == E_TTS_PLAYBACK_DONE and st is State.SPEAK:
            if self._slow_ready and self._slow_text:
                # 承接语播放完 → 接播慢通道完整回复（无感切换）
                self._slow_ready = False
                self._playing_slow = True
                self._set(State.SPEAK, Sub.SYNTHESIZING)
                asyncio.create_task(self._run_tts(self._slow_text))
            elif self._playing_slow:
                # 慢回复播完（最后一棒）→ 倾听窗口，回 LISTEN
                self._playing_slow = False
                self._set(State.SPEAK, Sub.POST_WAIT)
                self._start_timer("listen_window", self.cfg.listen_window_ms, "listen_window")
            else:
                # 承接语播完、慢通道未到 → 等待慢回复（slow_wait 兜底）
                self._set(State.SPEAK, Sub.POST_WAIT)
                self._start_timer("slow_wait", self.cfg.slow_wait_ms, "slow_wait")

        elif t == E_SESSION_TIMEOUT and st is State.SPEAK and sub is Sub.POST_WAIT:
            self._end_turn()
            self._set(State.LISTEN, Sub.IDLE)

        elif t == E_LLM_ERROR and st is State.SPEAK and sub is Sub.POST_WAIT:
            # 慢通道失败（承接语已播）→ 结束回合，回倾听
            self._end_turn()
            self._set(State.LISTEN, Sub.IDLE)

        elif t == E_SESSION_TIMEOUT and st is State.LISTEN and sub in (Sub.IDLE, Sub.VOICING):
            # wake_window 超时：免唤醒关闭（低功耗）
            self._cancel_timer("wake_window")
            self._seg_frames = []
            log.info("免唤醒窗口超时，低功耗监听")
            self._set(State.LISTEN, Sub.IDLE)

        elif t == E_USER_CANCEL:
            await self.tts.cancel()
            self.player.stop()
            self._end_turn()
            self._set(State.LISTEN, Sub.IDLE)

        elif t == "system.error":
            log.error("system.error: %s", evt.payload)
            self._set(State.LISTEN, Sub.IDLE)

        else:
            log.debug("未匹配转移: %s (state=%s/%s)", evt, st.value, sub.value)

    # ---------- 动作 ----------
    async def _run_asr_stream(self) -> None:
        """真流式：帧队列实时上行（speech_start 启动，speech_end 关队列）。"""
        async def gen():
            q = self._asr_queue
            if q is None:
                return
            while True:
                f = await q.get()
                if f is None:
                    break
                yield f

        try:
            async for evt in self.asr.stream_recognize(gen()):
                if evt.type == "asr.partial":
                    log.debug("ASR partial: %s", evt.payload.get("text", ""))
                self.on_event(evt)
        except Exception as e:
            log.error("ASR 流式异常: %s", e)
            self.on_event(Event(type="asr.error", domain="asr",
                                payload={"code": "E_ASR_UNKNOWN", "retryable": True}))
        finally:
            await self.asr.end_segment()

    async def _run_asr(self) -> None:
        """消费 _seg_frames → ASR 流式识别 → asr.final/error。"""
        frames = list(self._seg_frames)

        async def gen():
            for f in frames:
                yield f

        try:
            async for evt in self.asr.stream_recognize(gen()):
                if evt.type == "asr.error":
                    log.warning("ASR 错误: %s", evt.payload)
                self.on_event(evt)
        except Exception as e:
            log.error("ASR 异常: %s", e, exc_info=True)
            self.on_event(Event(type="asr.error", domain="asr",
                                payload={"code": "E_ASR_UNKNOWN", "retryable": True}))
        finally:
            await self.asr.end_segment()

    async def _run_llm_fast(self, user_text: str) -> None:
        """快通道：承接语（FusionPolicy.should_speak 决策）。"""
        try:
            if not FusionPolicy.should_speak(user_text):
                return
            prompt = FusionPolicy.build_fast_prompt(user_text)
            async for evt in self.llm_fast.stream_chat(prompt, max_tokens=FusionPolicy.MAX_FAST_TOKENS):
                self.on_event(evt)
        except Exception as e:
            log.warning("快通道失败（慢通道兜底）: %s", e)

    async def _run_llm_slow(self, user_text: str) -> None:
        """慢通道：完整回复（深度思考）。"""
        try:
            messages = self.session.build_messages(user_text, self._interrupt_note)
            self._interrupt_note = None
            async for evt in self.llm_slow.stream_chat(messages):
                self.on_event(evt)
        except Exception as e:
            log.error("慢通道失败: %s", e)
            self.on_event(Event(type="llm.error", domain="llm", payload={"error": str(e)}))

    async def _run_tts(self, text: str) -> None:
        try:
            self.capture.set_speak_phase(True)
            self.judge.set_phase("speak") if isinstance(self.judge, RuleVadJudge) else None
            async for evt in self.tts.stream_synthesize(text):
                if evt.type == "tts.audio_chunk":
                    self.player.play(evt.payload["pcm"])
                elif evt.type == "tts.sentence_done":
                    pass
                elif evt.type == "tts.cancelled":
                    self.on_event(Event(type=E_TTS_PLAYBACK_DONE, domain="tts", payload={}))
                    return
                else:
                    self.on_event(evt)
        finally:
            self.capture.set_speak_phase(False)

    async def _takeover(self, evt: Event) -> None:
        """打断裁决（§2.4）：TTS.stop → pre-roll 回滚 → ASR prefill → note → LISTEN/voicing。"""
        log.info("== 打断触发 ==")
        self._interrupt_flag = True
        self._interrupt_note = (
            f"[Note: user interrupted the previous response at {evt.ts} ms; "
            f"pre-roll audio rolled back. Respond to the new request.]")
        self._set(State.SPEAK, Sub.STOPPING)
        await self.tts.cancel()                       # ① 停 TTS（≤30ms）
        self.player.stop()                            # ② flush 播放
        rolled = self.pre_roll.rollback()             # ③ pre-roll（采集侧）
        self._seg_frames = [rolled] if rolled.size else []
        self._user_text = ""
        self._set(State.LISTEN, Sub.VOICING)          # ④ 继续累积打断话语
        log.info("pre-roll 回滚 %d ms 进入 ASR 接管", len(rolled) // 480 * 30)

    def _end_turn(self) -> None:
        rec = TurnRecord(
            session_id=self.session.session_id, turn_id=self._turn_id,
            user_text=self._user_text, assistant_text=self._slow_text or self._fast_text,
            interrupt=self._interrupt_flag, interrupt_note=self._interrupt_note,
            ts_start=self._turn_start_ms, ts_end=int(time.time() * 1000),
            latency_ms={},
        )
        self.session.append_turn(rec)
        self._assistant_text = ""
        self._interrupt_flag = False

    # ---------- 工具 ----------
    def _set(self, state: State, sub: Sub) -> None:
        old = (self.state, self.sub)
        self.state, self.sub = state, sub
        for cb in self._listeners:
            try:
                cb(state, sub, old)
            except Exception:
                pass

    def _start_timer(self, name: str, ms: int, kind: str) -> None:
        self._cancel_timer(name)

        async def _fire():
            await asyncio.sleep(ms / 1000)
            self.on_event(Event(type=E_SESSION_TIMEOUT, domain="sess",
                                payload={"kind": kind, "timer": name}))

        self._timers[name] = asyncio.create_task(_fire())

    def _cancel_timer(self, name: str) -> None:
        t = self._timers.pop(name, None)
        if t and not t.done():
            t.cancel()

    def _to_event(self, d: VadDecision, frame) -> Event:
        domain = "vad"
        if d.event_type in ("speech_start", "speech_end", "eot", "takeover_noise",
                            "takeover_intent", "reject", "backchannel"):
            domain = "vad"
        return Event(type=d.event_type, domain=domain, payload=d.meta | {"confidence": d.confidence})
