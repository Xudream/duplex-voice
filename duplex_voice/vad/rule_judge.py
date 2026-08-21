"""RuleVadJudge —— 声学规则档（缺省=半双工，软件设计 §2.2 实现 A + §3.3）。

Silero VAD v6 双实例 + 阶段感知双钳制：
- segment 实例（listen 期）：判停分段（speech_start/speech_end）
- interrupt 实例（speak 期）：声学打断（takeover_noise）——系统基础能力，P0 必备

规则（Hermes 源码级验证）：
① 噪声底校准：仅 listen 期前 450ms 安静块 pct90 RMS = quiet_floor（播放期不重校准）
② 双钳制：trigger = max(quiet_floor×4, 2×SILENCE_RMS)，上限 = quiet_floor + 18dB
③ 播放起始 grace 500ms（条件化）
④ 窗口多数：最近 300ms 内 ≥80% 帧超触发
⑤ 触发后继续录 1250ms 静音 → 交付完整打断话语
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

from ..config import VadConfig
from .judge import VadDecision

log = logging.getLogger(__name__)

# 帧长常量（16k/30ms）
FRAME_LEN = 480


class RuleVadJudge:
    """Silero 双实例声学判定。第一版用 webrtcvad 作可测替代 + 能量双钳制
    （Silero ONNX 推理放 P0a 尾接入，接口不变）。"""

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._segment = _SileroInstance(
            threshold=cfg.threshold_segment,
            min_silence_ms=cfg.min_silence_ms,
            min_speech_ms=cfg.min_speech_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )
        self._interrupt = _SileroInstance(
            threshold=cfg.threshold_interrupt,
            min_silence_ms=300,
            min_speech_ms=cfg.min_speech_ms,
            speech_pad_ms=0,
        )
        # 双钳制状态（§3.3）
        self._quiet_floor: float | None = None
        self._floor_locked = False
        self._floor_frames: list[float] = []
        self._playback_grace_until: int = 0
        self._last_playback_end_ms: int | None = None
        self._window: deque[bool] = deque(maxlen=10)  # 300ms / 30ms = 10 帧
        self._pending_trigger = False                  # 已命中窗口，等待话语收尾
        self._tail_silence_ms = 0
        self._phase = "listen"
        self._seq = 0

    # ---- 状态接口 ----
    def set_phase(self, phase: str) -> None:
        """FSM 下发 phase：listen → segment 实例；speak → interrupt 实例。"""
        self._phase = phase
        if phase == "speak":
            # 播放起始 grace（条件化：播放前 ≥1s 真实间隔才给）
            pass  # 由 playback_start() 显式设置
        elif phase == "listen":
            self._window.clear()

    def playback_start(self, now_ms: int, gap_ms: int | None = None) -> None:
        """播放开始：条件化 grace 500ms（播放前 ≥1s 间隔）。"""
        if gap_ms is None or gap_ms >= 1000:
            self._playback_grace_until = now_ms + self.cfg.grace_ms
        else:
            self._playback_grace_until = 0

    def playback_end(self, now_ms: int) -> None:
        self._last_playback_end_ms = now_ms

    # ---- 主入口 ----
    def feed(self, frame, phase: str | None = None) -> list[VadDecision]:
        """frame: AudioFrame 或 np.ndarray(480 int16)。phase 覆盖时切换实例。"""
        if phase is not None:
            self.set_phase(phase)
        pcm = frame.pcm if hasattr(frame, "pcm") else frame
        if pcm.shape[0] != FRAME_LEN:
            pcm = pcm[:FRAME_LEN] if pcm.shape[0] > FRAME_LEN else np.pad(pcm, (0, FRAME_LEN - pcm.shape[0]))
        now_ms = getattr(frame, "ts_ms", 0)
        rms_db = _rms_db(pcm)

        if self._phase == "speak":
            return self._interrupt_feed(pcm, rms_db, now_ms)
        return self._segment_feed(pcm, rms_db, now_ms)

    # ---- segment 实例（listen 期）----
    def _segment_feed(self, pcm, rms_db: float, now_ms: int) -> list[VadDecision]:
        out: list[VadDecision] = []
        # 噪声底校准（仅 listen 期，未锁定前）
        if not self._floor_locked:
            self._floor_frames.append(rms_db)
            if len(self._floor_frames) >= 15:  # 450ms
                q = np.percentile(self._floor_frames, 90)
                self._quiet_floor = q
                self._floor_locked = True
                log.info("quiet_floor 校准完成: %.1f dB", q)
        # Silero 判定（此处 webrtcvad 近似；Silero ONNX 接入点）
        is_speech = self._segment.vad(pcm)
        out.extend(self._segment.on_frame(is_speech, now_ms))
        return out

    # ---- interrupt 实例（speak 期，声学打断）----
    def _interrupt_feed(self, pcm, rms_db: float, now_ms: int) -> list[VadDecision]:
        out: list[VadDecision] = []
        if now_ms < self._playback_grace_until:
            return out  # 播放起始 grace
        if self._quiet_floor is None:
            self._quiet_floor = -45.0  # 未校准默认
        # 双钳制（§3.3 ②）
        floor = self._quiet_floor
        trigger_lo = max(floor + 12.0, 2 * _SILENCE_RMS)   # 下限防泄漏（×4 ≈ +12dB）
        trigger_hi = floor + self.cfg.interrupt_ceiling_db  # 上限保证真人语音可达
        trigger = min(max(trigger_lo, -35.0), trigger_hi)
        hit = rms_db >= trigger and self._segment.is_speech_likely(pcm)
        self._window.append(hit)

        if self._pending_trigger:
            # 已命中：继续录到 1250ms 静音 → 交付完整打断话语（§3.3 ⑤）
            if hit:
                self._tail_silence_ms = 0
            else:
                self._tail_silence_ms += 30
            if self._tail_silence_ms >= 1250:
                self._pending_trigger = False
                self._tail_silence_ms = 0
                self._window.clear()
                out.append(VadDecision(
                    event_type="takeover_noise", confidence=1.0,
                    meta={"rms_db": rms_db, "trigger_db": trigger,
                          "window_ratio": 1.0, "quiet_floor": floor},
                ))
            return out

        # 窗口多数（§3.3 ④）：300ms 内 ≥80%
        if len(self._window) == self._window.maxlen:
            ratio = sum(self._window) / len(self._window)
            if ratio >= self.cfg.window_ratio:
                self._pending_trigger = True
                self._tail_silence_ms = 0
                log.debug("打断窗口命中 ratio=%.2f (trigger=%.1fdB)", ratio, trigger)
        return out

    def is_voicing(self) -> bool:
        """judge 内部语音状态（帧收集用，不依赖事件循环状态）。"""
        return self._segment._in_speech

    def reset(self) -> None:
        self._segment.reset()
        self._interrupt.reset()
        self._window.clear()
        self._pending_trigger = False
        self._tail_silence_ms = 0


# ---------- Silero 实例封装 ----------

class _SileroInstance:
    """单实例：Silero VAD v6 判定 + 静音计时（2026-08-18 集成真 v6：torch 2.10 支持 3.14）。

    引擎链：silero_v6（神经网络，固定 512 帧输入）→ webrtcvad → 能量+过零率。
    判定粒度：采集帧 480 样本缓冲到 512（32ms@16k）再喂 Silero，state 内部保持。
    """

    def __init__(self, threshold: float, min_silence_ms: int, min_speech_ms: int, speech_pad_ms: int):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.speech_pad_ms = speech_pad_ms
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False
        self._silero = None
        self._vad512 = bytearray()   # 480→512 缓冲（Silero 固定输入）
        self._last_vad = False       # 未判定帧保持上次结果（窗口多数不被稀释）
        try:
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad()
            log.info("Silero VAD v6 引擎就绪（threshold=%.2f）", threshold)
        except Exception as e:
            log.warning("silero_vad 不可用（%s），回退 webrtcvad 近似", e)
        self._webrtc = None
        try:
            import webrtcvad
            self._webrtc = webrtcvad.Vad(2)
        except Exception:
            pass

    def vad(self, pcm: np.ndarray) -> bool:
        if self._silero is not None:
            try:
                # 480 帧缓冲到 512（Silero v6 固定输入长度）；未满帧保持上次判定
                self._vad512 += pcm.astype(np.int16).tobytes()
                if len(self._vad512) < 1024:
                    return self._last_vad
                chunk = np.frombuffer(self._vad512[:1024], dtype=np.int16)
                del self._vad512[:1024]
                self._last_vad = bool(self._silero(torch_tensor(chunk), 16000).item() > self.threshold)
                return self._last_vad
            except Exception as e:
                log.debug("silero 推理失败，回退: %s", e)
                self._silero = None
        if self._webrtc is not None:
            try:
                return self._webrtc.is_speech(pcm.tobytes(), 16000)
            except Exception:
                self._webrtc = None
        # 最终兜底：能量 + 过零率（无外部依赖；生产可换真 VAD）
        p = pcm.astype(np.float64)
        rms = float(np.sqrt(np.mean(p ** 2)))
        zc = float(np.mean(np.abs(np.diff(p)) > 200))
        return rms > 400 and zc > 0.03

    def is_speech_likely(self, pcm: np.ndarray) -> bool:
        return self.vad(pcm)

    def on_frame(self, is_speech: bool, now_ms: int) -> list[VadDecision]:
        out: list[VadDecision] = []
        if is_speech:
            self._speech_ms += 30
            self._silence_ms = 0
            if not self._in_speech and self._speech_ms >= self.min_speech_ms:
                self._in_speech = True
                out.append(VadDecision(event_type="speech_start", confidence=1.0,
                                       meta={"ts_ms": now_ms}))
        else:
            self._silence_ms += 30
            if self._in_speech and self._silence_ms >= self.min_silence_ms:
                self._in_speech = False
                out.append(VadDecision(event_type="speech_end", confidence=1.0,
                                       meta={"ts_ms": now_ms, "duration_ms": self._speech_ms}))
                self._speech_ms = 0
        return out

    def reset(self) -> None:
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False
        self._vad512.clear()
        if self._silero is not None:
            try:
                self._silero.reset_states()   # LSTM state 重置（长静音后防残留）
            except Exception:
                pass


# ---------- 工具 ----------

_SILENCE_RMS = -50.0  # 绝对静音底（dBFS）


def _rms_db(pcm: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-6)) if rms > 0 else -120.0


def torch_tensor(pcm: np.ndarray):
    """np.int16(480) → torch float32 tensor（silero_vad 输入）。"""
    import torch
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)
