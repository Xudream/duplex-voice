"""入口：装配 + 启动（P0a-P0c 第一版可运行闭环）。

用法:
    python -m duplex_voice.main            # 真实麦克风（mock 模型）
    python -m duplex_voice.main --text     # 无麦克风文本驱动模式（CI/冒烟）
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .adapter.registry import build_providers
from .audio import AudioCapture, AudioPlayer, PreRollBuffer
from .config import Config
from .events import Event
from .fsm import DuplexFSM
from .session import SessionManager
from .vad import RuleVadJudge, SemanticVadJudge

log = logging.getLogger(__name__)


def build_fsm(cfg: Config, text_mode: bool = False) -> DuplexFSM:
    capture = AudioCapture(cfg.audio)
    player = AudioPlayer()
    judge: RuleVadJudge | SemanticVadJudge
    if cfg.vad.judge == "semantic":
        judge = SemanticVadJudge(cfg.vad, cfg.model.semantic_vad_url)
        log.warning("semantic 档为 P1 占位，实际以 rule 档行为兜底")
    else:
        judge = RuleVadJudge(cfg.vad)
    providers = build_providers(cfg.model)
    session = SessionManager(cfg.session_jsonl, cfg.fsm.history_max_rounds,
                             cfg.fsm.history_max_tokens)
    pre_roll = PreRollBuffer(cfg.vad.pre_roll_ms, cfg.audio.frame_ms, cfg.audio.sample_rate)
    return DuplexFSM(cfg, capture, player, judge,
                     providers["asr"], providers["llm_fast"], providers["llm_slow"],
                     providers["tts"], session, pre_roll)


async def audio_loop(fsm: DuplexFSM) -> None:
    """T1→T2：采集帧 → FSM。"""
    fsm.capture.start()
    fsm.player.start()
    try:
        async for frame in fsm.capture.frames():
            fsm.feed_audio(frame)
    finally:
        fsm.capture.stop()
        fsm.player.close()


async def text_loop(fsm: DuplexFSM) -> None:
    """文本驱动模式（无麦克风）：stdin 输入模拟用户话语。"""
    fsm.player.start()
    print(">>> 文本模式（输入 q 退出）：")
    while True:
        line = (await asyncio.to_thread(sys.stdin.readline)).strip()
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            fsm.stop()
            break
        # 模拟 VAD 事件链：speech_start → (帧) → speech_end → final
        fsm.on_event(Event(type="speech_start", domain="vad", payload={}))
        fsm.on_event(Event(type="speech_end", domain="vad", payload={}))
        fsm.on_event(Event(type="asr.final", domain="asr",
                           payload={"text": line, "confidence": 0.99}))
        await asyncio.sleep(0.1)
    fsm.player.close()


async def main() -> None:
    ap = argparse.ArgumentParser(description="级联全双工语音交互系统 v0.1")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--text", action="store_true", help="文本驱动模式（无麦克风）")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)
    fsm = build_fsm(cfg, text_mode=args.text)
    fsm.on_state_change(lambda s, sub, old: log.info("状态: %s/%s", s.value, sub.value))

    loop_task = text_loop(fsm) if args.text else audio_loop(fsm)
    try:
        await asyncio.gather(loop_task, fsm.run())
    except KeyboardInterrupt:
        log.info("退出")


if __name__ == "__main__":
    asyncio.run(main())
