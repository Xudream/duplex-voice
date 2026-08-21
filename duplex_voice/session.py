"""会话管理（软件设计 §2.7）：上下文维护 + JSONL 落盘。"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "你是家庭语音助手，用简短自然的中文回答。"


@dataclass(slots=True)
class TurnRecord:
    """会话单轮记录（JSONL 落盘单元，§3.2）。"""
    session_id: str
    turn_id: int
    user_text: str
    assistant_text: str
    interrupt: bool
    interrupt_note: str | None
    ts_start: int
    ts_end: int
    latency_ms: dict


class SessionManager:
    def __init__(self, jsonl_path: str, history_max_rounds: int = 20, history_max_tokens: int = 8000):
        self.session_id = f"s_{uuid.uuid4().hex[:8]}"
        self.jsonl_path = jsonl_path
        self.history_max_rounds = history_max_rounds
        self.history_max_tokens = history_max_tokens
        self._history: list[dict] = []   # [{role, content}]
        self._turns: list[TurnRecord] = []

    def build_messages(self, user_text: str, interrupt_note: str | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 历史裁剪：最近 N 轮
        msgs.extend(self._history[-self.history_max_rounds * 2:])
        if interrupt_note:
            user_text = f"{interrupt_note}\n{user_text}"
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def append_turn(self, rec: TurnRecord) -> None:
        self._turns.append(rec)
        if rec.user_text:
            self._history.append({"role": "user", "content": rec.user_text})
        if rec.assistant_text:
            self._history.append({"role": "assistant", "content": rec.assistant_text})
        # 简单 token 裁剪（字符/4 近似）
        while sum(len(m["content"]) for m in self._history) > self.history_max_tokens * 4:
            self._history.pop(0)
        # JSONL 落盘（§2.7：每 turn）；":memory:" 表示不落盘
        if self.jsonl_path == ":memory:":
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "session_id": rec.session_id, "turn_id": rec.turn_id,
                    "user_text": rec.user_text, "assistant_text": rec.assistant_text,
                    "interrupt": rec.interrupt, "interrupt_note": rec.interrupt_note,
                    "ts_start": rec.ts_start, "ts_end": rec.ts_end,
                    "latency_ms": rec.latency_ms,
                }, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("JSONL 落盘失败（纯内存运行）: %s", e)
