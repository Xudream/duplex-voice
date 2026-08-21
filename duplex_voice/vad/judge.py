"""VAD 判定层统一接口（软件设计 §2.2）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VadDecision:
    """判定结果（统一事件词汇：声学子集 ∪ 语义子集）。"""
    event_type: str            # speech_start | speech_end | eot | takeover_noise | ...
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)
    segment_id: str | None = None


class VadJudge(Protocol):
    """判定层接缝 —— 唯一替换点（Rule / Semantic 两实现）。"""

    def feed(self, frame: Any, phase: str) -> list[VadDecision]:
        """喂入 30ms 帧（AEC 后），phase: "listen" | "speak" | "yield"。

        返回本帧产生的判定列表（通常 0-1 个）。
        """
        ...

    def reset(self) -> None:
        """分段结束 / 状态切换时复位。"""
        ...
