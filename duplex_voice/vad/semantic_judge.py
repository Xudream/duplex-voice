"""SemanticVadJudge —— 语义档（插上=全双工，P1）。

第一版占位：配置 vad.judge="semantic" 时加载 WS 推理服务客户端（SoulX-Duplug 同构），
返回 8 状态 token 判定。训练方案见《语义VAD训练设计方案》。
"""
from __future__ import annotations

import logging

from ..config import VadConfig
from .judge import VadDecision

log = logging.getLogger(__name__)


class SemanticVadJudge:
    """8 状态 token 模型判定（eot/takeover_intent/reject/addressed_to_me/backchannel）。"""

    def __init__(self, cfg: VadConfig, ws_url: str | None = None):
        self.cfg = cfg
        self.ws_url = ws_url
        self._client = None  # P1: WebSocket 客户端
        log.info("SemanticVadJudge 占位：P1 接入 8 状态 token 推理服务 %s", ws_url)

    def feed(self, frame, phase: str | None = None) -> list[VadDecision]:
        # P1: 音频帧 + ASR 文本引导 → 模型 → token 判定
        # 第一版：未训练，返回空（系统以 rule 档兜底运行）
        return []

    def reset(self) -> None:
        pass
