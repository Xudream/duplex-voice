"""回复融合策略（可插拔）：fastslow（快慢融合）| direct（慢回复直达）。

- FastSlowStrategy（插上）：快通道承接语（本地 Ollama 4b）+ 慢通道完整回复，
  承接语先播遮挡生成时延（现状行为）
- DirectStrategy（拔下）：仅慢通道完整回复直达，无承接语（无 fast 调用）

两套机制独立互不影响：策略只决定"是否启动快通道"与"慢通道提示词"；
播放链 idx 偏移由 fast_tts_url 是否产生自适应（direct 模式恒无 fast → 偏移 0）。
"""


class FusionStrategy:
    """融合策略抽象：慢通道提示词 + 是否启动快通道。"""

    name = "base"

    def slow_system_prompt(self) -> str:
        """慢通道 system prompt（fastslow 告知'已回应过'防重复；direct 直接给结果）。"""
        raise NotImplementedError

    def should_fast(self, text: str) -> bool:
        """是否启动快通道承接语（direct 恒 False——不调用 fast 模型）。"""
        raise NotImplementedError


class FastSlowStrategy(FusionStrategy):
    """快慢融合（默认）：承接语遮挡生成时延 + 慢回复接播。"""

    name = "fastslow"

    def slow_system_prompt(self) -> str:
        return ("你是智能家居语音助手，回复简洁口语化，不超过两句话。"
                "注意：AI 已先用简短的过渡语（如'好的，马上为您处理'）回应过用户，你回复时"
                "不要再说'好的''嗯''没问题'等开头客套，直接给出具体内容或结果。"
                "要求首句尽量简短（15字内）直接给结果（首句短→TTS 合成快，避免播放停顿），"
                "细节放第二句。")

    def should_fast(self, text: str) -> bool:
        # 快慢融合的判断沿用 FusionPolicy（指令类才发承接语）
        from duplex_voice.adapter.llm import FusionPolicy
        return FusionPolicy.should_speak(text)


class DirectStrategy(FusionStrategy):
    """慢回复直达：无承接语，慢通道直接给完整回复。"""

    name = "direct"

    def slow_system_prompt(self) -> str:
        return ("你是智能家居语音助手，回复简洁口语化，不超过两句话。"
                "首句直接给出核心结果，细节放第二句。")

    def should_fast(self, text: str) -> bool:
        return False


STRATEGIES = {
    "fastslow": FastSlowStrategy(),
    "direct": DirectStrategy(),
}


def get_fusion_strategy(mode: str) -> FusionStrategy:
    """按模式名取策略（未知 → 默认快慢融合，保守）。"""
    return STRATEGIES.get(mode, STRATEGIES["fastslow"])
