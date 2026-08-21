"""打断决策状态机：语义状态 × 播放状态 → 系统行为。

分工：模型（VadJudge）只判断"语义状态"（是否要响应）；
打断与否由确定性事实决定——AI 当前是否在播放 TTS（前端 is_replying 上报）。

转移表（semantic × is_playing → action, label）：

    semantic      | is_playing=True          | is_playing=False
    --------------|--------------------------|------------------
    complete      | barge_in（打断接管）      | respond（正常回复）
    incomplete    | wait（等待续说）          | wait
    backchannel   | ignore（继续播，无感）    | ignore
    noise         | ignore                   | ignore
    tts_echo      | ignore（继续播）          | ignore
    reject        | ignore（播放中不澄清）    | reject（澄清）
"""


class VadState:
    """语义 VAD 状态 token（8 状态设计：完整/未完成/应声/打断/噪声/回声/拒识）。"""

    COMPLETE = "complete"        # 完整指令，等待回复
    INCOMPLETE = "incomplete"    # 语义不完整（没说完）
    BACKCHANNEL = "backchannel"  # 应声（嗯/对），系统不应回应
    BARGE_IN = "barge_in"        # 打断 AI 播放且内容完整（接管）
    NOISE = "noise"              # 无意义/转录噪声/幻觉
    TTS_ECHO = "tts_echo"        # 回声（与刚播放内容几乎逐字一致）
    REJECT = "reject"            # 拒识（听不清，请求重复）


class BargeDecisionFSM:
    """打断决策状态机：语义状态 × 播放状态 → (行为, 对外状态标签)。"""

    # 行为常量
    RESPOND = "respond"     # 触发回复
    BARGE_IN = "barge_in"   # 打断播放 + 触发回复
    WAIT = "wait"           # 不回复，等续说
    IGNORE = "ignore"       # 不回复不打断
    REJECT = "reject"       # 澄清（"没听清，再说一遍"）

    def decide(self, semantic: str, is_playing: bool) -> tuple[str, str]:
        """输入语义状态 + 播放状态 → (行为, 对外状态标签)。"""
        if semantic == VadState.COMPLETE:
            return (self.BARGE_IN, VadState.BARGE_IN) if is_playing else (self.RESPOND, VadState.COMPLETE)
        if semantic == VadState.INCOMPLETE:
            return self.WAIT, VadState.INCOMPLETE
        if semantic == VadState.REJECT:
            return (self.IGNORE, VadState.REJECT) if is_playing else (self.REJECT, VadState.REJECT)
        # backchannel / noise / tts_echo → 一律忽略（不打断、不回复）
        return self.IGNORE, semantic
