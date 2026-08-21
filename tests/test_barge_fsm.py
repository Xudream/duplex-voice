"""打断决策状态机测试（BargeDecisionFSM 全转移表）。

设计分工：模型（VadJudge）只判断语义状态；打断与否由确定性事实
（AI 是否在播放 TTS）决定——语义 × 播放 → 行为。
"""
import pytest

from duplex_voice.fsm.barge_decision import BargeDecisionFSM, VadState


@pytest.fixture
def fsm():
    return BargeDecisionFSM()


# 全转移表：6 语义状态 × 2 播放状态 = 12 场景
CASES = [
    # (semantic, is_playing, 期望行为, 期望对外标签)
    (VadState.COMPLETE, True,  BargeDecisionFSM.BARGE_IN, VadState.BARGE_IN),   # 播放中完整指令 → 打断
    (VadState.COMPLETE, False, BargeDecisionFSM.RESPOND,  VadState.COMPLETE),   # 非播放完整指令 → 回复
    (VadState.INCOMPLETE, True,  BargeDecisionFSM.WAIT, VadState.INCOMPLETE),   # 未说完 → 等待续说
    (VadState.INCOMPLETE, False, BargeDecisionFSM.WAIT, VadState.INCOMPLETE),
    (VadState.BACKCHANNEL, True,  BargeDecisionFSM.IGNORE, VadState.BACKCHANNEL),  # 应声 → 忽略（不打断）
    (VadState.BACKCHANNEL, False, BargeDecisionFSM.IGNORE, VadState.BACKCHANNEL),
    (VadState.NOISE, True,  BargeDecisionFSM.IGNORE, VadState.NOISE),           # 噪声 → 忽略
    (VadState.NOISE, False, BargeDecisionFSM.IGNORE, VadState.NOISE),
    (VadState.TTS_ECHO, True,  BargeDecisionFSM.IGNORE, VadState.TTS_ECHO),     # 回声 → 忽略（继续播）
    (VadState.TTS_ECHO, False, BargeDecisionFSM.IGNORE, VadState.TTS_ECHO),
    (VadState.REJECT, True,  BargeDecisionFSM.IGNORE, VadState.REJECT),         # 播放中拒识 → 忽略
    (VadState.REJECT, False, BargeDecisionFSM.REJECT, VadState.REJECT),         # 非播放拒识 → 澄清
]


@pytest.mark.parametrize("semantic,is_playing,exp_action,exp_label", CASES,
                         ids=[f"{s}x{'playing' if p else 'idle'}" for s, p, _, _ in CASES])
def test_decide_full_table(fsm, semantic, is_playing, exp_action, exp_label):
    action, label = fsm.decide(semantic, is_playing)
    assert action == exp_action
    assert label == exp_label


def test_playing_decides_barge_not_model():
    """核心分工验证：complete 的打断与否完全由播放状态决定（模型不参与）。"""
    fsm = BargeDecisionFSM()
    assert fsm.decide(VadState.COMPLETE, True) == (BargeDecisionFSM.BARGE_IN, VadState.BARGE_IN)
    assert fsm.decide(VadState.COMPLETE, False) == (BargeDecisionFSM.RESPOND, VadState.COMPLETE)


def test_ignore_never_interrupts_playback():
    """backchannel/noise/echo 在任何播放状态下都不打断（continue 播放）。"""
    fsm = BargeDecisionFSM()
    for s in (VadState.BACKCHANNEL, VadState.NOISE, VadState.TTS_ECHO):
        assert fsm.decide(s, True)[0] == BargeDecisionFSM.IGNORE
        assert fsm.decide(s, False)[0] == BargeDecisionFSM.IGNORE
