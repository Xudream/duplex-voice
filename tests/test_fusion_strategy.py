"""回复融合策略测试（FusionStrategy：fastslow | direct 可插拔）。

设计：插上=快慢融合（承接语+慢回复），拔下=慢回复直达；两套独立。
"""
import pytest

from duplex_voice.adapter.fusion import (
    DirectStrategy, FastSlowStrategy, FusionStrategy, get_fusion_strategy,
)


def test_fastslow_starts_fast_for_commands():
    """快慢融合：指令类启动快通道（承接语）。"""
    s = FastSlowStrategy()
    assert s.name == "fastslow"
    assert s.should_fast("打开客厅的灯") is True
    assert s.should_fast("讲个故事") is True


def test_fastslow_prompt_mentions_transition():
    """快慢融合：慢通道提示词告知'已回应过'（防重复客套）+ 首句 15 字。"""
    p = FastSlowStrategy().slow_system_prompt()
    assert "已先用简短的过渡语" in p
    assert "首句尽量简短（15字内）" in p


def test_direct_never_starts_fast():
    """慢回复直达：恒不启动快通道（fast 模型零调用）——两套独立。"""
    s = DirectStrategy()
    assert s.name == "direct"
    assert s.should_fast("打开客厅的灯") is False
    assert s.should_fast("讲个故事") is False
    assert s.should_fast("随便什么") is False


def test_direct_prompt_direct_result():
    """慢回复直达：提示词首句直接给结果，无'已回应过'。"""
    p = DirectStrategy().slow_system_prompt()
    assert "已先用简短的过渡语" not in p
    assert "首句直接给出核心结果" in p


def test_unknown_mode_falls_back_fastslow():
    """未知模式 → 快慢融合（保守默认）。"""
    assert isinstance(get_fusion_strategy("unknown"), FastSlowStrategy)


def test_strategies_are_plug_isolated():
    """两套机制独立：direct 不依赖 FusionPolicy（无快通道逻辑耦合）。"""
    direct = DirectStrategy()
    fastslow = FastSlowStrategy()
    # 同一输入下两者决策互不影响
    assert direct.should_fast("打开灯") != fastslow.should_fast("打开灯")
    # 抽象接口完整
    assert isinstance(direct, FusionStrategy) and isinstance(fastslow, FusionStrategy)
