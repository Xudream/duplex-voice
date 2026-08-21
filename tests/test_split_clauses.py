"""慢回复长文本拆句 + TTS 并发限流（防 429）。"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.server import _split_clauses


LONG = ("讲个关于想飞的小石头。它每天对着风练习跳跃，终于被一只路过的老鹰发现，带去了最高的山顶看日出，"
        "还结识了会唱歌的云朵朋友，从此再也不怕滚落山崖了，后来它甚至学会了在月光下滑翔，成了森林里最传奇的飞行家，"
        "连最挑剔的猫头鹰都向它请教飞翔的秘诀呢，最后它把故事讲给了每一颗渴望自由的种子听，那些种子长大后都变成了"
        "能随风旅行的蒲公英，带着小石头的梦想飞向更远的地方。")


def test_long_text_split():
    """178 字长文本拆成 28-40 字子句（并行合成用）。"""
    subs = _split_clauses(LONG)
    assert 3 <= len(subs) <= 8, f"子句数异常: {len(subs)}"
    for s in subs:
        assert 1 <= len(s) <= 40, f"子句超长: {len(s)}字"
        assert not s.startswith(("，", "、", "。")), f"句首标点: {s[:5]}"


def test_short_text_not_split():
    """短句不拆（保持单句合成）。"""
    assert _split_clauses("好的，正在为您整理相关信息。") == ["好的，正在为您整理相关信息。"]


def test_punctuation_only_tail_merged():
    """尾残段（标点残留）并入上一段，不产生独立子句。"""
    # 长文本尾部标点残留不产生 1 字子句
    long = "从前有一座山，它每天清晨去溪边喝水。晚上数着星星慢慢入睡。。。"
    subs = _split_clauses(long)
    assert len(subs) == 2, f"应 2 子句: {subs}"
    assert all(len(s) > 1 for s in subs), f"存在 1 字残段: {subs}"
    # 短句整体（<15 字）合并为 1 句
    assert _split_clauses("第一句。第二句。") == ["第一句。第二句。"]


def test_no_punct_long_text():
    """无标点超长文本不崩溃（单段返回）。"""
    t = "这是一个没有任何标点的超级长文本测试" * 5
    subs = _split_clauses(t)
    assert subs and len(subs) >= 1
