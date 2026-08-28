"""_local_tts 分段计时测试（公网下载时延单独显示 dl_ms）。

设计：_local_tts 返回 (本地URL, 合成耗时ms, 公网下载耗时ms)——
合成 API 与云端音频下载分别计时；缓存命中均 0（无公网调用）。
"""
import time
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
_SRC = (WEB / "server.py").read_text(encoding="utf-8")
_start = _SRC.index("def _local_tts")
_end = _SRC.index("def _strip_leading_filler")


def _make_local_tts(tmp_path, tts_mock):
    # 函数内 import urllib.request 拿真实模块——用 monkeypatch 替换 urlopen
    ns = {"tts": tts_mock, "TTS_CACHE": tmp_path / "tts_cache", "time": time,
          "TTS_VOICE": "Cherry"}   # 面板音色全局（修复后 _local_tts 缺省取它）
    exec(_SRC[_start:_end], ns)
    return ns["_local_tts"]


def _fake_urlopen(monkeypatch, data=b"audio-data"):
    class FakeResp:
        def __init__(self):
            self._d = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._d

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=30: FakeResp())


@pytest.fixture
def env(tmp_path, monkeypatch):
    tts_mock = MagicMock()
    tts_mock._synthesize_url.return_value = "http://remote.example/audio.wav"
    _fake_urlopen(monkeypatch)
    return _make_local_tts(tmp_path, tts_mock), tts_mock


def test_first_call_returns_tuple_with_dl(env):
    """首次：返回 (本地URL, 合成耗时, 下载耗时)——合成/下载分别计时。"""
    local_tts, tts_mock = env
    url, tts_ms, dl_ms = local_tts("打开客厅的灯")
    assert url.startswith("/tts_cache/") and url.endswith(".wav")
    assert isinstance(tts_ms, int) and tts_ms >= 0
    assert isinstance(dl_ms, int) and dl_ms >= 0
    assert tts_mock._synthesize_url.called  # 云端合成被调用


def test_cache_hit_returns_zero(tmp_path, monkeypatch):
    """缓存命中：不再调云端合成，合成/下载均 0（无公网调用）。"""
    tts_mock = MagicMock()
    tts_mock._synthesize_url.return_value = "http://remote.example/audio.wav"
    _fake_urlopen(monkeypatch)
    local_tts = _make_local_tts(tmp_path, tts_mock)
    url1, tts1, dl1 = local_tts("讲个故事")
    calls = tts_mock._synthesize_url.call_count
    url2, tts2, dl2 = local_tts("讲个故事")
    assert url1 == url2
    assert tts2 == 0 and dl2 == 0          # 缓存命中零公网
    assert tts_mock._synthesize_url.call_count == calls  # 未再次合成
    assert list((tmp_path / "tts_cache").glob("*.wav"))  # 本地文件已写入


def test_different_text_different_cache(tmp_path, monkeypatch):
    """不同文本不同缓存文件（hash 隔离）。"""
    tts_mock = MagicMock()
    tts_mock._synthesize_url.return_value = "http://remote.example/a.wav"
    _fake_urlopen(monkeypatch)
    local_tts = _make_local_tts(tmp_path, tts_mock)
    u1, _, _ = local_tts("打开灯")
    u2, _, _ = local_tts("关灯")
    assert u1 != u2
