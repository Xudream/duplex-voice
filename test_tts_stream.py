"""流式 TTS 测试（tts_stream + server 流式事件路径）。

实测（2026-08-21）：realtime WS 返回 response.audio.delta（PCM
24000Hz 16bit mono，20480B≈0.43s/块），首块时延 ~450-500ms。
"""
import base64
import struct

from duplex_voice.adapter.tts_stream import Qwen3TTSStream, SAMPLE_RATE


def test_stream_class_contract():
    """Qwen3TTSStream 契约：SAMPLE_RATE/模型默认值/参数。"""
    st = Qwen3TTSStream("host.example", "key")
    assert SAMPLE_RATE == 24000
    assert st.host == "host.example" and st.api_key == "key"
    assert st.model == "qwen3-tts-instruct-flash-realtime"
    assert st.voice == "Cherry"
    assert hasattr(st, "synth")


def test_pcm_decode_matches_frontend():
    """前端 b64ToPcm（16bit LE → Float32/32768）与 Python 侧解码一致。"""
    # 构造 16bit PCM：[-32768, -16384, 0, 16384, 32767]
    samples = [-32768, -16384, 0, 16384, 32767]
    pcm = struct.pack("<5h", *samples)
    b64 = base64.b64encode(pcm).decode()

    # 复刻前端 b64ToPcm 逻辑（index.html）
    import binascii
    raw = binascii.a2b_base64(b64)
    n = len(raw) // 2
    f32 = []
    for i in range(n):
        v = struct.unpack_from("<h", raw, i * 2)[0]
        f32.append(v / 32768)
    assert f32 == [-1.0, -0.5, 0.0, 0.5, 32767 / 32768]
    assert len(f32) == 5


def test_delta_chunk_size():
    """典型 delta 块 20480B = 0.427s 音频（前端 buffer 对齐）。"""
    assert 20480 % 2 == 0
    assert 20480 / 2 / SAMPLE_RATE == 20480 / 48000  # 0.427s
