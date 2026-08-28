"""手动脚本：连接 /ws/app，流式发送 test_speech_16k.wav 的 PCM 帧，打印收到的所有事件。
用法: python3 tests/manual_ws_client_test.py
"""
import asyncio
import base64
import json
import struct
import wave

import websockets

WS_URL = "ws://127.0.0.1:8787/ws/app"
WAV_PATH = "tests/assets/test_speech_16k.wav"
FRAME_SAMPLES = 480  # 30ms @ 16k


async def main():
    with wave.open(WAV_PATH, "rb") as wf:
        assert wf.getframerate() == 16000, wf.getframerate()
        assert wf.getsampwidth() == 2
        pcm = wf.readframes(wf.getnframes())

    frame_bytes = FRAME_SAMPLES * 2
    frames = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    print(f"总帧数: {len(frames)}, 总时长约 {len(frames) * 30}ms")

    events = []

    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "auth"}))

        async def receiver():
            try:
                async for msg in ws:
                    d = json.loads(msg)
                    et = d.get("type")
                    events.append(et)
                    if et in ("audio_chunk",):
                        print(f"<< {et} idx={d.get('idx')}")
                    else:
                        print(f"<< {json.dumps({k:v for k,v in d.items() if k!='b64'}, ensure_ascii=False)[:200]}")
                    if et == "done":
                        break
            except websockets.exceptions.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        # 发送音频帧 + 结尾静音以触发 speech_end
        for i, f in enumerate(frames):
            await ws.send(json.dumps({"type": "audio_frame", "pcm_b64": base64.b64encode(f).decode()}))
            await asyncio.sleep(0.03)

        # 追加约1.5秒静音帧，触发声学VAD的 speech_end
        silence_frame = b"\x00\x00" * FRAME_SAMPLES
        for i in range(50):
            await ws.send(json.dumps({"type": "audio_frame", "pcm_b64": base64.b64encode(silence_frame).decode()}))
            await asyncio.sleep(0.03)

        # 等待收到 done 或超时
        try:
            await asyncio.wait_for(recv_task, timeout=30)
        except asyncio.TimeoutError:
            print("!!! 超时，未收到 done 事件")
            recv_task.cancel()

    print("\n=== 事件类型统计 ===")
    from collections import Counter
    print(Counter(events))


if __name__ == "__main__":
    asyncio.run(main())
