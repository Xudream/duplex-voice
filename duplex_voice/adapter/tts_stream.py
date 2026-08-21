"""Qwen3-TTS 流式合成（realtime WS：边合成边返回 PCM 块）。

协议（实测 2026-08-21，dashscope qwen_tts_realtime SDK 同源）：
- WS: wss://{host}/api-ws/v1/realtime?model={model}
- 上行: session.update（voice/response_format=pcm/sample_rate=24000/
  mode=server_commit）→ input_text_buffer.append(text) → session.finish
- 下行: session.created → response.created → response.audio.delta(PCM块)
  → response.audio.done → response.done
- 音频: PCM 24000Hz 16bit mono（20480B≈0.43s/块）

首块时延实测 ~300-800ms（文本提交 → 首个 delta）——相比整段合成+
下载（1-3s），流式可显著降低'说完→开始播'。
"""
import asyncio
import base64
import json
import uuid

SAMPLE_RATE = 24000
CHUNK_DELTA = 20480          # 典型 delta 字节数（0.427s 音频）
AUTH_HEADERS = {}            # 由调用方注入（web/server.py 或 provider）


class Qwen3TTSStream:
    """流式 TTS 会话（每句一个 WS，server_commit 模式）。"""

    def __init__(self, host: str, api_key: str, model: str = "qwen3-tts-instruct-flash-realtime",
                 voice: str = "Cherry"):
        self.host = host
        self.api_key = api_key
        self.model = model
        self.voice = voice

    async def synth(self, text: str, chunk_cb):
        """流式合成：chunk_cb(pcm_bytes) 逐块回调（边合成边返回）。

        返回首块时延 ms；异常抛给调用方（可回退整段合成）。
        """
        import websockets
        url = f"wss://{self.host}/api-ws/v1/realtime?model={self.model}"
        t0 = asyncio.get_event_loop().time()
        first_latency = None
        async with websockets.connect(url, proxy=None, open_timeout=10,
                                      additional_headers={"Authorization": f"Bearer {self.api_key}"}) as ws:
            await ws.send(json.dumps({
                "event_id": "ev_" + uuid.uuid4().hex[:8],
                "type": "session.update",
                "session": {"voice": self.voice, "response_format": "pcm",
                            "sample_rate": SAMPLE_RATE, "mode": "server_commit",
                            "volume": 50}}))
            await ws.send(json.dumps({
                "event_id": "ev_" + uuid.uuid4().hex[:8],
                "type": "input_text_buffer.append",
                "text": text}))
            await ws.send(json.dumps({
                "event_id": "ev_" + uuid.uuid4().hex[:8],
                "type": "session.finish"}))
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=15)
                except asyncio.TimeoutError:
                    raise TimeoutError("TTS 流式超时")
                if isinstance(msg, bytes):
                    if first_latency is None:
                        first_latency = int((asyncio.get_event_loop().time() - t0) * 1000)
                    chunk_cb(msg)   # 同步回调（put_nowait 不阻塞）
                    continue
                try:
                    evt = json.loads(msg)
                except Exception:
                    continue
                t = evt.get("type", "")
                if t == "response.audio.delta":
                    if first_latency is None:
                        first_latency = int((asyncio.get_event_loop().time() - t0) * 1000)
                    chunk_cb(base64.b64decode(evt.get("delta", "")))
                elif t == "response.done":
                    break
                elif t == "error":
                    raise RuntimeError(f"TTS 流式错误: {str(evt)[:150]}")
        return first_latency
