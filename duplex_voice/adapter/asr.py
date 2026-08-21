"""ASR Provider —— 统一接口 + Mock + Qwen3-ASR-Flash（百炼实测协议 2026-08-17）。

【实测协议（文档: qwen-asr-api-reference）】：
  POST {host}/compatible-mode/v1/chat/completions
  {"model":"qwen3-asr-flash",
   "messages":[{"role":"user","content":[{"type":"input_audio",
      "input_audio":{"data":"<公网URL 或 data:audio/wav;base64,...>"}}]}]}
  → choices[0].message.content = 识别文本
  （stream=True 可开流式增量；audio_tokens 计费）
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import urllib.request
from typing import AsyncIterator

import numpy as np

from ..events import Event

log = logging.getLogger(__name__)


class ASRProvider:  # Protocol
    async def stream_recognize(self, frames, *, language: str = "zh", partial: bool = True) -> AsyncIterator[Event]: ...
    async def end_segment(self) -> None: ...
    async def close(self) -> None: ...


class MockASRProvider:
    """模拟 ASR：识别固定文本（可配置），模拟 partial/final 时序。"""

    def __init__(self, text: str = "打开客厅的灯", confidence: float = 0.96,
                 partial_delay_ms: int = 300, final_delay_ms: int = 800):
        self.text = text
        self.confidence = confidence
        self.partial_delay_ms = partial_delay_ms
        self.final_delay_ms = final_delay_ms
        self._segment: list[np.ndarray] = []

    async def stream_recognize(self, frames, *, language="zh", partial=True) -> AsyncIterator[Event]:
        async for frame in frames:
            self._segment.append(frame)
        if partial:
            await asyncio.sleep(self.partial_delay_ms / 1000)
            yield Event(type="asr.partial", domain="asr",
                        payload={"text": self.text[:3] + "…", "confidence": self.confidence})
        await asyncio.sleep((self.final_delay_ms - self.partial_delay_ms) / 1000)
        yield Event(type="asr.final", domain="asr",
                    payload={"text": self.text, "confidence": self.confidence,
                             "duration_ms": len(self._segment) * 30})

    async def end_segment(self) -> None:
        pass

    async def close(self) -> None:
        self._segment.clear()


class Qwen3ASRProvider:
    """Qwen3-ASR-Flash 语音识别（百炼 compatible-mode chat/completions + input_audio）。

    实测：本地音频 base64 data URL 可用；识别文本在 choices[0].message.content。
    分段式：每次 stream_recognize 一整段（VAD 分段驱动）。
    key：config.dashscope_api_key 或 DASHSCOPE_API_KEY 环境变量
    """

    def __init__(self, host: str = "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com",
                 api_key: str = "", model: str = "qwen3-asr-flash"):
        self.host = host
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self._segment: list[np.ndarray] = []

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "缺少 DashScope API key：配置 model.dashscope_api_key 或环境变量 DASHSCOPE_API_KEY")

    def _opener(self):
        # 禁用系统代理（Clash）——DashScope 国内直连
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _recognize(self, audio_data_url: str) -> str:
        """同步识别（线程内执行 urllib）。"""
        self._ensure_key()
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_data_url}}]}],
        }).encode()
        req = urllib.request.Request(
            f"https://{self.host}/compatible-mode/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with self._opener().open(req, timeout=90) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    async def stream_recognize(self, frames, *, language="zh", partial=True) -> AsyncIterator[Event]:
        async for frame in frames:
            self._segment.append(frame.pcm if hasattr(frame, "pcm") else frame)
        if not self._segment:
            return
        pcm = np.concatenate(self._segment).astype(np.int16)
        b64 = base64.b64encode(pcm.tobytes()).decode()
        data_url = f"data:audio/wav;base64,{b64}"
        text = await asyncio.to_thread(self._recognize, data_url)
        duration_ms = len(self._segment) * 30
        if text:
            yield Event(type="asr.final", domain="asr",
                        payload={"text": text, "confidence": 1.0, "duration_ms": duration_ms})
        else:
            yield Event(type="asr.error", domain="asr",
                        payload={"code": "E_ASR_EMPTY", "retryable": True})

    async def end_segment(self) -> None:
        self._segment.clear()

    async def close(self) -> None:
        self._segment.clear()


class FunASRStreamProvider:
    """ASR 真流式（api-ws + fun-asr-flash-2026-06-15，SDK 源码级协议 2026-08-17 实测）。

    协议（与 SDK Recognition 完全一致）:
      run-task:  header{streaming:duplex, task_id, action:run-task}
                 payload{model, parameters{sample_rate,format,stream}, input:{},
                         task:"asr", task_group:"audio", function:"recognition"}
      音频帧:    裸二进制 PCM 块（12800 字节 = 400ms @16k——SDK 块大小）
      收尾:      finish-task + payload{input:{}}
      结果:      result-generated → output.sentence{sentence_id,begin_time,end_time,
                 text,sentence_end} —— sentence_end=False=partial / True=final
    实测: 帧流 → 流式返回句子（"打开客厅的灯。" 时间戳 [120-1350]）。
    """

    def __init__(self, host: str = "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com",
                 api_key: str = "", model: str = "fun-asr-flash-2026-06-15",
                 sample_rate: int = 16000, block_bytes: int = 12800):
        self.host = host
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.sample_rate = sample_rate
        self.block_bytes = block_bytes
        self._cancelled = False

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("缺少 DashScope API key")

    async def stream_recognize(self, frames, *, language="zh", partial=True) -> AsyncIterator[Event]:
        """帧流 → asr.partial / asr.final（真流式：帧边到边上行，句子边出边报）。"""
        import websockets
        self._ensure_key()
        url = f"wss://{self.host}/api-ws/v1/inference"
        task_id = f"t_{os.urandom(4).hex()}"
        buffer = bytearray()

        async with websockets.connect(url,
                                      additional_headers={"Authorization": f"Bearer {self.api_key}"},
                                      proxy=None, open_timeout=10, close_timeout=0) as ws:
            await ws.send(json.dumps({
                "header": {"streaming": "duplex", "task_id": task_id, "action": "run-task"},
                "payload": {"model": self.model,
                            "parameters": {"sample_rate": self.sample_rate,
                                           "format": "wav", "stream": True},
                            "input": {}, "task": "asr", "task_group": "audio",
                            "function": "recognition"}}))
            while True:
                d = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                evt = d.get("header", {}).get("event")
                if evt == "task-started":
                    break
                if evt == "task-failed":
                    log.error("ASR 启动失败: %s", d.get("header", {}))
                    yield Event(type="asr.error", domain="asr",
                                payload={"code": "E_ASR_CONN", "retryable": True})
                    return

            async def send_blocks():
                """帧 → 12800B 块实时上行（块间节流），发完立即 finish-task。"""
                async for frame in frames:
                    pcm = frame.pcm if hasattr(frame, "pcm") else frame
                    if isinstance(pcm, np.ndarray):
                        pcm = pcm.tobytes()
                    buffer.extend(pcm)
                    while len(buffer) >= self.block_bytes:
                        await ws.send(bytes(buffer[:self.block_bytes]))
                        del buffer[:self.block_bytes]
                        await asyncio.sleep(0.05)   # 节流（400ms 块按实时节奏）
                if buffer:
                    await ws.send(bytes(buffer))
                # 帧发完 → 立即 finish-task（服务端收到收尾才 finalize 出 final）
                try:
                    await ws.send(json.dumps({
                        "header": {"streaming": "duplex", "task_id": task_id,
                                   "action": "finish-task"},
                        "payload": {"input": {}}}))
                except Exception:
                    pass

            sender = asyncio.create_task(send_blocks())
            finished_task_sent = False

            async def handle_msg(d) -> tuple[list[Event], bool]:
                """处理一条服务端消息；返回 (产出事件, 是否结束)。"""
                nonlocal finished_task_sent
                evt = d.get("header", {}).get("event")
                out_events: list[Event] = []
                if evt == "result-generated":
                    out = d.get("payload", {}).get("output", {})
                    s = out.get("sentence") if isinstance(out, dict) else None
                    if s and s.get("text"):
                        if s.get("sentence_end"):
                            out_events.append(Event(type="asr.final", domain="asr",
                                                    payload={"text": s["text"], "confidence": 1.0,
                                                             "begin_ms": s.get("begin_time"),
                                                             "end_ms": s.get("end_time")}))
                        elif partial:
                            out_events.append(Event(type="asr.partial", domain="asr",
                                                    payload={"text": s["text"],
                                                             "begin_ms": s.get("begin_time")}))
                    return out_events, False
                if evt == "task-finished":
                    return out_events, True
                if evt == "task-failed":
                    log.error("ASR 任务失败: %s", d.get("header", {}))
                    return out_events, True
                return out_events, False

            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    except asyncio.TimeoutError:
                        # 帧已发完（finish-task 已在 send_blocks 尾部发出）
                        if sender.done():
                            break
                        continue
                    try:
                        d = json.loads(msg)
                    except Exception:
                        continue
                    evts, finished = await handle_msg(d)
                    for evt in evts:
                        yield evt
                        if evt.type == "asr.final":
                            finished = True   # final 已出 → 结束（不等 task-finished）
                    if finished:
                        break
            finally:
                sender.cancel()

    async def end_segment(self) -> None:
        pass

    async def close(self) -> None:
        pass
