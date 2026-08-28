"""WS 端点 /api/stream——App 客户端双工通道。

协议（JSON 消息，双向）：

客户端 → 服务端：
  {"type":"hello","token":"dvt_xxx","client_id":"..."}          # 首条：鉴权+绑定
  {"type":"audio","b64":"<16kHz mono s16le 分块>","is_final":false}
  {"type":"audio","b64":"...","is_final":true}                 # 末块（判停后发）
  {"type":"text","text":"..."}                                  # 文本输入（等价 /api/chat_text）
  {"type":"cancel"}                                             # 取消当前轮

服务端 → 客户端：
  {"type":"ready","device_id":"..."}                            # 鉴权通过
  {"type":"auth_error"}                                         # 鉴权失败（1s 后断开）
  {"type":"asr_partial","text":"..."}                           # 流式 ASR 中间结果
  {"type":"asr_final","text":"..."}                             # ASR 最终结果
  {"type":"vad","state":"complete|noise|barge_in|..."}          # 语义 VAD 判定
  {"type":"fast","text":"...","ms":123}                         # 快通道承接语
  {"type":"slow_first","text":"...","ms":4567}
  {"type":"slow_delta","text":"..."}                            # 慢通道流式增量
  {"type":"tts","b64":"<wav>","sample_rate":24000,"voice":"Cherry"}  # TTS 音频（流式句）
  {"type":"latency","fast_ms":7,"slow_first_ms":4544,...}       # 时延汇总（前端面板）
  {"type":"done"}                                               # 本轮结束
  {"type":"error","message":"..."}

音频路径：客户端采集（含系统 AEC）→ 16kHz mono s16le → base64 分块（~320ms/块）上行；
TTS 音频服务端本地合成后 base64 下发（复用 _local_tts 缓存），客户端 AudioTrack/AVAudioEngine 播放。
"""
import asyncio
import base64
import json
import time
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

import device_auth


def _auth_required() -> bool:
    """读 config.json server.auth.required（与 /api/config 热生效联动）。"""
    try:
        cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))
        return bool(cfg.get("server", {}).get("auth", {}).get("required", False))
    except Exception:
        return False


async def stream_ws(websocket: WebSocket, server_module) -> None:
    """主 WS 处理循环。server_module=web.server 模块（复用其管道/全局）。

    设计：轻壳——本文件只做协议编解码与连接管理，业务全部调用 server.py
    现有函数（chat 管道/ASR/TTS），不复制逻辑。
    """
    await websocket.accept()
    sm = server_module
    client_id = None
    authed = False

    def send(obj: dict) -> asyncio.Task:
        return asyncio.create_task(websocket.send_text(json.dumps(obj, ensure_ascii=False)))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await send({"type": "error", "message": "bad json"})
                continue

            t = msg.get("type")

            # ---- 鉴权（首条 hello）----
            if t == "hello":
                token = msg.get("token", "")
                if _auth_required():
                    did = device_auth.verify(token)
                    if not did:
                        await send({"type": "auth_error"})
                        await asyncio.sleep(1)
                        return
                client_id = msg.get("client_id") or f"ws_{int(time.time()*1000)%100000}"
                authed = True
                await send({"type": "ready",
                            "device_id": device_auth.verify(token) if token else None,
                            "client_id": client_id,
                            "vad_mode": getattr(sm, "VAD_JUDGE_MODE", "rule"),
                            "fusion": getattr(sm, "FUSION_MODE", {}).get("mode")})
                continue

            if not authed:
                await send({"type": "error", "message": "hello first"})
                continue
            assert client_id is not None

            # ---- 音频轮（采集分块 → 判停后 is_final → 走 chat 管道）----
            if t == "audio":
                # 流式 ASR：分块喂入（复用 server 的 ASR 实例与续说合并逻辑）
                is_final = bool(msg.get("is_final", False))
                await send({"type": "asr_partial", "text": ""})   # 占位：P2 接 server 流式 ASR 逐块反馈
                if is_final:
                    # 末块到达 = 客户端已判停 -> 触发完整管道（VAD 判断/快慢融合/TTS）
                    # P2：把 b64 累积写入临时文件走 server 的音频 ASR + judge_audio
                    await send({"type": "done"})
                continue

            # ---- 文本轮（等价 /api/chat_text，复用同一管道）----
            if t == "text":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                await _run_chat_text(sm, websocket, send, client_id, text)
                continue

            if t == "cancel":
                await send({"type": "done"})
                continue

            await send({"type": "error", "message": f"unknown type {t}"})

    except WebSocketDisconnect:
        return
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await send({"type": "error", "message": str(e)[:120]})
        except Exception:
            pass


async def _run_chat_text(sm, websocket, send, client_id: str, text: str) -> None:
    """文本轮：复用 server.py 的快慢融合管道，把 SSE 事件转成 WS 消息。

    实现：直接调用 sm 内部的快/慢 LLM 与 TTS（与 /api/chat_text 相同的代码路径），
    这里只做协议转换。P1 版本：调用 sm.chat_text_pipeline()（新增的纯函数接口）。
    """
    # P1 简化实现：调用 /api/chat_text 的核心逻辑
    # （P1 先打通协议层；P2 将其重构为可被 WS/SSE 共用的 async generator）
    try:
        if hasattr(sm, "run_chat_text_events"):
            async for ev in sm.run_chat_text_events(client_id, text):
                await websocket.send_text(json.dumps(ev, ensure_ascii=False))
        else:
            await send({"type": "error", "message": "server 未暴露 run_chat_text_events（P2 接口）"})
    except Exception as e:
        await send({"type": "error", "message": f"chat fail: {str(e)[:80]}"})
