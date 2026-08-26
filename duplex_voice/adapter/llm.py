"""LLM Provider —— 统一接口 + Mock + OpenAI 兼容（SSE）。

设计 §4.4：OpenAI 兼容 /chat/completions 流式（httpx-sse）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Protocol

from ..events import Event

log = logging.getLogger(__name__)


class LLMProvider(Protocol):
    async def stream_chat(self, messages, *, temperature: float = 0.3,
                          max_tokens: int = 1024, tools=None) -> AsyncIterator[Event]: ...
    async def close(self) -> None: ...


class MockLLMProvider:
    """模拟 LLM：固定回复（可配置），模拟 token 流。"""

    def __init__(self, reply: str = "好的，正在为您打开客厅的灯。",
                 first_token_ms: int = 200, token_gap_ms: int = 60):
        self.reply = reply
        self.first_token_ms = first_token_ms
        self.token_gap_ms = token_gap_ms

    async def stream_chat(self, messages, *, temperature=0.3, max_tokens=1024, tools=None) -> AsyncIterator[Event]:
        await asyncio.sleep(self.first_token_ms / 1000)
        yield Event(type="llm.first_token", domain="llm", payload={"channel": "slow"})
        chunks = [self.reply[i:i + 4] for i in range(0, len(self.reply), 4)] or [self.reply]
        for i, chunk in enumerate(chunks):
            await asyncio.sleep(self.token_gap_ms / 1000)
            yield Event(type="llm.token", domain="llm",
                        payload={"delta": chunk, "channel": "slow"})
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": self.reply, "channel": "slow"})

    async def close(self) -> None:
        pass


class OpenAICompatLLMProvider:
    """OpenAI 兼容流式（httpx-sse /chat/completions）。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 use_chat_template_kwargs: bool = False, trust_env: bool = True,
                 extra_headers: dict | None = None):
        """use_chat_template_kwargs=True：关思考用 chat_template_kwargs 结构
        （部分本地服务端/模板只认这个，顶层 enable_thinking 不生效——2026-08-26 用户提供结构）。
        trust_env=False：绕过系统代理直连（本地/内网端点——代理会连不通内网 IP，2026-08-26 实测）。
        extra_headers：额外请求头（如 Session-ID——用户参考代码指定）。"""
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.use_chat_template_kwargs = use_chat_template_kwargs
        self.extra_headers = extra_headers or {}
        self._client = httpx.AsyncClient(timeout=60, trust_env=trust_env)

    async def stream_chat(self, messages, *, temperature=0.3, max_tokens=1024, tools=None) -> AsyncIterator[Event]:
        payload: dict = {
            "model": self.model, "messages": messages,
            "stream": True, "temperature": temperature, "max_tokens": max_tokens,
        }
        if self.use_chat_template_kwargs:
            # 本地 OpenAI 兼容服务端：chat_template_kwargs 结构关思考链
            # （用户指定结构——部分服务端/模板不支持顶层 enable_thinking）
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["enable_thinking"] = False   # 禁思考链：dashscope.aliyuncs.com 老端点 qwen3.5-27b 默认思考（1138 块 reasoning_content 40s+）→ 1s 直接内容（2026-08-24 实测）
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(self.extra_headers)   # 额外头（如 Session-ID——本地服务端可能要求）
        first = True
        acc = ""
        req = self._client.build_request("POST", f"{self.base_url}/chat/completions",
                                         json=payload, headers=headers)
        try:
            resp = await self._client.send(req, stream=True)
            try:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                        delta = data["choices"][0]["delta"]["content"]
                        if first:
                            yield Event(type="llm.first_token", domain="llm",
                                        payload={"channel": "slow"})
                            first = False
                        acc += delta
                        yield Event(type="llm.token", domain="llm",
                                    payload={"delta": delta, "channel": "slow"})
            finally:
                await resp.aclose()
        except Exception as e:
            # 流式中断（长回复被服务端掐断）→ 降级非流式一次性全文
            log.warning("流式失败（降级非流式）: %s", e)
            non_stream = dict(payload, stream=False)
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions",
                                            json=non_stream, headers=headers, timeout=120)
                r.raise_for_status()
                data = r.json()
                full = data["choices"][0]["message"].get("content", "")
                yield Event(type="llm.first_token", domain="llm",
                            payload={"channel": "slow"})
                yield Event(type="llm.token", domain="llm",
                            payload={"delta": full, "channel": "slow"})
                acc = full
            except Exception as e2:
                yield Event(type="llm.error", domain="llm",
                            payload={"code": "E_LLM_CONN", "retryable": True, "error": str(e2)})
                return
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": acc, "channel": "slow"})

    async def close(self) -> None:
        await self._client.aclose()


class LocalOpenAICompatProvider:
    """本地 OpenAI 兼容端点（严格按用户验证过的调用方式，2026-08-26）。

    用户参考代码（实测可用）：
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1,<内网IP>'
        BASE_URL = "http://<ip>:8000/v1/chat/completions"   # 完整路径
        headers  = {"Content-Type": "application/json", "Authorization": ..., "Session-ID": ""}
        data     = {"model": ..., "messages": ..., "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
        response = requests.post(BASE_URL, headers=headers, json=data)

    差异点（vs OpenAICompatLLMProvider）：① requests 同步库（async 内 to_thread 包装）
    ② stream=False 非流式（本地服务端不支持/不响应流式——之前 stream=True 调用失败）
    ③ NO_PROXY 环境变量绕过系统代理直连内网 ④ 完整 /chat/completions URL
    """

    def __init__(self, base_url: str, api_key: str = "", model: str = ""):
        import os
        import re as _re
        # NO_PROXY：绕过系统代理直连内网（requests 读环境变量）
        host = _re.sub(r"^https?://([^:/]+).*", r"\1", base_url)
        existing = os.environ.get("NO_PROXY", "")
        hosts = {h for h in existing.split(",") if h}
        hosts.add("localhost")
        hosts.add("127.0.0.1")
        if host and host not in hosts:
            hosts.add(host)
        os.environ["NO_PROXY"] = ",".join(sorted(hosts))
        # 兼容两种 base_url 形态：http://ip:8000/v1 或 http://ip:8000/v1/chat/completions
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"
        self.api_key = api_key
        self.model = model

    async def stream_chat(self, messages, *, temperature=0.3, max_tokens=1024, tools=None) -> AsyncIterator[Event]:
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Session-ID": "",   # 本地服务端要求（用户参考代码）
        }
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,   # 非流式（用户验证过的调用方式——本地服务端不支持流式）
            "chat_template_kwargs": {"enable_thinking": False},   # 关思考链
        }
        if tools:
            data["tools"] = tools
        if max_tokens:
            data["max_tokens"] = max_tokens
        # 同步 requests 在 async 环境用 to_thread 包装（不阻塞事件循环）
        resp = await asyncio.to_thread(
            requests.post, self.base_url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        # 非流式 JSON 响应
        try:
            obj = resp.json()
            full = ""
            choices = obj.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                # 兼容部分服务端把 reasoning 也放进 content / 只有 reasoning_content
                full = msg.get("content") or msg.get("reasoning_content") or ""
                if not full and "content" in msg and msg["content"] is None:
                    full = msg.get("reasoning_content", "")
        except Exception:
            # 非 JSON（如纯文本/SSE 残片）——按用户参考代码逐 chunk 拼接
            full = ""
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    full += chunk.decode("utf-8", errors="ignore").strip()
        full = (full or "").strip()
        if full:
            yield Event(type="llm.first_token", domain="llm", payload={"channel": "fast"})
            yield Event(type="llm.token", domain="llm", payload={"delta": full, "channel": "fast"})
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": full, "channel": "fast"})

    async def close(self) -> None:
        pass


# ================= 快慢融合（软件设计 v2.2 §2.5） =================

class FusionPolicy:
    """融合策略：should_speak(承接) / transcribe(转述) / merge(增量)。

    快通道（本地小模型）先出承接语 → 慢通道（云端大模型）完整回复增量接管。
    第一版：承接语独立短句先播；慢通道完整回复到达后接播（播放队列）。
    P1：慢通道首句到达即增量转述（切句级接管）。
    """

    MAX_FAST_TOKENS = 32          # 承接语 ≤15 字
    FAST_SYSTEM = (
        "你是语音助手的快通道（承接语生成）。用户刚说完一句话，你生成一句简短口语化的过渡语，"
        "让用户知道系统正在处理。\n"
        "【严格约束】\n"
        "1. 承接语中出现的时间、地点、对象等关键信息，必须逐字取自用户原话，严禁替换、臆测或"
        "'顺口'改写——例如用户问的是'今天'，绝不能说成'明天'；用户说'上海'，绝不能说成'北京'。\n"
        "2. 对任何关键信息没有把握时，改用通用承接语：'好的，正在帮您处理。'\n"
        "3. 承接语要有内容感（不能只有'好的''收到'这种单字——太短会造成播放停顿），"
        "简短口语化，不提指令的具体执行细节（慢通道会给出具体结果）。\n"
        "【配对示例】（注意关键词逐字来自用户原话）\n"
        "用户问'今天上海天气怎么样'→'好的，正在帮您查询今天上海的天气。'\n"
        "用户问'找一下周星驰的电影'→'好的，正在帮您检索周星驰的电影。'\n"
        "用户问'帮我定个明早七点的闹钟'→'好的，正在帮您设置明早七点的闹钟。'\n"
        "用户问'播放一首周杰伦的歌'→'好的，正在为您播放周杰伦的歌。'\n"
        "只输出承接语，不要任何解释。"
    )

    @staticmethod
    def should_speak(user_text: str) -> bool:
        """是否需要先出承接语（默认 True；极短指令可直出结果）。"""
        return len(user_text) >= 2

    @staticmethod
    def build_fast_prompt(user_text: str) -> list[dict]:
        return [
            {"role": "system", "content": FusionPolicy.FAST_SYSTEM},
            {"role": "user", "content": user_text},
        ]

    @staticmethod
    def transcribe(fast_text: str, slow_text: str) -> str:
        """转述：慢通道首句到达时的播放文本（第一版：直接接管慢文本）。"""
        return slow_text

    @staticmethod
    def merge(acc: str, new_slow: str) -> str:
        """增量融合（P1 切句级）；第一版为累积。"""
        return acc + new_slow


class OllamaFastProvider:
    """快通道：Ollama 原生 /api/chat（think:false 关思考 → 承接语直出）。

    实测（2026-08-17）：qwen3.5:4b-mlx 走 OpenAI 兼容端点默认开思考
    （content 空、reasoning 输出思考过程）；原生 /api/chat + think:false
    才直出承接语（"马上为您撰写周报"）。
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 model: str = "qwen3.5:4b-mlx", max_tokens: int = 32):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=60)

    async def stream_chat(self, messages, *, temperature=0.3, max_tokens=32, tools=None) -> AsyncIterator[Event]:
        payload = {
            "model": self.model, "messages": messages,
            "stream": True, "think": False,
            "max_tokens": max_tokens or self.max_tokens,
            "options": {"temperature": temperature},
        }
        first = True
        acc = ""
        async with self._client.stream("POST", f"{self.base_url}/api/chat",
                                       json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                if delta:
                    if first:
                        yield Event(type="llm.first_token", domain="llm",
                                    payload={"channel": "fast"})
                        first = False
                    acc += delta
                    yield Event(type="llm.token", domain="llm",
                                payload={"delta": delta, "channel": "fast"})
                if data.get("done"):
                    break
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": acc, "channel": "fast"})

    async def close(self) -> None:
        await self._client.aclose()


class MockFastLLMProvider:
    """模拟快通道：固定承接语，首 token 快（100ms）。"""

    def __init__(self, reply: str = "好的", first_token_ms: int = 100):
        self.reply = reply
        self.first_token_ms = first_token_ms

    async def stream_chat(self, messages, *, temperature=0.3, max_tokens=32, tools=None) -> AsyncIterator[Event]:
        await asyncio.sleep(self.first_token_ms / 1000)
        yield Event(type="llm.first_token", domain="llm", payload={"channel": "fast"})
        yield Event(type="llm.token", domain="llm", payload={"delta": self.reply, "channel": "fast"})
        yield Event(type="llm.complete", domain="llm",
                    payload={"full_text": self.reply, "channel": "fast"})

    async def close(self) -> None:
        pass
