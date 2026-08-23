"""语音助手 Web 服务：音频 → ASR → 快慢融合 LLM → 回复（文字 + TTS 音频）。

启动：
    cd duplex-voice/web && DASHSCOPE_API_KEY=<key> python3 server.py
    （或先 export DASHSCOPE_API_KEY；key 也会从 ../config.yaml 的 dashscope_api_key 读取）

页面：http://127.0.0.1:8787/
接口：POST /api/chat  {"audio_b64": "<16k wav base64>"}
"""
import asyncio
import base64
import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

# 日志：统一时间戳格式，与 uvicorn 输出混排可区分（[voice] 前缀）
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d %(levelname)s [voice] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("voice")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from duplex_voice.adapter.asr import FunASRStreamProvider, Qwen3ASRProvider
from duplex_voice.adapter.llm import (
    FusionPolicy, OllamaFastProvider, OpenAICompatLLMProvider,
)
from duplex_voice.adapter.tts import Qwen3TTSProvider
from duplex_voice.adapter.tts_stream import Qwen3TTSStream

HOST = "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com"
WEB_DIR = Path(__file__).resolve().parent
TTS_CACHE = WEB_DIR / "tts_cache"   # 本地 TTS 音频缓存（server 代下载，前端播放不走公网）


def _load_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        try:
            cfg = json.loads((WEB_DIR.parent / "config.yaml").read_text())
            key = cfg.get("model", {}).get("dashscope_api_key", "")
        except Exception:
            pass
    return key


KEY = _load_key()
print(f"API key: {'已加载' if KEY else '缺失（export DASHSCOPE_API_KEY 或填 config.yaml）'}")

# ==================== 配置系统（config.json 可配置化——模型/prompt/前端参数） ====================
CONFIG_PATH = WEB_DIR / "config.json"

DEFAULT_CONFIG: dict = {
    "server": {
        "asr": {"provider": "dashscope", "model": "qwen3-asr-flash",
                "ws_endpoint": "api-ws", "sample_rate": 16000, "task": "asr"},
        "fast_llm": {"provider": "ollama", "model": "qwen3.5:4b-mlx",
                     "base_url": "http://127.0.0.1:11434", "max_chars": 10,
                     "prompt": ("你是全双工语音助手的快速响应模块。用户说完话后，你只生成一句"
                                "6-10 个字的简短的承接语（如'好的，正在为您处理'）。要求：口语化、"
                                "有内容感、不提指令具体内容（防与慢回复重复）、不要标点符号。")},
        "slow_llm": {"provider": "dashscope", "model": "qwen3.5-27b",
                     "base_url": "", "max_tokens": 1024,
                     "prompt_fastslow": ("你是智能家居语音助手，回复简洁口语化，不超过两句话。"
                                         "注意：AI 已先用简短的过渡语（如'好的，马上为您处理'）回应过用户，"
                                         "你回复时不要再说'好的''嗯''没问题'等开头客套，直接给出具体内容或结果。"
                                         "要求首句尽量简短（15字内）直接给结果（首句短→TTS 合成快，避免播放停顿），"
                                         "细节放第二句。"),
                     "prompt_direct": ("你是智能家居语音助手，回复简洁口语化，不超过两句话。"
                                       "首句直接给出核心结果，细节放第二句。")},
        "omni": {"provider": "dashscope", "model": "qwen3.5-omni-flash",
                 "temperature": 0, "max_tokens": 64,
                 "prompt": ("你是全双工语音对话系统的语义 VAD（语音活动检测）。给定用户语音转写文本、"
                            "AI 最近播放的回复、对话历史，判断用户当前状态。只输出 JSON，不要其他内容："
                            '{"state": "complete|incomplete|backchannel|barge_in|noise|tts_echo|reject", "reason": "简短理由"}'
                            "\n\n状态定义："
                            "- complete：用户说完完整指令/问题，等待系统回复"
                            "- incomplete：用户在对 AI 说话但被打断（有指令/问题意图、话没说完），"
                            "如'打开客厅的'（未完的指令）"
                            "- backchannel：仅应声词（嗯/对/好/明白）无新内容，系统不应回应"
                            "- barge_in：用户打断 AI 播放且内容完整（有新指令），系统应立即接管"
                            "- noise：无意义内容/转录噪声/幻觉——典型如：环境音/背景人声/与人闲聊的转写"
                            "（'要么就是整张图的分布''这个地方实际上就是之前我们'这类无指令意图、"
                            "像与人交谈的句子）、平台话术（'谢谢观看''请订阅'）、笑声（'哈哈'）、碎片句"
                            "- tts_echo：内容与 AI 刚播放的回复几乎逐字一致（麦克风拾取 AI 声音的回声）"
                            "- reject：听不清/不理解（用户请求重复，如'什么？''再说一遍'）"
                            "\n\n关键规则："
                            "- 当前 AI 是否正在播放回复：{playing}（这是权威状态，即使播放历史为空也以此为准）"
                            "- 若'正在播放'（是）且用户内容为完整新指令 → 判定 barge_in（不是 complete），无论播放历史是否为空"
                            "- 若'不在播放'（否）且内容完整 → 判定 complete"
                            "- 判断是否'对 AI 的指令/问题'：有明确动作/请求意图（打开/关闭/帮我/讲/问/调…）"
                            "→ 完整=complete，未完=incomplete"
                            "- 无指令意图、像闲聊/陈述/与人交谈（'这个地方''实际上''之前我们'）→ 判定 noise"
                            "（不是 incomplete——incomplete 是'对 AI 说话被打断'）"
                            "- 拿不准时 → 判定 noise（宁可丢弃，不可误触发回复）")},
        "tts": {"batch_model": "qwen3-tts-instruct-flash",
                "stream_model": "qwen3-tts-instruct-flash-realtime",
                "voice": "Cherry", "sample_rate": 24000},
    },
    "frontend": {
        "vad": {"silero_threshold": 0.5, "silence_ms": 800, "vote_in": 7, "vote_exit": 5,
                "energy_ratio": 4.0, "energy_floor_frame": 0.005, "energy_floor_seg": 0.012,
                "seg_speech_min_ratio": 0.35, "env_silence_ms": 2000, "max_seg_ms": 15000,
                "cooldown_ms": 10000, "pre_roll_ms": 500, "echo_sim": 0.7},
        "scenes": {
            "headset": {"surge_threshold": 2.2, "surge_floor": 0.008, "rms_floor": 0.02,
                        "surge_ms": 400, "cut_on_surge": False, "pre_roll_ms": 250, "tts_volume": 1.0},
            "speaker": {"surge_threshold": 1.6, "surge_floor": 0.006, "rms_floor": 0.015,
                        "surge_ms": 200, "cut_on_surge": True, "pre_roll_ms": 250, "tts_volume": 0.6},
        },
        "behavior": {"rule": {"barge_in": "immediate", "silence_ms": 800},
                     "omni": {"barge_in": "semantic", "silence_ms": 800},
                     "soulx": {"barge_in": "semantic", "silence_ms": 800}},
        "busy_ttl_ms": 60000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并（override 缺失键保留 base 值）——config.json 部分覆盖安全。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _deep_merge(base[k], v)
        else:
            out[k] = v
    return out


def _load_config() -> dict:
    """读 config.json；缺失/损坏 → 回退内置默认（不覆盖损坏文件，日志告警）。"""
    try:
        if CONFIG_PATH.exists():
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return _deep_merge(DEFAULT_CONFIG, raw)
    except Exception as e:
        print(f"[config] 读取失败，回退默认配置: {e}")
    return _deep_merge(DEFAULT_CONFIG, {})


def _save_config(cfg: dict) -> bool:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"[config] 保存失败: {e}")
        return False


def _mask_cfg(cfg: dict) -> dict:
    """返回给前端的配置（api_key 一律掩码，绝不回传明文）。"""
    import copy
    out = copy.deepcopy(cfg)
    for sec in out.get("server", {}).values():
        if isinstance(sec, dict) and sec.get("api_key"):
            sec["api_key"] = "****"
    return out


CFG = _load_config()
print(f"[config] 已加载: {CONFIG_PATH.name if CONFIG_PATH.exists() else '内置默认'}")

# 模型变量（供 Provider 实例化——热生效时更新）
ASR_MODEL = CFG["server"]["asr"]["model"]
FAST_MODEL = CFG["server"]["fast_llm"]["model"]
FAST_PROVIDER = CFG["server"]["fast_llm"].get("provider", "ollama")   # ollama 本地 | dashscope 云端
FAST_BASE = CFG["server"]["fast_llm"].get("base_url", "http://127.0.0.1:11434")
SLOW_MODEL = CFG["server"]["slow_llm"]["model"]
OMNI_MODEL = CFG["server"]["omni"]["model"]
TTS_BATCH_MODEL = CFG["server"]["tts"]["batch_model"]
TTS_STREAM_MODEL = CFG["server"]["tts"]["stream_model"]
TTS_VOICE = CFG["server"]["tts"].get("voice", "Cherry")
TTS_SAMPLE_RATE = CFG["server"]["tts"].get("sample_rate", 24000)

_NOISE_HINTS = ("嗯", "啊", "哦", "呃", "emm", "hmm", "哼", "哈", "嘿")

# ==================== 语义 VAD 可插拔接缝 ====================
# config: vad.judge（rule | omni）——RuleVadJudge=现有规则（默认，稳定）；
#         OmniVadJudge=qwen3.5-omni-flash prompt 引导（语义判断，验证架构用）。
# 接缝目标：语义 VAD 作为独立模块，实现可替换，系统其余部分（ASR→VAD→LLM→TTS）不变。
VAD_JUDGE = os.environ.get("SEMANTIC_VAD", "rule")   # 或从 config.yaml vad.judge 读取


from duplex_voice.fsm.barge_decision import BargeDecisionFSM, VadState
from duplex_voice.adapter.fusion import FusionStrategy, get_fusion_strategy


class VadJudge:
    """语义 VAD 抽象接口：judge(text, history, last_replies, is_replying) -> (state, reason)。"""

    async def judge(self, text: str, history: list[dict], last_replies=(),
                    is_replying: bool = False) -> tuple[str, str]:
        raise NotImplementedError


class RuleVadJudge(VadJudge):
    """规则实现（现有 _is_noise_text + _is_tts_echo 封装）：零成本、确定、可作兜底。"""

    async def judge(self, text: str, history: list[dict], last_replies=(),
                    is_replying: bool = False) -> tuple[str, str]:
        if _is_noise_text(text):
            return VadState.NOISE, "noise_text"
        if _is_tts_echo(text, history, last_replies):
            return VadState.TTS_ECHO, f"tts_echo sim>0.7"
        return VadState.COMPLETE, "ok"


class OmniVadJudge(VadJudge):
    """Omni 实现：qwen3.5-omni-flash + prompt 引导判断语义（可插拔架构可行性验证）。
    模型与 prompt 从 config.json（server.omni）读取——界面可配置。
    失败/超时自动回退 RuleVadJudge（接缝容错）。"""

    def __init__(self, host: str, api_key: str, model: str | None = None, prompt: str | None = None):
        self.host = host
        self.api_key = api_key
        self.model = model or OMNI_MODEL
        self.prompt = prompt or CFG["server"]["omni"]["prompt"]
        self._fallback = RuleVadJudge()

    async def judge(self, text: str, history: list[dict], last_replies=(),
                    is_replying: bool = False) -> tuple[str, str]:
        # 组合实现：确定性规则先行（回声 0.7 相似度、纯语气词）→ Omni 语义补充
        # （可插拔接缝的体现：实现内部可组合，不牺牲规则 VAD 已解决的确定场景）
        if _is_noise_text(text):
            return VadState.NOISE, "noise_text(rule)"
        if _is_tts_echo(text, history, last_replies):
            return VadState.TTS_ECHO, "tts_echo sim>0.7(rule)"
        try:
            playing = "是" if is_replying else "否"
        except Exception:
            playing = "否"
        # prompt 支持 {playing} 占位符（config 可配——无占位符则原样使用）
        system = self.prompt.format(playing=playing) if "{playing}" in self.prompt else self.prompt
        user = (
            f"AI 最近播放的回复：{json.dumps(list(last_replies or []), ensure_ascii=False)}\n"
            f"对话历史（最近3轮）：{json.dumps([m for m in history[-6:] if m.get('role') == 'user'][-3:] or history[-3:], ensure_ascii=False)}\n"
            f"用户语音转写：{text}\n"
            "请判断状态："
        )
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        try:
            r = await _post_json(f"https://{self.host}/compatible-mode/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                                 timeout=8.0)
            content = r["choices"][0]["message"]["content"]
            state = _extract_state(content)
            if state is None:
                log.warning("VADJUDGE cid=- omni 输出无法解析: %r", content[:80])
                return await self._fallback.judge(text, history, last_replies)
            return state, content[:60]
        except Exception as e:
            # 失败重试一次（瞬时超时/网络抖动可恢复——避免回退 rule 把环境音
            # 长句误判 complete 触发完整回复，实测 11.3s 噪音回复即此根因）
            try:
                log.warning("VADJUDGE cid=- omni 调用失败(%s) → 重试", str(e)[:80])
                r = await _post_json(f"https://{self.host}/compatible-mode/v1/chat/completions",
                                     headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                                     timeout=8.0)
                content = r["choices"][0]["message"]["content"]
                state = _extract_state(content)
                if state is not None:
                    return state, content[:60]
            except Exception:
                pass
            log.warning("VADJUDGE cid=- omni 重试仍失败 → 回退 rule")
            return await self._fallback.judge(text, history, last_replies)


def _extract_state(content: str) -> str | None:
    """从 Omni 输出提取 state（容忍 JSON 包裹/代码块/多余文本）。"""
    import re
    m = re.search(r'"state"\s*:\s*"([a-z_]+)"', content)
    if not m:
        return None
    s = m.group(1)
    valid = {VadState.COMPLETE, VadState.INCOMPLETE, VadState.BACKCHANNEL, VadState.BARGE_IN,
             VadState.NOISE, VadState.TTS_ECHO, VadState.REJECT}
    return s if s in valid else None


async def _post_json(url: str, headers: dict, json: dict, timeout: float = 8.0) -> dict:
    """httpx POST JSON（语义 VAD 判断用，轻量同步等待）。"""
    import httpx
    async with httpx.AsyncClient(timeout=timeout, proxy=None) as client:
        r = await client.post(url, headers=headers, json=json)
        r.raise_for_status()
        return r.json()


# 实例化（可插拔：SEMANTIC_VAD=omni 切换；运行中可由前端 /api/vad_switch 动态切换）
if VAD_JUDGE == "omni":
    vad_judge: VadJudge = OmniVadJudge(host=HOST, api_key=KEY)
    print(f"✅ 语义 VAD = Omni（{OMNI_MODEL}）——架构验证模式")
else:
    vad_judge = RuleVadJudge()
    print("✅ 语义 VAD = Rule（默认）——SEMANTIC_VAD=omni 切换 Omni 验证")

VAD_MODE = {"mode": VAD_JUDGE}   # 运行中可切换（前端开关）
FUSION_MODE = {"mode": os.environ.get("FUSION_MODE", "fastslow")}   # fastslow | direct（运行中可切换）
fusion: FusionStrategy = get_fusion_strategy(FUSION_MODE["mode"])   # 全局融合策略（可插拔）
TTS_STREAM = os.environ.get("TTS_STREAM", "1") == "1"   # 启动默认（流式）
TTS_MODE = {"mode": "stream" if TTS_STREAM else "batch"}   # 运行中可切换：stream=承接语流式+慢句整段并行 / batch=全整段
tts_stream = Qwen3TTSStream(HOST, KEY)   # 流式 TTS 实例（每句一个 WS 会话）


def _is_noise_text(text: str) -> bool:
    """噪声转录过滤：过短或纯语气词 → 视为无效语音（不触发回复）。"""
    t = text.strip()
    if not t:
        return True
    if len(t) < 2:
        return True
    if all(ch in "嗯啊哦呃哈嘿唉啧" for ch in t):
        return True
    return False


def _local_tts(text: str, voice: str = "Cherry") -> tuple:
    """合成 TTS 并转本地缓存 → (本地 URL, 合成耗时ms, 公网下载耗时ms)。

    公网部分（DashScope 合成 API + 云端音频下载）单独计时——下载抖动
    会反映在'响应'里，需单独显示（前端 tts_sentence.dl_ms）。
    同文本同音色缓存命中时合成/下载均 0（无公网调用）。
    """
    import hashlib
    import urllib.request as _urlreq
    name = hashlib.md5((text + voice).encode()).hexdigest()[:16] + ".wav"
    local = TTS_CACHE / name
    if local.exists():
        return f"/tts_cache/{name}", 0, 0
    TTS_CACHE.mkdir(parents=True, exist_ok=True)   # 目录缺失自动重建（服务重启后）
    t_tts = time.time()
    remote = tts._synthesize_url(text, voice)
    tts_ms = int((time.time() - t_tts) * 1000)
    with _urlreq.urlopen(remote, timeout=30) as r:
        data = r.read()
    dl_ms = int((time.time() - t_tts) * 1000) - tts_ms
    local.write_bytes(data)
    return f"/tts_cache/{name}", tts_ms, dl_ms


def _strip_leading_filler(text: str) -> str:
    """剥离慢回复首句的开头客套（防与快通道承接语重复割裂）：
    '好的，''嗯，''行，''没问题，''收到，'等（'好的方面'这类无标点的不剥）。
    整句都是客套（'好的'）返回空串——由调用方跳过（否则 TTS 仍播'好的'与承接语重复）。"""
    import re
    # 客套词后必须跟标点/空白/句尾（'好的方面'直接跟汉字不剥）
    m = re.match(r'^(好的|嗯|行|没问题|收到|可以|好嘞|好呀|好的呢|ok|OK|好|嗯嗯|稍等|稍候)(?:[，,、\s]|$)', text)
    if m:
        return text[m.end():].lstrip()
    return text


def _is_tts_echo(text: str, history: list[dict], last_replies=()) -> bool:
    """残余回声过滤：AEC 不完美时 AI 播放声音被拾取，ASR 文本与最近播放内容
    **逐字高度相似**（>0.7）→ 丢弃。
    阈值 0.7：真回声是播放音频的逐字复制（0.8-1.0）；用户重复指令与上轮回复
    仅部分重叠（实测 0.4-0.5）→ 放行。历史只查最近几轮（防跨轮误杀）。"""
    import difflib
    for cand in last_replies:
        if cand and difflib.SequenceMatcher(None, text, cand).ratio() > 0.7:
            return True
    for m in reversed(history[-4:]):
        if m.get("role") == "assistant":
            prev = m.get("content", "")
            if prev and difflib.SequenceMatcher(None, text, prev).ratio() > 0.7:
                return True
    return False

# 多轮会话历史（按 client_id；简单内存实现，重启清空）
HISTORIES: dict[str, list[dict]] = {}
LAST_REPLY: dict[str, list[str]] = {}   # client_id → 最近实际播放文本列表 [承接语, 完整回复]
PENDING_INCOMPLETE: dict[str, dict] = {}  # client_id → {text, ts} 未说完段缓存（incomplete 等待续说）
MAX_HISTORY = 20
PENDING_TTL_S = 3   # incomplete 缓存 3s 无续说则过期丢弃（用户要求：没说完等 3s 就行，不要一直等）
FRONTEND_LOGS: deque[str] = deque(maxlen=3000)   # 前端自动上报日志（定位问题用）

asr = Qwen3ASRProvider(host=HOST, api_key=KEY, model=ASR_MODEL)
asr_stream = FunASRStreamProvider(host=HOST, api_key=KEY)   # 真流式（fun-asr，partial 实时）


def _make_fast_llm():
    """快 LLM 按 provider 实例化：ollama 本地 / dashscope 云端（OpenAI 兼容）——界面可配置。"""
    if FAST_PROVIDER == "dashscope":
        return OpenAICompatLLMProvider(
            base_url=f"https://{HOST}/compatible-mode/v1", api_key=KEY, model=FAST_MODEL)
    # 默认 ollama 本地（base_url 可配自定义端点）
    return OllamaFastProvider(base_url=FAST_BASE, model=FAST_MODEL)


fast_llm = _make_fast_llm()
slow_llm = OpenAICompatLLMProvider(
    base_url=f"https://{HOST}/compatible-mode/v1", api_key=KEY, model=SLOW_MODEL)
tts = Qwen3TTSProvider(host=HOST, api_key=KEY)


def _apply_config(cfg: dict) -> dict:
    """应用新配置（热生效，不重启）：更新模型变量 + prompt 注入 + 重建 Provider。"""
    global CFG, ASR_MODEL, FAST_MODEL, FAST_BASE, SLOW_MODEL, OMNI_MODEL
    global TTS_BATCH_MODEL, TTS_STREAM_MODEL, TTS_VOICE, TTS_SAMPLE_RATE
    global asr, fast_llm, slow_llm, tts, tts_stream, vad_judge
    global FAST_PROVIDER
    CFG = cfg
    s = cfg["server"]
    ASR_MODEL = s["asr"]["model"]
    FAST_MODEL = s["fast_llm"]["model"]
    FAST_PROVIDER = s["fast_llm"].get("provider", "ollama")
    FAST_BASE = s["fast_llm"].get("base_url", "http://127.0.0.1:11434")
    SLOW_MODEL = s["slow_llm"]["model"]
    OMNI_MODEL = s["omni"]["model"]
    TTS_BATCH_MODEL = s["tts"]["batch_model"]
    TTS_STREAM_MODEL = s["tts"]["stream_model"]
    TTS_VOICE = s["tts"].get("voice", "Cherry")
    TTS_SAMPLE_RATE = s["tts"].get("sample_rate", 24000)
    # prompt 注入（fast 系统 prompt / slow 融合策略 / Omni VAD）
    FusionPolicy.FAST_SYSTEM = s["fast_llm"]["prompt"]
    from duplex_voice.adapter.fusion import STRATEGIES as _STRATS
    _STRATS["fastslow"].custom_slow_prompt = s["slow_llm"]["prompt_fastslow"]
    _STRATS["direct"].custom_slow_prompt = s["slow_llm"]["prompt_direct"]
    # 重建 Provider（模型/端点变化即时生效）
    asr = Qwen3ASRProvider(host=HOST, api_key=KEY, model=ASR_MODEL)
    fast_llm = _make_fast_llm()
    slow_llm = OpenAICompatLLMProvider(
        base_url=f"https://{HOST}/compatible-mode/v1", api_key=KEY, model=SLOW_MODEL)
    tts = Qwen3TTSProvider(host=HOST, api_key=KEY)
    tts_stream = Qwen3TTSStream(HOST, KEY)
    if VAD_JUDGE == "omni":
        vad_judge = OmniVadJudge(host=HOST, api_key=KEY, model=OMNI_MODEL, prompt=s["omni"]["prompt"])
    print(f"[config] 热生效: asr={ASR_MODEL} fast={FAST_MODEL} slow={SLOW_MODEL} "
          f"omni={OMNI_MODEL} tts_voice={TTS_VOICE}")
    return {"ok": True,
            "models": {"asr": ASR_MODEL, "fast": FAST_MODEL, "slow": SLOW_MODEL, "omni": OMNI_MODEL}}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 预热快通道（Ollama 冷加载 1.4s → 预热后 TTF ~110ms），首个用户请求不等冷启动
    try:
        async for _ in fast_llm.stream_chat([{"role": "user", "content": "好"}]):
            break
        print("✅ Ollama 快通道预热完成")
    except Exception as e:
        print(f"[warmup] Ollama 预热失败: {e}")
    yield


app = FastAPI(title="语音助手 Web 页面", lifespan=lifespan)


# ---- 配置 API（界面详细配置面板读写；模型/prompt 热生效，不重启）----
@app.get("/api/config")
async def get_config():
    return {"ok": True, "config": _mask_cfg(CFG)}


@app.post("/api/config")
async def post_config(body: dict):
    cfg = body.get("config")
    if not isinstance(cfg, dict):
        return JSONResponse({"ok": False, "error": "config 缺失"}, status_code=400)
    merged = _deep_merge(DEFAULT_CONFIG, cfg)   # 部分覆盖安全（缺键保留默认）
    if not _save_config(merged):
        return JSONResponse({"ok": False, "error": "保存失败"}, status_code=500)
    applied = _apply_config(merged)
    return {"ok": True, **applied, "frontend": merged.get("frontend", {})}


@app.post("/api/config/reset")
async def reset_config():
    """恢复内置默认配置（写文件 + 热生效）。"""
    merged = _deep_merge(DEFAULT_CONFIG, {})
    _save_config(merged)
    applied = _apply_config(merged)
    return {"ok": True, **applied, "frontend": merged.get("frontend", {})}

# 本地 TTS 音频缓存静态服务（前端播放不走公网——下载时延不计入响应）
app.mount("/tts_cache", StaticFiles(directory=TTS_CACHE), name="tts_cache")


class VadSwitchRequest(BaseModel):
    mode: str   # rule | omni


class FusionSwitchRequest(BaseModel):
    mode: str   # fastslow | direct


@app.post("/api/fusion_switch")
async def fusion_switch(req: FusionSwitchRequest):
    """动态切换回复融合策略（可插拔：快慢融合 | 慢回复直达，不重启）。"""
    if req.mode not in ("fastslow", "direct"):
        return JSONResponse({"ok": False, "msg": "mode 须为 fastslow|direct"}, status_code=400)
    global fusion
    fusion = get_fusion_strategy(req.mode)
    FUSION_MODE["mode"] = req.mode
    log.info("FUSION_SWITCH mode=%s", req.mode)
    return {"ok": True, "mode": req.mode}


@app.get("/api/fusion_mode")
async def fusion_mode():
    return {"mode": FUSION_MODE["mode"]}


@app.post("/api/tts_switch")
async def tts_switch(body: dict):
    """动态切换 TTS 模式：stream（承接语流式+慢句整段并行）| batch（全整段），不重启。"""
    mode = body.get("mode", "")
    if mode not in ("stream", "batch"):
        return JSONResponse({"error": "mode 必须是 stream 或 batch"}, status_code=400)
    TTS_MODE["mode"] = mode
    log.info("TTS_SWITCH mode=%s", mode)
    return {"ok": True, "mode": mode}


@app.get("/api/tts_mode")
async def tts_mode():
    return {"mode": TTS_MODE["mode"]}


@app.post("/api/vad_switch")
async def vad_switch(req: VadSwitchRequest):
    """动态切换语义 VAD 实现（不重启，前端开关调用）。"""
    global vad_judge, VAD_JUDGE
    if req.mode not in ("rule", "omni"):
        return JSONResponse({"error": "mode 必须是 rule 或 omni"}, status_code=400)
    VAD_MODE["mode"] = req.mode
    VAD_JUDGE = req.mode   # 同步更新（vad_state 事件/日志用——否则 rule 模式仍显示 omni）
    vad_judge = RuleVadJudge() if req.mode == "rule" else OmniVadJudge(host=HOST, api_key=KEY)
    log.info("VAD_SWITCH mode=%s", req.mode)
    return {"ok": True, "mode": req.mode}


@app.get("/api/vad_mode")
async def vad_mode():
    return {"mode": VAD_MODE["mode"]}


class ChatRequest(BaseModel):
    audio_b64: str          # 16k 16bit mono WAV 的 base64
    client_id: str = "default"   # 多轮会话标识（前端生成）
    speech_start_ms: int = 0     # 人开始说话的 epoch 毫秒（前端 VAD/按钮记录）
    speech_end_ms: int = 0       # 人说完话的 epoch 毫秒（静音判停/松开按钮）
    is_replying: bool = False    # 发送时刻 AI 是否正在播放回复（语义 VAD 判断 barge_in 的关键输入）


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """音频 → ASR → 快慢融合 LLM → TTS。SSE 流式返回阶段事件。"""
    if not KEY:
        return JSONResponse({"error": "DASHSCOPE_API_KEY 未配置"}, status_code=500)

    async def gen():
        t_start = time.time()
        try:
            wav_bytes = base64.b64decode(req.audio_b64)
            log.info("REQ cid=%s audio=%.2fs speech_start=%s speech_end=%s",
                     req.client_id, len(wav_bytes) / 32000,
                     req.speech_start_ms or "-", req.speech_end_ms or "-")
            data_url = f"data:audio/wav;base64,{req.audio_b64}"
            # 1. ASR 真流式（fun-asr：整段 wav → 帧流上行 → partial 实时推送）
            yield 'data: {"type":"stage","stage":"asr"}\n\n'
            t0 = time.time()
            flen = 960
            frames = [wav_bytes[i:i + flen] for i in range(0, len(wav_bytes) - flen + 1, flen)]

            async def gen_frames():
                for f in frames:
                    yield f

            asr_text = ""
            asr_partial_count = 0
            asr_first_partial_ms = None   # 首个 partial 出现的 epoch 毫秒
            asr_final_ms = None           # final 定稿的 epoch 毫秒
            async for evt in asr_stream.stream_recognize(gen_frames()):
                if evt.type == "asr.partial":
                    asr_partial_count += 1
                    if asr_first_partial_ms is None:
                        asr_first_partial_ms = time.time() * 1000
                    yield 'data: {"type":"asr_partial","text":%s}\n\n' % json.dumps(evt.payload.get("text", ""))
                elif evt.type == "asr.final":
                    asr_text = evt.payload.get("text", "")
                    asr_final_ms = time.time() * 1000
            asr_ms = int((time.time() - t0) * 1000)
            # 从人开始说话统计：出文字时延（start→首个 partial）；定稿时延（说完→final）
            partial_first_ms = (asr_first_partial_ms - req.speech_start_ms) if (asr_first_partial_ms and req.speech_start_ms) else None
            final_ms = (asr_final_ms - req.speech_end_ms) if (asr_final_ms and req.speech_end_ms) else None
            log.info("ASR cid=%s partial=%d text=%r asr_ms=%d final_ms=%s", req.client_id,
                     asr_partial_count, asr_text, asr_ms,
                     int(final_ms) if final_ms is not None else "-")
            yield 'data: {"type":"asr","text":%s,"latency_ms":%d,"partial_first_ms":%s,"final_ms":%s}\n\n' % (
                json.dumps(asr_text), asr_ms,
                json.dumps(int(partial_first_ms)) if partial_first_ms is not None else "null",
                json.dumps(int(final_ms)) if final_ms is not None else "null")
            if not asr_text:
                log.warning("FILTER cid=%s reason=empty_asr", req.client_id)
                yield 'data: {"type":"error","msg":"未识别到有效语音（环境噪声）"}\n\n'
                return
            # 语义 VAD（可插拔：rule 规则 / omni prompt 引导）——判断噪声/回声/拒识/应声/未说完
            # incomplete 续说合并：上一段判"没说完"则缓存，本段到达合并再判断（complete 才响应）
            pending = PENDING_INCOMPLETE.get(req.client_id)
            merged_text = asr_text
            if pending:
                if time.time() - pending["ts"] > PENDING_TTL_S:
                    PENDING_INCOMPLETE.pop(req.client_id, None)   # 过期丢弃
                    log.info("VADJUDGE cid=%s pending 过期丢弃", req.client_id)
                else:
                    merged_text = pending["text"] + asr_text
                    log.info("VADJUDGE cid=%s 续说合并: %r + %r", req.client_id,
                             pending["text"], asr_text)
            t_vad = time.time()   # 语义 VAD 判断时延起点
            vstate, vreason = await vad_judge.judge(
                merged_text, HISTORIES.get(req.client_id, []), LAST_REPLY.get(req.client_id, ""),
                is_replying=req.is_replying)
            # 打断决策状态机：语义状态（模型判断）× 播放状态（确定性事实）→ 行为
            # 模型只判断"是否要响应"；打断与否由"AI 是否在播放 TTS"决定
            fsm = BargeDecisionFSM()
            action, vstate = fsm.decide(vstate, req.is_replying)
            vad_ms = int((time.time() - t_vad) * 1000)   # 语义 VAD 判断时延（仅 omni 有意义）
            log.info("VADJUDGE cid=%s semantic=%s playing=%s action=%s state=%s vad=%s vad_ms=%d",
                     req.client_id, vreason[:30] if "(" in vreason else "omni", req.is_replying,
                     action, vstate, VAD_JUDGE, vad_ms)
            yield 'data: {"type":"vad_state","state":%s,"vad":%s,"latency_ms":%d}\n\n' % (
                json.dumps(vstate), json.dumps(VAD_JUDGE), vad_ms)
            if vstate == VadState.INCOMPLETE:
                # 没说完：缓存文本等待续说，不触发回复；前端继续聆听
                PENDING_INCOMPLETE[req.client_id] = {"text": merged_text, "ts": time.time()}
                log.info("VADJUDGE cid=%s incomplete 缓存待续 text=%r", req.client_id, merged_text)
                yield 'data: {"type":"vad_incomplete","state":"incomplete","vad":%s}\n\n' % (
                    json.dumps(VAD_JUDGE))
                return
            if pending:
                PENDING_INCOMPLETE.pop(req.client_id, None)   # 本段完整，清缓存
                asr_text = merged_text                        # 回复基于完整语义
            if vstate == VadState.NOISE:
                log.warning("FILTER cid=%s reason=noise(%s) text=%r", req.client_id, vreason, asr_text)
                yield 'data: {"type":"error","msg":"未识别到有效语音（环境噪声）"}\n\n'
                return
            if vstate == VadState.TTS_ECHO:
                log.warning("FILTER cid=%s reason=tts_echo(%s) text=%r", req.client_id, vreason, asr_text)
                yield 'data: {"type":"error","msg":"检测到回声，已忽略"}\n\n'
                return
            if vstate == VadState.BACKCHANNEL:
                log.info("FILTER cid=%s reason=backchannel(%s) text=%r", req.client_id, vreason, asr_text)
                yield 'data: {"type":"error","msg":"收到应声（嗯/对）"}\n\n'
                return
            if vstate == VadState.REJECT:
                log.info("FILTER cid=%s reason=reject(%s) text=%r", req.client_id, vreason, asr_text)
                yield 'data: {"type":"error","msg":"没听清，请再说一遍"}\n\n'
                return
            if vstate == VadState.BARGE_IN:
                log.info("BARGE_IN cid=%s text=%r", req.client_id, asr_text)
                # barge-in 语义上等同完整指令（接管对话），走正常回复
            # complete/incomplete → 正常处理
            # 3. 慢通道与快通道并行：ASR 完成即启动，事件入队，承接语播完按序接播
            slow_events: asyncio.Queue = asyncio.Queue()
            import re as _re

            async def run_slow():
                """慢通道生成 + 句子级 TTS 并行合成；事件推入 slow_events。"""
                try:
                    history = HISTORIES.get(req.client_id, [])[-10:]  # 最近 10 轮
                    messages = [{"role": "system",
                                 "content": fusion.slow_system_prompt()}]
                    messages += history
                    messages.append({"role": "user", "content": asr_text})
                    t0 = time.time()
                    # 静音时延：人说完话 → 送 LLM（含上传 + ASR 定稿 + 判停；slow_start_ms 在
                    # create_task 前记录，避免 SSE 发送与客户端读取的耦合延迟污染统计）
                    silence_ms = (slow_start_ms - req.speech_end_ms) if req.speech_end_ms else None
                    buf = ""
                    sidx = 0          # 内部计数（0 起）；推送时按 fast_tts_url 偏移
                    pending = {}   # idx → 合成任务
                    tts_t0 = {}
                    first_done = False
                    # DashScope TTS 限流：5 路并发实测 429 Too Many Requests——
                    # 信号量限 2 并发（安全且首句 2.5s 就绪；后续句合成 2.5s/句
                    # 被播放 5.6s/句掩盖 → 流水线无缝，无需全并行）
                    tts_sem = asyncio.Semaphore(2)

                    async def _synth(sub: str):
                        async with tts_sem:
                            return await asyncio.to_thread(_local_tts, sub)

                    def flush():
                        nonlocal buf, sidx
                        s = buf.strip()
                        buf = ""
                        if not s:
                            return
                        if sidx == 0:
                            # 首句剥离开头客套（防与快通道承接语'好的'重复割裂）
                            s = _strip_leading_filler(s)
                            if not s.strip():
                                return
                        # 长文本拆子句（15-40 字）并行合成——200 字整段 10.6s 会
                        # 让慢句久等（承接语播完 gap 7.6s）；拆后每句 2-3s 就绪
                        for sub in _split_clauses(s):
                            sub = sub.strip()
                            if not sub:
                                continue
                            i = sidx
                            sidx += 1
                            ts = time.time()
                            # 慢句一律整段并行合成（实测 realtime WS 合成 2.4x 慢于实时，
                            # 长句流式播放会中间等块卡顿；整段 ~2.5s 就绪，承接语播完无缝接播）
                            pending[i] = asyncio.create_task(_synth(sub))
                            tts_t0[i] = ts

                    try:
                        async for evt in slow_llm.stream_chat(messages):
                            if evt.type == "llm.token":
                                delta = evt.payload.get("delta", "")
                                if not first_done:
                                    first_done = True
                                    slow_events.put_nowait(
                                        ("slow_first", int((time.time() - t0) * 1000),
                                         int(silence_ms) if silence_ms is not None else None))
                                slow_events.put_nowait(("slow_delta", delta))
                                buf += delta
                                if _re.search(r"[。！？；.!?;]$", delta):
                                    flush()
                        if buf.strip():
                            flush()
                    except Exception as e:
                        print(f"[slow] 失败: {e}")
                    for i in sorted(pending):
                        try:
                            u, tts_ms, dl_ms = await pending[i]
                            slow_events.put_nowait(
                                ("tts_sentence", i, u, int((time.time() - tts_t0[i]) * 1000), dl_ms))
                        except Exception as e:
                            log.error("TTS_ERR cid=%s sentence=%d err=%s", req.client_id, i, e)
                    slow_events.put_nowait((("slow_done",)))
                except Exception as e:
                    log.error("SLOW_ERR cid=%s err=%s", req.client_id, e)
                    slow_events.put_nowait((("slow_error", str(e))))

            slow_start_ms = time.time() * 1000   # 送 LLM 时刻（静音时延基准，create_task 前记录）
            slow_task = asyncio.create_task(run_slow())

            # 2. 快通道承接语（本地 4b）→ 生成完立即 TTS 合成播放（不等慢通道）
            #    可插拔：direct 策略（慢回复直达）不启动快通道——两套机制独立
            fast_text = ""
            fast_tts_url = None
            if fusion.should_fast(asr_text):
                yield 'data: {"type":"stage","stage":"fast"}\n\n'
                t0 = time.time()
                try:
                    fast_prompt = FusionPolicy.build_fast_prompt(asr_text)
                    acc = ""
                    async for evt in fast_llm.stream_chat(fast_prompt):
                        if evt.type == "llm.token":
                            acc += evt.payload.get("delta", "")
                    fast_text = acc.strip()
                    LAST_REPLY[req.client_id] = [fast_text]   # 回声过滤：记录实际播放的承接语
                    fast_ms = int((time.time() - t0) * 1000)
                    log.info("FAST cid=%s text=%r ms=%d", req.client_id, fast_text, fast_ms)
                    yield 'data: {"type":"fast","text":%s,"latency_ms":%d}\n\n' % (json.dumps(fast_text), fast_ms)
                    # 承接语立即流式 TTS → 事件入统一队列（i=-1 映射 idx=0），
                    # 与慢句并行流式下发（不阻塞——慢句块无需等承接语合成完）
                    if fast_text:
                        t0 = time.time()
                        if TTS_MODE["mode"] == "stream":

                            async def _fast_stream(text, q):
                                try:
                                    first_ms = await tts_stream.synth(
                                        text, lambda pcm: q.put_nowait(
                                            ("audio_chunk", -1, base64.b64encode(pcm).decode())))
                                    q.put_nowait(("audio_end", -1, first_ms))
                                except Exception as e:
                                    log.error("STREAM_TTS_ERR cid=%s fast err=%s → 回退整段",
                                              req.client_id, str(e)[:80])
                                    try:
                                        u, tts_ms, dl_ms = await asyncio.to_thread(_local_tts, text)
                                        q.put_nowait(("tts_sentence_fb", -1, u, tts_ms, dl_ms))
                                    except Exception as e2:
                                        log.error("TTS_FB_ERR cid=%s fast err=%s",
                                                  req.client_id, str(e2)[:80])
                                finally:
                                    q.put_nowait(("fast_done",))

                            fast_task = asyncio.create_task(_fast_stream(fast_text, slow_events))
                            fast_tts_url = "stream"   # 标记流式（慢句 idx 偏移用）
                        else:
                            u, tts_ms, dl_ms = await asyncio.to_thread(_local_tts, fast_text)
                            fast_tts_ms = int((time.time() - t0) * 1000)
                            fast_tts_url = u
                            log.info("TTS cid=%s idx=0 fast ms=%d dl_ms=%d", req.client_id, fast_tts_ms, dl_ms)
                            yield 'data: {"type":"tts_sentence","idx":0,"channel":"fast","url":%s,"latency_ms":%d,"dl_ms":%d}\n\n' % (json.dumps(u), fast_tts_ms, dl_ms)
                except Exception as e:
                    log.error("FAST_ERR cid=%s err=%s", req.client_id, e)
            # 4. 消费慢通道事件（边生成边流式下发——承接语/慢句块并行，
            #    不再等 slow_task 全部完成才消费——修复慢句播放高延迟卡顿）
            yield 'data: {"type":"stage","stage":"slow"}\n\n'
            slow_text = ""
            got_slow_tts = False
            slow_done_evt = False
            fast_done_evt = False
            need_fast = fast_tts_url == "stream"   # 流式承接语已启动（等其 fast_done）
            while True:
                ev = await slow_events.get()
                if ev[0] == "slow_first":
                    _, sf_ms, sil_ms = ev
                    log.info("SLOW_FIRST cid=%s latency_ms=%d silence_ms=%s", req.client_id,
                             sf_ms, sil_ms if sil_ms is not None else "-")
                    yield 'data: {"type":"slow_first","latency_ms":%d,"silence_ms":%s}\n\n' % (
                        sf_ms, json.dumps(sil_ms) if sil_ms is not None else "null")
                elif ev[0] == "slow_delta":
                    slow_text += ev[1]
                    yield 'data: {"type":"slow_delta","delta":%s}\n\n' % json.dumps(ev[1])
                elif ev[0] == "audio_chunk":
                    # 流式 TTS：PCM 块（base64）→ 前端 Web Audio 边收边播
                    got_slow_tts = True   # 已有流式音频 → 兜底不再重复合成
                    _, i, b64s = ev
                    real_idx = i + (1 if fast_tts_url else 0)
                    yield 'data: {"type":"audio_chunk","idx":%d,"b64":%s}\n\n' % (real_idx, json.dumps(b64s))
                elif ev[0] == "audio_end":
                    got_slow_tts = True
                    _, i, first_ms = ev
                    real_idx = i + (1 if fast_tts_url else 0)
                    yield 'data: {"type":"audio_end","idx":%d,"first_ms":%s}\n\n' % (
                        real_idx, json.dumps(first_ms) if first_ms is not None else "null")
                elif ev[0] == "tts_sentence_fb":
                    # 流式失败回退整段（tts_sentence 事件兼容前端）
                    _, i, u, tts_ms, dl_ms = ev
                    real_idx = i + (1 if fast_tts_url else 0)
                    got_slow_tts = True
                    yield 'data: {"type":"tts_sentence","idx":%d,"channel":"slow","url":%s,"latency_ms":%d,"dl_ms":%d}\n\n' % (real_idx, json.dumps(u), tts_ms, dl_ms)
                elif ev[0] == "tts_sentence":
                    got_slow_tts = True
                    _, i, u, ms, dl_ms = ev
                    real_idx = i + (1 if fast_tts_url else 0)   # 承接语占 idx=0 → 慢句子偏移
                    log.info("TTS cid=%s idx=%d slow ms=%d dl_ms=%d", req.client_id, real_idx, ms, dl_ms)
                    yield 'data: {"type":"tts_sentence","idx":%d,"channel":"slow","url":%s,"latency_ms":%d,"dl_ms":%d}\n\n' % (real_idx, json.dumps(u), ms, dl_ms)
                elif ev[0] == "slow_done":
                    slow_done_evt = True
                elif ev[0] == "slow_error":
                    # run_slow 异常退出——按 slow_done 处理（防止消费循环永久等待）
                    log.error("SLOW_DONE_ERR cid=%s err=%s", req.client_id, str(ev[1])[:80])
                    slow_done_evt = True
                elif ev[0] == "fast_done":
                    fast_done_evt = True
                if slow_done_evt and (not need_fast or fast_done_evt):
                    break
            # 兜底：快/慢通道均无音频 → 整段合成（承接语已合成过则跳过）
            reply_for_tts = slow_text or fast_text
            if slow_text:
                last = LAST_REPLY.get(req.client_id, [])
                if fast_text not in last:
                    last = [fast_text] if fast_text else []
                LAST_REPLY[req.client_id] = last + [slow_text]   # 实际播放列表 [承接语, 完整回复]
            if not fast_tts_url and not got_slow_tts and reply_for_tts:
                try:
                    t0 = time.time()
                    u, _tts_ms, _dl_ms = _local_tts(reply_for_tts)
                    tts_ms = int((time.time() - t0) * 1000)
                    log.info("TTS cid=%s fallback ms=%d", req.client_id, tts_ms)
                    yield 'data: {"type":"tts_sentence","idx":0,"url":%s,"latency_ms":%d}\n\n' % (json.dumps(u), tts_ms)
                except Exception as e:
                    log.error("TTS_ERR cid=%s fallback err=%s", req.client_id, e)
            # 记入会话历史（多轮上下文）
            reply_for_hist = slow_text or fast_text or ""
            if asr_text:
                h = HISTORIES.setdefault(req.client_id, [])
                h.append({"role": "user", "content": asr_text})
                if reply_for_hist:
                    h.append({"role": "assistant", "content": reply_for_hist})
                HISTORIES[req.client_id] = h[-MAX_HISTORY:]
            total_ms = int((time.time() - t_start) * 1000)
            log.info("DONE cid=%s total=%dms slow_text=%r fast_tts=%s slow_tts=%s hist=%d",
                     req.client_id, total_ms, slow_text[:40], bool(fast_tts_url), got_slow_tts,
                     len(HISTORIES.get(req.client_id, [])))
            yield 'data: {"type":"done","total_ms":%d}\n\n' % total_ms
        except Exception as e:
            log.error("CHAT_ERR cid=%s err=%s\n%s", req.client_id, e, traceback.format_exc())
            yield 'data: {"type":"error","msg":%s}\n\n' % json.dumps(str(e))

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _split_clauses(text: str, max_len: int = 40, min_len: int = 15):
    """长文本拆子句（并行 TTS 合成用）：强标点（。！？；）必切；
    超 max_len 按弱标点（，、）再切（加入前检查——每段 ≤max_len）；
    短段（<min_len）与尾残段（标点残留）并入上一段。
    目的：200 字整段合成 10.6s → 拆成 25-40 字子句并行合成各 2-3s，
    承接语播完（~3s）时慢句全部就绪——消除'慢回复卡顿很久才开始播放'。"""
    import re
    parts = re.split(r"([。！？；.!?])", text)
    pieces = [p for p in parts if p]
    merged, cur = [], ""
    for p in pieces:
        cur += p
        if len(cur) >= min_len:
            merged.append(cur)
            cur = ""
    if cur.strip():
        if merged:
            merged[-1] += cur   # 尾残段（标点残留/短尾句）并入上一段
        else:
            merged.append(cur)
    out = []
    for c in merged:
        c = c.strip()
        if not c:
            continue
        if len(c) <= max_len:
            out.append(c)
            continue
        sub = re.split(r"([，、])", c)
        cur = ""
        for p in sub:
            if not p:
                continue
            if cur and len(cur) + len(p) > max_len:   # 加入前检查——防超切
                out.append(cur.lstrip("，、 "))
                cur = ""
            cur += p
        if cur.strip():
            out.append(cur.lstrip("，、 "))
    return out or [text]


@app.get("/api/health")
async def health():
    return {"ok": True, "key": bool(KEY), "fast_model": FAST_MODEL, "slow_model": SLOW_MODEL}


@app.post("/api/log")
async def fe_log(req: dict):
    """前端日志自动上报（vlog 批量 2s 一批，定位 VAD/播放问题用）。"""
    lines = req.get("lines") or []
    ts = time.strftime("%H:%M:%S")
    for ln in lines:
        FRONTEND_LOGS.append(f"{ts} {ln}")
    return {"ok": True, "n": len(lines)}


@app.get("/api/logs")
async def get_logs(n: int = 300):
    """查看最近前端上报日志。"""
    return {"logs": list(FRONTEND_LOGS)[-n:]}


@app.get("/")
async def index():
    # no-store：页面迭代频繁，禁止浏览器缓存旧版 index.html（否则用户测到旧逻辑）
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


# 前端资源（onnxruntime-web + Silero v6 ONNX 模型）
app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787)
