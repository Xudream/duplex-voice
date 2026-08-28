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

import numpy as np
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

# 日志：统一时间戳格式，与 uvicorn 输出混排可区分（[voice] 前缀）
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d %(levelname)s [voice] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("voice")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from duplex_voice.adapter.asr import FunASRStreamProvider, Qwen3ASRProvider
from duplex_voice.adapter.llm import (
    FusionPolicy, OllamaFastProvider, OpenAICompatLLMProvider, LocalOpenAICompatProvider,
)
from duplex_voice.adapter.tts import Qwen3TTSProvider
from duplex_voice.adapter.tts_stream import Qwen3TTSStream
from duplex_voice.config import VadConfig as _AcousticVadConfig
# 声学分段 VAD（帧级、判"何时开始/结束说话"）——与本文件下方 RuleVadJudge（语义 VAD，
# 判"说的是什么状态"）同名不同义，改名导入避免混淆（App 网关 /ws/app 用，Web 端浏览器
# 侧 JS 仍自己做声学分段，不受影响）。
from duplex_voice.vad.rule_judge import RuleVadJudge as AcousticVadJudge, FRAME_LEN

HOST = "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com"   # 默认专属空间（config server.host 可覆盖——dashscope.aliyuncs.com 老公共端点）
WEB_DIR = Path(__file__).resolve().parent
TTS_CACHE = WEB_DIR / "tts_cache"   # 本地 TTS 音频缓存（server 代下载，前端播放不走公网）


def _load_key() -> str:
    # 只读配置（duplex-voice/config.yaml 的 model.dashscope_api_key）——不读环境变量（2026-08-24 用户要求）
    try:
        cfg = json.loads((WEB_DIR.parent / "config.yaml").read_text(encoding="utf-8"))
        return cfg.get("model", {}).get("dashscope_api_key", "")
    except Exception:
        return ""


KEY = _load_key()
print(f"API key: {'已加载' if KEY else '缺失（export DASHSCOPE_API_KEY 或填 config.yaml）'}")

# ==================== 配置系统（config.json 可配置化——模型/prompt/前端参数） ====================
CONFIG_PATH = WEB_DIR / "config.json"

DEFAULT_CONFIG: dict = {
    "server": {
        "host": "llm-5ienv5iasbci5bt7.cn-beijing.maas.aliyuncs.com",   # 专属空间 | dashscope.aliyuncs.com（老公共端点）
        "asr": {"provider": "dashscope", "model": "fun-asr-flash-2026-06-15",   # 默认流式模型（mode=stream 联动）
                "mode": "stream",   # stream 流式（fun-asr，partial 实时）/ batch 非流式（qwen3-asr-flash 整段）
                "ws_endpoint": "api-ws", "sample_rate": 16000, "task": "asr"},
        "fast_llm": {"provider": "ollama", "model": "qwen3.5:4b-mlx",
                     "base_url": "http://127.0.0.1:11434", "max_chars": 10,
                     "prompt": ("你是语音助手的快通道（承接语生成）。用户刚说完一句话，你生成一句简短口语化的过渡语，"
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
                                "只输出承接语，不要任何解释。")},
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
                 "judge_mode": "text",   # 语义 VAD 判断模式：text 文本判断（老方案，ASR 后）/ audio 音频直判（新方案，当前轮音频+文本上下文，与 ASR 并行）——并存可切换
                 "prompt": ("你是全双工语音对话系统的语义 VAD（语音活动检测）。给定用户语音转写文本、"
                            "AI 最近播放的回复、对话历史，判断用户当前状态。只输出 JSON，不要其他内容："
                            '{"state": "complete|incomplete|backchannel|barge_in|noise|tts_echo|reject", "reason": "简短理由"}'
                            "\n\n状态定义："
                            "- complete：用户说完完整指令/问题，等待系统回复"
                            "- incomplete：用户在对 AI 说话但被打断（有指令/问题意图、话没说完），"
                            "如'打开客厅的'（未完的指令）"
                            "- backchannel：仅应声词（嗯/对/好/明白）无新内容，系统不应回应"
                            "- barge_in：用户打断 AI 播放且内容完整（有新指令），系统应立即接管"
                            "- noise：环境音/背景人声/转录噪声/无意义碎片（笑声'哈哈'、平台话术'谢谢观看'、过短无意义片段）"
                            "——注意：用户对 AI 的陈述/描述（即使不是指令，如'我这边有个问题'）不是 noise——是完整表达，应判 complete"
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
        "vad": {"silero_threshold": 0.5, "silence_ms": 800, "vote_in": 7, "vote_exit": 5, "vote_win": 10,
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
HOST = CFG["server"].get("host", HOST)   # config server.host 覆盖（专属空间 | dashscope.aliyuncs.com 老公共端点）
VAD_JUDGE_MODE = CFG["server"]["omni"].get("judge_mode", "text")   # 语义 VAD 判断模式：text（老方案）/ audio（新方案音频直判）——并存可切换
print(f"[config] 已加载: {CONFIG_PATH.name if CONFIG_PATH.exists() else '内置默认'}")

# 模型变量（供 Provider 实例化——热生效时更新）
ASR_MODEL = CFG["server"]["asr"]["model"]
ASR_MODE = CFG["server"]["asr"].get("mode", "stream")   # stream 流式 / batch 非流式（⚙️ 面板可配）
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
        # prompt 支持 {playing} 占位符（config 可配）。用 replace 而非 format——
        # format 会把 prompt 里 JSON 模板的花括号（如 {"state": ...}）误当占位符
        # 抛 KeyError（2026-08-22 实测：chat 500 err='"state"'）
        system = self.prompt.replace("{playing}", playing)
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

    async def judge_audio(self, audio_b64: str, history: list[dict], last_replies=(),
                          is_replying: bool = False) -> tuple[str | None, str]:
        """音频直判：当前轮给音频（multimodal-generation），上下文给文本——不等 ASR 结果。
        实验分支 feat/audio-vad-judge：与 ASR 并行启动，判断完成即决策（打断/噪声更早）。
        返回 (None, reason)：调用失败/解析失败——外层调用方用已识别的 ASR 文本做文本判断兜底。"""
        try:
            playing = "是" if is_replying else "否"
        except Exception:
            playing = "否"
        system = self.prompt.replace("{playing}", playing)
        user_ctx = (
            f"AI 最近播放的回复：{json.dumps(list(last_replies or []), ensure_ascii=False)}\n"
            f"对话历史（最近3轮）：{json.dumps([m for m in history[-6:] if m.get('role') == 'user'][-3:] or history[-3:], ensure_ascii=False)}\n"
            "请结合上面上下文，判断用户这段音频（语音）的状态："
        )
        b64 = audio_b64 if "base64" in audio_b64 else f"data:audio/wav;base64,{audio_b64}"
        body = {
            "model": self.model,
            "input": {"messages": [
                {"role": "system", "content": [{"text": system}]},   # 状态定义+JSON 输出要求（漏了会导致模型自由回答非 JSON——实测"用户正在说话。"解析失败回退 rule→noise 误判）
                {"role": "user", "content": [{"audio": b64}, {"text": user_ctx}]}
            ]},
            "parameters": {"temperature": 0.0, "max_tokens": 64},
        }
        try:
            r = await _post_json(f"https://{self.host}/api/v1/services/aigc/multimodal-generation/generation",
                                 headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                                 timeout=15.0)
            out = r["output"]["choices"][0]["message"]["content"]
            if isinstance(out, list):
                out = "".join(x.get("text", "") for x in out if isinstance(x, dict))
            state = _extract_state(out)
            if state is None:
                log.warning("VADJUDGE cid=- omni-audio 输出无法解析: %r", out[:80])
                return None, "audio_parse_fail"   # 外层文本判断兜底（用 merged_text，勿用空串）
            return state, out[:60]
        except Exception as e:
            # 音频直判失败 → 返回 (None, reason)：让外层调用方按设计走"文本判断兜底"
            # （用已识别的 merged_text——传空串会 _is_noise_text('')=True 误判 noise 丢用户指令，
            #   2026-08-26 Bug 修复：原实现在内部吞异常 + 空串兜底 → 外层 L729/L736 兜底永远走不到）
            log.warning("VADJUDGE cid=- omni-audio 调用失败(%s) → 文本判断兜底", str(e)[:80])
            return None, f"audio_fail:{str(e)[:40]}"


def _extract_state(content: str) -> str | None:
    """从 Omni 输出提取 state（容忍 JSON 包裹/代码块/多余文本/纯文本状态词）。"""
    import re
    m = re.search(r'"state"\s*:\s*"([a-z_]+)"', content)
    if m:
        s = m.group(1)
    else:
        # 非 JSON 兜底：纯文本里的状态词（"状态是 complete"/"判断为 noise"——音频直判实测模型
        # 偶发自由回答"用户正在说话"——含状态词可提取，不含则 None 走回退）
        m2 = re.search(r'\b(complete|incomplete|backchannel|barge_in|tts_echo|reject|noise)\b', content)
        s = m2.group(1) if m2 else None
    if s is None:
        return None
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
tts_stream = Qwen3TTSStream(HOST, KEY, model=TTS_STREAM_MODEL, voice=TTS_VOICE)   # 流式 TTS 实例（每句一个 WS 会话；voice/model 取配置——修复死配置）


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


def _sse(d: dict) -> str:
    """事件 dict → SSE 帧文本（/api/chat、/api/chat_text、run_reply_pipeline 共用格式）。"""
    return "data: %s\n\n" % json.dumps(d, ensure_ascii=False)


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """原始 16bit/mono PCM 字节 → 16k WAV 字节（/ws/app 声学分段完成后拼给 ASR，
    与浏览器端 pcmToWavBase64() 输出格式一致，供 run_asr_semantic_vad 直接复用）。"""
    import struct
    n = len(pcm_bytes)
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", n)
    return header + pcm_bytes


def _local_tts(text: str, voice: str | None = None) -> tuple:
    """合成 TTS 并转本地缓存 → (本地 URL, 合成耗时ms, 公网下载耗时ms)。

    公网部分（DashScope 合成 API + 云端音频下载）单独计时——下载抖动
    会反映在'响应'里，需单独显示（前端 tts_sentence.dl_ms）。
    同文本同音色缓存命中时合成/下载均 0（无公网调用）。
    voice 缺省取全局 TTS_VOICE（面板 server.tts.voice——修复死配置：原硬编码 Cherry）。
    """
    import hashlib
    import urllib.request as _urlreq
    if voice is None:
        voice = TTS_VOICE   # 面板配置音色（修复死配置：原硬编码 Cherry）
    assert voice is not None
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
asr_stream = FunASRStreamProvider(host=HOST, api_key=KEY, model=ASR_MODEL)   # 真流式（fun-asr/paraformer——config asr.model 可配）


def _make_fast_llm():
    """快 LLM 按 provider 实例化（界面可配置）：
    ollama 本地（Ollama 原生 /api/chat）/ dashscope 云端（OpenAI 兼容）
    / openai 本地（任意 OpenAI 兼容端点 /v1/chat/completions，如内部部署 qwen3.5-4B）。"""
    if FAST_PROVIDER == "dashscope":
        # trust_env=False：绕过系统代理直连（macOS 系统代理指向 mihomo 7890，
        # 代理未运行时 httpx 全走死代理 → All connection attempts failed，2026-08-26 实测）
        return OpenAICompatLLMProvider(
            base_url=f"https://{HOST}/compatible-mode/v1", api_key=KEY, model=FAST_MODEL,
            trust_env=False)
    if FAST_PROVIDER == "openai":
        # 本地/内部 OpenAI 兼容端点——严格按用户验证过的调用方式（2026-08-26）：
        # requests 同步 + stream=False + NO_PROXY 直连 + Session-ID + 完整 /chat/completions URL
        # model 传空字符串（用户参考代码 MODEL_NAME=""——本地服务端不校验/用默认模型）
        return LocalOpenAICompatProvider(
            base_url=FAST_BASE or "http://127.0.0.1:8000/v1",
            api_key="", model="")
    # 默认 ollama 本地（base_url 可配自定义端点）
    return OllamaFastProvider(base_url=FAST_BASE, model=FAST_MODEL)


fast_llm = _make_fast_llm()
slow_llm = OpenAICompatLLMProvider(
    base_url=f"https://{HOST}/compatible-mode/v1", api_key=KEY, model=SLOW_MODEL,
    trust_env=False)
# TTS 构造带 voice/model（面板 server.tts.voice/batch_model 真正生效——修复死配置）
tts = Qwen3TTSProvider(host=HOST, api_key=KEY, model=TTS_BATCH_MODEL, voice=TTS_VOICE)


def _apply_config(cfg: dict) -> dict:
    """应用新配置（热生效，不重启）：更新模型变量 + prompt 注入 + 重建 Provider。"""
    global CFG, ASR_MODEL, FAST_MODEL, FAST_BASE, SLOW_MODEL, OMNI_MODEL
    global TTS_BATCH_MODEL, TTS_STREAM_MODEL, TTS_VOICE, TTS_SAMPLE_RATE
    global asr, asr_stream, fast_llm, slow_llm, tts, tts_stream, vad_judge
    global FAST_PROVIDER, ASR_MODE, HOST, VAD_JUDGE_MODE
    CFG = cfg
    s = cfg["server"]
    HOST = s.get("host", HOST)   # host 热生效（dashscope.aliyuncs.com 老公共端点切换）
    VAD_JUDGE_MODE = s["omni"].get("judge_mode", "text")   # 判断模式热生效（text/audio）
    ASR_MODEL = s["asr"]["model"]
    ASR_MODE = s["asr"].get("mode", "stream")
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
    asr_stream = FunASRStreamProvider(host=HOST, api_key=KEY, model=ASR_MODEL)   # 流式 ASR 也重建（host/model 切换即时生效——2026-08-24 补）
    fast_llm = _make_fast_llm()
    slow_llm = OpenAICompatLLMProvider(
        base_url=f"https://{HOST}/compatible-mode/v1", api_key=KEY, model=SLOW_MODEL,
        trust_env=False)
    # TTS 构造带 voice/model（面板 server.tts.voice/batch_model/stream_model 真正生效——
    # 修复死配置：原实现只读全局变量从不传给构造，永远用类默认 Cherry/默认模型）
    tts = Qwen3TTSProvider(host=HOST, api_key=KEY, model=TTS_BATCH_MODEL, voice=TTS_VOICE)
    tts_stream = Qwen3TTSStream(HOST, KEY, model=TTS_STREAM_MODEL, voice=TTS_VOICE)
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
TTS_CACHE.mkdir(parents=True, exist_ok=True)   # 运行时目录自动创建（git clone 不带空目录——Windows 必踩）
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


class ChatTextRequest(BaseModel):
    text: str                    # 用户输入文本（文字聊天——跳过 ASR/语义VAD，直接生成语音回复）
    client_id: str = "default"   # 多轮会话标识（与语音共享会话历史）


# ==================== 共享回复管线（快慢融合 LLM + 句子级并行 TTS） ====================
# /api/chat（SSE，语音）、/api/chat_text（SSE，文字）、/ws/app（WS，App 客户端）三处
# 复用同一份业务逻辑——避免后续三份维护漂移。事件 schema 与既有 SSE 完全一致
# （stage/fast/slow_first/slow_delta/audio_chunk/audio_end/tts_sentence/done），
# 调用方只需决定"如何发送"（SSE yield 字符串 / WS send_json 字典）。
async def run_asr_semantic_vad(wav_bytes: bytes, client_id: str, speech_start_ms: int = 0,
                                speech_end_ms: int = 0, is_replying: bool = False,
                                audio_b64: str | None = None):
    """整段 16k/16bit/mono WAV → ASR 流式识别 + 语义 VAD 判断 + 打断决策。

    /api/chat（SSE）、/ws/app（WS）共用——原 /api/chat 内联逻辑原样抽出，仅将
    req.xxx 换成参数（行为完全不变）。

    异步生成器：产出阶段事件 dict（stage/asr_partial/asr/vad_state/vad_incomplete/error，
    与既有 SSE JSON 载荷同构）。若最终判定"需要回复"（complete/barge_in），额外在末尾
    多 yield 一个内部标记事件 {"type": "_resolved", "text": <待回复文本>}；若判定为
    不回复（noise/tts_echo/backchannel/reject/incomplete）或 ASR 为空，则不产出该
    标记——调用方据此判断是否继续调 run_reply_pipeline。
    """
    audio_b64 = audio_b64 if audio_b64 is not None else base64.b64encode(wav_bytes).decode()
    log.info("REQ cid=%s audio=%.2fs speech_start=%s speech_end=%s",
             client_id, len(wav_bytes) / 32000, speech_start_ms or "-", speech_end_ms or "-")
    # 1.5 音频直判语义 VAD（实验分支 feat/audio-vad-judge）：当前轮音频直接给 omni
    # （multimodal-generation），上下文给文本——与 ASR 并行，判断完成即决策
    vad_task = None
    if VAD_JUDGE_MODE == "audio" and hasattr(vad_judge, "judge_audio"):
        vad_task = asyncio.create_task(vad_judge.judge_audio(
            audio_b64, HISTORIES.get(client_id, []),
            LAST_REPLY.get(client_id, ""), is_replying=is_replying))
        log.info("VADJUDGE cid=%s 音频直判并行启动（不等 ASR）", client_id)
    # 1. ASR 真流式（fun-asr：整段 wav → 帧流上行 → partial 实时推送）
    yield {"type": "stage", "stage": "asr"}
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
    # 按 ASR 模式选 provider（config server.asr.mode：stream 流式 partial / batch 整段）
    _asr = asr_stream if ASR_MODE == "stream" else asr
    async for evt in _asr.stream_recognize(gen_frames()):
        if evt.type == "asr.partial":
            asr_partial_count += 1
            if asr_first_partial_ms is None:
                asr_first_partial_ms = time.time() * 1000
            yield {"type": "asr_partial", "text": evt.payload.get("text", "")}
        elif evt.type == "asr.final":
            asr_text = evt.payload.get("text", "")
            asr_final_ms = time.time() * 1000
    asr_ms = int((time.time() - t0) * 1000)
    # 从人开始说话统计：出文字时延（start→首个 partial）；定稿时延（说完→final）
    partial_first_ms = (asr_first_partial_ms - speech_start_ms) if (asr_first_partial_ms and speech_start_ms) else None
    final_ms = (asr_final_ms - speech_end_ms) if (asr_final_ms and speech_end_ms) else None
    log.info("ASR cid=%s partial=%d text=%r asr_ms=%d final_ms=%s", client_id,
             asr_partial_count, asr_text, asr_ms,
             int(final_ms) if final_ms is not None else "-")
    yield {"type": "asr", "text": asr_text, "latency_ms": asr_ms,
           "partial_first_ms": int(partial_first_ms) if partial_first_ms is not None else None,
           "final_ms": int(final_ms) if final_ms is not None else None}
    if not asr_text:
        log.warning("FILTER cid=%s reason=empty_asr", client_id)
        yield {"type": "error", "msg": "未识别到有效语音（环境噪声）"}
        return
    # 语义 VAD（可插拔：rule 规则 / omni prompt 引导）——判断噪声/回声/拒识/应声/未说完
    # incomplete 续说合并：上一段判"没说完"则缓存，本段到达合并再判断（complete 才响应）
    pending = PENDING_INCOMPLETE.get(client_id)
    merged_text = asr_text
    if pending:
        if time.time() - pending["ts"] > PENDING_TTL_S:
            PENDING_INCOMPLETE.pop(client_id, None)   # 过期丢弃
            log.info("VADJUDGE cid=%s pending 过期丢弃", client_id)
        else:
            merged_text = pending["text"] + asr_text
            log.info("VADJUDGE cid=%s 续说合并: %r + %r", client_id, pending["text"], asr_text)
    t_vad = time.time()   # 语义 VAD 判断时延起点
    if vad_task is not None:
        # 音频直判（新方案 judge_mode=audio）：vad_task 与 ASR 并行——已完成直接取
        # （省判断时间）；失败/异常 → 回退文本判断（双保险——老方案并存）
        try:
            vstate, vreason = await vad_task
            if vstate is None:
                vstate, vreason = await vad_judge.judge(
                    merged_text, HISTORIES.get(client_id, []),
                    LAST_REPLY.get(client_id, ""), is_replying=is_replying)
            else:
                log.info("VADJUDGE cid=%s 音频直判: state=%s vad=%dms", client_id, vstate,
                         int((time.time() - t_vad) * 1000))
        except Exception as e:
            log.warning("VADJUDGE cid=%s 音频直判异常(%s) → 文本判断兜底", client_id, str(e)[:60])
            vstate, vreason = await vad_judge.judge(
                merged_text, HISTORIES.get(client_id, []),
                LAST_REPLY.get(client_id, ""), is_replying=is_replying)
    else:
        vstate, vreason = await vad_judge.judge(
            merged_text, HISTORIES.get(client_id, []), LAST_REPLY.get(client_id, ""),
            is_replying=is_replying)
    # 打断决策状态机：语义状态（模型判断）× 播放状态（确定性事实）→ 行为
    # 模型只判断"是否要响应"；打断与否由"AI 是否在播放 TTS"决定
    fsm = BargeDecisionFSM()
    action, vstate = fsm.decide(vstate, is_replying)
    vad_ms = int((time.time() - t_vad) * 1000)   # 语义 VAD 判断时延（仅 omni 有意义）
    log.info("VADJUDGE cid=%s semantic=%s playing=%s action=%s state=%s vad=%s vad_ms=%d",
             client_id, vreason[:30] if "(" in vreason else "omni", is_replying,
             action, vstate, VAD_JUDGE, vad_ms)
    yield {"type": "vad_state", "state": vstate, "vad": VAD_JUDGE, "latency_ms": vad_ms}
    if vstate == VadState.INCOMPLETE:
        # 没说完：缓存文本等待续说，不触发回复；前端继续聆听
        PENDING_INCOMPLETE[client_id] = {"text": merged_text, "ts": time.time()}
        log.info("VADJUDGE cid=%s incomplete 缓存待续 text=%r", client_id, merged_text)
        yield {"type": "vad_incomplete", "state": "incomplete", "vad": VAD_JUDGE}
        return
    if pending:
        PENDING_INCOMPLETE.pop(client_id, None)   # 本段完整，清缓存
        asr_text = merged_text                    # 回复基于完整语义
    if vstate == VadState.NOISE:
        log.warning("FILTER cid=%s reason=noise(%s) text=%r", client_id, vreason, asr_text)
        yield {"type": "error", "msg": "未识别到有效语音（环境噪声）"}
        return
    if vstate == VadState.TTS_ECHO:
        log.warning("FILTER cid=%s reason=tts_echo(%s) text=%r", client_id, vreason, asr_text)
        yield {"type": "error", "msg": "检测到回声，已忽略"}
        return
    if vstate == VadState.BACKCHANNEL:
        log.info("FILTER cid=%s reason=backchannel(%s) text=%r", client_id, vreason, asr_text)
        yield {"type": "error", "msg": "收到应声（嗯/对）"}
        return
    if vstate == VadState.REJECT:
        log.info("FILTER cid=%s reason=reject(%s) text=%r", client_id, vreason, asr_text)
        yield {"type": "error", "msg": "没听清，请再说一遍"}
        return
    if vstate == VadState.BARGE_IN:
        log.info("BARGE_IN cid=%s text=%r", client_id, asr_text)
        # barge-in 语义上等同完整指令（接管对话），走正常回复——WS 端另需额外
        # 发一条 {"type":"barge_in"} 通知客户端停播（见 /ws/app，语义 VAD 层面的
        # complete/barge_in 判定这里统一产出 _resolved，barge_in 专属通知由调用方叠加）
    # complete/incomplete → 正常处理
    yield {"type": "_resolved", "text": asr_text, "vstate": vstate}


async def run_reply_pipeline(user_text: str, client_id: str, t_start: float,
                              speech_end_ms: int = 0):
    """已确认的用户文本 → 快通道承接语 + 慢通道完整回复（并行）→ 句子级 TTS。

    异步生成器：yield 出事件 dict（含 "type" 字段，与 SSE JSON 载荷同构）。
    末尾 yield {"type":"done","total_ms":...}（total_ms 相对 t_start）。
    副作用：写 HISTORIES[client_id]/LAST_REPLY[client_id]（多轮上下文与回声过滤）。
    """
    import re as _re
    slow_events: asyncio.Queue = asyncio.Queue()

    async def run_slow():
        """慢通道生成 + 句子级 TTS 并行合成；事件推入 slow_events。"""
        try:
            history = HISTORIES.get(client_id, [])[-10:]  # 最近 10 轮
            messages = [{"role": "system", "content": fusion.slow_system_prompt()}]
            messages += history
            messages.append({"role": "user", "content": user_text})
            t0 = time.time()
            # 静音时延：人说完话 → 送 LLM（speech_end_ms=0 表示无该基准，如文字聊天）
            silence_ms = (slow_start_ms - speech_end_ms) if speech_end_ms else None
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

            async def _slow_stream(sub: str, i: int, q):
                """慢句分句流式（TTS_MODE=stream）：realtime WS 边合成边播——
                qwen3-tts-realtime 首块 ~0.5s 出声；长句拆短句（15-40 字）
                规避'长句流式中间等块卡顿'；失败回退整段。"""
                async with tts_sem:
                    try:
                        first_ms = await tts_stream.synth(
                            sub, lambda pcm, i=i: q.put_nowait(
                                ("audio_chunk", i, base64.b64encode(pcm).decode())))
                        q.put_nowait(("audio_end", i, first_ms))
                    except Exception as e:
                        log.error("STREAM_TTS_ERR cid=%s slow idx=%d err=%s → 回退整段",
                                  client_id, i, str(e)[:80])
                        try:
                            u, tts_ms, dl_ms = await asyncio.to_thread(_local_tts, sub)
                            q.put_nowait(("tts_sentence_fb", i, u, tts_ms, dl_ms))
                        except Exception as e2:
                            log.error("TTS_FB_ERR cid=%s slow idx=%d err=%s",
                                      client_id, i, str(e2)[:80])

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
                    # 慢句合成：流式模式=分句流式（realtime WS 首块 ~0.5s，拆短句防卡顿）；
                    # 整段模式=整段并行（~2.5s 就绪，承接语播完无缝接播）
                    if TTS_MODE["mode"] == "stream":
                        pending[i] = asyncio.create_task(_slow_stream(sub, i, slow_events))
                    else:
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
                    r = await pending[i]
                    if TTS_MODE["mode"] != "stream":
                        # 整段：事件在此发（流式任务已自行入队 audio_chunk/audio_end）
                        u, tts_ms, dl_ms = r
                        slow_events.put_nowait(
                            ("tts_sentence", i, u, int((time.time() - tts_t0[i]) * 1000), dl_ms))
                except Exception as e:
                    log.error("TTS_ERR cid=%s sentence=%d err=%s", client_id, i, e)
            slow_events.put_nowait((("slow_done",)))
        except Exception as e:
            log.error("SLOW_ERR cid=%s err=%s", client_id, e)
            slow_events.put_nowait((("slow_error", str(e))))

    slow_start_ms = time.time() * 1000   # 送 LLM 时刻（静音时延基准，create_task 前记录）
    slow_task = asyncio.create_task(run_slow())

    # 2. 快通道承接语（本地 4b）→ 生成完立即 TTS 合成播放（不等慢通道）
    #    可插拔：direct 策略（慢回复直达）不启动快通道——两套机制独立
    fast_text = ""
    fast_tts_url = None
    if fusion.should_fast(user_text):
        yield {"type": "stage", "stage": "fast"}
        t0 = time.time()
        try:
            fast_prompt = FusionPolicy.build_fast_prompt(user_text)
            acc = ""
            async for evt in fast_llm.stream_chat(fast_prompt):
                if evt.type == "llm.token":
                    acc += evt.payload.get("delta", "")
            fast_text = acc.strip()
            LAST_REPLY[client_id] = [fast_text]   # 回声过滤：记录实际播放的承接语
            fast_ms = int((time.time() - t0) * 1000)
            log.info("FAST cid=%s text=%r ms=%d", client_id, fast_text, fast_ms)
            yield {"type": "fast", "text": fast_text, "latency_ms": fast_ms,
                   "provider": FAST_PROVIDER, "model": FAST_MODEL}
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
                                      client_id, str(e)[:80])
                            try:
                                u, tts_ms, dl_ms = await asyncio.to_thread(_local_tts, text)
                                q.put_nowait(("tts_sentence_fb", -1, u, tts_ms, dl_ms))
                            except Exception as e2:
                                log.error("TTS_FB_ERR cid=%s fast err=%s",
                                          client_id, str(e2)[:80])
                        finally:
                            q.put_nowait(("fast_done",))

                    fast_task = asyncio.create_task(_fast_stream(fast_text, slow_events))
                    fast_tts_url = "stream"   # 标记流式（慢句 idx 偏移用）
                else:
                    u, tts_ms, dl_ms = await asyncio.to_thread(_local_tts, fast_text)
                    fast_tts_ms = int((time.time() - t0) * 1000)
                    fast_tts_url = u
                    log.info("TTS cid=%s idx=0 fast ms=%d dl_ms=%d", client_id, fast_tts_ms, dl_ms)
                    yield {"type": "tts_sentence", "idx": 0, "channel": "fast", "url": u,
                           "latency_ms": fast_tts_ms, "dl_ms": dl_ms}
        except Exception as e:
            log.error("FAST_ERR cid=%s err=%s", client_id, e)
    # 4. 消费慢通道事件（边生成边流式下发——承接语/慢句块并行，
    #    不再等 slow_task 全部完成才消费——修复慢句播放高延迟卡顿）
    yield {"type": "stage", "stage": "slow"}
    slow_text = ""
    got_slow_tts = False
    slow_done_evt = False
    fast_done_evt = False
    need_fast = fast_tts_url == "stream"   # 流式承接语已启动（等其 fast_done）
    while True:
        ev = await slow_events.get()
        if ev[0] == "slow_first":
            _, sf_ms, sil_ms = ev
            log.info("SLOW_FIRST cid=%s latency_ms=%d silence_ms=%s", client_id,
                     sf_ms, sil_ms if sil_ms is not None else "-")
            yield {"type": "slow_first", "latency_ms": sf_ms, "silence_ms": sil_ms}
        elif ev[0] == "slow_delta":
            slow_text += ev[1]
            yield {"type": "slow_delta", "delta": ev[1]}
        elif ev[0] == "audio_chunk":
            # 流式 TTS：PCM 块（base64）→ 前端 Web Audio 边收边播
            got_slow_tts = True   # 已有流式音频 → 兜底不再重复合成
            _, i, b64s = ev
            real_idx = i + (1 if fast_tts_url else 0)
            yield {"type": "audio_chunk", "idx": real_idx, "b64": b64s}
        elif ev[0] == "audio_end":
            got_slow_tts = True
            _, i, first_ms = ev
            real_idx = i + (1 if fast_tts_url else 0)
            yield {"type": "audio_end", "idx": real_idx, "first_ms": first_ms}
        elif ev[0] == "tts_sentence_fb":
            # 流式失败回退整段（tts_sentence 事件兼容前端）
            _, i, u, tts_ms, dl_ms = ev
            real_idx = i + (1 if fast_tts_url else 0)
            got_slow_tts = True
            yield {"type": "tts_sentence", "idx": real_idx, "channel": "slow", "url": u,
                   "latency_ms": tts_ms, "dl_ms": dl_ms}
        elif ev[0] == "tts_sentence":
            got_slow_tts = True
            _, i, u, ms, dl_ms = ev
            real_idx = i + (1 if fast_tts_url else 0)   # 承接语占 idx=0 → 慢句子偏移
            log.info("TTS cid=%s idx=%d slow ms=%d dl_ms=%d", client_id, real_idx, ms, dl_ms)
            yield {"type": "tts_sentence", "idx": real_idx, "channel": "slow", "url": u,
                   "latency_ms": ms, "dl_ms": dl_ms}
        elif ev[0] == "slow_done":
            slow_done_evt = True
        elif ev[0] == "slow_error":
            # run_slow 异常退出——按 slow_done 处理（防止消费循环永久等待）
            log.error("SLOW_DONE_ERR cid=%s err=%s", client_id, str(ev[1])[:80])
            slow_done_evt = True
        elif ev[0] == "fast_done":
            fast_done_evt = True
        if slow_done_evt and (not need_fast or fast_done_evt):
            break
    # 兜底：快/慢通道均无音频 → 整段合成（承接语已合成过则跳过）
    reply_for_tts = slow_text or fast_text
    if slow_text:
        last = LAST_REPLY.get(client_id, [])
        if fast_text not in last:
            last = [fast_text] if fast_text else []
        LAST_REPLY[client_id] = last + [slow_text]   # 实际播放列表 [承接语, 完整回复]
    if not fast_tts_url and not got_slow_tts and reply_for_tts:
        try:
            t0 = time.time()
            u, _tts_ms, _dl_ms = _local_tts(reply_for_tts)
            tts_ms = int((time.time() - t0) * 1000)
            log.info("TTS cid=%s fallback ms=%d", client_id, tts_ms)
            yield {"type": "tts_sentence", "idx": 0, "url": u, "latency_ms": tts_ms}
        except Exception as e:
            log.error("TTS_ERR cid=%s fallback err=%s", client_id, e)
    # 记入会话历史（多轮上下文）
    reply_for_hist = slow_text or fast_text or ""
    if user_text:
        h = HISTORIES.setdefault(client_id, [])
        h.append({"role": "user", "content": user_text})
        if reply_for_hist:
            h.append({"role": "assistant", "content": reply_for_hist})
        HISTORIES[client_id] = h[-MAX_HISTORY:]
    total_ms = int((time.time() - t_start) * 1000)
    log.info("DONE cid=%s total=%dms slow_text=%r fast_tts=%s slow_tts=%s hist=%d",
             client_id, total_ms, slow_text[:40], bool(fast_tts_url), got_slow_tts,
             len(HISTORIES.get(client_id, [])))
    yield {"type": "done", "total_ms": total_ms}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """音频 → ASR → 语义 VAD → 快慢融合 LLM → TTS。SSE 流式返回阶段事件。

    实现：run_asr_semantic_vad（ASR+语义VAD+打断决策） + run_reply_pipeline
    （快慢融合+TTS）两段共享逻辑串联——与 /api/chat_text、/ws/app 复用同一套核心，
    仅本函数负责"解 base64 wav + SSE 帧发送"。
    """
    if not KEY:
        return JSONResponse({"error": "DASHSCOPE_API_KEY 未配置"}, status_code=500)

    async def gen():
        t_start = time.time()
        try:
            wav_bytes = base64.b64decode(req.audio_b64)
            resolved_text = None
            async for ev in run_asr_semantic_vad(
                    wav_bytes, req.client_id, speech_start_ms=req.speech_start_ms,
                    speech_end_ms=req.speech_end_ms, is_replying=req.is_replying,
                    audio_b64=req.audio_b64):
                if ev["type"] == "_resolved":
                    resolved_text = ev["text"]
                    continue
                yield _sse(ev)
            if resolved_text is None:
                return   # 未判定需要回复（噪声/回声/应声/拒识/未说完/ASR空）
            async for pev in run_reply_pipeline(resolved_text, req.client_id, t_start,
                                                 speech_end_ms=req.speech_end_ms):
                yield _sse(pev)
        except Exception as e:
            log.error("CHAT_ERR cid=%s err=%s\n%s", req.client_id, e, traceback.format_exc())
            yield 'data: {"type":"error","msg":%s}\n\n' % json.dumps(str(e))

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/chat_text")
async def chat_text(req: ChatTextRequest):
    """文字聊天：用户发文字 → 快慢融合 LLM → TTS 语音回复。SSE 流式返回阶段事件。

    与 /api/chat（语音）共享快慢融合 + TTS 生成逻辑，但跳过 ASR 与语义 VAD——
    user_text 直接作为慢通道 LLM 输入，回复经 TTS 合成下发，前端播放。
    会话历史与语音共享（同一 client_id 多轮上下文连贯）。
    """
    if not KEY:
        return JSONResponse({"error": "DASHSCOPE_API_KEY 未配置"}, status_code=500)
    user_text = (req.text or "").strip()
    if not user_text:
        return JSONResponse({"error": "text 为空"}, status_code=400)
    cid = req.client_id

    async def gen():
        t_start = time.time()
        try:
            log.info("TXT_REQ cid=%s text=%r", cid, user_text[:60])
            # 快慢融合生成 + TTS：与 /api/chat（语音）、/ws/app 复用同一份实现
            async for pev in run_reply_pipeline(user_text, cid, t_start):
                yield _sse(pev)
        except Exception as e:
            log.error("TXT_CHAT_ERR cid=%s err=%s\n%s", cid, e, traceback.format_exc())
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


# ==================== App 客户端 WS 网关（/ws/app） ====================
# 方案：Flutter 客户端 + 服务端 VAD（架构方案 §3 Option B）+ 仅前台交互。
# 客户端只做本地能量粗判（UX 提示，local_duck），真正的"何时开始/结束说话"分段
# 由服务端 AcousticVadJudge（duplex_voice/vad/rule_judge.py）帧级判定——与浏览器
# 端现有的 JS 端 VAD（web/index.html）并存，互不影响。
# 分段完成后复用 run_asr_semantic_vad + run_reply_pipeline（与 /api/chat 同一套
# 核心逻辑），事件通过 WS send_json 推给客户端，事件 schema 与 SSE 版语义一致
# （架构方案 §4.2）。
WS_ACOUSTIC_CFG = _AcousticVadConfig()   # 声学分段参数（可后续接入 config.json 热调）


@app.websocket("/ws/app")
async def ws_app(ws: WebSocket):
    """App 客户端 WS 网关：持续 PCM 帧上行 → 声学分段 → ASR/语义VAD/快慢融合/TTS。

    协议（架构方案 §4.2）：
      ↑ {"type":"auth","token":...}                     鉴权（当前占位，未接入真实校验）
      ↑ {"type":"audio_frame","seq":n,"pcm_b64":"..."}   16k/16bit/mono，建议每帧480采样(30ms)
      ↑ {"type":"local_duck","on":true|false}            客户端本地粗判打断提示（仅UX，不裁决）
      ↑ {"type":"cancel"}                                用户主动打断：服务端停止当前生成
      ↑ {"type":"config_get"|"config_set", ...}          复用 /api/config 语义（当前占位）
      ↓ {"type":"stage"|"asr_partial"|"asr"|"vad_state"|"vad_incomplete"|
          "fast"|"slow_first"|"slow_delta"|"audio_chunk"|"audio_end"|
          "tts_sentence"|"done"|"error"}                 与既有 SSE 事件同构
      ↓ {"type":"barge_in"}                              服务端主动通知：确认打断，请停止播放
    """
    await ws.accept()
    client_id = f"app_{id(ws)}"
    acoustic = AcousticVadJudge(WS_ACOUSTIC_CFG)
    seg_buf = bytearray()          # 声学分段期间累积的 PCM 字节（16bit mono）
    seg_seq = 0                    # 帧序号计数（480 samples/frame 累计 → ts_ms 换算）
    is_replying = False            # 服务端确定性事实：当前是否在给该连接推 TTS 音频
    reply_task: asyncio.Task | None = None   # 当前正在跑的 ASR→回复 pipeline 任务（cancel 用）
    authed = True   # 占位：真实部署需校验 token 后才置 True（架构方案 §5 鉴权待办）

    async def send(d: dict):
        try:
            await ws.send_json(d)
        except Exception:
            pass

    async def run_one_segment(wav_bytes: bytes):
        """声学分段交付的一段音频 → ASR/语义VAD → （需要回复则）快慢融合+TTS。"""
        nonlocal is_replying
        t_start = time.time()
        try:
            resolved_text = None
            async for ev in run_asr_semantic_vad(
                    wav_bytes, client_id, is_replying=is_replying):
                if ev["type"] == "_resolved":
                    resolved_text = ev["text"]
                    if ev.get("vstate") == VadState.BARGE_IN:
                        # 语义确认打断：主动通知客户端停止当前播放（架构方案新增事件）
                        await send({"type": "barge_in"})
                        is_replying = False
                    continue
                await send(ev)
            if resolved_text is None:
                return
            is_replying = True
            acoustic.set_phase("speak")   # 声学层同步进入播放期（声学打断钳制生效）
            async for pev in run_reply_pipeline(resolved_text, client_id, t_start):
                await send(pev)
                if pev.get("type") == "done":
                    break
        except asyncio.CancelledError:
            log.info("WS_APP cid=%s 回复任务被 cancel 中断", client_id)
            raise
        except Exception as e:
            log.error("WS_APP_ERR cid=%s err=%s\n%s", client_id, e, traceback.format_exc())
            await send({"type": "error", "msg": str(e)})
        finally:
            is_replying = False
            acoustic.set_phase("listen")

    log.info("WS_APP cid=%s 连接建立", client_id)
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "auth":
                # TODO(架构方案 §5)：接入真实 token 校验（App 登录态），当前占位放行
                authed = True
                await send({"type": "auth_ok"})
            elif mtype == "audio_frame":
                if not authed:
                    continue
                pcm = base64.b64decode(msg.get("pcm_b64", ""))
                if not pcm:
                    continue
                seg_buf.extend(pcm)
                # 按 480 样本(960字节)/30ms 切帧喂 AcousticVadJudge（服务端声学分段——
                # 架构方案确定的"VAD放服务端"落地点，替代浏览器端 JS 分段）
                flen_bytes = FRAME_LEN * 2
                off = 0
                while off + flen_bytes <= len(pcm):
                    frame_pcm = np.frombuffer(pcm[off:off + flen_bytes], dtype=np.int16)
                    seg_seq += 1
                    now_ms = seg_seq * 30
                    decisions = acoustic.feed(_SimpleFrame(frame_pcm, now_ms))
                    for d in decisions:
                        if d.event_type == "speech_end" and not is_replying:
                            # 交付完整语音段给 ASR（听期分段完成）
                            wav_bytes = _pcm16_to_wav(bytes(seg_buf))
                            seg_buf.clear()
                            if reply_task and not reply_task.done():
                                reply_task.cancel()
                            reply_task = asyncio.create_task(run_one_segment(wav_bytes))
                        elif d.event_type == "takeover_noise" and is_replying:
                            # 播放期声学打断：先行通知（语义确认在 ASR 完成后由
                            # run_one_segment 内部 barge_in 事件补发/撤销）
                            log.info("WS_APP cid=%s 声学打断触发", client_id)
                    off += flen_bytes
                if not acoustic.is_voicing():
                    # 非说话中（静音）：不在语音段内的音频不入 seg_buf，防止累积过大
                    if len(seg_buf) > 16000 * 2 * 30:   # 30s 兜底上限，防异常持续增长
                        seg_buf.clear()
            elif mtype == "local_duck":
                # 客户端本地能量粗判打断提示——仅日志/UX参考，不参与裁决（架构方案 §4.2）
                log.debug("WS_APP cid=%s local_duck=%s", client_id, msg.get("on"))
            elif mtype == "cancel":
                # 用户主动打断：取消当前回复任务，通知客户端停播
                if reply_task and not reply_task.done():
                    reply_task.cancel()
                is_replying = False
                acoustic.set_phase("listen")
                await send({"type": "barge_in"})
                log.info("WS_APP cid=%s 用户主动 cancel", client_id)
            elif mtype == "config_get":
                await send({"type": "config", "config": _mask_cfg(CFG)})
            elif mtype == "config_set":
                # TODO(架构方案 §5)：App 端仅暴露用户级设置（音色/语速），当前占位不落盘
                await send({"type": "config_set_ack", "ok": False,
                            "msg": "config_set 暂未实现（占位，见架构方案§5待办）"})
            else:
                await send({"type": "error", "msg": f"未知消息类型: {mtype}"})
    except WebSocketDisconnect:
        log.info("WS_APP cid=%s 连接断开", client_id)
    except Exception as e:
        log.error("WS_APP_FATAL cid=%s err=%s\n%s", client_id, e, traceback.format_exc())
    finally:
        if reply_task and not reply_task.done():
            reply_task.cancel()


class _SimpleFrame:
    """AcousticVadJudge.feed() 期望的 frame 对象最小实现（.pcm/.ts_ms）。"""

    __slots__ = ("pcm", "ts_ms")

    def __init__(self, pcm: np.ndarray, ts_ms: int):
        self.pcm = pcm
        self.ts_ms = ts_ms


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
