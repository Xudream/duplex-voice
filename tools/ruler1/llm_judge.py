#!/usr/bin/env python3
"""被测系统适配器：把 manifest 里的样本送进"裁判"模型，产出 predictions.jsonl。

支持三种后端（同一套动作协议）：
  --backend openai_compat   任意 OpenAI 兼容端点（DashScope 兼容模式 / vLLM / Ollama / DeepSeek）
  --backend jev             TypeSafe System One（有界决策：动作集合作为 Choice 选项）
  --backend dryrun          不调模型，只校验 prompt 组装与解析链路（CI 用）

用法:
  python3 llm_judge.py --manifest manifests/chorus.jsonl --backend dryrun --limit 20 \
      --out predictions/dryrun__chorus.jsonl
  DASHSCOPE_API_KEY=... python3 llm_judge.py --manifest manifests/chorus.jsonl \
      --backend openai_compat --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
      --model qwen-plus --out predictions/qwen-plus__chorus.jsonl

  # 跑道 A 两版：--text-source gold（用集自带转写） / asr（用 ASR 结果，需先跑 asr.py）
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.request

ACTIONS = ["respond", "ack", "wait", "clarify", "silent", "silent_record"]
NEG = {"S1", "S2", "S3", "S4"}

SYSTEM_PROMPT = """你是语音助手的"拒识裁判"，只判断这段语音是否该响应，不生成回答内容。

判定规则：
1. 语音是用户在跟助手说话（下指令、提问、回答助手刚才的问题）→ respond
2. 语音是用户对身边的其他人说话，或旁人之间的对话，或电视/媒体人声 → silent
3. 语音是"嗯/啊/对对"这类附和，或用户还在思考没有说完 → ack（附和）或 wait（等待）
4. 无法判断但可能是对助手说的 → clarify（轻确认）

场景标签：S0=对助手说（应响应）；S1=主用户转向旁人；S2=旁人对此用户说；S3=旁人之间；S4=媒体/电子人声。
输出必须是严格 JSON，不要任何解释文字：
{"action": "respond|ack|wait|clarify|silent|silent_record", "confidence": 0.0-1.0, "reason": "不超过 20 字"}"""


SYSTEM_V2 = SYSTEM_PROMPT + """

补充判定规则（v2）：
1. 转写可能来自**多个说话人**且没有说话人标记（无空间线索时的 ASR 流）。以**最后一句**为准；前面的话只当背景。
2. 后面这几类话**不是**在对设备说话，应静默：对旁人陈述/闲聊、评价、抱怨、提及第三方、自言自语、念稿或念屏幕上的文字。
3. 只有"需要设备提供信息或执行动作"的请求才 respond；单纯的情感表达或附和用 ack。""" 


SYSTEM_V3 = SYSTEM_V2 + """

输出格式（v3）：
{"action": "...", "p_silent": 0-100 的整数, "confidence": 0.0-1.0, "reason": "≤20 字"}
p_silent = 你认为"这句话不该由助手回"的概率（百分比），请给出连续分布的判断，不要总是 95/98。"""


def build_user_prompt(sample, text, text_mode="full"):
    hist = sample.get("history") or []
    hist_txt = "\n".join(f"{'助手' if h.get('role')=='assistant' else '用户'}：{h.get('text','')}"
                         for h in hist[-5:]) or "（无）"
    ctx = ""
    if text_mode == "last":
        segs = sample.get("segments") or []
        cur = sample.get("target_text") or text
        if len(segs) > 1:
            ctx = "\n".join(x["text"] for x in segs[:-1])
    else:
        cur = text
    if ctx:
        return (f"系统状态：{'正在播报' if sample.get('system_state')=='speaking' else '空闲'}\n"
                f"最近对话：\n{hist_txt}\n\n"
                f"此前转写片段（背景，可能来自旁人/助手，不是待判内容）：{ctx}\n\n"
                f"当前语音（待判，仅这一段）：{cur}\n\n请输出 JSON。")
    return (f"系统状态：{'正在播报' if sample.get('system_state')=='speaking' else '空闲'}\n"
            f"最近对话：\n{hist_txt}\n\n"
            f"待判语音（转写）：{cur}\n\n"
            f"请输出 JSON。")


def parse_action(raw):
    if not raw:
        return "invalid", 0.0, "empty"
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        low = raw.lower()
        for a in ACTIONS:
            if a in low:
                return a, 0.5, "no-json"
        return "invalid", 0.0, raw[:40]
    try:
        j = json.loads(m.group(0))
    except Exception:
        return "invalid", 0.0, "bad-json"
    a = str(j.get("action", "")).strip().lower()
    if a not in ACTIONS:
        return "invalid", float(j.get("confidence", 0) or 0), "bad-action:" + a[:20]
    try:
        c = float(j.get("confidence", 0.5))
    except Exception:
        c = 0.5
    return a, c, str(j.get("reason", ""))[:60]


def call_openai_compat(base_url, model, api_key, system, user, timeout=60, temperature=0):
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 120}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def call_jev(base_url, model, api_key, system, user, timeout=60):
    """TypeSafe System One：state + questions → 概率分布。动作集合作为 Choice 选项。"""
    body = {"model": model, "state": f"{system}\n\n{user}",
            "questions": [{"type": "choice", "id": "action", "choices": ACTIONS}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    # 期望返回 {"answers":[{"id":"action","distribution":{...},"value":...}]}
    ans = (d.get("answers") or [{}])[0]
    dist = ans.get("distribution") or {}
    if dist:
        a = max(dist.items(), key=lambda kv: kv[1])[0]
        return json.dumps({"action": a, "confidence": dist[a], "reason": "jev"})
    return json.dumps({"action": ans.get("value", ""), "confidence": 0.5, "reason": "jev"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["openai_compat", "jev", "dryrun"], default="dryrun")
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v1")
    ap.add_argument("--text-mode", choices=["full", "last"], default="last",
                    help="last=只判当前 VAD 段（产品口径，HumDial 等多说话人集必需）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    key = a.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if a.backend != "dryrun" and not key:
        sys.exit("缺少 API key（--api-key 或环境变量）")

    samples = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        samples = samples[:a.limit]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n_ok = n_invalid = 0
    sys_prompt = {"v2": SYSTEM_V2, "v3": SYSTEM_V3}.get(a.prompt_version, SYSTEM_PROMPT)
    with open(a.out, "w", encoding="utf-8") as o:
        for s in samples:
            text = s.get("transcript") or ""
            user_prompt = build_user_prompt(s, text, a.text_mode)
            t0 = time.time()
            if a.backend == "dryrun":
                raw = json.dumps({"action": "silent" if s.get("gold_scene") in NEG else "respond",
                                  "confidence": 0.9, "reason": "dryrun-oracle"})
                lat = 0
            else:
                try:
                    if a.backend == "jev":
                        raw = call_jev(a.base_url, a.model, key, sys_prompt, user_prompt, a.timeout)
                    else:
                        raw = call_openai_compat(a.base_url, a.model, key, sys_prompt, user_prompt, a.timeout)
                except urllib.error.HTTPError as e:
                    raw = json.dumps({"action": "invalid", "confidence": 0.0,
                                      "reason": f"http{e.code}"})
                except Exception as e:
                    raw = json.dumps({"action": "invalid", "confidence": 0.0,
                                      "reason": type(e).__name__})
                lat = int((time.time() - t0) * 1000)
                if a.sleep:
                    time.sleep(a.sleep)
            act, conf, reason = parse_action(raw)
            p_silent = None
            try:
                m2 = re.search(r'\{.*\}', raw or "", re.S)
                if m2:
                    p_silent = json.loads(m2.group(0)).get("p_silent")
            except Exception:
                p_silent = None
            n_ok += act != "invalid"
            n_invalid += act == "invalid"
            o.write(json.dumps({"id": s["id"], "model": a.model, "p_silent": p_silent, "action": act,
                                "confidence": conf, "reason": reason, "latency_ms": lat,
                                "raw": raw[:500] if isinstance(raw, str) else ""},
                               ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(samples)} 条，可解析 {n_ok}，invalid {n_invalid}")


if __name__ == "__main__":
    main()
