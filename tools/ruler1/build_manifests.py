#!/usr/bin/env python3
"""尺子1 manifest 生成器：把各公开集映射成统一的 S0–S4 任务清单。

v1.1 修订：
  - 每条样本带 audio_root，路径统一为「audio_root + audio」，避免相对/绝对混用
  - Chorus：非主用户对助手说 → 单列 S0_guest（访客指令，设计文档 §1.3.1 范围外，不计入 FRR）
  - HumDial：raw 与 clean_ 是同一句的两个版本 → 只取 raw，clean 挂 audio_clean 字段；约一半样本因此不再重复计数
  - HumDial 转写：优先 <stem>.json，缺则回退 <stem>_timestamp.json 的 chunks
  - HumDial 默认只收 cn（en 只解压了 3 个文件，避免污染统计）

用法:
  python3 build_manifests.py --dataset all --data ~/.hermes/workspace/ruler1/data --out manifests
"""
import argparse, glob, json, os, sys
from collections import Counter

HISTORY_N = 5


def mark_audio(rows):
    """标注音频是否已就位，判分时可按 audio_ok 过滤（避免把缺文件算成模型错）。"""
    for r in rows:
        r["audio_ok"] = os.path.exists(os.path.join(r.get("audio_root", ""), r["audio"]))
    miss = sum(1 for r in rows if not r["audio_ok"])
    if miss:
        print(f"  [警告] {miss}/{len(rows)} 条音频缺失（未下载完整），已标 audio_ok=false", file=sys.stderr)
    return rows


def w(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {path}  {len(rows)} 条")


# ---------------- ChorusDuplex-SideTalk ----------------
def build_chorus(root, out):
    rows, missing_tl = [], []
    ann = os.path.join(root, "annotations.jsonl")
    dialogues = [json.loads(l) for l in open(ann, encoding="utf-8") if l.strip()]
    for d in dialogues:
        did = d["dialogue_id"]
        tls = glob.glob(os.path.join(root, "audio", f"{did}.timeline.json"))
        if not tls:
            missing_tl.append(did)
            continue
        timeline = json.load(open(tls[0], encoding="utf-8"))
        if not os.path.exists(os.path.join(root, "audio", f"{did}.flac")):
            continue

        to_assistant = Counter()
        for t in timeline:
            addr = t.get("addressee")
            al = addr if isinstance(addr, list) else [addr]
            if "assistant" in al and str(t.get("speaker", "")).startswith("user"):
                to_assistant[t["speaker"]] += 1
        primary = to_assistant.most_common(1)[0][0] if to_assistant else "user_01"

        hist = []
        for t in timeline:
            spk, kind = t.get("speaker", ""), t.get("kind", "speech")
            addr = t.get("addressee")
            al = addr if isinstance(addr, list) else [addr]

            if not (spk.startswith("user") or spk.startswith("third_party")):
                hist.append({"role": "assistant", "text": t.get("text", "")})
                hist = hist[-HISTORY_N:]
                continue

            if kind in ("backchannel", "interruption"):
                scene, gold = "REG", "silent"
            elif "assistant" in al and spk == primary:
                scene, gold = "S0", "respond"
            elif "assistant" in al:
                scene, gold = "S0_guest", "respond"   # 访客指令：设计文档范围外，单列不参与 FRR
            elif spk == primary:
                scene, gold = "S1", "silent"
            elif primary in al:
                scene, gold = "S2", "silent"
            else:
                scene, gold = "S3", "silent"

            rows.append({
                "id": f"chorus-{did}-{t['turn_id']:03d}",
                "dataset": "chorus", "split": "test",
                "audio_root": os.path.join(root, "audio"), "audio": f"{did}.flac",
                "start_sec": t["start_sec"], "end_sec": t["end_sec"],
                "channels": 2, "layout": "stereo_L_nonassistant_R_assistant",
                "system_state": "speaking" if hist and hist[-1]["role"] == "assistant" else "idle",
                "history": hist[-HISTORY_N:],
                "gold_action": gold, "gold_scene": scene, "gold_lang": "zh",
                "transcript": t.get("text", ""),
                "meta": {"dialogue": did, "turn_id": t["turn_id"], "kind": kind,
                         "speaker": spk, "addressee": addr, "primary": primary, "synth": True},
            })
            hist.append({"role": "user", "text": t.get("text", "")})
            hist = hist[-HISTORY_N:]
    mark_audio(rows)
    w(os.path.join(out, "chorus.jsonl"), rows)
    if missing_tl:
        print(f"  [Chorus] {len(missing_tl)} 段对话缺 timeline，已跳过：{missing_tl}", file=sys.stderr)
    return rows


# ---------------- WearVox ----------------
def build_wearvox(root, out):
    pub = json.load(open(os.path.join(root, "data_public.json"), encoding="utf-8"))
    items = [x for x in pub if x.get("task") == "non-assistant-directed"]
    rows = []
    for x in items:
        gt, sid = str(x.get("ground_truth")), x.get("id", "")
        if gt == "[Mute]":
            scene = "S3" if "bystander_only" in sid else "S1"
            gold = "silent"
        else:
            scene, gold = "S0", "respond"
        rows.append({
            "id": f"wearvox-{sid}",
            "dataset": "wearvox", "split": "test",
            "audio_root": root, "audio": x.get("audio_query"),
            "audio_mc": x.get("audio_query_mc"),
            "channels": 1, "layout": "mono_glasses",
            "system_state": "idle", "history": [],
            "gold_action": gold, "gold_scene": scene, "gold_lang": "en",
            "transcript": x.get("gt_transcript", ""),
            "meta": {"task": x["task"], "raw_gt": gt,
                     "subtype": "bystander_only" if "bystander_only" in sid else "unrelated_talk"},
        })
    mark_audio(rows)
    w(os.path.join(out, "wearvox.jsonl"), rows)
    return rows


# ---------------- HumDial-FDBench ----------------
HUMDIAL_MAP = {
    "talk_to_others": ("S1", "silent"),
    "others_talk_to_user_before": ("S2", "silent"),
    "others_talk_to_user_after": ("S2", "silent"),
    "backchannel": ("REG", "silent"),
    "pause": ("REG", "silent"),
}


def _humdial_text(idir, stem):
    p = os.path.join(idir, stem + ".json")
    if os.path.exists(p):
        try:
            j = json.load(open(p, encoding="utf-8"))
            if j.get("text"):
                return j["text"]
            segs = j.get("speech_segments") or []
            if segs:
                return " ".join(str(s.get("text", "")) for s in segs)
        except Exception:
            pass
    p2 = os.path.join(idir, stem + "_timestamp.json")
    if os.path.exists(p2):
        try:
            j = json.load(open(p2, encoding="utf-8"))
            chunks = j.get("chunks") or []
            if chunks:
                return "".join(str(c.get("text", "")) for c in chunks)
            return j.get("text", "") or ""
        except Exception:
            pass
    return ""


def _humdial_segments(idir, stem):
    """返回 [{start,end,text}]；产品口径只判最后一段（当前 VAD 段）。"""
    for name in (stem + ".json", stem + "_timestamp.json"):
        fp = os.path.join(idir, name)
        if not os.path.exists(fp):
            continue
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        segs = d.get("speech_segments") or d.get("segments") or []
        out = []
        for x in segs:
            t = (x.get("text") or "").strip()
            if t:
                out.append({"start": x.get("xmin", x.get("start")), "end": x.get("xmax", x.get("end")), "text": t})
        if out:
            return out
    return []


def build_humdial(root, out, splits=("test",), include_en=False):
    rows = []
    for split in splits:
        sroot = os.path.join(root, split)
        if not os.path.isdir(sroot):
            continue
        for lang_dir in sorted(os.listdir(sroot)):
            lroot = os.path.join(sroot, lang_dir)
            if not os.path.isdir(lroot):
                continue
            lang = "zh" if lang_dir.startswith("cn") else "en"
            if lang == "en" and not include_en:
                continue
            for intent, (scene, gold) in HUMDIAL_MAP.items():
                idir = os.path.join(lroot, intent)
                if not os.path.isdir(idir):
                    continue
                for f in sorted(os.listdir(idir)):
                    # raw 与 clean_ 是同一句的两版：只收 raw，clean 作为附加字段
                    if not f.endswith(".wav") or f.startswith("clean_"):
                        continue
                    stem = f[:-4]
                    clean_rel = f"clean_{f}"
                    _seg = _humdial_segments(idir, stem)
                    rows.append({
                        "id": f"humdial-{split}-{lang}-{intent}-{stem}",
                        "dataset": "humdial", "split": split,
                        "audio_root": idir, "audio": f,
                        "audio_clean": clean_rel if os.path.exists(os.path.join(idir, clean_rel)) else None,
                        "channels": 2, "layout": "dual_channel",
                        "system_state": "idle", "history": [],
                        "gold_action": gold, "gold_scene": scene, "gold_lang": lang,
                        "transcript": _humdial_text(idir, stem),
                        "segments": _seg,
                        "target_text": (_seg[-1]["text"] if _seg else ""),
                        "meta": {"intent": intent, "lang_dir": lang_dir},
                    })
    if rows:
        mark_audio(rows)
        w(os.path.join(out, "humdial_test.jsonl"), rows)
    else:
        print("  [HumDial] 未找到 wav（确认已解压 test/<cn|en>_test_nondev/<intent>/）", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["chorus", "wearvox", "humdial", "all"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--include-en", action="store_true", help="HumDial 英文部分（需先解压）")
    a = ap.parse_args()

    def run(name, fn, sub):
        # 总目录模式：子集在 <data>/<sub>；单集模式：直接给到该集目录
        cand = os.path.join(a.data, sub)
        root = cand if os.path.isdir(cand) else a.data
        if not os.path.isdir(root):
            print(f"[{name}] 目录不存在：{root}", file=sys.stderr)
            return []
        print(f"[{name}] {root}")
        return fn(root, a.out)

    for name, fn, sub in [("chorus", build_chorus, "chorus"),
                          ("wearvox", build_wearvox, "wearvox"),
                          ("humdial", lambda r, o: build_humdial(r, o, include_en=a.include_en), "humdial")]:
        if a.dataset in (name, "all"):
            r = run(name, fn, sub)
            if r:
                print("  scene:", dict(sorted(Counter(x["gold_scene"] for x in r).items())))


if __name__ == "__main__":
    main()
