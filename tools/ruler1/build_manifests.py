#!/usr/bin/env python3
"""尺子1 manifest 生成器：把各公开集映射成统一的 S0–S4 任务清单。

用法:
  python3 build_manifests.py --dataset chorus   --data ~/.hermes/workspace/ruler1/data/chorus   --out manifests
  python3 build_manifests.py --dataset wearvox  --data ~/.hermes/workspace/ruler1/data/wearvox  --out manifests
  python3 build_manifests.py --dataset humdial  --data ~/.hermes/workspace/ruler1/data/humdial  --out manifests
  python3 build_manifests.py --dataset all      --data ~/.hermes/workspace/ruler1/data          --out manifests
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

HISTORY_N = 5


def w(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {path}  {len(rows)} 条")


# ---------------- ChorusDuplex-SideTalk ----------------
def build_chorus(root, out):
    rows = []
    for ann_path in sorted(glob.glob(os.path.join(root, "annotations.jsonl"))):
        dialogues = [json.loads(l) for l in open(ann_path, encoding="utf-8") if l.strip()]
        for d in dialogues:
            did = d["dialogue_id"]
            tls = glob.glob(os.path.join(root, "audio", f"{did}.timeline.json"))
            if not tls:
                continue
            timeline = json.load(open(tls[0], encoding="utf-8"))
            flac = os.path.join(root, "audio", f"{did}.flac")
            if not os.path.exists(flac):
                continue

            # 主用户 = 对助手说话最多的人（数据驱动，避免硬编码 user_01）
            to_assistant = Counter()
            for t in timeline:
                addr = t.get("addressee")
                al = addr if isinstance(addr, list) else [addr]
                if "assistant" in al and t.get("speaker", "").startswith("user"):
                    to_assistant[t["speaker"]] += 1
            primary = to_assistant.most_common(1)[0][0] if to_assistant else "user_01"

            hist = []
            for idx, t in enumerate(timeline):
                spk, kind = t.get("speaker", ""), t.get("kind", "speech")
                addr = t.get("addressee")
                al = addr if isinstance(addr, list) else [addr]
                is_human = spk.startswith("user") or spk.startswith("third_party")

                # 只把"人说的话"作为待判样本；assistant 轮是系统输出，跳过
                if not is_human:
                    hist.append({"role": "assistant", "text": t.get("text", "")})
                    hist = hist[-HISTORY_N:]
                    continue

                if kind in ("backchannel", "interruption"):
                    scene, gold = "REG", "silent"      # 仅回归，不进 sideTalk 指标
                elif "assistant" in al and spk == primary:
                    scene, gold = "S0", "respond"
                elif "assistant" in al:
                    scene, gold = "S0", "respond"      # 旁人直接对设备说（访客）→ 归 S0 对照
                else:
                    if spk == primary:
                        scene, gold = "S1", "silent"   # 主用户转向旁人
                    elif primary in al:
                        scene, gold = "S2", "silent"   # 旁人对主用户说
                    else:
                        scene, gold = "S3", "silent"   # 旁人之间

                rows.append({
                    "id": f"chorus-{did}-{t['turn_id']:03d}",
                    "dataset": "chorus", "split": "test",
                    "audio": f"{did}.flac", "start_sec": t["start_sec"], "end_sec": t["end_sec"],
                    "channels": 2, "layout": "stereo_L_nonassistant_R_assistant",
                    "system_state": "speaking" if hist and hist[-1]["role"] == "assistant" else "idle",
                    "history": hist[-HISTORY_N:],
                    "gold_action": gold, "gold_scene": scene, "gold_lang": "zh",
                    "kind": kind, "speaker": spk, "addressee": addr,
                    "primary": primary, "transcript": t.get("text", ""),
                    "meta": {"dialogue": did, "turn_id": t["turn_id"], "synth": True},
                })
                hist.append({"role": "user", "text": t.get("text", "")})
                hist = hist[-HISTORY_N:]
    w(os.path.join(out, "chorus.jsonl"), rows)
    return rows


# ---------------- WearVox ----------------
def build_wearvox(root, out):
    pub = json.load(open(os.path.join(root, "data_public.json"), encoding="utf-8"))
    items = [x for x in pub if x.get("task") == "non-assistant-directed"]
    rows = []
    for x in items:
        gt = str(x.get("ground_truth"))
        sid = x.get("id", "")
        if gt == "[Mute]":
            scene = "S3" if "bystander_only" in sid else "S1"
            gold = "silent"
        else:
            scene, gold = "S0", "respond"
        single = x.get("audio_query")
        rows.append({
            "id": f"wearvox-{sid}",
            "dataset": "wearvox", "split": "test",
            "audio": single, "audio_mc": x.get("audio_query_mc"),
            "channels": 1, "layout": "mono_glasses",
            "system_state": "idle", "history": [],
            "gold_action": gold, "gold_scene": scene, "gold_lang": "en",
            "transcript": x.get("gt_transcript", ""),
            "meta": {"task": x["task"], "raw_gt": gt, "subtype": "bystander_only" if "bystander_only" in sid else "unrelated_talk"},
        })
    w(os.path.join(out, "wearvox.jsonl"), rows)
    return rows


# ---------------- HumDial-FDBench ----------------
# 解压后的目录结构：test/<cn|en>_test_nondev/<intent>/*.wav + *.json
# 意图 → 场景：
#   talk_to_others              → S1（主用户转向旁人 SDO）
#   others_talk_to_user_before  → S2（旁人对主用户，插话前）
#   others_talk_to_user_after   → S2（旁人对主用户，插话后）
#   backchannel                 → REG（URB，仅回归）
#   pause                       → REG（PH，仅回归）
HUMDIAL_MAP = {
    "talk_to_others": ("S1", "silent"),
    "others_talk_to_user_before": ("S2", "silent"),
    "others_talk_to_user_after": ("S2", "silent"),
    "backchannel": ("REG", "silent"),
    "pause": ("REG", "silent"),
}


def build_humdial(root, out, splits=("test",)):
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
            for intent, (scene, gold) in HUMDIAL_MAP.items():
                idir = os.path.join(lroot, intent)
                if not os.path.isdir(idir):
                    continue
                for f in sorted(os.listdir(idir)):
                    if not f.endswith(".wav"):
                        continue
                    # 同名 json 提供分段与转写（xmin/xmax/text）
                    stem = f[:-4]
                    jp = os.path.join(idir, stem + ".json")
                    segs, text = [], ""
                    if os.path.exists(jp):
                        try:
                            j = json.load(open(jp, encoding="utf-8"))
                            segs = j.get("speech_segments", []) or j.get("chunks", [])
                            text = j.get("text") or " ".join(
                                str(x.get("text", "")) for x in segs)
                        except Exception:
                            pass
                    rows.append({
                        "id": f"humdial-{split}-{lang}-{intent}-{stem}",
                        "dataset": "humdial", "split": split,
                        "audio": os.path.join(idir, f), "channels": 2,
                        "layout": "dual_channel", "system_state": "idle", "history": [],
                        "gold_action": gold, "gold_scene": scene, "gold_lang": lang,
                        "transcript": text[:2000],
                        "meta": {"intent": intent, "lang_dir": lang_dir,
                                 "segments": len(segs) if isinstance(segs, list) else 0},
                    })
    if rows:
        w(os.path.join(out, "humdial_test.jsonl"), rows)
    else:
        print("  [HumDial] 未找到 wav（确认已解压 test/<cn|en>_test_nondev/<intent>/）", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["chorus", "wearvox", "humdial", "all"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="manifests")
    a = ap.parse_args()

    def run(name, fn, sub):
        root = os.path.join(a.data, sub) if name == "all" else a.data
        if not os.path.isdir(root):
            print(f"[{name}] 目录不存在：{root}", file=sys.stderr); return []
        print(f"[{name}] {root}")
        return fn(root, a.out)

    if a.dataset in ("chorus", "all"):
        r = run("chorus", build_chorus, "chorus")
        if r: print("  scene:", dict(Counter(x["gold_scene"] for x in r)))
    if a.dataset in ("wearvox", "all"):
        r = run("wearvox", build_wearvox, "wearvox")
        if r: print("  scene:", dict(Counter(x["gold_scene"] for x in r)))
    if a.dataset in ("humdial", "all"):
        r = run("humdial", build_humdial, "humdial")
        if r: print("  scene:", dict(Counter(x["gold_scene"] for x in r)))


if __name__ == "__main__":
    main()
