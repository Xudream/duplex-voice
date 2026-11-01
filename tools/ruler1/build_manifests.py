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
def _walk(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            yield os.path.join(dirpath, f)


def build_humdial(root, out):
    """HumDial 的标注文件名/字段依赖解压结果；这里做尽力解析并在缺失时给出明确报错。"""
    cmap = {}
    for p in _walk(root):
        if p.endswith((".json", ".jsonl", ".csv", ".tsv", ".txt")) and "labels" in p.lower() or p.endswith(".jsonl"):
            cmap[p] = True
    files = list(_walk(root))
    if not files:
        print("  [HumDial] 数据目录为空，跳过（需先解压 Humdial-Track2-Test.zip）", file=sys.stderr)
        return []
    print(f"  [HumDial] 发现 {len(files)} 个文件；标注候选：{list(cmap)[:5]}", file=sys.stderr)
    rows = []
    # 通用兜底：目录/文件名里含 URB/PH/TPS/SDO 即取子类
    for p in files:
        low = p.lower()
        sub = next((s for s in ("urb", "ph", "tps", "sdo") if f"/{s}" in low or f"_{s}" in low or f"-{s}" in low), None)
        if sub and p.endswith((".wav", ".flac")):
            scene = {"sdo": "S1", "tps": "S2", "urb": "REG", "ph": "REG"}[sub]
            rows.append({
                "id": "humdial-" + os.path.splitext(os.path.basename(p))[0],
                "dataset": "humdial", "split": "test", "audio": p,
                "channels": 2, "layout": "dual_channel", "system_state": "unknown", "history": [],
                "gold_action": "respond" if scene == "S0" else "silent",
                "gold_scene": scene, "gold_lang": "unknown", "meta": {"subtype": sub},
            })
    if not rows:
        print("  [HumDial] 未能从目录结构推断子类 —— 需要按解压后的标注文件补映射规则", file=sys.stderr)
    else:
        w(os.path.join(out, "humdial_test.jsonl"), rows)
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
