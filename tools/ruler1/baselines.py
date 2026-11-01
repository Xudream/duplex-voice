#!/usr/bin/env python3
"""平凡基线：all_silent / all_respond。用于给每个集定"地板"与"天花板"，并验证判分链路。

用法:
  python3 baselines.py --manifest manifests/chorus.jsonl --out predictions/all_silent__chorus.jsonl
  python3 baselines.py --manifest manifests/chorus.jsonl --mode respond --out predictions/all_respond__chorus.jsonl
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["silent", "respond", "oracle"], default="silent")
    a = ap.parse_args()

    n = 0
    with open(a.manifest, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as o:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            if a.mode == "silent":
                action = "silent"
            elif a.mode == "respond":
                action = "respond"
            else:  # oracle：上限参考，不可作为被测系统
                action = "respond" if m.get("gold_action") == "respond" else "silent"
            o.write(json.dumps({"id": m["id"], "model": f"baseline-{a.mode}",
                                "action": action, "confidence": 1.0, "latency_ms": 0},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"{a.out}: {n} 条（{a.mode}）")


if __name__ == "__main__":
    main()
