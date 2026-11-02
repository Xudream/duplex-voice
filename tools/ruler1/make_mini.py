#!/usr/bin/env python3
"""从全量清单里分层抽一个最小集（冒烟/首轮选型用）。

默认配额：S1 120 / S2 90 / S3 60 / S0 60 / S0_guest 20 / REG 30 = 380
只取 audio_ok=true 的样本；按 dataset 轮转，避免某一集独大。

用法:
  python3 make_mini.py --manifests manifests --out manifests/mini.jsonl
  python3 make_mini.py --manifests manifests --out manifests/mini.jsonl --per-scene S1=60,S2=40
"""
import argparse, glob, json, os, random
from collections import defaultdict

DEFAULT_QUOTA = {"S1": 120, "S2": 90, "S3": 60, "S0": 60, "S0_guest": 20, "REG": 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out", default="manifests/mini.jsonl")
    ap.add_argument("--per-scene", default="", help="如 S1=60,S2=40")
    ap.add_argument("--seed", type=int, default=20261101)
    a = ap.parse_args()

    quota = dict(DEFAULT_QUOTA)
    if a.per_scene:
        quota = {}
        for kv in a.per_scene.split(","):
            k, v = kv.split("=")
            quota[k.strip()] = int(v)

    rows = []
    for f in sorted(glob.glob(os.path.join(a.manifests, "*.jsonl"))):
        if os.path.basename(f).startswith("mini"):
            continue
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("audio_ok", True)]

    by_scene = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_scene[r["gold_scene"]][r.get("dataset", "na")].append(r)

    rng = random.Random(a.seed)
    picked = []
    for scene, n_want in quota.items():
        pool = by_scene.get(scene, {})
        if not pool:
            continue
        # 按数据集轮转抽取，保证跨集均衡
        sets = list(pool)
        for v in pool.values():
            rng.shuffle(v)
        idx = {s: 0 for s in sets}
        got = 0
        while got < n_want:
            progressed = False
            for s in sets:
                if got >= n_want:
                    break
                if idx[s] < len(pool[s]):
                    picked.append(pool[s][idx[s]])
                    idx[s] += 1
                    got += 1
                    progressed = True
            if not progressed:
                break
    rng.shuffle(picked)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"{a.out}: {len(picked)} 条")
    print("  scene:", dict(sorted(Counter(r['gold_scene'] for r in picked).items())))
    print("  dataset:", dict(sorted(Counter(r['dataset'] for r in picked).items())))
    print("  lang:", dict(sorted(Counter(r['gold_lang'] for r in picked).items())))


if __name__ == "__main__":
    main()
