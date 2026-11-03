#!/usr/bin/env python3
"""尺子1 最小集生成器（模型选型专用，v2）

设计目标：用最少的样本，支撑"淘汰 / 保留 / 待定"三态决策。

两级结构：
  --stage smoke   120 条  —— 只验协议（invalid 率、JSON 可解析、延迟），不做任何结论
  --stage screen 1320 条  —— 判定集：S1 420 / S2 300 / S3 220 / S0 300 / S0_guest 40 / REG 40

抽样规则：
  1) 分层 = 数据集 × 场景，按全量盘的比例配比（避免某一集主导）
  2) 每格最少 10 条（避免单格为 0，无法看出该集退化）
  3) 固定随机种子（默认 20261102），输出 sha256 便于报告头固定版本
  4) smoke 是 screen 的子集（同一批样本里抽），保证两级结果可比

用法:
  python3 make_mini.py --manifests manifests --out manifests/mini_screen.jsonl --stage screen
  python3 make_mini.py --manifests manifests --out manifests/mini_smoke.jsonl  --stage smoke
"""
import argparse, glob, hashlib, json, os, random
from collections import defaultdict

STAGE_QUOTA = {
    "smoke":  {"S1": 40,  "S2": 30, "S3": 20, "S0": 20, "S0_guest": 5, "REG": 5},
    "screen": {"S1": 420, "S2": 300, "S3": 220, "S0": 300, "S0_guest": 40, "REG": 40},
}
MIN_CELL = 10


def load_all(manifests_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(manifests_dir, "*.jsonl"))):
        if os.path.basename(f).startswith("mini"):
            continue
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("audio_ok", True)]


def allocate(pool_by_dataset, quota, min_cell=MIN_CELL):
    """按各数据集在全量中的占比分配 quota，最少 MIN_CELL，余数按小数部分补齐。"""
    total = sum(len(v) for v in pool_by_dataset.values())
    if total == 0:
        return {}
    raw = {d: quota * len(v) / total for d, v in pool_by_dataset.items()}
    out = {d: max(min_cell, int(round(x))) for d, x in raw.items()}
    # 修正总和：先削掉超出该集可用量的，再按小数部分从大到小补
    for d in out:
        out[d] = min(out[d], len(pool_by_dataset[d]))
    diff = quota - sum(out.values())
    order = sorted(raw, key=lambda d: -(raw[d] - int(raw[d])))
    i = 0
    while diff != 0 and order:
        d = order[i % len(order)]
        if diff > 0 and out[d] < len(pool_by_dataset[d]):
            out[d] += 1; diff -= 1
        elif diff < 0 and out[d] > min_cell:
            out[d] -= 1; diff += 1
        i += 1
        if i > 10000:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["smoke", "screen"], default="screen")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--min-cell", type=int, default=MIN_CELL, help="每格最少条数（smoke 建议 1）")
    ap.add_argument("--from-file", default=None, help="从已有清单里抽（保证 smoke ⊂ screen）")
    a = ap.parse_args()

    if a.from_file:
        rows = [json.loads(l) for l in open(a.from_file, encoding="utf-8") if l.strip()]
    else:
        rows = load_all(a.manifests)
    by_scene = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_scene[r["gold_scene"]][r.get("dataset", "na")].append(r)

    rng = random.Random(a.seed)
    picked, plan = [], []
    for scene, quota in STAGE_QUOTA[a.stage].items():
        pool = by_scene.get(scene)
        if not pool:
            continue
        alloc = allocate(pool, quota, a.min_cell)
        for ds in sorted(alloc):
            pool[ds].sort(key=lambda r: r["id"])          # 先按 id 定序，保证可复现
            rng.shuffle(pool[ds])
            take = pool[ds][:alloc[ds]]
            picked += take
            plan.append((scene, ds, alloc[ds], len(pool[ds])))
    rng.shuffle(picked)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    blob = open(a.out, "rb").read()

    print(f"{a.out}: {len(picked)} 条  sha256={hashlib.sha256(blob).hexdigest()[:16]}")
    print(f"{'场景':10s}{'数据集':10s}{'抽取':>6s}{'全量':>6s}")
    for scene, ds, n, tot in plan:
        print(f"{scene:10s}{ds:10s}{n:6d}{tot:6d}")


if __name__ == "__main__":
    main()
