#!/usr/bin/env python3
"""统计工具：McNemar 配对检验、Bootstrap AUC 差、Wilson 区间。"""
import math, random


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(rows_a, rows_b, key="silent", label_a="A", label_b="B"):
    """同一批样本上两个系统的配对差异。silent 判定：pred in {silent, silent_record}。"""
    def silent(r):
        return r["pred"] in ("silent", "silent_record")

    a = {r["id"]: silent(r) for r in rows_a}
    b = {r["id"]: silent(r) for r in rows_b}
    ids = set(a) & set(b)
    n01 = sum(1 for i in ids if not a[i] and b[i])   # A 错 B 对
    n10 = sum(1 for i in ids if a[i] and not b[i])   # A 对 B 错
    n = n01 + n10
    if n == 0:
        return f"McNemar: 两系统在 {len(ids)} 条配对样本上完全一致，无法区分"
    # 精确二项检验（双侧）
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    p = min(1.0, p)
    verdict = "显著" if p < 0.05 else "不显著（差异在噪声内）"
    return (f"McNemar: {label_a} 独对 {n10} 条 / {label_b} 独对 {n01} 条，"
            f"n={n}，p={p:.4f} → {verdict}")


def bootstrap_auc_diff(scores_a, scores_b, labels, n_boot=1000, seed=0):
    """返回 AUC 差的点值与 95% CI。scores_*: list[float]，labels: list[int]（1=应静默）。"""
    def auc(scores, labels):
        pairs = sorted(zip(scores, labels))
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            return 0.5
        rank_sum, i = 0.0, 1
        ranks = [0.0] * len(pairs)
        while i <= len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i - 1][0]:
                j += 1
            avg = (i + j - 1) / 2
            for k in range(i - 1, j - 1):
                ranks[k] = avg
            i = j
        for r, (_, y) in zip(ranks, pairs):
            if y == 1:
                rank_sum += r
        return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)

    point = auc(scores_a, labels) - auc(scores_b, labels)
    rng = random.Random(seed)
    idx = list(range(len(labels)))
    diffs = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        la = [scores_a[i] for i in s]; lb = [scores_b[i] for i in s]; ly = [labels[i] for i in s]
        diffs.append(auc(la, ly) - auc(lb, ly))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return point, lo, hi


def min_detectable_diff(p=0.9, n=1000, z=1.96):
    """可分辨的最小差异（≈2×Wilson 半宽），单位 pt。"""
    hw = z * math.sqrt(p * (1 - p) / n)
    return 2 * hw * 100
