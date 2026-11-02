#!/usr/bin/env python3
"""尺子1 判分器：分场景拒识率 / FAR_w / FRR / clarify / invalid / 误执行 + Wilson CI + 工作点扫描。

用法:
  python3 score.py --manifest manifests/chorus.jsonl --pred predictions/all_silent__chorus.jsonl
  python3 score.py --manifest ... --pred ... --baseline-pred predictions/all_respond__chorus.jsonl --report out.md
"""
import argparse, json, math, sys
from collections import Counter, defaultdict

SILENT = {"silent", "silent_record"}
RESPOND = {"respond", "ack", "wait"}
CLARIFY = {"clarify"}
VALID = SILENT | RESPOND | CLARIFY
DEFAULT_W = {"S1": 0.40, "S2": 0.15, "S3": 0.25, "S4": 0.20}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def norm_action(a):
    if a is None:
        return "invalid"
    a = str(a).strip().lower()
    alias = {
        "silent_record": "silent_record", "silent": "silent", "ignore": "silent",
        "dismiss": "silent", "keep_silent": "silent", "no_response": "silent",
        "respond": "respond", "takeover": "respond", "answer": "respond",
        "ack": "ack", "wait": "wait", "stop_listen": "wait", "maintain": "wait",
        "clarify": "clarify", "ask": "clarify", "confirm": "clarify",
        "not_enough_information": "not_enough_information",
    }
    return alias.get(a, a if a in VALID else "invalid")


def evaluate(manifest, preds, weights=DEFAULT_W, only_predicted=False):
    gold = {m["id"]: m for m in manifest}
    pred = {}
    for p in preds:
        if p.get("id") in gold:
            pred[p["id"]] = p

    missing = len(gold) - len(pred)
    if only_predicted:
        # 冒烟场景：只对"有预测"的样本出分（缺预测不算错，避免小样本跑分被拉低）
        gold = {i: g for i, g in gold.items() if i in pred}
        missing = 0
    rows = []
    for i, g in gold.items():
        p = pred.get(i)
        action = norm_action(p.get("action")) if p else "missing"
        rows.append({
            "id": i, "scene": g.get("gold_scene", "NA"),
            "gold": g.get("gold_action", ""), "pred": action,
            "conf": float(p.get("confidence", 1.0)) if p else 0.0,
            "p_silent": (p.get("p_silent") if p else None),
            "exec": bool((p or {}).get("tool_call")) and action in RESPOND,
            "lang": g.get("gold_lang", "NA"), "dataset": g.get("dataset", "NA"),
        })

    # 分场景
    per = defaultdict(lambda: Counter())
    for r in rows:
        per[r["scene"]][r["pred"]] += 1
    scene_stats = {}
    for scene, c in sorted(per.items()):
        total = sum(c.values())
        silent = sum(v for k, v in c.items() if k in SILENT)
        p, lo, hi = wilson(silent / total if total else 0.0, total)
        scene_stats[scene] = {
            "n": total, "silent": silent, "rate": p, "ci": (lo, hi),
            "clarify": sum(v for k, v in c.items() if k in CLARIFY),
            "invalid": c.get("invalid", 0) + c.get("missing", 0),
        }

    neg = [r for r in rows if r["scene"] in DEFAULT_W]
    far_w = 0.0
    wsum = 0.0
    for s, w in weights.items():
        if s in scene_stats:
            far_w += w * (1 - scene_stats[s]["rate"])
            wsum += w
    far_w = far_w / wsum if wsum else float("nan")

    s0 = scene_stats.get("S0")
    # 注意：S0 是"应响应"样本，silent 即漏响应；FRR = 静默数 / S0 总数
    frr = s0["rate"] if s0 else float("nan")

    n_neg = len(neg)
    n_exec = sum(1 for r in rows if r["exec"])
    n_invalid = sum(1 for r in rows if r["pred"] in ("invalid", "missing"))
    n_clarify = sum(1 for r in rows if r["pred"] in CLARIFY)

    return {
        "n": len(rows), "missing": missing, "scene": scene_stats,
        "far_w": far_w, "frr": frr, "exec": n_exec,
        "invalid_rate": n_invalid / len(rows) if rows else 0,
        "clarify_rate": n_clarify / len(rows) if rows else 0,
        "rows": rows,
    }


def work_point(res, thresholds=None):
    """扫 FAR_w 与 FRR 的权衡。
    有 p_silent（0-100）时按 p_silent >= t 判静默（细粒度，推荐）；
    否则退回置信度回退（阈值 0-1，qwen 常饱和在 0.95/0.98，区分力差）。
    """
    has_p = any(r.get("p_silent") is not None for r in res["rows"])
    if thresholds is None:
        thresholds = list(range(50, 100, 5)) if has_p else (0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95)
    out = []
    for t in thresholds:
        per = defaultdict(lambda: [0, 0])
        for r in res["rows"]:
            if r["scene"] not in DEFAULT_W and r["scene"] != "S0":
                continue
            # 任何非静默动作都可在低置信度下回退静默（原先漏了 ack/wait，导致阈值扫不动）
            if has_p and r.get("p_silent") is not None:
                silent = r["pred"] in SILENT or r["p_silent"] >= t
            else:
                silent = r["pred"] in SILENT or (r["pred"] not in SILENT and r["conf"] < t)
            bucket = per[r["scene"]]
            bucket[1] += 1
            if silent:
                bucket[0] += 1
        rej = {s: (v[0] / v[1] if v[1] else 1.0) for s, v in per.items()}
        far_w = 0.0
        wsum = 0.0
        for s, w in DEFAULT_W.items():
            if s in rej:
                far_w += w * (1 - rej[s]); wsum += w
        far_w = far_w / wsum if wsum else float("nan")
        frr = rej.get("S0", 1.0)   # S0 被置为静默的比例即 FRR
        out.append((t, far_w, frr))
    return out


def fmt_md(res, extra=None):
    L = []
    L.append(f"样本 {res['n']}　缺失预测 {res['missing']}")
    L.append("")
    L.append("| 场景 | n | 正确静默 | 拒识率 | 95% CI | clarify | invalid/缺失 |")
    L.append("|---|---:|---:|---:|---|---:|---:|")
    for s, v in res["scene"].items():
        lo, hi = v["ci"]
        label = f"{s}（静默率=FRR）" if s == "S0" else s
        L.append(f"| {label} | {v['n']} | {v['silent']} | {v['rate']*100:.1f}% | "
                 f"[{lo*100:.1f}, {hi*100:.1f}] | {v['clarify']} | {v['invalid']} |")
    L.append("")
    L.append(f"- FAR_w = {res['far_w']*100:.2f}%　（w: " +
             ", ".join(f"{k} {v}" for k, v in DEFAULT_W.items()) + "）")
    L.append(f"- FRR(S0) = {res['frr']*100:.2f}%" if res["frr"] == res["frr"] else "- FRR(S0) = n/a（无 S0 样本）")
    L.append(f"- 误执行 = {res['exec']}（一票否决）")
    L.append(f"- invalid/缺失率 = {res['invalid_rate']*100:.2f}%　clarify 率 = {res['clarify_rate']*100:.2f}%")
    if extra:
        L.append("")
        L.extend(extra)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--baseline-pred", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only-predicted", action="store_true",
                    help="只评有预测的样本（冒烟/子集用）")
    args = ap.parse_args()

    manifest = load_jsonl(args.manifest)
    preds = load_jsonl(args.pred)
    res = evaluate(manifest, preds, only_predicted=args.only_predicted)

    extra = []
    if args.sweep:
        extra.append("**工作点扫描**（低置信度回退静默）")
        extra.append("")
        extra.append("| 阈值 | FAR_w | FRR(S0) |")
        extra.append("|---:|---:|---:|")
        for t, f, fr in work_point(res):
            extra.append(f"| {t:.2f} | {f*100:.2f}% | {fr*100:.2f}% |")

    if args.baseline_pred:
        try:
            from stats import mcnemar  # 可选：同仓库 stats.py
        except Exception:
            mcnemar = None
        base = evaluate(manifest, load_jsonl(args.baseline_pred))
        extra += ["", f"对照基线 FAR_w={base['far_w']*100:.2f}% FRR={base['frr']*100:.2f}%"]
        if mcnemar:
            extra.append(mcnemar(res["rows"], base["rows"]))

    text = fmt_md(res, extra)
    if args.report:
        open(args.report, "w", encoding="utf-8").write(text + "\n")
        print(f"写入 {args.report}")
    print(text)


if __name__ == "__main__":
    main()
