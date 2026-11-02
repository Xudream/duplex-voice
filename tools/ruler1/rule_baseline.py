#!/usr/bin/env python3
"""A0 基线：字符 n-gram + 逻辑回归（传统分类器，非 LLM）。

为什么必须有它：不跑 A0，就无法回答"LLM 判官到底比传统分类器强多少"，
选型结论会被质疑"你只是没试便宜方案"。

口径与 LLM 跑道一致：
  - 文本：优先 target_text（当前 VAD 段），否则 transcript
  - 标签：gold_scene in {S1,S2,S3,S4} → 静默(1)；S0 → 响应(0)
  - S0_guest / REG 不参与训练与评估（与 score.py 口径一致）
  - 评估：K 折交叉验证（默认 5）——因为 S1 没有公开训练集（FDBench 的 SDO 训练集是 0 条），
    只能用交叉验证给一个"同分布上限"参考值

用法:
  python3 rule_baseline.py --manifest manifests/mini.jsonl --folds 5 \
      --out predictions/a0__mini.jsonl --report reports/a0__mini.md
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

NEG = {"S1", "S2", "S3", "S4"}
SKIP = {"S0_guest", "REG"}


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        sc = r.get("gold_scene")
        if sc in SKIP or sc not in (NEG | {"S0"}):
            continue
        text = (r.get("target_text") or r.get("transcript") or "").strip()
        if not text:
            continue
        out.append({"id": r["id"], "scene": sc, "text": text,
                    "y": 1 if sc in NEG else 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261101)
    a = ap.parse_args()

    data = load(a.manifest)
    n_pos = sum(d["y"] for d in data)
    print(f"{len(data)} 条（应静默 {n_pos} / 应响应 {len(data)-n_pos}）")

    X_text = [d["text"] for d in data]
    y = np.array([d["y"] for d in data])
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)

    proba = np.zeros(len(data))
    for tr, te in skf.split(X_text, y):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([X_text[i] for i in tr])
        Xte = vec.transform([X_text[i] for i in te])
        base = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        # 概率校准：不做校准的话 LR 概率集中在 0/1，工作点扫描会完全扫不动
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(Xtr, y[tr])
        proba[te] = clf.predict_proba(Xte)[:, 1]

    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, pred)
    print(f"交叉验证 AUC={auc:.3f}  F1(静默类)={f1:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d, p in zip(data, proba):
            f.write(json.dumps({
                "id": d["id"], "model": "A0-char-ngram-lr",
                "p_silent": round(float(p) * 100, 1),
                "action": "silent" if p >= 0.5 else "respond",
                "confidence": round(abs(float(p) - 0.5) * 2, 3),
                "reason": "tfidf+lr", "latency_ms": 0, "raw": "",
            }, ensure_ascii=False) + "\n")
    print(f"写入 {a.out}")

    # 分场景
    per = {}
    for d, p in zip(data, proba):
        s = per.setdefault(d["scene"], [0, 0])
        s[1] += 1
        s[0] += 1 if p >= 0.5 else 0
    lines = ["| 场景 | n | 判静默 | 拒识率 |", "|---|---:|---:|---:|"]
    for s in ["S1", "S2", "S3"]:
        if s in per:
            c, n = per[s]
            lines.append(f"| {s} | {n} | {c} | {c/n*100:.1f}% |")
    if "S0" in per:
        c, n = per["S0"]
        lines.append(f"| S0（静默率=FRR） | {n} | {c} | {c/n*100:.1f}% |")
    lines += ["", f"- AUC = {auc:.3f}", f"- F1 = {f1:.3f}", f"- {a.folds} 折交叉验证（非留出集，仅作同分布参考）"]
    report = "\n".join(lines)
    print(report)
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        open(a.report, "w", encoding="utf-8").write(report + "\n")


if __name__ == "__main__":
    main()
