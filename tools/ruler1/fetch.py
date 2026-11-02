#!/usr/bin/env python3
"""从 hf-mirror 直连下载数据集（绕开 hf CLI 的 metadata 校验问题）。

v1.1 修正：**跟分页**。HF 的 tree API 每页最多 1,000 条，只请求一次会静默截断
（WearVox 实际 7,688 个文件，单次请求只看到 998 个，导致清单里 714 条音频从未下载）。

用法:
  python3 fetch.py <repo_id> <dest_dir> [--pattern wav/] [--jobs 6]
  python3 fetch.py <repo_id> <dest_dir> --only-file list.txt     # 只下清单里列出的相对路径
  python3 fetch.py <repo_id> <dest_dir> --only-file list.txt --dry-run
"""
import argparse, json, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"
RAW = "https://hf-mirror.com/datasets/{repo}/resolve/main/{path}"
UA = {"User-Agent": "Mozilla/5.0"}


def list_files(repo, limit_pages=200):
    """递归列出全部文件；按 Link: rel="next" 翻页，直到没有下一页。"""
    url = API.format(repo=repo)
    files, pages = [], 0
    while url and pages < limit_pages:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
            link = r.headers.get("Link", "") or ""
        files += [(x["path"], x.get("size", 0)) for x in data if x.get("type") == "file"]
        pages += 1
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split("<")[1].split(">")[0]
        url = nxt
    if pages >= limit_pages:
        print(f"[警告] 翻页停在 {limit_pages} 页，可能仍有遗漏", file=sys.stderr)
    return files, pages


def fetch_one(repo, path, size, dest):
    out = os.path.join(dest, path)
    if os.path.exists(out) and os.path.getsize(out) == size and size > 0:
        return ("skip", path, size)
    os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
    url = RAW.format(repo=repo, path=path)
    # --speed-limit/--speed-time：hf-mirror 会偶发"连上但不传数据"的僵死，
    # 低于 1KB/s 持续 30s 直接放弃并重试，否则整个跑批会卡死在一个文件上
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "2",
           "--max-time", "240", "--speed-limit", "1024", "--speed-time", "30",
           "-C", "-", "--create-dirs", "-o", out, url]
    for _ in range(3):
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode == 0 and os.path.exists(out) and (size == 0 or os.path.getsize(out) == size):
            return ("ok", path, size)
    return ("fail", path, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("dest")
    ap.add_argument("--pattern", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only-file", default=None, help="只下这个文件里列出的相对路径（每行一个）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files, pages = list_files(a.repo)
    print(f"{a.repo}: 共 {len(files)} 个文件（翻了 {pages} 页）", flush=True)

    if a.only_file:
        want = {l.strip() for l in open(a.only_file, encoding="utf-8") if l.strip()}
        known = dict(files)
        unknown = sorted(want - set(known))
        files = [(p, known[p]) for p in sorted(want & set(known))]
        if unknown:
            print(f"[注意] {len(unknown)} 个路径不在仓库列表里（如 {unknown[0]}）", file=sys.stderr)
    if a.pattern:
        files = [(p, s) for p, s in files if p.startswith(a.pattern)]

    missing = [(p, s) for p, s in files
               if not (os.path.exists(os.path.join(a.dest, p))
                       and os.path.getsize(os.path.join(a.dest, p)) == s and s > 0)]
    total = sum(s for _, s in missing)
    print(f"待下载 {len(missing)} 个（已存在 {len(files)-len(missing)}），合计 {total/1e6:.1f} MB → {a.dest}", flush=True)
    if a.dry_run or not missing:
        return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for status, path, size in ex.map(lambda t: fetch_one(a.repo, t[0], t[1], a.dest), missing):
            if status in ("ok", "skip"):
                ok += 1
            else:
                fail += 1
                print("FAIL", path, flush=True)
    print(f"完成: ok={ok} fail={fail}", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
