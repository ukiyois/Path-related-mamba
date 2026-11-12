"""Parallel, resumable download of a Hugging Face dataset (or a sub-folder) through the HF mirror.

  python explore3d/data/hf_fetch.py <repo_id> <dest_dir> [prefix] [--threads 24]

Files larger than 32 MiB are fetched as 16 byte ranges in parallel (one connection to the mirror is slow);
files already present with the right size are skipped.
"""
import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def listing(repo, prefix):
    url = f"{E}/api/datasets/{repo}/tree/main" + (f"/{prefix}" if prefix else "") + "?recursive=1"
    return [(f["path"], f.get("size") or (f.get("lfs") or {}).get("size") or 0)
            for f in json.load(urllib.request.urlopen(url, timeout=120)) if f.get("type") == "file"]


def fetch_range(url, dest, start, end, tries=30):
    for t in range(tries):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= end - start + 1:
            return
        if t >= 3 and start == 0 and have == 0:   # small files: plain GET without a Range header
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            time.sleep((30 if "429" in str(e) else 3) + 2 * t)
    raise RuntimeError(f"failed {url} {start}-{end}")


def fetch(repo, path, size, root, pool_ranges=16):
    out = root / path
    if out.exists() and out.stat().st_size == size:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{E}/datasets/{repo}/resolve/main/{path}"
    if size <= 32 << 20:
        tmp = out.with_suffix(out.suffix + ".part")
        fetch_range(url, tmp, 0, max(size - 1, 0))
        tmp.rename(out)
        return "ok"
    n = pool_ranges
    chunk = (size + n - 1) // n
    parts = [out.with_suffix(out.suffix + f".p{i}") for i in range(n)]
    ths = [threading.Thread(target=fetch_range, args=(url, parts[i], i * chunk, min(size, (i + 1) * chunk) - 1))
           for i in range(n)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    with open(out, "wb") as f:
        for p in parts:
            f.write(p.read_bytes())
            p.unlink()
    assert out.stat().st_size == size, (path, out.stat().st_size, size)
    return "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("prefix", nargs="?", default="")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    root = Path(a.dest)
    files = listing(a.repo, a.prefix)
    total = sum(s for _, s in files)
    print(f"{a.repo}/{a.prefix}: {len(files)} files, {total / 2**30:.2f} GiB", flush=True)
    t0 = time.time()
    files = [f for f in files if not f[0].endswith(".gitattributes")]

    def safe(f):
        for t in range(5):                               # per-file retries with backoff (mirror rate limits: 429)
            try:
                return fetch(a.repo, f[0], f[1], root)
            except Exception as e:
                err = e
                time.sleep(20 * (t + 1))
        print(f"FAILED {f[0]}: {err}", flush=True)
        return "fail"
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(safe, files))
    print(f"done: {res.count('ok')} fetched, {res.count('skip')} skipped, {res.count('fail')} failed, "
          f"{time.time() - t0:.0f} s", flush=True)
