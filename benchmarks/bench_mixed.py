#!/usr/bin/env python3
"""混长负载：长请求把 prefill 占满时，短请求要等多久。

先发 --long 条超长请求，--delay 秒后再发 --short 条短请求，
分别统计短请求的 p50/p99/max 与长请求的完成时间。

这个脚本是分档调度的判据：FCFS 下短请求 p50 = 206 s，
开分档 + 关 admission gate 后降到 69 s（见 docs/ARCHITECTURE.md 第 6 节）。

用法：
    python3 bench_mixed.py --long 4 --long-chars 600000 --short 32 --short-chars 2000
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

_LOCK = threading.Lock()
RESULTS = []


def corpus(n_chars, nonce):
    parts = [f"[{nonce}] 观测站运行日志。\n"]
    i = 0
    while sum(len(x) for x in parts) < n_chars:
        parts.append(f"第{i}日：风速{3 + (i * 7) % 21}米每秒，气温{-8 + (i * 3) % 40}摄氏度。\n")
        i += 1
    return "".join(parts)[:n_chars] + "\n请概括。"


def one(base, model, prompt, max_tokens, kind, idx):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(base, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    try:
        with opener.open(req, timeout=7200) as r:
            out = json.loads(r.read())
        dt = time.time() - t0
        with _LOCK:
            RESULTS.append({"kind": kind, "idx": idx, "ok": True, "dt": dt,
                            "pt": out["usage"]["prompt_tokens"]})
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            RESULTS.append({"kind": kind, "idx": idx, "ok": False, "err": str(exc)[:120]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1/completions")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--long", type=int, default=4)
    ap.add_argument("--long-chars", type=int, default=600000)
    ap.add_argument("--short", type=int, default=32)
    ap.add_argument("--short-chars", type=int, default=2000)
    ap.add_argument("--out-tokens", type=int, default=32)
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    t_start = time.time()
    threads = []
    for i in range(args.long):
        p = corpus(args.long_chars, f"L{i}-{uuid.uuid4().hex[:8]}")
        t = threading.Thread(target=one, args=(args.base_url, args.model, p,
                                               args.out_tokens, "long", i))
        t.start()
        threads.append(t)

    time.sleep(args.delay)
    for i in range(args.short):
        p = corpus(args.short_chars, f"S{i}-{uuid.uuid4().hex[:8]}")
        t = threading.Thread(target=one, args=(args.base_url, args.model, p,
                                               args.out_tokens, "short", i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    wall = time.time() - t_start
    for kind in ("short", "long"):
        rows = [r for r in RESULTS if r["kind"] == kind]
        ok = [r["dt"] for r in rows if r["ok"]]
        fail = len(rows) - len(ok)
        if not ok:
            print(f"[{kind}] 全部失败 fail={fail}")
            continue
        ok.sort()
        print(f"[{kind}] ok={len(ok)}/{len(rows)} fail={fail} "
              f"p50={statistics.median(ok):.1f}s p99={ok[int(len(ok) * 0.99) - 1]:.1f}s "
              f"max={ok[-1]:.1f}s")
    print(f"[total] wall={wall:.1f}s")


if __name__ == "__main__":
    main()
