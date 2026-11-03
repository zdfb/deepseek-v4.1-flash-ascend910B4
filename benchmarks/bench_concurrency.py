#!/usr/bin/env python3
"""并发吞吐 / 单流速率测量。

每个并发档发 N 条请求，统计：
  - 单流速率 = 每条请求 decode 窗口速率的**中位数**（tok/s）
  - 聚合吞吐 = 全部 completion token / decode 窗口长（tok/s）
  - TTFT 中位数

请求都带唯一 nonce，避免 prefix cache 命中把 prefill 变成 0。

用法：
    python3 bench_concurrency.py --concurrency 1,32,64,96,128,192 --requests 192
    python3 bench_concurrency.py --concurrency 192 --requests 384 --repeats 2
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

_LOCK = threading.Lock()
ROWS = []


def corpus(n_chars, nonce):
    parts = [f"[{nonce}] 观测站运行日志。\n"]
    i = 0
    while sum(len(x) for x in parts) < n_chars:
        parts.append(f"第{i}日：风速{3 + (i * 7) % 21}米每秒，气温{-8 + (i * 3) % 40}摄氏度。\n")
        i += 1
    return "".join(parts)[:n_chars] + "\n继续写下去。"


def one(base, model, chars, out_tokens, timeout):
    nonce = uuid.uuid4().hex[:12]
    payload = {
        "model": model,
        "prompt": corpus(chars, nonce),
        "max_tokens": out_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    t_first = None
    t_last = None
    usage = None
    try:
        with opener.open(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                obj = json.loads(line[6:])
                if obj.get("usage"):
                    usage = obj["usage"]
                for _ in obj.get("choices") or []:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            ROWS.append({"ok": False, "err": str(exc)[:100]})
        return
    with _LOCK:
        ROWS.append({
            "ok": True,
            "ttft": (t_first or t_last or t0) - t0,
            "decode": max((t_last or t0) - (t_first or t_last or t0), 1e-6),
            "tokens": (usage or {}).get("completion_tokens", 0),
            "start": t0,
            "end": t_last or t0,
        })


def run_level(base, model, conc, requests, chars, out_tokens, timeout):
    ROWS.clear()
    threads = []
    for _ in range(requests):
        while sum(1 for t in threads if t.is_alive()) >= conc:
            time.sleep(0.005)
        t = threading.Thread(target=one, args=(base, model, chars, out_tokens, timeout))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    ok = [r for r in ROWS if r["ok"]]
    if not ok:
        print(f"[bench] conc={conc:>4} 全部失败")
        return None
    per_stream = [r["tokens"] / r["decode"] for r in ok if r["tokens"] > 1]
    total_tokens = sum(r["tokens"] for r in ok)
    window = max(r["end"] for r in ok) - min(r["start"] for r in ok)
    ttfts = [r["ttft"] for r in ok]
    row = {
        "conc": conc,
        "ok": len(ok),
        "requests": requests,
        "per_stream": statistics.median(per_stream) if per_stream else 0.0,
        "aggregate": total_tokens / window if window > 0 else 0.0,
        "ttft": statistics.median(ttfts),
    }
    print(f"[bench] conc={conc:>4} ok={len(ok)}/{requests} "
          f"单流中位={row['per_stream']:.1f} tok/s 聚合={row['aggregate']:.1f} tok/s "
          f"TTFT中位={row['ttft']:.2f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1/completions")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--concurrency", default="1,32,64,96,128,192")
    ap.add_argument("--requests", type=int, default=192,
                    help="每个并发档发出的请求总数（并发 1 默认取 16）")
    ap.add_argument("--prompt-chars", type=int, default=1700, help="约 1024 token")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    for rep in range(args.repeats):
        if args.repeats > 1:
            print(f"[bench] ==== rep {rep + 1}/{args.repeats} ====", flush=True)
        for conc in levels:
            n = 16 if conc == 1 else args.requests
            run_level(args.base_url, args.model, conc, n,
                      args.prompt_chars, args.output_tokens, args.timeout)


if __name__ == "__main__":
    main()
