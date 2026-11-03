#!/usr/bin/env python3
"""prefix cache 容量与复用测量。

流程：连续发 N 条互不相同的长请求把 HBM KV 池填满，然后原样重发第 1 条。
- 命中：TTFT 数量级下降（例如 147K prompt 18.9 s → 1.71 s）
- 淘汰：TTFT 回到冷启水平（实测 N=8、385K prompt 时重发 83.3 s）

用法：
    python3 bench_prefix_reuse.py --fill 8 --chars 595000
"""
import argparse
import json
import time
import urllib.request
import uuid


def corpus(n_chars, nonce):
    parts = [f"[{nonce}] 观测站运行日志。\n"]
    i = 0
    while sum(len(x) for x in parts) < n_chars:
        parts.append(f"第{i}日：风速{3 + (i * 7) % 21}，气温{-8 + (i * 3) % 40}，"
                     f"编号E{10000 + i * 17 % 80000}。\n")
        i += 1
    return "".join(parts)[:n_chars] + "\n请概括。"


def ask(base, model, prompt):
    payload = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0, t_first, usage = time.time(), None, None
    with opener.open(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices") and t_first is None:
                t_first = time.time()
    return t_first - t0, usage["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1/completions")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--fill", type=int, default=8)
    ap.add_argument("--chars", type=int, default=595000)
    ap.add_argument("--hit-threshold", type=float, default=10.0,
                    help="TTFT 低于该值视为命中（秒）")
    args = ap.parse_args()

    first = None
    for i in range(args.fill):
        prompt = corpus(args.chars, f"fill{i}-{uuid.uuid4().hex[:8]}")
        if i == 0:
            first = prompt
        ttft, pt = ask(args.base_url, args.model, prompt)
        print(f"[fill {i + 1}/{args.fill}] tokens={pt} TTFT={ttft:.1f}s", flush=True)

    ttft, pt = ask(args.base_url, args.model, first)
    verdict = "HIT" if ttft < args.hit_threshold else "MISS (evicted)"
    print(f"[reuse #1] tokens={pt} TTFT={ttft:.2f}s -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
