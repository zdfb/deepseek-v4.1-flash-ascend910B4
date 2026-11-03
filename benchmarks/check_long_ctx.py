#!/usr/bin/env python3
"""长上下文 TTFT / prefill 速率测量。

关键点：每个请求带唯一 nonce，否则第二次会命中 prefix cache，
测出来的就不是真实 prefill（59K prompt 冷启 6.9 s、命中 0.64 s）。

用法：
    python3 check_long_ctx.py --chars 390000            # ≈252K token
    python3 check_long_ctx.py --chars 1590000 --repeat 2
"""
import argparse
import json
import time
import urllib.request
import uuid


def corpus(n_chars, nonce):
    parts = [f"[{nonce}] 观测站运行日志。以下为逐日记录。\n"]
    i = 0
    while sum(len(x) for x in parts) < n_chars:
        parts.append(
            f"第{i}日：风速{3 + (i * 7) % 21}米每秒，气温{-8 + (i * 3) % 40}摄氏度，"
            f"核对编号E{10000 + i * 17 % 80000}的备件并签字确认。\n"
        )
        i += 1
    return "".join(parts)[:n_chars] + "\n请用一句话概括主题。"


def ask(base, model, prompt, max_tokens):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0, t_first, usage, text = time.time(), None, None, []
    with opener.open(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if t_first is None:
                    t_first = time.time()
                if ch.get("text"):
                    text.append(ch["text"])
    return t_first - t0, usage, "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1/completions")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--chars", type=int, default=390000)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    for k in range(args.repeat):
        nonce = uuid.uuid4().hex[:16]
        ttft, usage, text = ask(args.base_url, args.model, corpus(args.chars, nonce), args.max_tokens)
        pt = usage["prompt_tokens"]
        print(f"[long] run{k + 1} chars={args.chars} prompt_tokens={pt} "
              f"TTFT={ttft:.2f}s prefill={pt / ttft:.0f} tok/s")
        print(f"[long] out={text[:80]!r}")


if __name__ == "__main__":
    main()
