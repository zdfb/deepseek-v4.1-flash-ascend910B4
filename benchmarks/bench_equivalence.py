#!/usr/bin/env python3
"""贪心输出等价性对比。

用途：验证「截断」等改动是否改变模型输出。对固定 prompt 生成固定 token，
把结果写成 sha256 供不同配置间比对。

注意：本栈**跨进程不是逐位确定的**（ACL 图/批形状改变 bf16 归约顺序），
所以「不同实例跑出不同 token」不能直接判定改动有错；反之亦然。
严格的等价性要在**同一进程内**比较 logprob 差值。

用法：
    python3 bench_equivalence.py --tag baseline       # 存 /tmp/eq_baseline.json
    python3 bench_equivalence.py --tag yoco           # 另一配置
    python3 bench_equivalence.py --compare baseline yoco
"""
import argparse
import hashlib
import json
import os
import urllib.request


def corpus(n_chars, nonce):
    seed = (f"[{nonce}] DeepSeek V4.1 的混合缓存把长上下文压缩到少数几层，"
            "剩余层只保留 128 token 的滑窗。以下材料用于验证截断是否改变输出。\n")
    row = "第{i}段：观测站记录风速{ws}米每秒，温度{t}摄氏度，签字确认编号E{lid}。\n"
    out, i = [seed], 0
    while sum(len(x) for x in out) < n_chars:
        out.append(row.format(i=i, ws=3 + (i * 7) % 20, t=-5 + (i * 3) % 30,
                              lid=1000 + i * 13 % 9000))
        i += 1
    return "".join(out)[:n_chars]


def ask(base, model, prompt, max_tokens):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "top_p": 1.0, "seed": 0}
    req = urllib.request.Request(base, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1/completions")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--nonce", default="eqv1", help="固定 nonce：两个配置必须用同一个 prompt")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        a, b = (json.load(open(f"/tmp/eq_{t}.json")) for t in args.compare)
        for key in sorted(a):
            same = a[key]["sha256"] == b[key]["sha256"]
            print(f"[cmp] {key:<6} tokens={a[key]['prompt_tokens']:<7} "
                  f"{'IDENTICAL' if same else 'DIFFERS'}")
            if not same:
                print(f"       A={a[key]['text'][:60]!r}")
                print(f"       B={b[key]['text'][:60]!r}")
        return

    result = {}
    for name, chars in (("short", 900), ("mid", 16000)):
        prompt = corpus(chars, args.nonce) + "\n\n请用一句话总结上面材料的主题，然后输出标记 TLDR:"
        out = ask(args.base_url, args.model, prompt, args.max_tokens)
        text = out["choices"][0]["text"]
        result[name] = {
            "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        print(f"[eq] {name} prompt_tokens={result[name]['prompt_tokens']} "
              f"sha={result[name]['sha256'][:16]}")
        print(f"[eq] {name} text={text[:100]!r}")

    path = f"/tmp/eq_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[eq] wrote {path}")


if __name__ == "__main__":
    main()
