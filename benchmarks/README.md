# 测量脚本

四个脚本都只依赖 Python 标准库，直接打 OpenAI 兼容端点。

| 脚本 | 测什么 | 关键参数 |
|---|---|---|
| `check_long_ctx.py` | 长上下文 TTFT / prefill 速率 | `--chars 390000`（≈252K token） |
| `bench_mixed.py` | 混长负载下短请求延迟 | `--long 4 --short 32` |
| `bench_prefix_reuse.py` | prefix cache 命中 / 容量拐点 | `--fill 8 --chars 595000` |
| `bench_equivalence.py` | 贪心输出 sha256 对比 | `--tag A` 然后 `--compare A B` |

## 用法

```bash
# 长上下文（每个请求带唯一 nonce，避免 prefix cache 干扰）
python3 check_long_ctx.py --chars 390000 --repeat 2

# 混长：4 条 200K + 32 条 2K，看短请求 p50/p99/max
python3 bench_mixed.py --long 4 --long-chars 600000 --short 32 --short-chars 2000

# 填满 KV 池后重发第一条，判断是否被淘汰
python3 bench_prefix_reuse.py --fill 8 --chars 595000

# 等价性：两个配置用同一个 nonce 各跑一次，再比对
python3 bench_equivalence.py --tag baseline
python3 bench_equivalence.py --tag yoco
python3 bench_equivalence.py --compare baseline yoco
```

聚合吞吐（1K prompt / 128 输出 × 192 并发）用上游的 `tools/bench_concurrency.py`，
见 README §1.2。

## 三个必须遵守的测量纪律

1. **绝对 tok/s 不能跨 run 比**——语料随机导致接受长度 A 在 2.1–2.6 波动，
   同一配置差 13%。用 `聚合 ÷ A` 归一。
2. **长上下文必须加唯一 nonce**——否则第二次命中 prefix cache（6.9 s → 0.64 s）。
3. **等价性只在同一进程内可信**——本栈跨进程非确定。
