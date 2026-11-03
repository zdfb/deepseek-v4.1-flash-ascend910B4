# 实测数据与测量方法

所有数字都在同一台机器上测得：**8 × Ascend 910B4（64 GiB/卡，512 GiB 合计）**、
Kunpeng-920 192 核、1005 GB 内存、8 NUMA 节点、CANN 9.1.0、
vLLM 0.27.1 + vllm-ascend 0.19.1rc2、W4A8 真权重（490 GB）。

## 0. 测量前必读：两个陷阱

**(a) 绝对吞吐不能跨 run 比。** 压测语料每次随机，投机接受长度 `A` 在 2.1–2.6 之间波动。
同一份配置能测出 989.3 和 1138.5 tok/s，差 13%。**必须用 `聚合吞吐 ÷ A` 归一**：

| 配置 | 聚合 tok/s | A | tok/s ÷ A |
|---|---:|---:|---:|
| 基线 | 1138.5 / 1221.9 / 989.3 | 2.56 / 2.63 / 2.26 | 445 / 465 / 438 |
| BAT 16384 | 1004.8 | 2.27 | 443 |
| YOCO 精确 | 1005.8 | 2.29 | 439 |
| MAX_SEQS 384 | 438–592 | 2.27 | 193–261 |

前三个都在 438–465 —— 说明**每步算力是同一套，差异全来自 A**。只有 SEQ=384 是真回归。

**(b) prefix cache 会污染长上下文测量。** 重复的 prompt 第二次会命中缓存
（59K prompt：6.9 s → 0.64 s）。测真实 prefill 必须给每个请求加唯一 nonce，
见 `benchmarks/check_long_ctx.py`。

## 1. decode

| 指标 | 值 | 条件 |
|---|---:|---|
| 单流 | **34.1 tok/s** | 1K prompt，A=2.12，TTFT 0.41 s |
| 每步耗时 | ~62 ms | 换算 16.1 tok/s/step |
| 聚合 @96 | 451–473 tok/s | 1K/128，同口径 |
| **聚合 @192** | **1,138–1,222 tok/s** | 256 与 512 输出长度同量级 |

MAX_SEQS 扫描（作者默认 32 → 192，唯一有效的配置杠杆，+85%）：

| MAX_SEQS | 32 | 64 | 96 | 128 | **192** |
|---|---:|---:|---:|---:|---:|
| 聚合 tok/s | 621.9 | 882.3 | 972.9 | 1043.8 | **1148.8** |
| 单流 tok/s | 28.8 | 20.3 | 15.8 | 12.5 | 9.2 |

192 不是随便选的：`cudagraph_capture_sizes` 上限 1152 = 192 × (1 + 5 draft)。

## 2. prefill（快速模式）

| prompt | TTFT | prefill 速率 |
|---:|---:|---:|
| 147K token | 18.9 s | 7,780 tok/s |
| 252K token | 33.7 s | 7,531 tok/s |
| 365K token | 50.9 s | 7,180 tok/s |
| 973K token | 183.4 s | 5,306 tok/s |

基线对照（不截断）：252K 49.5 s（5,099）、365K 80.1 s（4,563）。

## 3. 上下文与容量

- 上限 **1,048,576 token**（模型 `max_position_embeddings`），输入+输出共用；
  超限返回 400：`maximum context length is 1048576 tokens … total of at least 1048577`
- 实测通过 973,206 token
- KV 池 2,941,744 token（13.6 GiB）：

| 场景 | 单条 KV | 可同时 |
|---|---:|---:|
| 1M | ~7.0 GiB | 2 |
| 200K | ~1.0 GiB | 14 |
| 32K | ~0.16 GiB | 90 |
| 1K | ~5 MB | 192（MAX_SEQS 限制） |

## 4. 缓存复用

| 场景 | 冷启 | 二次请求（prefix 命中） |
|---|---:|---:|
| 59K prompt | 6.9 s | **0.64 s** |
| 147K prompt | 18.9 s | **1.71 s** |

容量拐点：2.94M token 池被填满后旧前缀被淘汰 —— 8 条 385K 会话之后重发第 1 条，
TTFT 回到 83.3 s（重算）。**这就是 host KV tier 的动机**：那份 KV 只有 1.87 GB，
从主机内存回填约 0.13 s，差 600 倍。

## 5. 已排除的方向

| 方向 | 结果 |
|---|---|
| `BAT_TOKENS` 32768 | 起不来：activation 吃掉 11 GiB，KV 只剩 2.46 GiB < 1M 所需的 6.98 |
| `BAT_TOKENS` 16384 | 聚合归一后等价（443 vs 438），但 **252K 请求 NPU OOM** |
| `MAX_SEQS` 384 + 抓图扩到 2304 | 归一 193–261，**严重变差**；TTFT 4.6 → 22.9 s |
| `SP_TOKENS` 3 | 被 build 硬拒：`DSpark requires num_speculative_tokens >= dspark_block_size (5)` |
| `SP_TOKENS` 7 | −10%（验证成本 +33%，接受长度增加有限） |
| KV 量化 fp8 / int8 | `int8` 参数非法；`int8_per_token_head` / `fp8` **静默回退 bf16**（engine config 仍是 bfloat16） |
| prefill MC2 | 252K prefill **慢 20%**（49.5 → 59.3 s） |
| 关闭 engram（上界测量） | 只快 5%，不值得做热点缓存 |
| EPLB 冗余专家 | 物理专家 392 超出算子索引边界 384，本 build 不支持 |
| host KV tier（Mooncake store） | 链路打通但 **save 路径使 prefill +150%**，重入只省 31%（见 ROADMAP） |

## 6. 复现

```bash
# 长上下文（唯一 nonce，避免 prefix cache 干扰）
python3 benchmarks/check_long_ctx.py --chars 390000 --port 8100

# 混长负载（4 条长 + 32 条短，看短请求延迟）
python3 benchmarks/bench_mixed.py --long 4 --short 32

# prefix 复用 / 容量拐点
python3 benchmarks/bench_prefix_reuse.py --fill 8 --chars 595000

# 等价性（贪心输出 sha256；注意本栈跨进程非确定，必须同进程比较）
python3 benchmarks/bench_equivalence.py --tag baseline
```

聚合吞吐用上游的 `tools/bench_concurrency.py`（随上游分发，不在本仓库）。
