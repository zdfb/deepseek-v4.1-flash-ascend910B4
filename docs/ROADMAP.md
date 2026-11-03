# 路线图 / 未完成的方向

按「预期收益 × 可行性」排序。所有结论都有实测依据，不是猜测。

## 1. host KV tier（多轮/长会话场景，收益最大，工作量也最大）

**动机（实测）**：HBM KV 池 2.94M token 被填满后旧前缀被淘汰——
8 条 385K 会话之后重发第 1 条，TTFT 回到 83.3 s（从头重算）。
而那份 KV 只有 1.87 GB，从主机内存回填按 15 GB/s 算约 0.13 s，**差 600 倍**。
700 GB 空闲内存理论上能常驻约 1.44 亿 token（约 370 条 385K 会话）。

**已经做到哪一步**：

- `patches/files/dsa_v41_kvstore.py` 补上了 V4.1 自定义注意力的 save 钩子
  （上游的 save 钩子只覆盖 `dsa_v1`/`mla_v1`/`sfa_v1`，V4.1 走 `dsa_v41_forward`，从不调用）；
- 用上游自带的 `MooncakeConnectorStoreV1`（CPU 内存 KV store）验证通过：
  8/8 worker 注册成功、主机池 64 GB、实际存下 20.98 GB / 45,005 keys。

**为什么还没落地**：

| 问题 | 现象 | 需要做什么 |
|---|---|---|
| save 太贵 | 365K prompt prefill 从 80 s 涨到 200 s | 52 个 cache 面 × 45 个 chunk ≈ 2300 次 RPC，是延迟瓶颈。**只存 long_kv + index_k（12 个面），跳过 SWA** |
| 命中不全 | 重发只从 200 s 降到 137.6 s（约 1/3） | 打印 connector 的 matched tokens 与前缀链，确认是否 block 粒度匹配不完整 |

只存 12 个面的代价：重入时必须**重算 ≥128 个 token** 让 SWA 环自然重建
（这个约束在 `docs/ARCHITECTURE.md` 第 4 节推导过，成本约 0.03 s）。

**注意**：内置的 `sparse_kv_offload_config` 看起来正是现成方案（`dram_size_per_dp_GB`），
但它 `_validate_preconditions` 明确拒绝带 `compress_ratios` 的模型，本模型正好踩中。

## 2. 投机接受长度 A

吞吐 ≈ 440 × A。A 每 +0.1 ≈ +4% 吞吐，是**唯一不依赖缓存复用的**杠杆。
DSpark 有 confidence head（`dspark_proposer._compute_confidence`），
但本 build 没有暴露阈值/调度旋钮；`SP_TOKENS` 被锁在 5（≥`dspark_block_size`）。
想再进一步需要改 drafter 或换 checkpoint。

## 3. EP16 / EP32 wide-EP

EPLB 在本 build 上因算子索引边界（384）不可用；而且 EP8 下 2% 冗余本来收益有限。
真正的 EPLB 收益在更宽的专家并行，但那需要重新分片权重（4090 亿参数级别的重打包）。

## 4. 精确模式的性能

当前精确模式（`KEEP=2560 APPROX=0`）比不截断还慢 10%——
尾部省下的算力被每 chunk 的 gather/scatter + 元数据重建抵消。
如果能把元数据重建做成增量式（复用上一 chunk 的 device tensor），
精确模式有机会追平甚至超过基线。

## 5. 多请求批的截断粒度

当前 `KEEP` 按请求取末尾 N 个 token。如果同一批里长短差异极大，
短请求会被长请求的 KEEP 吃掉预算。可以做成按请求 `min(KEEP, 该请求剩余量)` 的细粒度版本
（现有实现已经是 `min(query_len, KEEP)`，但 chunk 级的 min-chunk 门槛是全局的）。
