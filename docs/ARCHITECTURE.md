# 架构与原理

## 1. 这个模型的 KV 结构与常规模型不同

`config.json`（text_config）里的三个字段决定了一切：

```json
"compress_ratios": [0,0,2,2,...,2,1,1,...,1],   // 40 层（后面 3 个 0 属于 MTP）
"kv_source_layer_ids": [2, 8, 14, 20],
"index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
"sliding_window": 128
```

- **只有 4 层产长上下文 KV**（2 / 8 / 14 ratio=2，20 ratio=1）。
- 其余层是**消费者层**：长 KV 全部读它之前最近的 source 层，自己只写一个 **128 token 的滑窗环**。
- 因此每 token 的 KV 只有 **4.85 KB**（13.6 GiB 池容纳 2,941,744 token），
  比常规 MLA 模型小一个量级——这也是单机能开 192 并发、1M 上下文的原因。

## 2. 核心观察：消费者层的 prefill 是浪费的

对第 21~39 层（19 个消费者层）来说：

- 它们在位置 `t` 的输出只可能被后续层在 `[t, t+127]` 范围内读到；
- 它们只写 128 宽的滑窗环，**写多了也会被覆盖**；
- 模型最终只取请求末尾位置的 logits（和 DSpark 需要的末尾几个 hidden state）。

所以 **prefill 时这 19 层只需要算每个请求末尾的一段 token**。

## 3. 截断怎么实现

在 `model.py` 的层循环里，第 20 层跑完后插入一步（`patches/files/v41_yoco.py`）：

1. 按请求取末尾 `KEEP` 个 token 的 gather 索引；
2. 为 `swa` / `long_kv` / `index_k` 三类 cache plane 构造**裁剪版 metadata**：
   - `query_start_loc` 换成新的前缀和（内核视角上这就是一个合法的「最后一个 chunk」）；
   - `slot_mapping` / `positions` / `cos` / `sin` 按索引 gather；
   - `seq_lens` 与 block table **保持不变**（可见域不变）；
   - 按新布局重建 SMLA / QLI 的 device metadata；
3. 把共享的 `topk_indices` / `candidates` 里属于保留 token 的行压缩到前面
   （第 20 层已按整批算好，尾部消费者按 `[:n]` 读）；
4. 尾部 19 层只在裁剪后的 batch 上跑，结束后 scatter 回整批形状；
5. `torch.compiler.is_compiling()` 为真时直接跳过 → 图捕获路径零影响；
   decode 步（每请求 1 token）天然 no-op。

## 4. KEEP 到底该取多少（这是最容易搞错的地方）

DeepInfer 的默认值是 `KEEP_TOKENS=140`（= 滑窗 128 + 投机回溯余量）。**在本模型上它是错的**，
因为那里的假设是消费者层共享同一份 KV；而本模型**每个消费者层有自己的滑窗环**，
污染会逐层向后传一个窗口：

```
第 k 层精确的位置区间下界 S_k = S_{k-1} + 127
19 个消费者层 → 要让最后一个 token 精确，需要 KEEP ≥ 127 × 19 + 1 ≈ 2,414
本实现取 KEEP = window × (消费者层数 + 1) = 128 × 20 = 2,560
```

两种模式的实测：

| 模式 | KEEP | 长 prefill（252K） | 等价性 |
|---|---:|---:|---|
| 基线（不截断） | 0 | 49.5 s / 5,099 tok/s | 完全等价 |
| 精确截断 | 2560 | 55.1 s / 4,958 tok/s | 理论等价 |
| **快速（本仓库默认）** | **140** | **33.6 s / 7,162 tok/s** | **近似** |

精确模式反而更慢：尾部 19 层省下的算力被每 chunk 的 gather/scatter + 元数据重建抵消。
所以本仓库默认 `APPROX=1`，并在文档里明确标注「长 prompt 输出不保证逐 token 等价」。

## 5. 两个必须同时满足的工程约束

**(a) 短请求不能被误伤。** 初版把 1024-token 的 prompt 也截断，聚合掉 5–17%。
加 `V41_YOCO_MIN_CHUNK=4096` 门槛：chunk 小于 4096 token 直接走原始路径。

**(b) 并发下不能崩。** 初版把裁剪后的 SMLA/QLI 元数据 `copy_` 回 builder 的共享 buffer，
而那个 buffer 被 ACL 图重放读取 —— 192 并发必崩（`AclmdlRIExecuteAsync` 507899）。
改为**每次 forward 新分配张量**，并让 `wait_for_device_metadata` 对未注册的 id 宽容
（新张量在当前 stream 上构建，无需跨流等待）。

## 6. 分档调度（`request_queue.py` + `scheduler.py`）

混长负载下，4 条 200K 长请求会把 32 条 1K 短请求饿死 200 秒。做法：

- 按 prompt 长度分 4 档（默认边界 16K / 64K / 256K），`TieredRequestQueue` 优先出短档；
- `scheduler.py` 加「正在跑的长 prefill 数量上限」（`V41_TIER_MAX_LONG_PREFILL`）。

**关键前提**：必须同时设 `VLLM_ADMISSION_GATE=0`。上游的 admission gate 在队列层之上
一刀切放行长 prefill，长请求一旦进入 running 就不再受队列顺序影响——实测不关 gate 时
分档完全无效（212 s vs 206 s）。

| 配置 | 短请求 p50 | 长请求 min–max |
|---|---:|---:|
| FCFS 基线 | 206.4 s | 208.4–208.6 s |
| 仅分档队列 | 212.0 s | 209.8–214.3 s |
| + 长 prefill 上限 1 | 201.0 s | 202.7–203.4 s |
| **+ `VLLM_ADMISSION_GATE=0`** | **69.3 s** | 142.5–371.0 s |

代价是长请求变慢（208 → 371 s），这是明确的取舍。

## 7. EPLB 为什么没做成

修了两个 bug 后仍卡在算子边界：

1. `serve_v2.sh` 只在 `EPLB_DYNAMIC=1` 时才拼 `eplb_config`，只给 `EPLB_MAP_PATH` 会静默失效；
2. DSpark drafter 有 128 个专家，而 `expert_file_to_tensor` 对 MTP 层复用 layer 0 的宽度（49/rank）
   → `EPLB local expert capacity mismatch: allocated=17, placement=49`；
3. 修完构造层后，带冗余专家时 MoE 算子报
   `Index out of range in dimension 0: index value ... exceeds bounds 384`（物理专家 392 > 384）。

结论：本 build 不支持带冗余的 EPLB。EP8 下 2% 冗余本来收益也有限，真正的空间在 EP16/32。
