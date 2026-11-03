# 踩过的坑（13 条，都是真实报错）

诊断顺序建议：先看 `docker logs` 里第一条 Python traceback，再看容器是否被信号杀掉
（`exit code: None` = 信号，通常是 OOM 或 C++ abort）。

## 启动期

**1. `ValueError: To serve at least one request with the model's max seq len (1048576), (6.98 GiB KV cache is needed, which is larger than the available KV cache memory (2.46 GiB)`**
`BAT_TOKENS` 调大后 activation 内存暴涨，KV 池被挤到 2.46 GiB。
8192 是本机甜点；16384 能起来但 252K 请求会 OOM。

**2. `Exited (1) OSError: [Errno 98] Address already in use`**
上一个容器没退干净，`--net=host` 下 8100 被占。启动实验前先确认
`ss -ltn | grep 8100` 为空，或 `docker rm -f` 旧容器并等待端口释放。

**3. `RuntimeError: DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3.`**
`SP_TOKENS` 被 build 硬限制，不能调小。7 可以但实测更慢。

**4. worker 被信号杀掉（`Worker proc died unexpectedly (exit code: None)`），日志里没有 Python traceback**
典型原因是主机内存被吃光或 C++ abort。我们遇到的具体例子：
Mooncake store 的 `global_segment_size=64GB/rank` × 8 = 512 GB，
而机器上 engram 已占 300 GB → worker 直接死。降到 8 GB/rank 后正常。

**5. `FileNotFoundError: optional/quarot.safetensors`**
权重的文件清单要从 `Root=""` 和 `Root="engram_int8"` 两个根拼，漏了 `optional/` 目录。

## 运行期

**6. `NPU out of memory. Tried to allocate 1.19 GiB (… 825.75 MiB free)`**
`BAT_TOKENS=16384` 时余量太小，长请求进来就 OOM。回到 8192。

**7. `replay:…NPUGraph.cpp:281 AclmdlRIExecuteAsync …, error code is 507899`（只在并发压测时）**
YOCO 初版把裁剪后的 SMLA/QLI 元数据 `copy_` 回 builder 的共享 buffer，
而那个 buffer 被 ACL 图重放读取 → 图重放读到错位内容。
修法：每次 forward **新分配**元数据张量 + 让 `wait_for_device_metadata` 对未注册 id 宽容。
（本仓库的 `v41_yoco.py` + `device_metadata.py` 已是修复版。）

**8. `AttributeError: 'RopeDataProxy' object has no attribute 'items'`**
我们的 cache bundle 里 `cos`/`sin` 不是 dict 而是 `RopeDataProxy`。
它支持张量索引（返回裁剪后的新 proxy），直接 `bundle.cos[idx]` 即可。

**9. `Attempted to call function marked as skipped`（torch._dynamo）**
在 fullgraph 捕获的 forward 里调用了 `os.path.exists` 之类的 Python 副作用。
截断逻辑必须包在 `not torch.compiler.is_compiling()` 里。

**10. `EPLB local expert capacity mismatch: allocated=17, placement=49`**
DSpark drafter 有 128 专家，而 `expert_file_to_tensor` 对 MTP 层复用 layer 0 的宽度。
`patches/files/eplb_utils.py` 已修（MTP 层按自己的专家数生成布局）。
修完仍会遇到第 11 条。

**11. `Index out of range in dimension 0: index value … exceeds bounds 384`（配了冗余专家）**
物理专家 392 > 算子索引边界 384。本 build 不支持带冗余的 EPLB。

**12. `NotImplementedError: MooncakeBackend does not support protocol 'tcp'.`**
Ascend 后端只认 `"protocol": "ascend"`。

**13. 压测中 kill 掉客户端后，引擎对后续请求**全部挂起**（单请求也超时）**
现象：`docker logs` 里不再出现 `Running: N reqs`，NPU 占用显示空闲，
但新请求一律超时；`docker restart` 也恢复不了，必须 `docker rm -f` 重建容器。
原因：大并发批次里客户端断连后请求没被正确清理，调度器卡住。
规避：压测脚本要能优雅收尾；出现该现象直接重建容器。

**14. `ValueError: Sparse KV offload don't support compress now.`**
内置的 `sparse_kv_offload_config` 明确拒绝带 `compress_ratios` 的模型——正是本模型。

## 测量陷阱（不算报错，但会让结论错）

**A. 冷启动的第一档并发偏低 2–2.5×。** 同一配置 conc96 首测 463.9、次测 970.2。
跨 run 比较必须让被测档位不是首档。

**B. 绝对吞吐不可跨 run 比。** 语料随机导致接受长度 A 在 2.1–2.6 波动，
必须用 `聚合 ÷ A` 归一（见 README §1.5）。

**C. prefix cache 会污染长上下文测量。** 重复 prompt 第二次走缓存（6.9 s → 0.64 s），
测真实 prefill 要给每个请求加唯一 nonce。

**D. 本栈跨进程不是逐位确定的。** 同一份代码重启后贪心输出可能差 1 个 token
（ACL 图/批形状改变 bf16 归约顺序）。等价性验证必须在**同一进程内**做。
