# 补丁说明

所有补丁都是**整文件覆盖**（overlay），通过 `Dockerfile` COPY 到正式导入路径。
每个补丁都由环境变量门控，**默认不改变上游行为**。

| 文件 | 覆盖目标 | 开关 | 默认 |
|---|---|---|---|
| `model_v41.py` | `vllm_ascend/models/deepseek_v41/model.py` | `V41_YOCO_KEEP>0` | 镜像烘入 `140` |
| `v41_yoco.py` | 同目录新模块（被上面 import） | 同上 | — |
| `device_metadata.py` | `vllm_ascend/worker/device_metadata.py` | 无（对未注册 id 宽容，仅在 YOCO 新张量时生效） | 生效 |
| `dsa_v41_kvstore.py` | `vllm_ascend/attention/dsa_v41.py` | `V41_KV_STORE=1` | 关 |
| `request_queue.py` | `vllm/v1/core/sched/request_queue.py` | `V41_TIERED_SCHED=1` | 关 |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | `V41_TIER_MAX_LONG_PREFILL>0` | 关 |
| `eplb_utils.py` | `vllm_ascend/eplb/core/eplb_utils.py` | `EPLB_MAP_PATH` 非空时生效 | 关 |

## 关键开关

| 变量 | 取值 | 作用 |
|---|---|---|
| `V41_YOCO_KEEP` | 0 / 140 / 2560 | 0=关闭；140=快速（近似）；2560=精确 |
| `V41_YOCO_APPROX` | 0 / 1 | 1=允许 `KEEP` 小于精确值并接受近似 |
| `V41_YOCO_MIN_CHUNK` | 4096 | chunk 小于该值不截断（保护短请求） |
| `V41_TIERED_SCHED` | 0 / 1 | 按上下文分档排队 |
| `V41_TIER_BOUNDS` | `16384,65536,262144` | 分档边界 |
| `V41_TIER_MAX_LONG_PREFILL` | 0 / 1 / N | 同时进行的长 prefill 上限 |
| `V41_LONG_PREFILL_THRESHOLD` | 1024 | 长 prefill 的 chunk 上限（传给 vLLM） |
| `VLLM_ADMISSION_GATE` | 0 / 1 | **分档生效的必要条件**（设 0） |
| `V41_KV_STORE` | 0 / 1 | 打开 KV store 保存钩子（实验性） |

## 与上游同名补丁的关系

上游 `patches/files/` 里也有一批补丁（`v41_decode_guard.py`、`patch_cudagraph.py`、
`mainpy/` 等）。**本仓库不重复收录**，需要时从上游取——我们的 `Dockerfile`
只覆盖上表 7 个文件，其余保持上游版本。

`model_v41.py` / `scheduler.py` / `request_queue.py` / `eplb_utils.py` / `dsa_v41_kvstore.py`
都是从**上游对应版本**派生再改的，因此文件头保留上游版权，改动点用
`[YOCO-TAIL]` / `[ZCODE-TIER]` / `[V41-KV-STORE]` 之类标记标出，便于 code review 和 upstream PR。
