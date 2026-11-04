# 与上游的差异清单

对照对象：[`chiro2001/deepseek-v4.1-flash-ascend910B`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)（Apache-2.0）。

对比方式：clone 上游（494 MB / 9009 文件），与本仓库重叠的 5 个目录做逐文件 `cmp`。

## 1. 规模

| | 上游 | 本仓库 |
|---|---:|---:|
| 体积 | 494 MB | 570 KB（不含权重/镜像） |
| 文件数 | 9009 | 待见 git |
| 定位 | 完整研究仓（算子、量化、实验证据） | 可部署的增量优化包 |

## 2. 重叠文件：只有 2 个脚本不同

在 `scripts/`、`patches/`、`reports/`、`tools/`、`docs/` 五个目录里逐字节比对，
**除以下 2 个文件外全部一致**（上游补丁、tools 319 个、docs 236 个、reports 67 个）：

| 文件 | 差异 | 内容 |
|---|---:|---|
| `scripts/serve_a2.sh` | +32 行 | EPLB 变量定义与透传、3 个补丁挂载点（scheduler / request_queue / eplb_utils / YOCO）、`docker --init` 防僵尸、`--long-prefill-token-threshold` |
| `scripts/serve_v2.sh` | +11 行 | `EPLB_MAP_PATH` 支持（上游仅 `EPLB_DYNAMIC=1` 时才拼配置）、分档阈值透传、`--kv-transfer-config` 透传 |

## 3. 本仓库新增（上游没有）

| 文件 | 作用 |
|---|---|
| `patches/files/model_v41.py` | 带 YOCO 截断的 `model.py` 全量覆盖 |
| `patches/files/v41_yoco.py` | 消费者层截断实现（新模块） |
| `patches/files/device_metadata.py` | 让 device metadata 等待对未注册 id 宽容 |
| `patches/files/request_queue.py` | 按上下文分档的 `TieredRequestQueue` |
| `patches/files/scheduler.py` | 长 prefill 并发上限 |
| `patches/files/eplb_utils.py` | EPLB 加载器兼容 DSpark drafter |
| `patches/files/dsa_v41_kvstore.py` | 给 V4.1 自定义注意力补 KV store 保存钩子 |
| `scripts/eplb_map_384x8x49.json` | 40 层 × 8 rank × 49 专家的 EPLB 布局 |
| `benchmarks/` | 长上下文 / 混长 / 复用 / 等价性 四个测量脚本 |
| `docs/` | ARCHITECTURE / BENCHMARKS / DEPLOY / TROUBLESHOOTING / ROADMAP |

文本层面确认「上游没有」的 grep 证据：

```
YOCO               : 0 个文件
v41_yoco           : 0 个文件
TieredRequestQueue : 0 个文件
V41_TIERED_SCHED   : 0 个文件
```

而下面这些**上游本来就有**，本仓库是在其基础上做增量，不重复实现：

```
expert_map_path    : 6 处（EPLB 布局机制是上游的）
kv_transfer_config : 21 处（Mooncake/P2P 连接器栈是上游的）
sparse_kv_offload  : 12 处（内置 KV 卸载，但拒绝 compress_ratios 模型）
```

## 4. 上游有、本仓库未收录

`evidence/`（254 MB，全部实测证据）、`experiments/`（188 MB）、`a2/`、`data/`、
`experimental/`、`deploy/`、`upstream/`（vendored vLLM）、`quant/`（W4A8 重打包）、
`tests/`、`results/`、`kernels/`、`fixes/`、`build_scripts/`、`logs_meta/`。

需要这些请直接取上游。本仓库刻意只保留「部署 + 本次增量」所需的部分。

## 5. 性能口径差异（别直接比数字）

上游 `EXPECTED_PERF.md` 的主表是 **A3（8×910C、16 device）**：
128K 单流 30.23 ms/step、A=2.748、90.13 tok/s（峰值 110.5）。
本仓库所有数字是 **A2（8×910B4、8 device）**：单流约 38.6–50.9 tok/s（随设备状态/A 波动，
干净重启后更高；见 README §1.1/§1.5）。
硬件不同，只能做「同机同配置 A/B」，不能跨平台比绝对值。
