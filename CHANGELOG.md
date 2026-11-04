# Changelog

## [Unreleased]

### Changed

- **单流口径更新**：干净重启后单流（1K prompt / 128 输出）实测 **50.9 tok/s、TTFT 0.375 s**
  （早前一轮为 38.6 tok/s / 0.42 s）。README §1.1/§1.2/§1.5 与 `docs/UPSTREAM-DIFF.md`
  同步改为「38.6–50.9 区间 + 随设备状态/A 波动」，并明确记录单流数字需附带条件。

## [0.1.0] — 2026-11-03

首个公开版本。基于上游 `deepseek-v4.1-flash-ascend910B` 的部署包，新增长上下文
prefill 截断与调度修复，全部在 8×910B4 上实测。

### Added

- **消费者层 prefill 截断（YOCO）**：`patches/files/model_v41.py` + `v41_yoco.py`。
  - 长上下文 prefill：**5,099 → 7,162 tok/s（+40%）** @240K
  - `KEEP` 按 `window × (消费者层数 + 1)` 推导（2,560），提供 140 的快速近似档
  - `V41_YOCO_MIN_CHUNK=4096` 门槛，短 prompt 零损失
  - 并发安全：裁剪后元数据改为每次新分配（修掉 `NPUGraph 507899` 崩溃）
- **按上下文分档调度**：`request_queue.py` + `scheduler.py`。
  - 混长负载下短请求 p50 **206.4 s → 69.3 s**（需 `VLLM_ADMISSION_GATE=0`）
- **EPLB 加载器修复**：`eplb_utils.py` 让 MTP/draft 层按自身专家数生成布局。
  （修完仍受算子索引边界限制，见 `docs/ROADMAP.md`）
- **KV store 保存钩子**：`dsa_v41_kvstore.py`，让 host KV tier 具备前置条件
  （Mooncake store 验证通：8/8 client、64 GB 池；但 save 开销过大，见 ROADMAP）
- **device metadata 宽容等待**：`device_metadata.py`
- **测量工具**：`benchmarks/` 四个脚本（长上下文 / 混长 / 前缀复用 / 等价性）
- **文档**：ARCHITECTURE / BENCHMARKS / DEPLOY / TROUBLESHOOTING / UPSTREAM-DIFF / ROADMAP

### Changed

- `scripts/serve_a2.sh`：EPLB 透传、4 个补丁挂载点、`docker --init`、长 prefill 阈值
- `scripts/serve_v2.sh`：`EPLB_MAP_PATH` 支持、`--kv-transfer-config` 透传

### Known limitations

- 默认的快速模式（`APPROX=1`）**不保证长 prompt 输出与基线逐 token 等价**
- 精确模式（`KEEP=2560`）比不截断慢约 10%
- 分档调度会拖慢长请求（208 s → 371 s），是明确取舍
- 本 build 不支持带冗余专家的 EPLB；KV 量化（fp8/int8）静默回退 bf16
