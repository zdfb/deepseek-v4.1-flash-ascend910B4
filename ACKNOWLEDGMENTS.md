# 致谢 / Acknowledgments

这个项目能成立，几乎完全站在别人的肩膀上。按贡献顺序：

## 1. 上游项目：`deepseek-v4.1-flash-ascend910B`

**作者：chiro2001**（<chiro2001@163.com>）
https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B

本项目是它的**下游分支**。上游已经解决了这个模型在昇腾上跑起来最难的 90%：

- 完整的 vllm-ascend V4.1 补丁集（`dsa_v41` 注意力、`rope_dsv4`、MoE token dispatcher、
  engram JIT、QLI、slot 重排、admission gate 等）
- 一键起服脚本（`serve_a2.sh` / `serve_v2.sh`）与自检、验收口径
- engram 两层共 412 GiB 表的主机驻留方案，以及 `host_mem_pool=0` 时的降级路径
- 大量实测证据与调优记录（`evidence/`、`reports/` 共 60+ 份）

没有这些，我们连 1M 上下文都起不来。上游的 `EXPECTED_PERF.md`、`REPORTS` 也是本仓库
所有对照实验的起点。

**上游在 NOTICE 中还致谢了：** vLLM、vllm-ascend（V4.1 line）、msmodelslim、CPython。
本仓库沿用同一份致谢链，详见 [`NOTICE`](NOTICE)。

## 2. 开源的量化权重

**`chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend`**（ModelScope）
https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend

感谢作者把这套 **W4A8_DYNAMIC（per-channel 非对称 int4 专家 + W8A8 注意力）** 的
Ascend 版权重开放出来。本仓库的所有数字都是在这份权重上测得的；它让
「490 GB 权重塞进 8×64 GiB HBM、同时留出 13.6 GiB 给 KV」成为可能——
这也是本项目能谈「1M 上下文 + 192 并发」的前提。

也感谢 **DeepSeek** 开源 DeepSeek-V4.1-Flash 基座模型。

## 3. 生态

- **vLLM**（https://github.com/vllm-project/vllm）——调度器、KV 管理、投机解码框架
- **vllm-ascend**（https://github.com/GDzhu01/vllm-ascend-v41-private）——昇腾后端
- **Ascend CANN / msmodelslim**——算子库与量化工具链
- **Mooncake**——KV store / 传输引擎（本仓库的 host KV 探索基于它，见 `docs/ROADMAP.md`）

## 4. 方法借鉴

本项目的核心优化（消费者层 prefill 截断）思路来自 **DeepInfer** 的
`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` / `KEEP_TOKENS` 设计
（https://github.com/deepseek-ai/DeepInfer）。我们在它的基础上，
针对本模型「每个消费者层各自持有 128 token 滑窗」的差异，
把 KEEP 的取值从 140 修正为 `window × (消费者层数 + 1)`，并补上了并发安全实现
（详见 `docs/ARCHITECTURE.md`）。

## 5. 免责

本仓库只包含补丁、脚本、基准与文档，**不含模型权重、不含容器镜像**。
使用者需自行遵守上游项目与模型权重各自的许可证。
