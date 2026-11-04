# DeepSeek-V4.1-Flash 910B4 Turbo

[![CI](https://github.com/zdfb/deepseek-v4.1-flash-ascend910B4/actions/workflows/ci.yml/badge.svg)](https://github.com/zdfb/deepseek-v4.1-flash-ascend910B4/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/hardware-8%C3%97Ascend%20910B4-orange.svg)](docs/DEPLOY.md)
[![Context](https://img.shields.io/badge/context-1M%20tokens-green.svg)](docs/ARCHITECTURE.md)
[![Status](https://img.shields.io/badge/status-experimental-yellow.svg)](#8-项目状态)

在 **8 × Ascend 910B4（64 GiB/卡）** 上把 DeepSeek-V4.1-Flash **W4A8** 推理做到
**1M 上下文 + 192 并发 + 长 prefill 提速 40%**，并给出可复现的测量口径。

本项目是 [`chiro2001/deepseek-v4.1-flash-ascend910B`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)
的下游分支，只做**增量优化**，不重新造轮子。请先读 [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md)。

---

## 0. 一句话

> 这个模型 40 层里只有 **4 层产长上下文 KV**，第 21 层之后全是「消费者层」——
> 它们只写自己的 128-token 滑窗、长 KV 全读源层。
> 所以 **prefill 时尾部 19 层只需要算每个请求末尾的一小段 token**。
> 我们把这件事实做成了可用的补丁：长 prefill **从 5,099 提升到 7,162 tok/s**。

## 1. 性能（8 × 910B4，真权重）

测试条件：TP8 / EP8、`MAX_SEQS=192`、`BAT_TOKENS=8192`、`GPU_UTIL=0.92`、
`PREFIX=1`、`GRAPH=1`、DSpark `SP_TOKENS=5`、快速 prefill 档（`KEEP=140`）。

### 1.1 单流（并发 1，1K prompt / 128 输出）

| 指标 | 数值 |
|---|---|
| **单流吞吐** | **50.9 tok/s** |
| TTFT | 0.375 s |
| 每步耗时 | ~50 ms（依 A 波动，见下） |
| 投机接受长度 A | 随语料在 2.2–2.6 波动 |

> 该值为干净重启后、单流 1K prompt 的单次实测。单流 tok/s 强依赖投机接受长度 A 与
> 设备状态（长跑后有热漂移/显存碎片会偏低，早前一轮为 38.6 tok/s），**跨 run 不要直接比
> 绝对值**，见 §1.5 测量口径。

### 1.2 并发下的单流平均吞吐与聚合吞吐

每档 192 条请求（并发 1 用 16 条），请求带唯一 nonce 避免 prefix cache 命中。

| 并发 | 单流平均 tok/s | 聚合 tok/s | A | TTFT |
|---:|---:|---:|---:|---:|
| 1 | **50.9** | ~52 | —¹ | 0.375 s |
| 32 | 28.7 | 540.8 | 2.27 | 1.23 s |
| 64 | 17.8 | 723.5 | 2.27 | 2.33 s |
| 96 | 13.6 | 851.4 | 2.28 | 2.68 s |
| 128 | 12.8 | 850.9 | 2.26 | 2.87 s |
| **192** | **8.0** | **1,008.4** | 2.31 | 5.05 s |

¹ 并发 1 的 50.9 tok/s 为干净重启后单次实测（A 随语料 2.2–2.6 波动，该次未单独记录）；
其余档位为同批 192 请求、带 nonce 的整套测量。

192 是上限：`cudagraph_capture_sizes` 最大 1152 = 192 ×(1 + 5 个 draft token)，
再往上需要重建全部 ACL 图，实测反而更差。

吞吐 ≈ 每步 tok/s × A：A 每 +0.1 约 +4%。步时本身也随设备状态（热漂移、显存碎片）变化，
所以**跨 run 比较请用 `聚合 ÷ A`**，不要直接比绝对 tok/s。

### 1.3 长上下文

| 项目 | 数值 |
|---|---|
| **最大上下文** | **1,048,576 token**（模型 `max_position_embeddings`，输入+输出共用） |
| 实测通过 | 965,209 token |
| 超限行为 | 400 + `maximum context length is 1048576 tokens` |
| 240K prompt prefill | 33.6 s → **7,162 tok/s** |
| 965K prompt prefill | 186.5 s → 5,175 tok/s |

对照：同机不截断（`V41_YOCO_KEEP=0`）时 252K prompt 为 5,099 tok/s，
开启消费者层截断后 **+40%**（不同次运行，见 §1.5 口径说明）。

### 1.4 容量

KV 池 **2,941,853 token（13.6 GiB）**，权重占 512 GiB HBM 的约 96%：

| 场景 | 单条 KV | 可同时 |
|---|---:|---:|
| 1M 上下文 | ~7.0 GiB | 2 |
| 200K | ~1.0 GiB | 14 |
| 32K | ~0.16 GiB | 90 |
| 1K | ~5 MB | 192（受 `MAX_SEQS` 限制） |

prefix cache 命中时二次请求：147K prompt 从 18.9 s 降到 **1.71 s**。

### 1.5 测量口径（两组数字不要混着比）

- **绝对 tok/s 不能跨 run 比**：除语料随机导致接受长度 A 在 2.1–2.6 波动外，
  设备状态（热漂移、长跑后的显存碎片）也会让步时变化——同一配置单流实测从
  38.6 到 50.9 tok/s 都出现过。跨配置比较请用 `聚合 ÷ A`。
- **记录单流数字时请附带条件**（prompt 长度、是否干净重启、A），否则不可复现。
- **长上下文测量必须加唯一 nonce**：否则第二次命中 prefix cache（6.9 s → 0.64 s）。
- 复现脚本见 [`benchmarks/`](benchmarks/)。

## 2. 我们加了什么

| 补丁 | 作用 | 收益 |
|---|---|---|
| [`patches/files/model_v41.py`](patches/files/model_v41.py) + [`v41_yoco.py`](patches/files/v41_yoco.py) | 消费者层 prefill 截断（YOCO） | 长 prefill **+40%** |
| [`patches/files/request_queue.py`](patches/files/request_queue.py) | 按上下文长度分档的调度队列 | 混长负载短请求 p50 **206 s → 69 s** |
| [`patches/files/scheduler.py`](patches/files/scheduler.py) | 长 prefill 并发上限 | 同上（需配合关闭 admission gate） |
| [`patches/files/eplb_utils.py`](patches/files/eplb_utils.py) | EPLB 加载器兼容 DSpark drafter | 修通 `expert_map` 的构造期崩溃 |
| [`patches/files/dsa_v41_kvstore.py`](patches/files/dsa_v41_kvstore.py) | 给 V4.1 自定义注意力补 KV store 保存钩子 | host KV tier 的前置条件（见 `docs/ROADMAP.md`） |

**全部补丁都用环境变量门控，默认关闭**，不改变上游行为。

## 3. 快速开始

```bash
# 1) 准备：上游镜像 + 本项目补丁 + 量化权重（不随仓库分发）
#    权重：https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend

# 2) 构建镜像（把补丁烘进正式路径）
docker build -t deepseek-v4.1-flash-ascend910B4:910b4 .

# 3) 起服（业务挂载只有模型，必须可写）
bash deploy/docker-run.sh
```

启动约 40–55 分钟（490 GB 权重 + 静态算子编译 + 27 组图捕获）。
就绪判据：`curl http://127.0.0.1:8100/health` 返回 200。

## 4. 仓库结构

```
deepseek-v4.1-flash-ascend910B4/
├── README.md               本文件
├── LICENSE                 Apache-2.0（继承上游）
├── NOTICE                  完整版权与依赖声明
├── ACKNOWLEDGMENTS.md      致谢：上游项目、开源权重、生态
├── CHANGELOG.md
├── CONTRIBUTING.md         贡献指南（含"实测依据"硬要求）
├── CODE_OF_CONDUCT.md      行为准则
├── SECURITY.md             安全策略（含 --privileged 部署注意）
├── .github/                Issue / PR 模板 + 无硬件 CI
├── docs/
│   ├── ARCHITECTURE.md     为什么这样改能快（含 KEEP 的正确推导）
│   ├── DEPLOY.md           镜像、启动、验证、调模式
│   ├── PACKAGING.md        从原镜像打包成新镜像（照搬上游做法 + commit 路径的坑）
│   ├── TROUBLESHOOTING.md  13 个真实踩坑记录
│   ├── UPSTREAM-DIFF.md    与上游逐文件差异清单
│   └── ROADMAP.md          host KV tier、投机接受长度等后续方向
├── patches/
│   ├── README.md           每个补丁的作用、挂载点、开关
│   └── files/              6 个补丁文件
├── scripts/                起服脚本（上游基础上小改）+ build_image.sh + EPLB 布局表
├── benchmarks/             可复现的测量脚本（长上下文/混长/复用/等价性）
└── deploy/                 Dockerfile + docker run
```

## 5. 不要再去调的旋钮

这些都被验证过是死路，改之前先看 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)：
`BAT_TOKENS` 调大会吃掉 KV 并 OOM；`MAX_SEQS>192` 要重建全部图且更慢；
`SP_TOKENS` 被 build 硬限制为 ≥5；KV 量化（fp8/int8）在本 build 静默回退 bf16；
prefill MC2 反而慢 20%；EPLB 冗余专家超出算子索引边界 384。

## 6. 引用

如果这个项目对你有帮助，请同时引用上游项目与本项目的优化说明。

```bibtex
@misc{dsv41_flash_ascend910b4,
  title  = {DeepSeek-V4.1-Flash 910B4 Turbo: consumer-tail prefill truncation},
  year   = {2026},
  note   = {Derived from chiro2001/deepseek-v4.1-flash-ascend910B}
}
```

## 7. 许可证

Apache-2.0，与上游一致。模型权重与容器镜像不在本仓库分发，遵循各自许可证。

## 8. 项目状态

- **阶段**：实验性部署优化包（experimental），按"现状"提供，不承诺生产 SLA。
- **验证范围**：全部性能数字在 **单机 8 × Ascend 910B4**、CANN 9.1.0 上实测；
  未在其他昇腾型号（910B 其它 die、910C、Atlas 系列）上验证。
- **可复现性**：只做"取现成产物"的复现（预构建基础镜像 + ModelScope 权重），
  不从源码编译 vllm-ascend。详见 [`docs/DEPLOY.md`](docs/DEPLOY.md) 的"可复现性前置"。
- **已知取舍**：快速档（`APPROX=1`）不保证逐 token 等价；精确档比不截断慢约 10%；
  分档调度会拖慢长请求（详见 [`CHANGELOG.md`](CHANGELOG.md) 与 [`docs/ROADMAP.md`](docs/ROADMAP.md)）。
- **CI**：仅静态检查（语法 / 许可证头 / 敏感信息 / 相对链接），真机测试需在 910B 上手动跑。

## 9. 参与贡献

欢迎 issue 与 PR。**提性能相关改动前请先读**
[`CONTRIBUTING.md`](CONTRIBUTING.md)（尤其是"测量口径"与"整文件 overlay、env 门控、
默认关闭"的硬要求）以及 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) /
[`docs/ROADMAP.md`](docs/ROADMAP.md) 里的已验证死路，避免重复负结果。

- 行为准则：[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- 安全漏洞：请勿公开提 issue，走 [`SECURITY.md`](SECURITY.md)
