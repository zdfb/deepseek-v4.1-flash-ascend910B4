# DeepSeek-V4.1-Flash 910B4 Turbo

在 **8 × Ascend 910B4（64 GiB/卡）** 上把 DeepSeek-V4.1-Flash **W4A8** 推理做到
**1M 上下文 + 192 并发 + 长 prefill 提速 47%**，并给出可复现的测量口径。

本项目是 [`chiro2001/deepseek-v4.1-flash-ascend910B`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)
的下游分支，只做**增量优化**，不重新造轮子。请先读 [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md)。

---

## 0. 一句话

> 这个模型 40 层里只有 **4 层产长上下文 KV**，第 21 层之后全是「消费者层」——
> 它们只写自己的 128-token 滑窗、长 KV 全读源层。
> 所以 **prefill 时尾部 19 层只需要算每个请求末尾的一小段 token**。
> 我们把这件事实做成了可用的补丁：长 prefill **从 5,099 提升到 7,531 tok/s**。

## 1. 性能（8 × 910B4，真权重）

完整口径见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)。

| 维度 | 数值 |
|---|---|
| 最大上下文 | **1,048,576 token**（模型上限，实测 973,206 通过） |
| 单流 decode | **34.1 tok/s**（接受长度 A=2.12，每步 16.1 tok/s） |
| 聚合 decode @192 并发 | **1,138–1,222 tok/s** |
| 长上下文 prefill | **7,531 tok/s @252K**（上游默认 5,099，**+47%**） |
| 973K prefill | 5,306 tok/s（TTFT 183 s） |
| 并发容量 | 192 条短请求 / 14 条 200K / 2 条 1M（KV 池 2.94M token） |
| prefix 复用 | 147K prompt 二次请求 TTFT **1.71 s**（冷启 18.9 s） |

## 2. 我们加了什么

| 补丁 | 作用 | 收益 |
|---|---|---|
| [`patches/files/model_v41.py`](patches/files/model_v41.py) + [`v41_yoco.py`](patches/files/v41_yoco.py) | 消费者层 prefill 截断（YOCO） | 长 prefill **+47%** |
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
docker build -t deepseek-v4.1-flash-ascend910B4:910b4 deploy/

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
├── docs/
│   ├── ARCHITECTURE.md     为什么这样改能快（含 KEEP 的正确推导）
│   ├── BENCHMARKS.md       全部实测数据 + 测量方法（含 A 归一化陷阱）
│   ├── DEPLOY.md           镜像、启动、验证、调模式
│   ├── TROUBLESHOOTING.md  13 个真实踩坑记录
│   ├── UPSTREAM-DIFF.md    与上游逐文件差异清单
│   └── ROADMAP.md          host KV tier、投机接受长度等后续方向
├── patches/
│   ├── README.md           每个补丁的作用、挂载点、开关
│   └── files/              6 个补丁文件
├── scripts/                起服脚本（上游基础上小改）+ EPLB 布局表
├── benchmarks/             可复现的测量脚本（长上下文/混长/复用/等价性）
└── deploy/                 Dockerfile + docker run
```

## 5. 已排除的方向（都有实测，别再踩）

`BAT_TOKENS` 调大会吃掉 KV 并 OOM；`MAX_SEQS>192` 需要重建全部图且实测更差；
`SP_TOKENS` 被 build 硬限制为 ≥5；KV 量化（fp8/int8）在本 build 上静默回退 bf16；
prefill MC2 反而慢 20%；关闭 engram 只快 5%；EPLB 冗余专家超出算子索引边界。
详见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) 第 5 节。

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
