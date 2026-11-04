# 贡献指南 / Contributing

感谢你愿意为 **DeepSeek-V4.1-Flash 910B4 Turbo** 做贡献！本文件说明参与方式、
如何提 issue / PR、代码规范，以及本项目对「实测依据」的特殊要求。

## 1. 先读这几份

在动代码之前，请先读：

- [`README.md`](README.md) —— 项目做什么、性能数字、不要再去调的旋钮
- [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) —— 我们是站在谁的肩膀上
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) —— 13 个真实踩坑记录
- [`docs/ROADMAP.md`](docs/ROADMAP.md) —— 已知方向与死路

很多"看起来能做"的优化已经在 `TROUBLESHOOTING.md` / `ROADMAP.md` 里被验证为死路。
**提 PR 前先确认你没有重复别人的负结果。**

## 2. 项目性质与红线

这是一个**下游派生项目**（derived work）：

- 只做**增量优化**，不重新造轮子。核心 90% 来自上游
  [`chiro2001/deepseek-v4.1-flash-ascend910B`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)。
- 所有补丁必须是**整文件覆盖（overlay）**，通过 `Dockerfile` COPY 到正式导入路径，
  **不允许**靠运行时挂载 monkey-patch。
- 每个补丁**必须由环境变量门控，且默认关闭 / 默认不改变上游行为**（镜像烘入的
  `V41_YOCO_KEEP=140` 是唯一例外，且已在 README 说明）。

**不合规的 PR 会被直接拒绝：**
- 硬编码模型权重路径、私有镜像地址、内部主机名 / 凭据
- 未经门控、默认生效、改变上游行为的功能补丁
- 没有实测依据的性能声称

## 3. 如何提 Issue

用 [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE/) 下的模板。请尽量包含：

- **环境**：卡型号与数量、CANN / 驱动版本、镜像 tag、`npu-smi info` 摘要
- **配置**：`docker inspect` 的 CMD、关键环境变量
- **现象**：复现步骤 + 日志（`docker logs` 相关片段）+ `curl /health` 状态
- **期望 vs 实际**

## 4. 如何提 PR

1. Fork → 建分支（`fix/…`、`feat/…`、`docs/…`）
2. 改代码 + 更新文档 / CHANGELOG
3. 跑通你改动的验证（见下）
4. 提 PR，描述里写清：**动机 / 改了什么 / 实测收益 / 已排除的负结果**

### PR 检查清单

- [ ] 补丁是整文件 overlay，路径与 `docs/DEPLOY.md` 的覆盖表一致
- [ ] 新增行为有环境变量门控，默认关闭
- [ ] 已更新 `CHANGELOG.md`（`Added` / `Changed` / `Fixed`）
- [ ] 若改了性能相关逻辑，附**可复现的测量口径**（见第 5 节）
- [ ] 无密码 / 内网 IP / 私有仓库 / 大文件（权重、镜像）
- [ ] 派生文件的 Apache-2.0 头保留（见第 6 节）

## 5. 性能声称的硬要求

本项目的所有数字都讲究"可复现、可比"。提性能相关 PR 时：

- **绝对 tok/s 不能跨 run 比**：语料随机导致投机接受长度 A 在 2.1–2.6 波动，
  同一配置可差 13%。跨配置比较请用 **`聚合 tok/s ÷ A`**（本机各配置落在 240–390）。
- **长上下文测量必须加唯一 nonce**，否则第二次会命中 prefix cache（6.9 s → 0.64 s）。
- **等价性**：若声称"数值不变"，请用 [`benchmarks/bench_equivalence.py`](benchmarks/bench_equivalence.py)
  做 in-process 对比（跨 run 对比不可信）。
- 测量脚本统一放在 [`benchmarks/`](benchmarks/)，并在 PR 中说明如何复现。

## 6. 代码风格

- **Python**：遵循 PEP 8；每个文件保留其来源的 Apache-2.0 / SPDX 头
  （派生自 vLLM 的用 `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`，
  派生自华为 / vllm-ascend 的保留原头；本项目原创的用 `Copyright 2026 zdfb`）。
- **Shell**：`#!/usr/bin/env bash` + `set -euo pipefail`。
- **文档**：中文为主，专有名词保留英文；表格优先于长段落。
- **注释**：解释"为什么"而不是"是什么"，尤其是对上游行为的偏离。

## 7. 许可证

提交即表示你同意你的贡献以 **Apache-2.0** 授权（与仓库一致）。
若你贡献的代码来自其他项目，请确认其许可证兼容并在 PR 中注明来源。

## 8. 行为准则

参与本项目即表示你同意遵守 [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。

## 9. 安全

发现安全问题时**不要**公开提 issue，请走 [`SECURITY.md`](SECURITY.md) 的私下渠道。
