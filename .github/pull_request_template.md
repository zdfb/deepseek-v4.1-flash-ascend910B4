# Pull Request

## 动机

<!-- 解决什么问题？关联的 issue（如有）：Closes #123 -->

## 改了什么

<!-- 简要列出改动文件与逻辑 -->

## 实测收益 / 影响

<!--
性能类改动必填。请遵守 CONTRIBUTING.md 第 5 节的测量口径：
- 用 聚合 tok/s ÷ A 做跨配置比较（绝对 tok/s 不能跨 run 比）
- 长上下文加唯一 nonce
- 等价性用 benchmarks/bench_equivalence.py in-process 对比
-->

## 已排除的负结果

<!-- 说明没有重复 docs/ROADMAP.md / docs/TROUBLESHOOTING.md 里已有的负结果 -->

## 检查清单

- [ ] 补丁是整文件 overlay，路径与 `docs/DEPLOY.md` 覆盖表一致
- [ ] 新增行为有环境变量门控，且默认关闭
- [ ] 已更新 `CHANGELOG.md`
- [ ] 派生文件的 Apache-2.0 / SPDX 头保留
- [ ] 无密码 / 内网 IP / 私有仓库 / 大文件（权重、镜像）
- [ ] 文档已相应更新（README / docs / patches/README）
