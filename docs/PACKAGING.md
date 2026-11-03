# 从原镜像打包成新镜像

本项目的镜像**不是从零构建的**，而是在上游镜像之上「烘焙补丁」得到。
做法参考上游的 `Dockerfile` + `scripts/build_image.sh`。

---

## 1. 上游是怎么做的（我们照搬的部分）

上游仓库根目录有一个 `Dockerfile`，外面套 `scripts/build_image.sh`：

| 上游的做法 | 为什么这么做 | 本项目 |
|---|---|---|
| `ARG BASE_IMAGE` + `FROM ${BASE_IMAGE}` | 基础镜像用在哪台机器上可能不同，不写死 | ✅ 照搬 |
| `FROM` 之后**重新声明** `ARG BASE_IMAGE` | Docker 规则：`FROM` 之前的 ARG 在 stage 内不可见 | ✅ 照搬（这个坑踩过） |
| 补丁先 `COPY ... /tmp/bake/`，再由 `RUN` 安装 | 便于在一个 RUN 里做校验/备份/日志 | ✅ 照搬 |
| 覆盖前把原文件留成 `<file>.orig` | 出问题可就地回滚，不用重拉镜像 | ✅ 照搬 |
| 安装路径用 `ARG ASCEND_PKG / VLLM_ROOT` | 不同镜像的 vllm 安装路径不同 | ✅ 照搬 |
| 构建期 `md5sum` + `py_compile` 校验 | 坏补丁绝不产出镜像 | ✅ 用 `py_compile` |
| 镜像不含模型权重 | 490 GB 不适合打进镜像 | ✅ 照搬 |
| 外面套 `build_image.sh`，探测路径 + 构建 + 复核 + 打印下一条命令 | 一条命令闭环 | ✅ 照搬 |

**差别**：上游把 15 个补丁烘进去（engram 系列、dsa_v1、rope_dsv4 …）；
本项目只覆盖 7 个文件（见 `patches/README.md`），其余保持上游版本——
所以我们的镜像必须建立在上游镜像之上，不能反过来。

---

## 2. 本项目怎么打包（推荐路径）

```bash
# 0) 准备基础镜像（没有就拉）
docker pull quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler

# 1) 一条命令：探测路径 → 构建 → 复核 → 打印启动命令
bash scripts/build_image.sh

# 2) 启动
IMAGE=deepseek-v4.1-flash-ascend910B4:910b4 bash deploy/docker-run.sh
```

`build_image.sh` 内部做的事：

```
1) docker run --rm --entrypoint python3 $BASE_IMAGE  ← 用 importlib 探测
      vllm_ascend 的安装目录 → ASCEND_PKG
      vllm 的安装目录        → VLLM_ROOT
2) docker build --build-arg BASE_IMAGE=... \
                --build-arg ASCEND_PKG=... --build-arg VLLM_ROOT=... \
                -t $IMAGE_TAG .
3) docker run --rm --entrypoint bash $IMAGE_TAG  ← 复核
      7 个补丁是否落在正式路径、默认 ENV 是否生效、.orig 备份是否存在
4) 打印启动与验证命令
```

想换基础镜像：

```bash
BASE_IMAGE=<你的上游镜像> IMAGE_TAG=<你的 tag> bash scripts/build_image.sh
```

---

## 3. Dockerfile 的关键段落

```dockerfile
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
FROM ${BASE_IMAGE}
ARG BASE_IMAGE                      # 必须重新声明
ARG ASCEND_PKG=/vllm-workspace/vllm-ascend/vllm_ascend
ARG VLLM_ROOT=/vllm-workspace/vllm

COPY patches/files/model_v41.py /tmp/bake/model_v41.py
...

RUN set -euo pipefail; \
    install_overlay() {                                    # 覆盖已有文件
        [ -f "$2" ] || { echo "[bake][FAIL] 布局变了: $2" >&2; exit 1; }
        [ -f "$2.orig" ] || cp -a "$2" "$2.orig"           # 备份一次
        install -m 0644 "$1" "$2"; }; \
    install_new() {                                        # 新增模块（v41_yoco.py）
        mkdir -p "$(dirname "$2")"; install -m 0644 "$1" "$2"; }; \
    install_overlay /tmp/bake/model_v41.py "${ASCEND_PKG}/models/deepseek_v41/model.py"; \
    ...

RUN python3 -m py_compile ...  && echo "[bake] py_compile OK"    # 构建期校验

ENV V41_YOCO_KEEP=140 V41_YOCO_APPROX=1 V41_YOCO_MIN_CHUNK=4096
CMD ["vllm", "serve", "/models/deepseek-v4.1-flash", ...]        # 原生 vLLM 风格
```

三个容易错的点：

1. **`ARG` 必须在 `FROM` 后重新声明**，否则 `ENV BASE_IMAGE_TAG=${BASE_IMAGE}` 是空的。
2. **`CMD` 必须是单行 JSON**，多行（即使换行缩进）会被当作 shell 命令解析而报
   `unknown instruction`。
3. **`v41_yoco.py` 是新增文件**，不能走「覆盖」分支（目标不存在会 fail），要走 `install_new`。

---

## 4. 另一条路：运行容器 + `docker commit`（本次开发实际走的）

调试阶段更快的路径：

```bash
# 1) 起一个上游镜像的容器（带 --init，否则会攒僵尸）
docker run -d --name dev --init --net=host --shm-size=512g --privileged \
  --device /dev/davinci0 ... --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ... \
  -e V41_YOCO_KEEP=140 -e V41_YOCO_APPROX=1 \
  <BASE_IMAGE>

# 2) 把补丁拷进正式路径
docker cp patches/files/model_v41.py dev:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py
...

# 3) 验证通过后固化成镜像（容器的 Config.Env 会一起被提交）
docker stop dev && docker commit dev deepseek-v4.1-flash-ascend910B4:910b4
```

这一路能work，但有四个**必须知道的陷阱**（我们在开发中都踩过）：

| 陷阱 | 现象 | 规避 |
|---|---|---|
| bind-mount 的文件不会被 commit | 用 `-v` 挂进去的补丁在镜像里"消失" | 用 `docker cp` 写进可写层，或走 Dockerfile |
| 只读挂载会挡住写入 | `docker cp` 报 Read-only | 确认目标路径不是 `-v ...:ro` |
| commit 会把容器 Env 一起冻结 | 之后改 `-e` 仍可覆盖，但默认值变了 | 明确哪些 env 是要固化的 |
| commit 出来的容器没 init | 一次启动攒 1400+ 僵尸进程 | 运行时加 `--init` |

**结论**：发布用 `Dockerfile`（可复现、有校验、有 `.orig` 备份）；
`docker commit` 只用于开发期快速试错。

---

## 5. 产物验证清单

镜像建好后按顺序过一遍（`build_image.sh` 已自动做 1–3）：

```bash
# 1) 七个补丁都在正式路径
docker run --rm --entrypoint bash $TAG -c 'ls -l \
  /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/{model.py,v41_yoco.py} \
  /vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py \
  /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py \
  /vllm-workspace/vllm/vllm/v1/core/sched/{request_queue.py,scheduler.py} \
  /vllm-workspace/vllm-ascend/vllm_ascend/eplb/core/eplb_utils.py'

# 2) 默认 ENV 生效
docker run --rm --entrypoint env $TAG | grep -E 'V41_YOCO_(KEEP|APPROX|MIN_CHUNK)'

# 3) CMD 正确（原生 vLLM 风格）
docker inspect $TAG --format '{{json .Config.Cmd}}' | head -c 120

# 4) 起服后
curl -s http://127.0.0.1:8100/health -o /dev/null -w '%{http_code}\n'   # 200
docker logs dsv41 | grep 'GPU KV cache size'                            # 2,941,744 tokens
python3 benchmarks/check_long_ctx.py --chars 390000                     # TTFT ≈ 33 s
```

---

## 6. 分发

```bash
# 离线：导出/导入（压缩后约 6 GB）
docker save $TAG | zstd -T8 -3 -o deepseek-v41-910b4.tar.zst
zstd -d -c deepseek-v41-910b4.tar.zst | docker load

# 在线：推到你的 registry
docker tag $TAG <registry>/<namespace>/deepseek-v4.1-flash-ascend910B4:910b4
docker push <registry>/<namespace>/deepseek-v4.1-flash-ascend910B4:910b4
```

**镜像里只有代码补丁、启动参数和编译缓存，没有模型权重**。
分发时要同时告知权重来源与许可证（见 `NOTICE` / `ACKNOWLEDGMENTS.md`）。

---

## 7. 回滚

```bash
# 回到基础镜像：直接 `docker run $BASE_IMAGE` 即可
# 回到本项目上一版：镜像 tag 就是版本
docker run ... deepseek-v4.1-flash-ascend910B4:910b4-prev

# 只想关掉截断（不改镜像）：
docker run ... -e V41_YOCO_KEEP=0 $TAG
```
