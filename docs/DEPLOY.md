# 部署

## 1. 前置

| 项 | 要求 |
|---|---|
| 硬件 | 8 × Ascend 910B4（64 GiB/卡），单机 |
| 驱动 / CANN | 与上游镜像一致（CANN 9.1.0） |
| 权重 | `chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend`（约 490 GB，需要可写） |
| 基础镜像 | 上游 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler`（或其衍生标签） |

权重需**可写**挂载：engram 的 int8 表在启动时会回写索引文件。

## 2. 构建

```bash
docker build -t deepseek-v4.1-flash-ascend910B4:910b4 .
```

根目录 `Dockerfile` 做的事：

1. `FROM` 上游镜像（可用 `--build-arg BASE_IMAGE=...` 覆盖）；
2. 把 `patches/files/` 下的补丁 COPY 到**正式导入路径**（不是靠挂载）：

| 仓库内文件 | 容器内路径 |
|---|---|
| `model_v41.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py` |
| `v41_yoco.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/v41_yoco.py` |
| `device_metadata.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py` |
| `dsa_v41_kvstore.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py` |
| `request_queue.py` | `/vllm-workspace/vllm/vllm/v1/core/sched/request_queue.py` |
| `scheduler.py` | `/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py` |
| `eplb_utils.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/eplb/core/eplb_utils.py` |

3. 烘入默认环境变量（`V41_YOCO_KEEP=140`、`V41_YOCO_APPROX=1`、`V41_YOCO_MIN_CHUNK=4096`）；
4. `CMD` 固定成完整的 `vllm serve` 命令行（原生 vLLM 风格）。

## 3. 启动

```bash
bash deploy/docker-run.sh
```

或者手工：

```bash
docker run -d --name dsv41 --init \
  --net=host --shm-size=512g --privileged=true --ulimit memlock=-1 \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /data/xmodels/DeepSeek-V4.1-Flash-1017-w4a8-Ascend:/models/deepseek-v4.1-flash \
  deepseek-v4.1-flash-ascend910B4:910b4
```

`--init` 不是可选项：镜像里没有 init 时容器 PID 1 是 `sleep infinity`，
TBE 编译产生的子进程退出后无人回收，实测一次启动会累积 1400+ 僵尸进程。

## 4. 等待就绪

```bash
for i in $(seq 1 360); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health | grep -q 200 && echo READY && break
  sleep 10
done
```

约 40–55 分钟：490 GB 权重加载 + 静态算子编译 + 27 组 ACL 图捕获 + engram（412 GiB）注册。
就绪后 `GPU KV cache size` 应为 **2,941,744 token**、`MemAvailable` 仍充裕。

## 5. 验证

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","messages":[{"role":"user","content":"只回答两个字：就绪"}],"max_tokens":64,"temperature":0}'

python3 benchmarks/check_long_ctx.py --chars 390000     # 期望 TTFT ≈ 33 s / 7,500 tok/s
```

## 6. 切换模式

| 模式 | 设置 | 长 prefill | 等价性 |
|---|---|---|---|
| **默认（快速）** | 镜像已烘入 `KEEP=140 APPROX=1` | 7,162 tok/s | 长 prompt 近似 |
| 精确截断 | `-e V41_YOCO_KEEP=2560 -e V41_YOCO_APPROX=0` | ~4,950 tok/s | 理论等价 |
| 完全关闭 | `-e V41_YOCO_KEEP=0` | ~5,100 tok/s | 完全等价 |

分档调度（混长负载）：

```bash
-e V41_TIERED_SCHED=1 -e V41_TIER_MAX_LONG_PREFILL=1 \
-e V41_LONG_PREFILL_THRESHOLD=1024 -e VLLM_ADMISSION_GATE=0
```

注意最后一项是**必要条件**，原因见 `docs/ARCHITECTURE.md` 第 6 节。

## 7. 常见运维

```bash
docker logs -f dsv41 | grep -E "ERROR|KV cache size|Capturing"
docker exec dsv41 ps -eo stat | grep -c Z     # 应为 0
docker rm -f dsv41                             # 重启（启动时间会重来一遍）
```
