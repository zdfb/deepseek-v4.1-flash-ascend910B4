#!/usr/bin/env bash
# 起服脚本。业务挂载只有模型权重（必须可写：engram 需要回写索引）。
#
#   MODEL_HOST=/path/to/weights bash deploy/docker-run.sh
#
set -euo pipefail

MODEL_HOST=${MODEL_HOST:-/data/xmodels/DeepSeek-V4.1-Flash-1017-w4a8-Ascend}
MODEL_CONT=${MODEL_CONT:-/models/deepseek-v4.1-flash}
IMAGE=${IMAGE:-deepseek-v4.1-flash-ascend910B4:910b4}
NAME=${NAME:-dsv41}
PORT=${PORT:-8100}

[ -d "$MODEL_HOST" ] || { echo "模型目录不存在: $MODEL_HOST" >&2; exit 1; }
[ -w "$MODEL_HOST" ] || echo "警告: $MODEL_HOST 不可写，engram 可能启动失败" >&2

exec docker run -d --name "$NAME" --init \
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
  -v "$MODEL_HOST:$MODEL_CONT" \
  ${EXTRA_DOCKER_ARGS:-} \
  "$IMAGE"
