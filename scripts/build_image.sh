#!/usr/bin/env bash
# =============================================================================
# 【命令 ①】从上游镜像烘焙出本项目镜像
#
#   bash scripts/build_image.sh
#
# 它会：
#   1. 从基础镜像里探测 vllm_ascend / vLLM 的实际安装路径（不同镜像布局不同，不写死）
#   2. 把 patches/files/ 下的补丁烘焙进镜像（原文件留 <file>.orig 备份）
#   3. 构建期做 py_compile 校验
#   4. 在新镜像里复核补丁落位与默认 ENV，并打印下一条启动命令
#
# 可选环境变量：
#   BASE_IMAGE  基础镜像（默认上游镜像）
#   IMAGE_TAG   产出镜像名（默认 deepseek-v4.1-flash-ascend910B4:910b4）
#   NO_CACHE    1 = 不用构建缓存
#   SKIP_VERIFY 1 = 跳过产物复核
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
cd "$PKG" || exit 1

BASE_IMAGE=${BASE_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler}
IMAGE_TAG=${IMAGE_TAG:-deepseek-v4.1-flash-ascend910B4:910b4}
CACHE_ARGS=()
[ "${NO_CACHE:-0}" = "1" ] && CACHE_ARGS+=(--no-cache)

say() { printf '\n\033[1m[build]\033[0m %s\n' "$*"; }
die() { printf '\n\033[31m[build][FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 0) docker ----------
DOCKER="docker"
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo -n docker"
$DOCKER info >/dev/null 2>&1 || die "docker 不可用（当前用户无权限且 sudo -n 失败）"

# ---------- 1) 从基础镜像探测安装路径 ----------
say "探测基础镜像布局：$BASE_IMAGE"
$DOCKER image inspect "$BASE_IMAGE" >/dev/null 2>&1 || \
  die "基础镜像不存在：$BASE_IMAGE
  可选做法：
    docker pull $BASE_IMAGE
    或 BASE_IMAGE=<你本地的上游镜像> bash scripts/build_image.sh"

PROBE=$($DOCKER run --rm --entrypoint python3 "$BASE_IMAGE" - <<'PY' 2>/dev/null | tail -1
import importlib.util, os, sys
spec = importlib.util.find_spec("vllm_ascend")
vllm = importlib.util.find_spec("vllm")
if spec is None or vllm is None:
    print("MISSING"); sys.exit(1)
print(os.path.dirname(spec.origin), os.path.dirname(vllm.origin))
PY
)
[ -n "${PROBE:-}" ] && [ "$PROBE" != "MISSING" ] || die "探测失败：镜像里找不到 vllm_ascend / vllm"
ASCEND_PKG=${PROBE%% *}
VLLM_ROOT=${PROBE##* }
say "vllm_ascend = $ASCEND_PKG"
say "vllm        = $VLLM_ROOT"

# ---------- 2) 构建 ----------
say "构建 $IMAGE_TAG"
$DOCKER build "${CACHE_ARGS[@]}" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg ASCEND_PKG="$ASCEND_PKG" \
  --build-arg VLLM_ROOT="$VLLM_ROOT" \
  -t "$IMAGE_TAG" . || die "docker build 失败（看上面 [bake] 日志）"

# ---------- 3) 产物复核 ----------
if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  say "复核产物"
  $DOCKER run --rm --entrypoint bash "$IMAGE_TAG" -c "
    set -e
    echo '--- 补丁落位 ---'
    for f in \
      ${ASCEND_PKG}/models/deepseek_v41/model.py \
      ${ASCEND_PKG}/models/deepseek_v41/v41_yoco.py \
      ${ASCEND_PKG}/worker/device_metadata.py \
      ${ASCEND_PKG}/attention/dsa_v41.py \
      ${VLLM_ROOT}/vllm/v1/core/sched/request_queue.py \
      ${VLLM_ROOT}/vllm/v1/core/sched/scheduler.py \
      ${ASCEND_PKG}/eplb/core/eplb_utils.py ; do
      printf '  %-95s %s\n' \"\$f\" \"\$( [ -f \$f ] && echo OK || echo MISSING )\"
    done
    echo '--- 默认 ENV ---'
    echo \"  V41_YOCO_KEEP=\$V41_YOCO_KEEP V41_YOCO_APPROX=\$V41_YOCO_APPROX V41_YOCO_MIN_CHUNK=\$V41_YOCO_MIN_CHUNK\"
    echo '--- 备份文件 ---'
    ls -1 ${ASCEND_PKG}/models/deepseek_v41/model.py.orig 2>/dev/null && echo '  (orig 备份存在，可回滚)'
  " || die "产物复核失败"
fi

# ---------- 4) 下一条命令 ----------
cat <<EOF

$([ "${SKIP_VERIFY:-0}" = "1" ] && echo '')

  ✅ 镜像已就绪：$IMAGE_TAG

  启动：
    IMAGE=$IMAGE_TAG bash deploy/docker-run.sh

  切换模式：
    精确截断   -e V41_YOCO_KEEP=2560 -e V41_YOCO_APPROX=0
    完全关闭   -e V41_YOCO_KEEP=0

  验证（启动 40–55 分钟后）：
    curl -s http://127.0.0.1:8100/health
    python3 benchmarks/check_long_ctx.py --chars 390000     # 期望 TTFT ≈ 33 s
EOF
