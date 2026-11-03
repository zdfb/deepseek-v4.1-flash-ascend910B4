# syntax=docker/dockerfile:1
#
# deepseek-v4.1-flash-ascend910B4 —— 从上游镜像烘焙本项目补丁
#
# 用法（一般由 scripts/build_image.sh 调用，不手写）：
#   docker build \
#     --build-arg BASE_IMAGE=<上游镜像> \
#     -t deepseek-v4.1-flash-ascend910B4:910b4 .
#
# 设计沿用上游项目的做法：
#   1. 补丁先 COPY 到 /tmp/bake，再由 RUN 安装到目标路径（保留 <file>.orig 备份）；
#   2. 安装路径用 ARG 参数化（不同镜像布局可能不同，不写死）；
#   3. 构建期做 py_compile 校验，坏补丁不会产出镜像；
#   4. 镜像不含模型权重。
#
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
FROM ${BASE_IMAGE}

# FROM 之前声明的 ARG 在 FROM 之后不可见，必须重新声明。
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
ARG ASCEND_PKG=/vllm-workspace/vllm-ascend/vllm_ascend
ARG VLLM_ROOT=/vllm-workspace/vllm

ENV ASCEND_PKG=${ASCEND_PKG} \
    VLLM_ROOT=${VLLM_ROOT} \
    BASE_IMAGE_TAG=${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
USER root

# ---------- 1) 补丁暂存 ----------
COPY patches/files/model_v41.py        /tmp/bake/model_v41.py
COPY patches/files/v41_yoco.py         /tmp/bake/v41_yoco.py
COPY patches/files/device_metadata.py  /tmp/bake/device_metadata.py
COPY patches/files/dsa_v41_kvstore.py  /tmp/bake/dsa_v41_kvstore.py
COPY patches/files/request_queue.py    /tmp/bake/request_queue.py
COPY patches/files/scheduler.py        /tmp/bake/scheduler.py
COPY patches/files/eplb_utils.py       /tmp/bake/eplb_utils.py

# ---------- 2) 安装到正式导入路径（前 6 个覆盖已有文件，v41_yoco.py 是新增模块）----------
RUN set -euo pipefail; \
    install_overlay() { \
        src="$1"; dst="$2"; \
        [ -f "$dst" ] || { echo "[bake][FAIL] 目标不存在，基础镜像布局变了？ $dst" >&2; exit 1; }; \
        [ -f "$dst.orig" ] || cp -a "$dst" "$dst.orig"; \
        install -m 0644 "$src" "$dst"; \
        echo "[bake] overlay $(basename "$src") -> $dst"; \
    }; \
    install_new() { \
        src="$1"; dst="$2"; \
        mkdir -p "$(dirname "$dst")"; \
        install -m 0644 "$src" "$dst"; \
        echo "[bake] new     $(basename "$src") -> $dst"; \
    }; \
    install_overlay /tmp/bake/model_v41.py        "${ASCEND_PKG}/models/deepseek_v41/model.py"; \
    install_new     /tmp/bake/v41_yoco.py         "${ASCEND_PKG}/models/deepseek_v41/v41_yoco.py"; \
    install_overlay /tmp/bake/device_metadata.py  "${ASCEND_PKG}/worker/device_metadata.py"; \
    install_overlay /tmp/bake/dsa_v41_kvstore.py  "${ASCEND_PKG}/attention/dsa_v41.py"; \
    install_overlay /tmp/bake/request_queue.py    "${VLLM_ROOT}/vllm/v1/core/sched/request_queue.py"; \
    install_overlay /tmp/bake/scheduler.py        "${VLLM_ROOT}/vllm/v1/core/sched/scheduler.py"; \
    install_overlay /tmp/bake/eplb_utils.py       "${ASCEND_PKG}/eplb/core/eplb_utils.py"

# ---------- 3) 构建期校验：坏补丁不能产出镜像 ----------
RUN set -euo pipefail; \
    python3 -m py_compile \
        "${ASCEND_PKG}/models/deepseek_v41/model.py" \
        "${ASCEND_PKG}/models/deepseek_v41/v41_yoco.py" \
        "${ASCEND_PKG}/worker/device_metadata.py" \
        "${ASCEND_PKG}/attention/dsa_v41.py" \
        "${VLLM_ROOT}/vllm/v1/core/sched/request_queue.py" \
        "${VLLM_ROOT}/vllm/v1/core/sched/scheduler.py" \
        "${ASCEND_PKG}/eplb/core/eplb_utils.py"; \
    echo "[bake] py_compile OK"; \
    rm -rf /tmp/bake

# ---------- 4) 默认行为：快速 prefill 档（近似）。精确档用 -e 覆盖 ----------
ENV V41_YOCO_KEEP=140 \
    V41_YOCO_APPROX=1 \
    V41_YOCO_MIN_CHUNK=4096

WORKDIR /workspace

LABEL org.opencontainers.image.title="deepseek-v4.1-flash-ascend910B4" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash W4A8 on 8x Ascend 910B4: consumer-tail prefill truncation (+47% long-context prefill)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

# 基础镜像 ENTRYPOINT 会 source CANN/ATB 环境后 exec "$@"，所以这里直接给 vLLM 命令行。
CMD ["vllm", "serve", "/models/deepseek-v4.1-flash", \
     "--host", "0.0.0.0", "--port", "8100", \
     "--served-model-name", "deepseek-v41", \
     "--tensor-parallel-size", "8", "--enable-expert-parallel", \
     "--quantization", "ascend", "--dtype", "bfloat16", \
     "--kv-cache-dtype", "bfloat16", "--trust-remote-code", \
     "--max-model-len", "1048576", "--max-num-seqs", "192", \
     "--max-num-batched-tokens", "8192", "--block-size", "128", \
     "--gpu-memory-utilization", "0.92", "--enable-prefix-caching", \
     "--speculative-config", "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"enforce_eager\":true}", \
     "--additional-config", "{\"enable_engram\":true,\"enable_cpu_binding\":true,\"ascend_compilation_config\":{\"enable_npugraph_ex\":true,\"enable_static_kernel\":true},\"multistream_overlap_shared_expert\":true,\"multistream_dsv4_dsa_overlap\":true,\"enable_prefill_mc2\":false,\"enable_mc2_hierarchy_comm\":false,\"engram_storage\":\"int8\"}", \
     "--compilation-config", "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,3,4,6,12,18,24,30,36,42,48,60,72,96,120,144,192,240,288,336,384,432,480,576,672,768,864,960,1152]}", \
     "--model-loader-extra-config", "{\"enable_multithread_load\":true,\"num_threads\":128}", \
     "--safetensors-load-strategy", "lazy", \
     "--limit-mm-per-prompt", "{\"image\":4}", \
     "--tokenizer-mode=deepseek_v41", "--reasoning-parser=deepseek_v41", \
     "--tool-call-parser=deepseek_v41", "--enable-auto-tool-choice", \
     "--default-chat-template-kwargs", "{\"enable_thinking\":true}"]
