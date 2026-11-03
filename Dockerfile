# syntax=docker/dockerfile:1
#
# deepseek-v4.1-flash-ascend910B4
#
# 在上游部署镜像之上，把本项目的 7 个补丁烘到正式导入路径，并把验证过的
# vLLM 命令行固定成 CMD（原生 vLLM 镜像风格：运行时只需挂载模型）。
#
#   docker build -t deepseek-v4.1-flash-ascend910B4:910b4 .
#   docker build --build-arg BASE_IMAGE=<你的上游镜像> -t ... .
#
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
FROM ${BASE_IMAGE}
# 重新声明，供下面的 LABEL 使用（FROM 之前的 ARG 在 stage 内不可见）
ARG BASE_IMAGE

# ---- 补丁：整文件覆盖到正式导入路径 ----
COPY patches/files/model_v41.py          /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py
COPY patches/files/v41_yoco.py           /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/v41_yoco.py
COPY patches/files/device_metadata.py    /vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py
COPY patches/files/dsa_v41_kvstore.py    /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py
COPY patches/files/request_queue.py      /vllm-workspace/vllm/vllm/v1/core/sched/request_queue.py
COPY patches/files/scheduler.py          /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py
COPY patches/files/eplb_utils.py         /vllm-workspace/vllm-ascend/vllm_ascend/eplb/core/eplb_utils.py

# ---- 默认行为：快速 prefill 档（近似）；精确档用 -e 覆盖 ----
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
