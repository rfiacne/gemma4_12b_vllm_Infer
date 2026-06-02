#!/bin/bash
# start_vllm_server.sh — 安装 vllm (清华源) & 后台启动 LFM2.5-8B-A1B-NVFP4 推理服务
set -e

cd "$(dirname "$0")"

# ── 配置 ──────────────────────────────────────────────
MODEL_PATH="models/LFM2.5-8B-A1B-NVFP4"
SERVED_NAME="lfm25-8b-a1b"
PORT="${1:-1234}"                          # 默认 1234，可传参覆盖
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOG_FILE="vllm_server.log"

VENV_DIR=".venv_vllm"

# ── 创建/激活虚拟环境 ─────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "=== 创建虚拟环境 ${VENV_DIR} ==="
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# ── 安装 vllm (清华源) ──────────────────────────────
echo "=== 检查 vllm 是否已安装 ==="
if ! python -c "import vllm" 2>/dev/null; then
    echo "vllm 未安装，从清华源安装 (可能要几分钟)..."
    pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "vllm 安装完成。"
else
    echo "vllm 已安装，跳过。"
fi

# ── 修复: 确保 venv bin 在 PATH (ninja 等) ──────────────
export PATH="${VENV_DIR}/bin:${PATH}"
# 修复: FlashInfer MoE FP4 后端可能崩溃，用 VLLM_CUTLASS
export VLLM_USE_FLASHINFER_MOE_FP4=0
# 清理旧的 FlashInfer 编译缓存 (避免 ninja 编译失败)
rm -rf ~/.cache/flashinfer ~/.cache/vllm/torch_compile_cache

# ── 启动推理服务 (后台) ──────────────────────────────
echo ""
echo "=== 启动 vLLM 推理服务 (后台) ==="
echo "  Model:       ${MODEL_PATH}"
echo "  Served name: ${SERVED_NAME}"
echo "  Port:        ${PORT}"
echo "  GPU:         ${GPU}"
echo "  Log:         ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup "${VENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --quantization modelopt \
    --kv-cache-dtype fp8 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser lfm2 \
    --port "${PORT}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "  后台 PID: ${PID}"
echo "${PID}" > vllm_server.pid

# ── 等待就绪 ──────────────────────────────────────────
echo ""
echo "=== 等待服务就绪 (最多 5 分钟) ==="
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" "localhost:${PORT}/v1/models" 2>/dev/null | grep -q 200; then
        echo ""
        echo "✅ 服务已就绪！"
        echo "   Models endpoint: http://localhost:${PORT}/v1/models"
        echo "   Chat completions: http://localhost:${PORT}/v1/chat/completions"
        echo ""
        echo "   日志:  tail -f ${LOG_FILE}"
        echo "   停止:  kill \$(cat vllm_server.pid)"
        exit 0
    fi
    printf "."
    sleep 10
done

echo ""
echo "⚠️  超时 — 服务可能仍在加载中，查看日志: tail -f ${LOG_FILE}"
