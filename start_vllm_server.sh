#!/bin/bash
# start_vllm_server.sh — Docker 启动 Gemma-4-12B-it 推理服务
set -e

cd "$(dirname "$0")"

# ── 配置 ──────────────────────────────────────────────
MODEL_DIR="./models/gemma-4-12B-it-assistant"      # 本地模型目录（可仍在下载中）
MODEL_CONTAINER="/models/gemma-4-12B-it-assistant" # 容器内绝对路径
SERVED_NAME="gemma-4-12B-it-qat-w4a16"
PORT="${1:-1234}" # 默认 1234，可传参覆盖
CONTAINER_NAME="vllm-gemma4-server"
LOG_FILE="vllm_server.log"

DOCKER_IMAGE="docker.xuanyuan.run/vllm/vllm-openai:gemma4-unified-x86_64-cu130"

# ── 前置检查 ──────────────────────────────────────────
echo "=== 检查 Docker 是否可用 ==="
if ! command -v docker &>/dev/null; then
	echo "❌ 未找到 docker，请先安装 Docker。"
	exit 1
fi

echo "=== 检查模型目录 ${MODEL_DIR} ==="
if [ ! -d "${MODEL_DIR}" ]; then
	echo "⚠️  模型目录 ${MODEL_DIR} 不存在，服务启动会等待模型文件。"
else
	echo "✅ 模型目录存在，内容："
	ls -lh "${MODEL_DIR}" | head -20
fi

# ── 停止已有容器 ──────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
	echo ""
	echo "=== 停止已有容器 ${CONTAINER_NAME} ==="
	docker stop "${CONTAINER_NAME}" 2>/dev/null || true
	docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# ── 拉取镜像（如本地不存在）────────────────────────────
if ! docker image inspect "${DOCKER_IMAGE}" &>/dev/null; then
	echo ""
	echo "=== 拉取 Docker 镜像（首次可能较慢）==="
	echo "  Image: ${DOCKER_IMAGE}"
	docker pull "${DOCKER_IMAGE}"
	echo "✅ 镜像拉取完成。"
else
	echo ""
	echo "✅ 镜像 ${DOCKER_IMAGE} 已存在，跳过拉取。"
fi

# ── 启动容器 ──────────────────────────────────────────
echo ""
echo "=== 启动 vLLM Docker 容器 ==="
echo "  Model dir:   ${MODEL_DIR} -> ${MODEL_CONTAINER} (容器内)"
echo "  Served name: ${SERVED_NAME}"
echo "  Port:        ${PORT} -> 8000 (容器内)"
echo "  Log:         ${LOG_FILE}"
echo ""

docker run -d \
	--name "${CONTAINER_NAME}" \
	--entrypoint /bin/bash \
	--gpus all \
	--privileged \
	--ipc=host \
	-p "${PORT}:8000" \
	-v "${PWD}/${MODEL_DIR}:${MODEL_CONTAINER}:ro" \
	-v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
	-v "${PWD}/docker_entrypoint.sh:/docker_entrypoint.sh:ro" \
	"${DOCKER_IMAGE}" \
	-c "/docker_entrypoint.sh ${MODEL_CONTAINER} --tensor-parallel-size 1  --gpu-memory-utilization 0.95 --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja --reasoning-parser gemma4" \
	>"${LOG_FILE}" 2>&1

echo "✅ 容器已启动。"
echo "   查看日志:  docker logs -f ${CONTAINER_NAME}"
echo "   查看日志:  tail -f ${LOG_FILE}"
echo "   停止服务:  docker stop ${CONTAINER_NAME}"
echo "   删除容器:  docker rm   ${CONTAINER_NAME}"

# ── 等待就绪 ──────────────────────────────────────────
echo ""
echo "=== 等待服务就绪 (最多 10 分钟) ==="
for _ in $(seq 1 60); do
	if curl -s -o /dev/null -w "%{http_code}" "localhost:${PORT}/v1/models" 2>/dev/null | grep -q 200; then
		echo ""
		echo "✅ 服务已就绪！"
		echo "   Models endpoint:      http://localhost:${PORT}/v1/models"
		echo "   Chat completions:     http://localhost:${PORT}/v1/chat/completions"
		echo "   日志:                 docker logs -f ${CONTAINER_NAME}"
		echo "   停止服务:             docker stop ${CONTAINER_NAME}"
		exit 0
	fi
	printf "."
	sleep 10
done

echo ""
echo "⚠️  超时 — 模型可能仍在加载或下载中。"
echo "   查看日志:  docker logs -f ${CONTAINER_NAME}"
echo "   容器状态:  docker ps -a --filter name=${CONTAINER_NAME}"
