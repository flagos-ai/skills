#!/bin/bash
# E2E: Manage upstream vLLM server on GT machine via SSH -> docker exec -> conda run
# Usage:
#   bash e2e_remote_serve.sh start <config.json> <MODEL_DISPLAY_NAME>
#   bash e2e_remote_serve.sh stop  <config.json>
#   bash e2e_remote_serve.sh status <config.json>
set -euo pipefail

ACTION="${1:?Usage: e2e_remote_serve.sh <start|stop|status> <config.json> [MODEL_DISPLAY_NAME]}"
CONFIG="${2:?Usage: e2e_remote_serve.sh <start|stop|status> <config.json> [MODEL_DISPLAY_NAME]}"

# Parse config with python (no jq dependency)
eval "$(python3 -c "
import json
cfg = json.load(open('$CONFIG'))
gt = cfg['gt_machine']
ss = cfg['shared_storage']
env_vars = gt.get('env_vars', {})
env_str = ' '.join(f'{k}={v}' for k, v in env_vars.items())
print(f'GT_HOST={gt[\"host\"]}')
print(f'GT_USER={gt.get(\"ssh_user\", \"root\")}')
print(f'GT_SSH_PORT={gt.get(\"ssh_port\", 22)}')
print(f'GT_SSH_KEY={gt.get(\"ssh_key\", \"~/.ssh/id_ed25519\")}')
print(f'GT_CONTAINER={gt.get(\"docker_container\", \"\")}')
print(f'GT_CONDA_ENV={gt.get(\"conda_env\", \"\")}')
print(f'GT_VLLM_PORT={gt.get(\"vllm_port\", 8122)}')
print(f'GT_GPU_IDS={gt.get(\"gpu_ids\", \"0,1,2,3,4,5,6,7\")}')
print(f'GT_TP={gt.get(\"tensor_parallel_size\", 8)}')
print(f'GT_EXTRA_ARGS=\"{gt.get(\"extra_serve_args\", \"--trust-remote-code\")}\"')
print(f'MODELS_DIR={ss.get(\"models_dir\", \"/models\")}')
print(f'GT_ENV_VARS=\"{env_str}\"')
")"

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${GT_SSH_PORT} -i ${GT_SSH_KEY} ${GT_USER}@${GT_HOST}"

# Build the command prefix for running inside docker + conda
# Result: docker exec <container> bash -c 'conda run -n <env> bash -c "..."'
build_remote_cmd() {
    local inner_cmd="$1"
    local cmd=""
    if [ -n "${GT_CONDA_ENV}" ]; then
        inner_cmd="/root/miniconda3/bin/conda run --no-capture-output -n ${GT_CONDA_ENV} bash -c \"${inner_cmd}\""
    fi
    if [ -n "${GT_CONTAINER}" ]; then
        cmd="docker exec ${GT_CONTAINER} bash -c '${inner_cmd}'"
    else
        cmd="${inner_cmd}"
    fi
    echo "${cmd}"
}

case "$ACTION" in
    start)
        MODEL="${3:?Usage: e2e_remote_serve.sh start <config.json> <MODEL_DISPLAY_NAME>}"
        echo ">>> Starting GT server on ${GT_HOST}:${GT_VLLM_PORT} for model ${MODEL}..."

        # Kill any existing vllm process on GT
        echo "  Killing existing vLLM processes..."
        KILL_CMD=$(build_remote_cmd "pkill -f vllm.entrypoints || true")
        ${SSH_CMD} "${KILL_CMD}" 2>/dev/null || true
        sleep 3

        # Build the vllm serve command
        SERVE_INNER="CUDA_VISIBLE_DEVICES=${GT_GPU_IDS} ${GT_ENV_VARS} vllm serve ${MODELS_DIR}/${MODEL} --host 0.0.0.0 --port ${GT_VLLM_PORT} --tensor-parallel-size ${GT_TP} ${GT_EXTRA_ARGS}"

        # Start server in background: nohup docker exec ... &
        echo "  Starting upstream vLLM server..."
        echo "  Command: ${SERVE_INNER}"
        if [ -n "${GT_CONTAINER}" ]; then
            if [ -n "${GT_CONDA_ENV}" ]; then
                START_CMD="nohup docker exec ${GT_CONTAINER} /root/miniconda3/bin/conda run --no-capture-output -n ${GT_CONDA_ENV} bash -c '${SERVE_INNER}' > /tmp/vllm_gt_server.log 2>&1 &"
            else
                START_CMD="nohup docker exec ${GT_CONTAINER} bash -c '${SERVE_INNER}' > /tmp/vllm_gt_server.log 2>&1 &"
            fi
        else
            START_CMD="nohup bash -c '${SERVE_INNER}' > /tmp/vllm_gt_server.log 2>&1 &"
        fi
        ${SSH_CMD} "${START_CMD}"

        # Wait for health check
        echo "  Waiting for GT server health check..."
        MAX_WAIT=600
        ELAPSED=0
        while [ $ELAPSED -lt $MAX_WAIT ]; do
            if curl -s --connect-timeout 5 "http://${GT_HOST}:${GT_VLLM_PORT}/health" > /dev/null 2>&1; then
                echo "  GT server is ready at ${GT_HOST}:${GT_VLLM_PORT}"
                exit 0
            fi
            echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
            sleep 15
            ELAPSED=$((ELAPSED + 15))
        done
        echo "  GT server failed to start within ${MAX_WAIT}s"
        echo "  Check logs: ssh ${GT_USER}@${GT_HOST} cat /tmp/vllm_gt_server.log"
        exit 1
        ;;

    stop)
        echo ">>> Stopping GT server on ${GT_HOST}..."
        KILL_CMD=$(build_remote_cmd "pkill -f vllm.entrypoints || true")
        ${SSH_CMD} "${KILL_CMD}" 2>/dev/null || true
        echo "  GT server stopped"
        ;;

    status)
        echo ">>> Checking GT server status at ${GT_HOST}:${GT_VLLM_PORT}..."
        if curl -s --connect-timeout 5 "http://${GT_HOST}:${GT_VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "  GT server is running"
            exit 0
        else
            echo "  GT server is not reachable"
            exit 1
        fi
        ;;

    logs)
        echo ">>> Fetching GT server logs..."
        ${SSH_CMD} "tail -50 /tmp/vllm_gt_server.log 2>/dev/null || echo 'No log file found'"
        ;;

    *)
        echo "Unknown action: $ACTION"
        echo "Usage: e2e_remote_serve.sh <start|stop|status|logs> <config.json> [MODEL_DISPLAY_NAME]"
        exit 1
        ;;
esac
