#!/usr/bin/env bash
# Distills every SFT dataset with Qwen3.8-27B and Ornith-1.5-9B, both unmodified and after CPT,
# serving one model at a time. See docs/dataset_generator.md. Linux-only, because vLLM is.
#
# Usage: bash run_distill.sh [workers]
#
# Rerun to resume: distill continues after the last row of each output file. If a model fails, the
# next one still runs and the script exits 1 at the end. vLLM's output goes to outputs/distill-logs.

# `sh run_distill.sh` ignores the shebang, and dash can't run this script.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

cd "$(dirname "$0")"

WORKERS=${1:-16}
PORT=8000
export VLLM_BASE_URL="http://localhost:${PORT}"

# distill counts the first attempt, so this is 5 retries.
MAX_ATTEMPTS=6
# Long enough to download the model on first use.
STARTUP_TIMEOUT_SECONDS=3600
SHUTDOWN_TIMEOUT_SECONDS=120
# Holds the reference answer plus the response. The longest row is about 10k characters. A request
# over this limit fails, which stops distill. Capping it keeps the KV cache small.
MAX_MODEL_LEN=8192
# vLLM's default of 1024 needs more Mamba cache blocks than Qwen3.8-27B leaves on one GPU (678), so
# it fails to start. Keep this at or above the worker count, or requests queue.
MAX_NUM_SEQS=64
# Limits thinking plus the response, and also the instruction plus both. Keep it equal to sft.seq_len
# in the -distilled experiments, or the trainer cuts off rows that passed.
MAX_LEN=4096
LOG_DIR=outputs/distill-logs

# distilled/ holds rewrites by the model after CPT, which the -distilled experiments read. Rewrites
# by the unmodified models don't match an SFT run resuming from CPT, so they're kept apart.
# Local paths are relative to serving/vllm, where vLLM runs.
MODELS=(
    "qwen3.8-27b unsloth/Qwen3.8-27B distilled-unmodified"
    "qwen3.8-27b ../../outputs/py3.15-Qwen3.8-27B/cpt/merged distilled"
    "ornith-1.5-9b ornith-ai/Ornith-1.5-9B distilled-unmodified"
    "ornith-1.5-9b ../../outputs/py3.15-Ornith-1.5-9B/cpt/merged distilled"
)
DATASETS=(
    python3.14/sft
    python3.14/sft-PEP750
    python3.15/sft
    python3.15/sft2
)

SERVER_PID=""

server_is_up() {
    curl --silent --fail --max-time 5 --output /dev/null "${VLLM_BASE_URL}/health"
}

start_server() {
    local repo=$1 log_path=$2

    # distill would otherwise talk to whichever model that server has loaded.
    if server_is_up; then
        echo "Another server is already listening on port ${PORT}" >&2
        return 1
    fi

    # In its own process group, so stop_server also reaches uv's child and vLLM's engine process.
    setsid uv run --directory serving/vllm vllm serve "$repo" \
        --served-model-name seed --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" --reasoning-parser qwen3 \
        >"$log_path" 2>&1 &
    SERVER_PID=$!

    local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    until server_is_up; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM exited during startup; see ${log_path}" >&2
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "vLLM wasn't ready after ${STARTUP_TIMEOUT_SECONDS}s; see ${log_path}" >&2
            return 1
        fi
        sleep 5
    done
}

# Returns once the whole process group has gone, so the next model gets a free GPU.
stop_server() {
    if [[ -z $SERVER_PID ]]; then
        return 0
    fi

    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    local deadline=$((SECONDS + SHUTDOWN_TIMEOUT_SECONDS))
    while kill -0 -- "-${SERVER_PID}" 2>/dev/null; do
        if ((SECONDS >= deadline)); then
            echo "vLLM ignored SIGTERM; killing it" >&2
            kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
            break
        fi
        sleep 1
    done
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

distill_model() {
    local name=$1 repo=$2 output_dir=$3 dataset

    start_server "$repo" "${LOG_DIR}/vllm-${output_dir}-${name}.log" || return 1

    for dataset in "${DATASETS[@]}"; do
        echo "=== ${output_dir}/${name}: ${dataset}"
        uv run --project corpus_builder satyrn-dataset distill \
            -i "datasets/${dataset}.jsonl" \
            -o "datasets/${dataset%%/*}/${output_dir}/${name}/${dataset##*/}.jsonl" \
            --thinking --max-tokens "$MAX_LEN" --max-seq-len "$MAX_LEN" \
            --workers "$WORKERS" --max-attempts "$MAX_ATTEMPTS" || return 1
    done
}

# The server is in its own session, so Ctrl-C doesn't reach it.
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$LOG_DIR"

failed=()
for model in "${MODELS[@]}"; do
    read -r name repo output_dir <<<"$model"
    echo "=== Serving ${repo}"
    distill_model "$name" "$repo" "$output_dir" || failed+=("${output_dir}/${name}")
    stop_server
done

if ((${#failed[@]} > 0)); then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
echo "All models distilled"
