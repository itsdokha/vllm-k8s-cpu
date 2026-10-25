#!/usr/bin/env bash
# Запускает OpenAI-совместимый сервер vLLM.
#
# Веса берутся только из локального каталога MODEL_DIR (их туда кладёт
# initContainer model-downloader), сеть для загрузки модели не используется.
# Параметры инференса читаются из SERVE_CONFIG_FILE (ConfigMap), токен
# доступа к API - из переменной VLLM_API_KEY (Secret), её vLLM читает сам.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?MODEL_DIR is not set}"
SERVE_CONFIG_FILE="${SERVE_CONFIG_FILE:-/etc/vllm/config.yaml}"

if [[ ! -f "${MODEL_DIR}/.download-complete" ]]; then
    echo "Model is not found in ${MODEL_DIR}. Run model-downloader first." >&2
    exit 1
fi

if [[ -z "${VLLM_API_KEY:-}" ]]; then
    echo "VLLM_API_KEY is empty, refusing to start an unauthenticated endpoint." >&2
    exit 1
fi

# Тела ответов не должны попадать в лог ни при каких настройках.
unset VLLM_DEBUG_LOG_API_SERVER_RESPONSE

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec vllm serve "${MODEL_DIR}" --config "${SERVE_CONFIG_FILE}" "$@"
