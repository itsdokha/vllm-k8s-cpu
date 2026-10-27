#!/usr/bin/env bash
# Прогон vllm bench serve по сценариям "короткий/длинный промпт" x "1/8 параллельных клиентов".
# Запускается Job-ом (k8s/bench/job.yaml). Локально тоже работает, если стоит vllm и сделан port-forward.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-1.5b-instruct-w8a8}"
TOKENIZER_DIR="${TOKENIZER_DIR:?path to tokenizer is required}"
RESULTS="${RESULTS:-/results}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

# name input_len concurrency num_prompts
# Запросов немного: на CPU один ответ в 128 токенов занимает десятки секунд,
# а для p50/p95 и throughput этого достаточно.
SCENARIOS=(
  "short-c1  128  1  5"
  "short-c8  128  8 16"
  "long-c1  3072  1  3"
  "long-c8  3072  8  8"
)

for s in "${SCENARIOS[@]}"; do
  read -r name input_len conc num <<<"$s"
  echo ">>> ${name}: input=${input_len} output=${OUTPUT_LEN} concurrency=${conc} prompts=${num}"
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_NAME" \
    --tokenizer "$TOKENIZER_DIR" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$OUTPUT_LEN" \
    --ignore-eos \
    --num-prompts "$num" \
    --max-concurrency "$conc" \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,95 \
    --seed 42 \
    --save-result --result-dir "$RESULTS" --result-filename "${name}.json"
done

python3 "$(dirname "$0")/summarize.py" "$RESULTS"
