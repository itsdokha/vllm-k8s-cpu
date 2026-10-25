# Образ vLLM для инференса на CPU (x86_64, AVX2/AVX512).
# Сборка из корня репозитория:
#   docker build -f docker/vllm/Dockerfile -t vllm-cpu:0.32.0 .

ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY docker/vllm/requirements.txt /tmp/requirements.txt
# --require-hashes: ставится ровно то, что в requirements.txt, вплоть до байта.
# --no-deps: транзитивные зависимости тоже должны быть перечислены явно,
# иначе сборка упадёт, а не подтянет что-то новое.
RUN pip install --require-hashes --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt \
    && pip check


FROM ${PYTHON_IMAGE}

# libnuma1 нужен CPU-бэкенду vLLM для привязки потоков к ядрам,
# tcmalloc заметно ускоряет аллокации, g++ нужен torch.compile (inductor
# генерирует и компилирует C++ ядра для CPU прямо во время старта).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnuma1 \
        libtcmalloc-minimal4t64 \
        g++ \
    && rm -rf /var/lib/apt/lists/*

ARG UID=10001
RUN groupadd --gid ${UID} vllm \
    && useradd --uid ${UID} --gid ${UID} --create-home --home-dir /home/vllm --shell /usr/sbin/nologin vllm

COPY --from=builder /opt/venv /opt/venv
COPY privacy_logging /opt/app/privacy_logging
COPY docker/vllm/entrypoint.sh /opt/app/entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/app \
    PYTHONUNBUFFERED=1 \
    HOME=/home/vllm \
    TRITON_HOME=/home/vllm/.cache \
    TRITON_CACHE_DIR=/home/vllm/.cache/triton \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so \
    VLLM_CONFIGURE_LOGGING=1 \
    VLLM_LOGGING_CONFIG_PATH=/opt/app/privacy_logging/logging_config.json \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

USER ${UID}:${UID}
WORKDIR /home/vllm
EXPOSE 8000

ENTRYPOINT ["/opt/app/entrypoint.sh"]
