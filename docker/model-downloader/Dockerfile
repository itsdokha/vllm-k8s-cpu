# Лёгкий rootless-образ для initContainer, который скачивает веса модели.
# Сборка из корня репозитория:
#   docker build -f docker/model-downloader/Dockerfile -t model-downloader:0.1.0 .

ARG PYTHON_IMAGE=python:3.12-alpine3.22@sha256:a190708a2dec1bd18b1decb539f8e8f5407abaa9bf39cacda583f7f8c11db322

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/model-downloader/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes --no-deps --only-binary :all: -r /tmp/requirements.txt \
    && /opt/venv/bin/pip check


FROM ${PYTHON_IMAGE}

ARG UID=10001
RUN addgroup -g ${UID} app && adduser -D -u ${UID} -G app -h /home/app app

COPY --from=builder /opt/venv /opt/venv
COPY docker/model-downloader/download_model.py /opt/app/download_model.py

# HF_HOME в /tmp: корневая ФС в поде только для чтения.
# HF_HUB_DISABLE_XET: xet-загрузка на медленной сети minikube зависала без
# ошибки, обычная HTTP-загрузка надёжно докачивается после обрыва.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    HF_HOME=/tmp/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    HF_HUB_DISABLE_XET=1

USER ${UID}:${UID}
ENTRYPOINT ["python", "/opt/app/download_model.py"]
