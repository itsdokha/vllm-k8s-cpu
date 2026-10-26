"""Скачивает веса модели с Hugging Face в примонтированный volume.

Запускается как initContainer перед vLLM. Если в MODEL_DIR уже лежит
маркер .download-complete с тем же repo_id и revision, скрипт завершается
сразу и в сеть не ходит. Если ревизия в ConfigMap поменялась, докачиваются
только изменившиеся файлы.

Переменные окружения:
    MODEL_ID        репозиторий на HF, например Qwen/Qwen2.5-1.5B-Instruct
    MODEL_REVISION  коммит или тег (лучше коммит, чтобы веса не менялись сами)
    MODEL_DIR       куда класть файлы, например /models/qwen2.5-1.5b-instruct
    HF_TOKEN        токен HF, нужен только для gated/private моделей
    ALLOW_PATTERNS  какие файлы качать, через запятую
    PRUNE_OTHER_MODELS  true/false: удалить с volume другие модели, скачанные
                    этим же скриптом (каталоги с маркером), кроме текущей
    CACHE_DIR       необязательно: каталог под кэш torch.compile на том же
                    volume. Создаётся здесь, чтобы его владельцем был наш uid,
                    а не root (kubelet создаёт subPath от root).
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MARKER = ".download-complete"
DEFAULT_PATTERNS = "*.json,*.safetensors,*.txt,*.model,*.tiktoken,*.jinja"

log = logging.getLogger("model-downloader")


def read_marker(model_dir: Path) -> dict | None:
    try:
        return json.loads((model_dir / MARKER).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_marker(model_dir: Path, info: dict) -> None:
    # Пишем через временный файл, чтобы vLLM никогда не увидел
    # маркер от недокачанной модели.
    tmp = model_dir / f"{MARKER}.tmp"
    tmp.write_text(json.dumps(info, indent=2))
    tmp.replace(model_dir / MARKER)


def remove_partial_files(model_dir: Path) -> None:
    # Недокачанные файлы huggingface_hub после оборванных загрузок.
    for f in (model_dir / ".cache" / "huggingface").rglob("*.incomplete"):
        log.info("Removing partial file %s (%.0f MiB)", f.name, f.stat().st_size / 2**20)
        f.unlink()


def prune_other_models(model_dir: Path) -> None:
    # Трогаем только каталоги с нашим маркером: кэш компиляции и всё,
    # что положили на volume не мы, остаётся на месте.
    for other in model_dir.parent.iterdir():
        if other == model_dir or not (other / MARKER).is_file():
            continue
        log.info("Removing previous model %s", other)
        shutil.rmtree(other)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx на INFO пишет каждый запрос вместе с подписанными ссылками на CDN
    logging.getLogger("httpx").setLevel(logging.WARNING)

    model_id = os.environ["MODEL_ID"]
    revision = os.environ.get("MODEL_REVISION", "main")
    model_dir = Path(os.environ["MODEL_DIR"])
    patterns = [p.strip() for p in os.environ.get("ALLOW_PATTERNS", DEFAULT_PATTERNS).split(",") if p.strip()]
    token = os.environ.get("HF_TOKEN") or None

    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get("PRUNE_OTHER_MODELS", "false").lower() == "true":
        prune_other_models(model_dir)

    wanted = {"model_id": model_id, "revision": revision}
    marker = read_marker(model_dir)
    if marker and {k: marker.get(k) for k in wanted} == wanted:
        log.info("Model %s@%s is already in %s, skipping download", model_id, revision, model_dir)
        return 0

    log.info("Downloading %s@%s to %s (token: %s)", model_id, revision, model_dir, "yes" if token else "no")
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        allow_patterns=patterns,
        token=token,
    )

    remove_partial_files(model_dir)
    size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 2**20
    write_marker(model_dir, {**wanted, "downloaded_at": int(time.time())})
    log.info("Done in %.0fs, %.0f MiB on disk", time.monotonic() - started, size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
