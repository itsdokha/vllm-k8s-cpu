"""Фильтр логов vLLM, который не пропускает в stdout/stderr промпты и ответы.

В vLLM 0.32 пользовательские данные попадают в лог в трёх местах:

* ``vllm.entrypoints.serve.utils.request_logger`` (RequestLogger):
  промпт и тело запроса на DEBUG, текст ответа на INFO при
  ``--enable-log-outputs``, SamplingParams на INFO (в них бывают
  stop-строки, json-схемы для structured output, logit_bias);
* ``vllm.entrypoints.serve.middleware.log_response``: тело HTTP-ответа,
  включается через VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true;
* ``vllm.logging_utils.dump_input``: дамп scheduler output при падении
  движка, в нём prompt_token_ids, по которым промпт восстанавливается
  токенизатором.

Фильтр вешается на handler, а не на логгер, поэтому через него проходят
записи всех логгеров и всех процессов vLLM (API-сервер и EngineCore
поднимают логирование из одного и того же VLLM_LOGGING_CONFIG_PATH).

Для этих трёх логгеров работает белый список: известные сообщения
переписываются так, что остаются только метаданные, всё остальное
отбрасывается. Если в следующей версии vLLM поменяют текст сообщения,
оно пропадёт из лога, но не утечёт.
"""

from __future__ import annotations

import logging

REQUEST_LOGGER = "vllm.entrypoints.serve.utils.request_logger"
RESPONSE_LOGGER = "vllm.entrypoints.serve.middleware.log_response"
DUMP_LOGGER = "vllm.logging_utils.dump_input"

# Шаблоны сообщений из vllm/entrypoints/serve/utils/request_logger.py
RECEIVED_TEMPLATE = "Received request %s: params: %s, lora_request: %s."
GENERATED_TEMPLATE = "Generated response %s%s: output: %r, finish_reason: %s"
DUMP_HEADER_PREFIX = "Dumping input data for V1 LLM engine"

# Поля SamplingParams, в которых не бывает пользовательского текста.
SAFE_PARAM_FIELDS = ("n", "max_tokens", "min_tokens", "temperature", "top_p", "top_k")


def summarize_params(params: object) -> str:
    if params is None:
        return "-"
    parts = []
    for field in SAFE_PARAM_FIELDS:
        value = getattr(params, field, None)
        if value is not None:
            parts.append(f"{field}={value}")
    return ", ".join(parts) if parts else type(params).__name__


class PromptRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == REQUEST_LOGGER:
            return self._request_logger(record)
        if record.name == RESPONSE_LOGGER:
            return False
        if record.name == DUMP_LOGGER:
            # Первое сообщение содержит только конфиг движка, его оставляем,
            # чтобы было понятно, что движок упал. Сам дамп выкидываем.
            return isinstance(record.msg, str) and record.msg.startswith(DUMP_HEADER_PREFIX)
        return True

    def _request_logger(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()

        if record.msg == RECEIVED_TEMPLATE and len(args) == 3:
            request_id, params, lora_request = args
            record.msg = "Received request %s: %s, lora: %s"
            record.args = (request_id, summarize_params(params), lora_request is not None)
            return True

        if record.msg == GENERATED_TEMPLATE and len(args) == 4:
            request_id, stream_info, output, finish_reason = args
            record.msg = "Generated response %s%s: output_chars=%d, finish_reason: %s"
            record.args = (request_id, stream_info, len(output or ""), finish_reason)
            return True

        # Предупреждения об уровне логирования из RequestLogger.__init__
        # (вызываются без аргументов, данных запроса в них нет).
        if not args and isinstance(record.msg, str) and record.msg.startswith("`--enable-log-requests`"):
            return True

        return False
