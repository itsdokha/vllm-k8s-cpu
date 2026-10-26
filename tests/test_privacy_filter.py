import logging
from types import SimpleNamespace

import pytest

from privacy_logging.filters import (
    DUMP_LOGGER,
    GENERATED_TEMPLATE,
    RECEIVED_TEMPLATE,
    REQUEST_LOGGER,
    RESPONSE_LOGGER,
    PromptRedactionFilter,
)

SECRET = "мой пароль от почты qwerty123"


def make_record(name, msg, args=(), level=logging.INFO):
    return logging.LogRecord(name, level, __file__, 1, msg, args, None)


@pytest.fixture
def flt():
    return PromptRedactionFilter()


def test_received_request_keeps_only_safe_params(flt):
    params = SimpleNamespace(max_tokens=64, temperature=0.7, stop=[SECRET], n=1)
    record = make_record(REQUEST_LOGGER, RECEIVED_TEMPLATE, ("chatcmpl-1", params, None))

    assert flt.filter(record)
    text = record.getMessage()
    assert "chatcmpl-1" in text
    assert "max_tokens=64" in text
    assert SECRET not in text


def test_generated_response_has_no_text(flt):
    record = make_record(REQUEST_LOGGER, GENERATED_TEMPLATE, ("chatcmpl-1", "", SECRET, "stop"))

    assert flt.filter(record)
    text = record.getMessage()
    assert SECRET not in text
    assert f"output_chars={len(SECRET)}" in text
    assert "finish_reason: stop" in text


@pytest.mark.parametrize(
    "msg, args",
    [
        ("Request %s details: prompt: %r, prompt_token_ids: %s, prompt_embeds shape: %s.",
         ("r1", SECRET, [1, 2, 3], None)),
        ("Request %s JSON body: %s", ("r1", '{"messages": "%s"}' % SECRET)),
        ("Generated response %s%s details: output_token_ids: %s", ("r1", "", [1, 2])),
        ("Some new message in future vLLM: %s", (SECRET,)),
    ],
)
def test_other_request_logger_messages_are_dropped(flt, msg, args):
    assert not flt.filter(make_record(REQUEST_LOGGER, msg, args, logging.DEBUG))


def test_response_body_middleware_is_dropped(flt):
    record = make_record(RESPONSE_LOGGER, "response_body={%s}", ('{"content": "%s"}' % SECRET,))
    assert not flt.filter(record)


def test_engine_dump_keeps_header_only(flt):
    header = make_record(DUMP_LOGGER, "Dumping input data for V1 LLM engine (v%s) with config: %s, ", ("0.32.0", "cfg"))
    dump = make_record(DUMP_LOGGER, "Dumping scheduler output for model execution: %s", ("prompt_token_ids=[1, 2]",))

    assert flt.filter(header)
    assert not flt.filter(dump)


def test_unrelated_records_pass_unchanged(flt):
    record = make_record("vllm.engine", "Avg prompt throughput: %.1f tokens/s", (12.5,))
    assert flt.filter(record)
    assert record.getMessage() == "Avg prompt throughput: 12.5 tokens/s"
