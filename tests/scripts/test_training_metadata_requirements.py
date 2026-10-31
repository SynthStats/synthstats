from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from conftest import load_script_module


@pytest.mark.parametrize("with_logprobs", [True, False])
def test_api_generation_rejects_missing_native_token_metadata(monkeypatch, with_logprobs):
    token = SimpleNamespace(token="model", logprob=-0.25)
    choice = SimpleNamespace(
        message=SimpleNamespace(content="model"),
        logprobs=SimpleNamespace(content=[token]) if with_logprobs else None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[choice]))
        )
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    script = load_script_module("generate_pilot_programs")
    with pytest.raises(ValueError, match="token|logprob"):
        script.generate_programs_api(
            "test-model",
            "system",
            "user",
            generation=script.GenerationSettings(
                n=1, temperature=0.8, top_p=0.95, max_new_tokens=128
            ),
            api_base="https://example.invalid/v1",
            api_key="test-key",
        )


def test_sft_conversion_requires_a_tokenizer():
    script = load_script_module("sft_to_atif")
    with pytest.raises(ValueError, match="tokenizer|token"):
        script.build_atif_record(
            0,
            "Write a model.",
            '<submit_program language="haskell">module Submission where</submit_program>',
            tokenizer=None,
        )


def test_lazyppl_executor_requires_a_configured_library(monkeypatch):
    from synthstats.executors.lazyppl_sandbox import LazyPPLExecutor

    monkeypatch.delenv("LAZYPPL_DEV_DIR", raising=False)
    with pytest.raises(ValueError, match="LAZYPPL_DEV_DIR|lazyppl_dir"):
        LazyPPLExecutor()
