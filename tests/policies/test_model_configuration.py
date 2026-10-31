"""Load the default checkpoint's published model configuration."""

import pytest


def test_default_qwen_configuration_is_supported():
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    assert config.model_type == "qwen3_5"
    model_class = transformers.AutoModelForImageTextToText._model_mapping[type(config)]
    assert model_class.__name__ == "Qwen3_5ForConditionalGeneration"
