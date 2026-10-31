"""Stop behavior through HFPolicy and Transformers' real generation loop."""

from functools import partial

import pytest

from synthstats.core.policy import GenConfig
from synthstats.core.types import Message
from synthstats.policies.hf_policy import HFPolicy


@pytest.fixture
def local_policy(tmp_path):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: i for i, token in enumerate(["<pad>", "<eos>", *alphabet, "OP"])}
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[("O", "P")]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>"
    )
    tokenizer.chat_template = '{{ messages[0]["content"] }}'
    tokenizer.save_pretrained(tmp_path)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(tokenizer),
            n_layer=1,
            n_head=1,
            n_embd=8,
            bos_token_id=None,
            eos_token_id=[tokenizer.eos_token_id, vocabulary["!"]],
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model.save_pretrained(tmp_path)
    return HFPolicy(str(tmp_path))


def generate_text(policy, text, *, stops):
    """Constrain token choices while keeping model execution and stopping real."""
    tokens = policy.tokenizer.encode(text, add_special_tokens=False)
    policy.model.generate = partial(
        policy.model.generate,
        prefix_allowed_tokens_fn=lambda _batch, ids: [tokens[len(ids) - 1]],
    )
    return policy.generate(
        [Message(role="user", content="Q")],
        gen=GenConfig(max_tokens=len(tokens), temperature=0, stop_sequences=stops),
    )


def test_multi_token_stop_requires_the_complete_string(local_policy):
    result = generate_text(local_policy, "STOP tail", stops=["STOP"])

    assert result.text == "STOP"
    assert len(result.token_ids) == 3
    assert len(result.token_logprobs) == len(result.eos_logprobs) == 3
    assert result.finish_reason == "stop"


def test_partial_stop_prefix_does_not_stop_generation(local_policy):
    result = generate_text(local_policy, "ST other", stops=["STOP"])

    assert result.text == "ST other"
    assert result.finish_reason == "length"


@pytest.mark.parametrize("eos", ["<eos>", "!"])
@pytest.mark.parametrize("stops", [[], ["STOP"]])
def test_model_eos_tokens_still_stop_generation(local_policy, eos, stops):
    result = generate_text(local_policy, eos + " tail", stops=stops)

    assert len(result.token_ids) == 1
    assert result.finish_reason == "stop"


def test_single_token_stop_still_stops_generation(local_policy):
    result = generate_text(local_policy, "x tail", stops=["x"])

    assert result.text == "x"
    assert len(result.token_ids) == 1
    assert result.finish_reason == "stop"


def test_stop_inside_the_last_token_reports_stop(local_policy):
    result = generate_text(local_policy, "STOP tail", stops=["STO"])

    assert result.text == "STOP"
    assert len(result.token_ids) == 3
    assert result.finish_reason == "stop"
