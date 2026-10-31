"""Training retains the exact causal context of each generated action."""

import pytest

from synthstats.integrations.harbor.generator import extract_generation_data_from_atif


def trace():
    return {
        "schema_version": "ATIF-v1.6",
        "steps": [
            {
                "source": "agent",
                "metrics": {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4],
                    "logprobs": [-0.1, -0.2],
                },
            },
            {
                "source": "agent",
                "metrics": {
                    "prompt_token_ids": [1, 2, 3, 4, 5, 6],
                    "completion_token_ids": [7],
                    "logprobs": [-0.3],
                },
            },
        ],
    }


def test_observations_remain_in_causal_sequence_with_zero_loss_mask():
    result = extract_generation_data_from_atif(trace())
    assert result["prompt_ids"] == [1, 2]
    assert result["response_ids"] == [3, 4, 5, 6, 7]
    assert result["loss_mask"] == [1, 1, 0, 0, 1]
    assert result["logprobs"] == [-0.1, -0.2, 0.0, 0.0, -0.3]


@pytest.mark.parametrize("field", ["prompt_token_ids", "logprobs"])
def test_insufficient_training_trace_is_rejected(field):
    atif = trace()
    del atif["steps"][0]["metrics"][field]
    with pytest.raises(ValueError):
        extract_generation_data_from_atif(atif)


def test_rewritten_causal_history_is_rejected():
    atif = trace()
    atif["steps"][1]["metrics"]["prompt_token_ids"] = [9, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="prefix"):
        extract_generation_data_from_atif(atif)


@pytest.mark.parametrize(
    "bad_metadata",
    [
        {"token_ids": [3], "mask_ids": [1], "logprobs": [-0.1]},
        {"prompt_ids": [], "token_ids": [3], "mask_ids": [1], "logprobs": [-0.1]},
        {"prompt_ids": [1], "token_ids": [-7], "mask_ids": [1], "logprobs": [-0.1]},
        {"prompt_ids": [1], "token_ids": [3], "mask_ids": [1], "logprobs": [float("nan")]},
        {"prompt_ids": [1], "token_ids": [3], "mask_ids": [-2], "logprobs": [-0.1]},
    ],
)
def test_agent_metadata_route_enforces_strict_causal_validation(bad_metadata):
    from synthstats.integrations.harbor.generator import (
        extract_generation_data_from_agent_metadata,
    )

    with pytest.raises(ValueError):
        extract_generation_data_from_agent_metadata(bad_metadata)


def test_copied_context_agent_step_receives_zero_loss_mask():
    atif = trace()
    atif["steps"][0]["is_copied_context"] = True
    result = extract_generation_data_from_atif(atif)
    assert result is not None
    assert result["prompt_ids"] == [1, 2]
    assert result["response_ids"] == [3, 4, 5, 6, 7]
    assert result["loss_mask"] == [0, 0, 0, 0, 1]
