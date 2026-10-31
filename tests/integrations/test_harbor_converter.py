"""Tests for Harbor ATIF -> SynthStats type converter."""

from __future__ import annotations

import pytest

from synthstats.core.types import Reward
from synthstats.integrations.harbor.converter import (
    atif_to_messages,
    atif_to_trajectory,
    build_loss_mask,
    extract_agent_generations,
    extract_program,
    read_reward_sidecar,
    validate_atif_schema,
)

# -- fixtures ----------------------------------------------------------------


def _make_step(
    source: str = "agent",
    message: str | list[dict[str, str]] = "hello",
    *,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
    observation: dict | None = None,
    completion_token_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
) -> dict:
    step: dict = {"step_id": 1, "source": source, "message": message}
    if reasoning_content is not None:
        step["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        step["tool_calls"] = tool_calls
    if observation is not None:
        step["observation"] = observation
    metrics: dict = {}
    if completion_token_ids is not None:
        metrics["completion_token_ids"] = completion_token_ids
    if logprobs is not None:
        metrics["logprobs"] = logprobs
    if metrics:
        step["metrics"] = metrics
    return step


def _make_atif(*steps: dict, version: str = "ATIF-v1.5") -> dict:
    return {"schema_version": version, "steps": list(steps)}


DUMMY_REWARD = Reward(total=0.5, components={"test": 0.5}, info={})


# -- validate_atif_schema ---------------------------------------------------


class TestValidateSchema:
    def test_valid_v15(self):
        ok, err = validate_atif_schema(_make_atif(_make_step()))
        assert ok
        assert err == ""

    def test_valid_v16(self):
        ok, _ = validate_atif_schema(_make_atif(_make_step(), version="ATIF-v1.6"))
        assert ok

    def test_unsupported_version(self):
        ok, err = validate_atif_schema(_make_atif(version="ATIF-v2.0"))
        assert not ok
        assert "unsupported" in err

    def test_missing_steps(self):
        ok, err = validate_atif_schema({"schema_version": "ATIF-v1.5"})
        assert not ok
        assert "steps" in err


# -- atif_to_messages --------------------------------------------------------


class TestAtifToMessages:
    def test_single_agent_step(self):
        atif = _make_atif(_make_step("agent", "code here"))
        msgs = atif_to_messages(atif)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "code here"

    def test_source_mapping(self):
        atif = _make_atif(
            _make_step("system", "sys prompt"),
            _make_step("user", "user msg"),
            _make_step("agent", "agent msg"),
        )
        msgs = atif_to_messages(atif)
        assert [m.role for m in msgs] == ["system", "user", "assistant"]

    def test_reasoning_content_wrapped(self):
        atif = _make_atif(_make_step("agent", "output", reasoning_content="thinking..."))
        msgs = atif_to_messages(atif)
        assert msgs[0].content == "output"
        assert msgs[0].reasoning_content == "thinking..."

    def test_tool_calls_flattened(self):
        atif = _make_atif(
            _make_step(
                "agent",
                "invoking tool",
                tool_calls=[
                    {
                        "tool_call_id": "tc1",
                        "function_name": "query",
                        "arguments": {"q": "age=5"},
                    }
                ],
            )
        )
        msgs = atif_to_messages(atif)
        assert len(msgs) == 2
        assert msgs[0].role == "assistant"
        assert msgs[1].role == "assistant"
        assert msgs[1].tool_call_id == "tc1"
        assert "query" in msgs[1].content

    def test_observations_flattened(self):
        atif = _make_atif(
            _make_step(
                "agent",
                "response",
                observation={"results": [{"source_call_id": "tc1", "content": "observation text"}]},
            )
        )
        msgs = atif_to_messages(atif)
        assert len(msgs) == 2
        assert msgs[1].role == "tool"
        assert msgs[1].content == "observation text"
        assert msgs[1].tool_call_id == "tc1"

    def test_multimodal_content_v16(self):
        atif = _make_atif(
            _make_step(
                "user",
                [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ],
            ),
            version="ATIF-v1.6",
        )
        msgs = atif_to_messages(atif)
        assert msgs[0].content == "part1 part2"

    def test_empty_steps(self):
        atif = _make_atif()
        msgs = atif_to_messages(atif)
        assert msgs == []


# -- atif_to_trajectory ------------------------------------------------------


class TestAtifToTrajectory:
    def test_token_data_extracted(self):
        atif = _make_atif(
            _make_step("system", "sys"),
            _make_step(
                "agent",
                "code",
                completion_token_ids=[10, 20, 30],
                logprobs=[-0.1, -0.2, -0.3],
            ),
        )
        traj = atif_to_trajectory(atif, DUMMY_REWARD)
        assert traj.token_ids == [[10, 20, 30]]
        assert traj.token_logprobs == [[-0.1, -0.2, -0.3]]
        assert traj.reward.total == 0.5
        assert len(traj.messages) == 2

    def test_multiple_generations(self):
        atif = _make_atif(
            _make_step(
                "agent",
                "gen1",
                completion_token_ids=[1, 2],
                logprobs=[-0.1, -0.2],
            ),
            _make_step("user", "feedback"),
            _make_step(
                "agent",
                "gen2",
                completion_token_ids=[3, 4, 5],
                logprobs=[-0.3, -0.4, -0.5],
            ),
        )
        traj = atif_to_trajectory(atif, DUMMY_REWARD)
        assert len(traj.token_ids) == 2
        assert traj.token_ids[0] == [1, 2]
        assert traj.token_ids[1] == [3, 4, 5]

    def test_no_metrics_yields_empty_tokens(self):
        atif = _make_atif(
            _make_step("agent", "no metrics"),
        )
        traj = atif_to_trajectory(atif, DUMMY_REWARD)
        assert traj.token_ids == []

    def test_loss_mask_all_true_without_reasoning(self):
        atif = _make_atif(
            _make_step(
                "agent",
                "code",
                completion_token_ids=[1, 2, 3],
                logprobs=[-0.1, -0.2, -0.3],
            ),
        )
        traj = atif_to_trajectory(atif, DUMMY_REWARD)
        assert traj.loss_mask == [[True, True, True]]

    def test_default_trajectory_preserves_reasoning_tokens_when_tokenizer_available(self):
        tok = _make_char_tokenizer()
        reasoning = "abc"
        message = "xyz"
        full_text = f"<think>{reasoning}</think>{message}"
        token_ids = tok.encode(full_text)
        atif = _make_atif(
            _make_step(
                "agent",
                message,
                reasoning_content=reasoning,
                completion_token_ids=token_ids,
                logprobs=[-0.1] * len(token_ids),
            )
        )

        traj = atif_to_trajectory(atif, DUMMY_REWARD, tokenizer=tok)

        assert traj.loss_mask == [[True] * len(token_ids)]

    def test_explicit_mask_thinking_masks_reasoning_tokens(self):
        tok = _make_char_tokenizer()
        reasoning = "abc"
        message = "xyz"
        full_text = f"<think>{reasoning}</think>{message}"
        token_ids = tok.encode(full_text)
        atif = _make_atif(
            _make_step(
                "agent",
                message,
                reasoning_content=reasoning,
                completion_token_ids=token_ids,
                logprobs=[-0.1] * len(token_ids),
            )
        )

        traj = atif_to_trajectory(atif, DUMMY_REWARD, tokenizer=tok, mask_thinking=True)

        decoded_chars = [tok.reverse_vocab[tid] for tid in token_ids]
        for masked, char in zip(traj.loss_mask[0], decoded_chars, strict=True):
            if char in "abc":
                assert masked is False
            elif char in "xyz":
                assert masked is True


class TestExtractAgentGenerations:
    def test_preserves_generation_boundaries(self):
        generations = extract_agent_generations(
            _make_atif(
                _make_step(
                    "agent",
                    "gen1",
                    completion_token_ids=[1, 2],
                    logprobs=[-0.1, -0.2],
                ),
                _make_step("user", "feedback"),
                _make_step(
                    "agent",
                    "gen2",
                    completion_token_ids=[3, 4, 5],
                    logprobs=[-0.3, -0.4, -0.5],
                ),
            )
        )

        assert [generation.token_ids for generation in generations] == [[1, 2], [3, 4, 5]]
        assert [generation.logprobs for generation in generations] == [
            [-0.1, -0.2],
            [-0.3, -0.4, -0.5],
        ]
        assert [generation.loss_mask for generation in generations] == [
            [True, True],
            [True, True, True],
        ]


# -- extract_program ---------------------------------------------------------


class TestExtractProgram:
    def test_from_tool_call(self):
        atif = _make_atif(
            _make_step(
                "agent",
                "submitting",
                tool_calls=[
                    {
                        "function_name": "submit_program",
                        "arguments": {"code": "model :: Meas Int\nmodel = sample (normal 0 1)"},
                    }
                ],
            )
        )
        prog = extract_program(atif)
        assert prog is not None
        assert "model" in prog

    def test_from_submit_tag(self):
        atif = _make_atif(
            _make_step(
                "agent",
                '<submit_program language="haskell">model :: Meas Int</submit_program>',
            )
        )
        prog = extract_program(atif)
        assert prog == "model :: Meas Int"

    def test_no_program(self):
        atif = _make_atif(_make_step("agent", "just chatting"))
        assert extract_program(atif) is None


# -- build_loss_mask ---------------------------------------------------------


class _CharTokenizer:
    """Character-level tokenizer for testing build_loss_mask."""

    def __init__(self, chars: str):
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.reverse_vocab = {i: c for c, i in self.vocab.items()}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.reverse_vocab.get(tid, "") for tid in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[c] for c in text if c in self.vocab]


def _make_char_tokenizer() -> _CharTokenizer:
    return _CharTokenizer("abcdefghijklmnopqrstuvwxyz<>/! ")


class TestBuildLossMask:
    def test_no_reasoning_all_true(self):
        assert build_loss_mask([1, 2, 3], None, None) == [True, True, True]

    def test_reasoning_mask_requires_tokenizer(self):
        with pytest.raises(ValueError, match="requires the model tokenizer"):
            build_loss_mask([1, 2, 3], "reasoning", None)

    def test_reasoning_tokens_masked(self):
        tok = _make_char_tokenizer()
        reasoning = "abc"
        message = "xyz"
        # full text will be <think>abc</think>xyz
        full_text = f"<think>{reasoning}</think>{message}"
        token_ids = tok.encode(full_text)
        mask = build_loss_mask(
            token_ids,
            reasoning,
            tok,
            message_content=message,
        )
        assert len(mask) == len(token_ids)
        decoded_chars = [tok.reverse_vocab[tid] for tid in token_ids]
        for i, (m, c) in enumerate(zip(mask, decoded_chars, strict=True)):
            if c in "abc":
                # reasoning tokens inside <think> should NOT be trainable
                assert m is False, f"token {i} ({c!r}) should be False (reasoning)"
            elif c in "xyz":
                # message tokens should be trainable
                assert m is True, f"token {i} ({c!r}) should be True (message)"

    def test_no_reasoning_with_tokenizer_all_true(self):
        tok = _make_char_tokenizer()
        assert build_loss_mask([1, 2, 3], None, tok) == [True, True, True]

    def test_empty_reasoning_all_true(self):
        tok = _make_char_tokenizer()
        assert build_loss_mask([1, 2, 3], "", tok) == [True, True, True]

    def test_multimodal_non_text_warns(self, caplog):
        """Non-text content parts log a warning."""
        import logging

        atif = _make_atif(
            _make_step(
                "user",
                [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
            ),
            version="ATIF-v1.6",
        )
        with caplog.at_level(logging.WARNING):
            msgs = atif_to_messages(atif)
        assert msgs[0].content == "hello"
        assert any("dropping non-text" in r.message for r in caplog.records)


# -- read_reward_sidecar -----------------------------------------------------


class TestReadRewardSidecar:
    def test_json_reward(self, tmp_path):
        p = tmp_path / "reward.json"
        p.write_text('{"reward": 0.75, "elpd": -12.3}')
        r = read_reward_sidecar(str(p))
        assert r is not None
        assert r.total == 0.75
        assert r.components == {"elpd": -12.3}
        assert r.info["reward_source"] == "sidecar"
        assert r.info["evidence_kind"] == "heuristic"
        assert r.info["truth_owner"] == "heuristic"

    def test_structured_reward_json(self, tmp_path):
        p = tmp_path / "reward.json"
        p.write_text(
            '{"schema_version": "synthstats.harbor.reward.v1", '
            '"reward": 0.8, '
            '"reward_source": "verifier", '
            '"elpd": -4.0, '
            '"loo_status": "accepted", '
            '"diagnostics": {"warning": false}, '
            '"provenance": {"task": "dugongs"}, '
            '"components": {"elpd": -4.0}, '
            '"info": {"would_reject_strict": false}}'
        )

        r = read_reward_sidecar(str(p))

        assert r is not None
        assert r.total == 0.8
        assert r.components == {"elpd": -4.0}
        assert r.info["reward_source"] == "verifier"
        assert r.info["evidence_kind"] == "verifier"
        assert r.info["truth_owner"] == "verifier"
        assert r.info["loo_status"] == "accepted"
        assert r.info["diagnostics"] == {"warning": False}
        assert r.info["provenance"] == {"task": "dugongs"}
        assert r.info["would_reject_strict"] is False

    def test_plain_reward_json_preserves_explicit_provenance_fields(self, tmp_path):
        p = tmp_path / "reward.json"
        p.write_text(
            '{"reward": 0.8, '
            '"reward_source": "validation.elpd", '
            '"evidence_kind": "verifier", '
            '"truth_owner": "verifier", '
            '"elpd": -4.0, '
            '"info": {"would_reject_strict": false}}'
        )

        r = read_reward_sidecar(str(p))

        assert r is not None
        assert r.total == 0.8
        assert r.components == {"elpd": -4.0}
        assert r.info["reward_source"] == "validation.elpd"
        assert r.info["evidence_kind"] == "verifier"
        assert r.info["truth_owner"] == "verifier"
        assert r.info["would_reject_strict"] is False

    def test_plain_reward_json_preserves_top_level_metrics_with_provenance(self, tmp_path):
        p = tmp_path / "reward.json"
        p.write_text(
            '{"reward": -0.123, '
            '"reward_source": "validation.semantic_score", '
            '"evidence_kind": "verifier", '
            '"truth_owner": "verifier", '
            '"semantic_metric": "avg_holdout_logp", '
            '"semantic_score": -0.123}'
        )

        r = read_reward_sidecar(str(p))

        assert r is not None
        assert r.total == pytest.approx(-0.123)
        assert r.components == {"semantic_score": pytest.approx(-0.123)}
        assert r.info["reward_source"] == "validation.semantic_score"
        assert r.info["evidence_kind"] == "verifier"
        assert r.info["truth_owner"] == "verifier"
        assert r.info["semantic_metric"] == "avg_holdout_logp"
        assert r.info["semantic_score"] == pytest.approx(-0.123)

    def test_txt_reward(self, tmp_path):
        p = tmp_path / "reward.txt"
        p.write_text("0.9\n")
        r = read_reward_sidecar(str(p))
        assert r is not None
        assert r.total == 0.9
        assert r.info["reward_source"] == "reward_txt"
        assert r.info["evidence_kind"] == "heuristic"
        assert r.info["truth_owner"] == "heuristic"

    def test_txt_reward_prefers_structured_sibling_when_available(self, tmp_path):
        reward_json = tmp_path / "reward.json"
        reward_txt = tmp_path / "reward.txt"
        reward_json.write_text(
            '{"schema_version": "synthstats.harbor.reward.v1", '
            '"reward": 0.8, '
            '"reward_source": "validation.elpd", '
            '"elpd": -4.0, '
            '"loo_status": "accepted", '
            '"diagnostics": {"warning": false}, '
            '"provenance": {"task": "dugongs"}, '
            '"components": {"elpd": -4.0}, '
            '"info": {"would_reject_strict": false}}'
        )
        reward_txt.write_text("0.9\n")

        r = read_reward_sidecar(str(reward_txt))

        assert r is not None
        assert r.total == 0.8
        assert r.components == {"elpd": -4.0}
        assert r.info["reward_source"] == "validation.elpd"
        assert r.info["evidence_kind"] == "verifier"
        assert r.info["truth_owner"] == "verifier"

    def test_missing_file(self):
        assert read_reward_sidecar("/nonexistent/reward.json") is None
