"""Tests for HFPolicy (interface and mock behavior; no model loading)."""

from types import SimpleNamespace

import torch

from synthstats.core.policy import GenConfig
from synthstats.core.types import Message


class _FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        del text, return_tensors
        return {"input_ids": torch.tensor([[11, 12, 13]])}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return ",".join(str(token_id) for token_id in token_ids)


class _FakeGenerateOutput:
    def __init__(self):
        self.sequences = torch.tensor([[11, 12, 13, 21, 22]])
        self.scores = (
            torch.tensor([[0.2, 0.8] + [0.0] * 98], dtype=torch.float32),
            torch.tensor([[0.6, 0.4] + [0.0] * 98], dtype=torch.float32),
        )


class _FakeForwardOutput:
    def __init__(self):
        self.logits = torch.tensor(
            [[[0.1, 0.9], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]],
            dtype=torch.float32,
        )


class _FakeModel:
    def __init__(self, *, training: bool):
        self.training = training
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.eval_calls = 0
        self.train_calls = 0
        self.generate_saw_inference_mode = False
        self.forward_saw_inference_mode = False

    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.train_calls += 1
        self.training = mode
        return self

    def generate(self, input_ids, **kwargs):
        del input_ids, kwargs
        self.generate_saw_inference_mode = torch.is_inference_mode_enabled()
        return _FakeGenerateOutput()

    def __call__(self, full_input):
        del full_input
        self.forward_saw_inference_mode = torch.is_inference_mode_enabled()
        return _FakeForwardOutput()


def _make_uninitialized_hf_policy(*, training: bool):
    from synthstats.policies.hf_policy import HFPolicy

    policy = HFPolicy.__new__(HFPolicy)
    policy.device = "cpu"
    policy.tool_schemas = None
    policy.model = _FakeModel(training=training)
    policy.tokenizer = _FakeTokenizer()
    return policy


class TestHFPolicyConfig:
    def test_hf_policy_has_default_model_name(self):
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        assert "model_name" in sig.parameters

    def test_hf_policy_accepts_device(self):
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        assert "device" in sig.parameters

    def test_hf_policy_accepts_lora_config(self):
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        assert "lora_config" in sig.parameters

    def test_hf_policy_accepts_require_grad_logp(self):
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        assert "require_grad_logp" in sig.parameters

    def test_hf_policy_accepts_use_4bit(self):
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        assert "use_4bit" in sig.parameters


class TestHFPolicyMethods:
    def test_hf_policy_has_call_method(self):
        from synthstats.policies.hf_policy import HFPolicy

        assert callable(HFPolicy.__call__)

    def test_hf_policy_has_score_action_method(self):
        from synthstats.policies.hf_policy import HFPolicy

        assert hasattr(HFPolicy, "score_action")
        assert callable(HFPolicy.score_action)

    def test_hf_policy_has_parameters_method(self):
        from synthstats.policies.hf_policy import HFPolicy

        assert hasattr(HFPolicy, "parameters")

    def test_generate_uses_inference_mode_and_restores_training_state(self):
        policy = _make_uninitialized_hf_policy(training=True)

        generation = policy.generate(
            [Message(role="user", content="hello")],
            gen=GenConfig(max_tokens=2),
        )

        assert generation.token_ids == [21, 22]
        assert policy.model.generate_saw_inference_mode is True
        assert policy.model.eval_calls == 1
        assert policy.model.train_calls == 1
        assert policy.model.training is True

    def test_logprobs_uses_inference_mode_and_restores_training_state(self):
        policy = _make_uninitialized_hf_policy(training=True)

        result = policy.logprobs([Message(role="user", content="hello")], [0, 1])

        assert result.token_ids == [0, 1]
        assert len(result.logprobs) == 2
        assert policy.model.forward_saw_inference_mode is True
        assert policy.model.eval_calls == 1
        assert policy.model.train_calls == 1
        assert policy.model.training is True

    def test_inference_only_paths_do_not_toggle_back_to_train_when_already_eval(self):
        policy = _make_uninitialized_hf_policy(training=False)

        generation = policy.generate(
            [Message(role="user", content="hello")],
            gen=GenConfig(max_tokens=2),
        )
        result = policy.logprobs([Message(role="user", content="hello")], [0, 1])

        assert generation.token_ids == [21, 22]
        assert result.token_ids == [0, 1]
        assert policy.model.generate_saw_inference_mode is True
        assert policy.model.forward_saw_inference_mode is True
        assert policy.model.eval_calls == 2
        assert policy.model.train_calls == 0
        assert policy.model.training is False


class TestMockHFPolicy:
    def test_mock_policy_call_returns_action_logp_ent(self):
        from tests.fixtures import MockHFPolicy

        policy = MockHFPolicy()
        action, logp, ent = policy("Test observation")

        assert isinstance(action, dict)
        assert "type" in action
        assert isinstance(logp, float | torch.Tensor)
        assert isinstance(ent, float | torch.Tensor)

    def test_mock_policy_with_temperature(self):
        from tests.fixtures import MockHFPolicy

        policy = MockHFPolicy()
        action, logp, ent = policy("Test observation", temperature=0.5)

        assert action is not None

    def test_mock_policy_score_action_returns_tensors(self):
        from tests.fixtures import MockHFPolicy

        policy = MockHFPolicy()
        action = {"type": "answer", "payload": "42"}

        logp, ent = policy.score_action("Test observation", action)

        assert isinstance(logp, torch.Tensor)
        assert isinstance(ent, torch.Tensor)

    def test_mock_policy_with_require_grad_logp(self):
        from tests.fixtures import MockHFPolicy

        policy = MockHFPolicy(require_grad_logp=True)
        action, logp, ent = policy("Test observation")

        assert isinstance(logp, torch.Tensor)
        assert isinstance(ent, torch.Tensor)

    def test_mock_policy_parameters(self):
        from tests.fixtures import MockHFPolicy

        policy = MockHFPolicy()
        params = list(policy.parameters())

        assert len(params) >= 1

    def test_mock_policy_sample_with_eos_returns_final_eos_logprob(self):
        from tests.fixtures import MockHFPolicy

        eos_logprobs = [-4.0, -2.5, -0.75]
        policy = MockHFPolicy(
            fixed_token_ids=[101, 102, 103],
            fixed_token_logprobs=[-0.1, -0.2, -0.3],
            fixed_eos_logprobs=eos_logprobs,
        )

        action, logp, ent, eos_logprob = policy.sample_with_eos("Test observation")

        assert action["type"] == "answer"
        assert isinstance(logp, float | torch.Tensor)
        assert isinstance(ent, float | torch.Tensor)
        assert eos_logprob == eos_logprobs[-1]

    def test_mock_policy_score_action_with_eos_returns_final_eos_logprob(self):
        from tests.fixtures import MockHFPolicy

        eos_logprobs = [-4.0, -2.5, -0.75]
        policy = MockHFPolicy(
            fixed_token_ids=[101, 102, 103],
            fixed_token_logprobs=[-0.1, -0.2, -0.3],
            fixed_eos_logprobs=eos_logprobs,
        )

        logp, ent, eos_logprob = policy.score_action_with_eos(
            "Test observation",
            {"type": "answer", "payload": "42"},
        )

        assert isinstance(logp, torch.Tensor)
        assert isinstance(ent, torch.Tensor)
        assert eos_logprob == eos_logprobs[-1]


class TestPolicyOutput:
    def test_policy_output_tuple_structure(self):
        from synthstats.policies.hf_policy import PolicyOutput

        output: PolicyOutput = ({"type": "answer"}, -0.5, 0.1)
        action, logp, ent = output

        assert action["type"] == "answer"
        assert logp == -0.5
        assert ent == 0.1
