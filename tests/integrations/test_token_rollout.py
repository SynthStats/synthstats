import asyncio

import httpx
import pytest

from synthstats.integrations.harbor.converter import extract_causal_token_data
from synthstats.integrations.harbor.episode import TokenRolloutClient
from synthstats.integrations.harbor.generator import resolve_trial_sampling_params


def test_generation_records_the_exact_requested_prefix_and_behavior_probabilities():
    requests = []

    def endpoint(request):
        import json

        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "action",
                        "token_ids": [3, 4],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.2, -0.3]},
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://endpoint.test", transport=httpx.MockTransport(endpoint)
        ) as client:
            rollout = TokenRolloutClient(client, model="policy", sampling={"temperature": 0.7})
            text, metrics = await rollout.complete([1, 2], max_tokens=2)
            return text, extract_causal_token_data(
                {"steps": [{"source": "agent", "metrics": metrics}]}
            )

    text, row = asyncio.run(run())
    assert text == "action"
    assert requests[0]["prompt"] == [1, 2]
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["logprobs"] == 1
    assert row["prompt_ids"] == [1, 2]
    assert row["response_ids"] == [3, 4]
    assert row["logprobs"] == [-0.2, -0.3]


def test_resolved_per_call_stop_strings_reach_completion_request():
    requests = []

    def endpoint(request):
        import json

        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "action",
                        "token_ids": [3, 4],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.2, -0.3]},
                    }
                ]
            },
        )

    resolved = resolve_trial_sampling_params(
        {"max_generate_length": 64, "temperature": 0.7, "top_p": 0.95, "top_k": 50},
        {"max_tokens": 5, "temperature": 0, "stop": ["END"]},
    )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://endpoint.test", transport=httpx.MockTransport(endpoint)
        ) as client:
            rollout = TokenRolloutClient(client, model="policy", sampling=resolved)
            return await rollout.complete([1, 2], max_tokens=resolved["max_generate_length"])

    asyncio.run(run())
    assert requests[0]["max_tokens"] == 5
    assert requests[0]["temperature"] == 0
    assert requests[0]["stop"] == ["END"]
    assert "max_generate_length" not in requests[0]


def test_sampling_resolver_and_rollout_client_reject_unsupported_fields():
    with pytest.raises(ValueError, match="Unsupported sampling parameter"):
        resolve_trial_sampling_params(
            {"max_generate_length": 16, "temperature": 0.7},
            {"max_tokens": 5, "best_of": 2},
        )

    with pytest.raises(ValueError, match="Unsupported sampling parameter"):
        TokenRolloutClient(object(), model="policy", sampling={"best_of": 2})


def test_generation_rejects_an_endpoint_that_omits_behavior_probabilities():
    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"text": "action", "token_ids": [3]}],
                },
            )
        )
        async with httpx.AsyncClient(
            base_url="http://endpoint.test", transport=transport
        ) as client:
            rollout = TokenRolloutClient(client, model="policy", sampling={})
            await rollout.complete([1, 2], max_tokens=1)

    with pytest.raises(ValueError, match="log probabilities"):
        asyncio.run(run())


def test_run_episode_and_harbor_generator_round_trip(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import synthstats.integrations.harbor.episode as episode_mod
    import synthstats.integrations.harbor.generator as generator_mod
    from synthstats.integrations.harbor.episode import run_episode

    class _FakeEnv:
        def __init__(self):
            self.chat_history = []
            self.turns = 0

        def init(self):
            self.chat_history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "obs0"},
            ]
            return list(self.chat_history), {}

        def step(self, action: str):
            self.turns += 1
            if self.turns == 1:
                self.chat_history.append({"role": "assistant", "content": action})
                self.chat_history.append({"role": "user", "content": "obs1"})
                return {"done": False, "reward": 0.0, "metadata": {}}
            return {
                "done": True,
                "reward": 1.75,
                "metadata": {
                    "program": "import pymc as pm",
                    "language": "pymc",
                    "reward_info": {"source": "validation.elpd"},
                },
            }

    monkeypatch.setattr(episode_mod, "build_env_from_dict", lambda cfg: _FakeEnv())

    completion_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal completion_calls
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            if "prompt" in body:
                return httpx.Response(200, json={"tokens": [99]})
            messages = body["messages"]
            if len(messages) > 1:
                return httpx.Response(200, json={"tokens": [10, 11, 12]})
            return httpx.Response(200, json={"tokens": [30, 31]})
        assert request.url.path == "/v1/completions"
        completion_calls += 1
        if completion_calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "text": "<query>x=1</query>",
                            "token_ids": [20, 21],
                            "finish_reason": "stop",
                            "logprobs": {"token_logprobs": [-0.1, -0.2]},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "<submit>import pymc as pm</submit>",
                        "token_ids": [40, 41],
                        "finish_reason": "stop",
                        "logprobs": {"token_logprobs": [-0.3, -0.4]},
                    }
                ]
            },
        )

    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        episode_mod.httpx,
        "AsyncClient",
        lambda **kwargs: orig_async_client(
            base_url=kwargs.get("base_url", "http://endpoint.test"),
            transport=httpx.MockTransport(handler),
        ),
    )

    trial_dir = tmp_path / "trial-001"
    agent_dir = trial_dir / "agent"
    runtime = {
        "TASK_CONFIG": json.dumps({"env": {"name": "dugongs"}}),
        "TASK_PROVENANCE": json.dumps(
            {
                "task_name": "dugongs",
                "dataset_row": {
                    "id": "row-1",
                    "env_id": "dugongs",
                    "trace_id": "trace-1",
                },
            }
        ),
        "LLM_API_BASE": "http://endpoint.test/v1",
        "MODEL_NAME": "Qwen/Qwen3.5-0.8B",
        "SAMPLING_PARAMS": json.dumps({"max_generate_length": 16, "temperature": 0.7}),
        "MAX_SEQUENCE_LENGTH": "128",
        "MAX_TURNS": "4",
    }
    asyncio.run(run_episode("Fit dugongs.", runtime, agent_dir))

    trajectory = json.loads((agent_dir / "trajectory.json").read_text())
    result_log = json.loads((agent_dir / "synthstats_result.json").read_text())
    assert trajectory["schema_version"] == "ATIF-v1.6"
    assert result_log["reward"] == pytest.approx(1.75)

    from synthstats.core.types import Reward
    from synthstats.integrations.harbor.reward_artifacts import (
        build_reward_artifact,
        write_reward_artifacts,
    )

    write_reward_artifacts(
        build_reward_artifact(
            Reward(total=1.75, components={}, info={}), reward_source="validation.likelihood"
        ),
        output_dir=trial_dir / "verifier",
    )

    class _BaseGenerator:
        def __init__(self, *args, **kwargs):
            return None

    monkeypatch.setattr(generator_mod, "_GeneratorInterface", _BaseGenerator)
    monkeypatch.setattr(generator_mod, "_HARBOR_AVAILABLE", True)
    gen_cls = generator_mod._make_generator_class()
    assert gen_cls is not None
    generator = gen_cls(config=SimpleNamespace(), llm_api_base="http://endpoint.test/v1")

    fake_trial_result = SimpleNamespace(
        trial_uri=str(trial_dir),
        agent_result=SimpleNamespace(metadata=result_log),
        verifier_result=None,
    )

    async def _fake_run_single_trial(task_cfg):
        return fake_trial_result, generator._extract_atif(fake_trial_result)

    monkeypatch.setattr(generator, "_run_single_trial", _fake_run_single_trial)
    output = asyncio.run(generator.generate({"env_extras": [{"task_name": "dugongs"}]}))

    assert output["prompt_token_ids"] == [[10, 11, 12]]
    assert output["response_ids"] == [[20, 21, 99, 30, 31, 40, 41]]
    assert output["loss_masks"] == [[1, 1, 0, 0, 0, 1, 1]]
    assert output["rollout_logprobs"] == [[-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4]]
    assert output["rewards"] == [pytest.approx(1.75)]
    assert output["rollout_metrics"]["n_success"] == 1
    assert output["rollout_metrics"]["n_with_verifier_reward"] == 1


def test_harbor_generator_rejects_trajectory_without_token_payload(monkeypatch):
    from types import SimpleNamespace

    import synthstats.integrations.harbor.generator as generator_mod

    class _BaseGenerator:
        def __init__(self, *args, **kwargs):
            return None

    monkeypatch.setattr(generator_mod, "_GeneratorInterface", _BaseGenerator)
    monkeypatch.setattr(generator_mod, "_HARBOR_AVAILABLE", True)
    gen_cls = generator_mod._make_generator_class()
    assert gen_cls is not None
    generator = gen_cls(config=SimpleNamespace(), llm_api_base="http://endpoint.test/v1")

    async def _fake_run_single_trial(task_cfg):
        return SimpleNamespace(agent_result=None), {"schema_version": "ATIF-v1.6", "steps": []}

    monkeypatch.setattr(generator, "_run_single_trial", _fake_run_single_trial)
    with pytest.raises(ValueError, match="no token-level training payload"):
        asyncio.run(generator.generate({"env_extras": [{"task_name": "dugongs"}]}))
