"""Subtrajectory flow conservation with termination potentials."""

import pytest
import torch

from synthstats.train.objectives.losses import compute_modified_subtb_core


@pytest.mark.parametrize("force_loop", [False, True])
@pytest.mark.parametrize(
    ("pf", "eos", "mask"),
    [
        ([-1.0], [-2.0], [True]),
        ([-1.0, -1.0], [-3.0, -2.0], [True, True]),
        ([-1.0, -1.0, 99.0], [-3.0, -2.0, 99.0], [True, True, False]),
        ([-1.0, 99.0, -1.0], [-3.0, 99.0, -2.0], [True, False, True]),
    ],
)
def test_balanced_termination_flows(pf, eos, mask, force_loop):
    logpf = torch.tensor([pf], requires_grad=True)
    termination = torch.tensor([eos], requires_grad=True)
    loss = compute_modified_subtb_core(
        logpf,
        termination,
        torch.tensor([1.0]),
        torch.tensor([mask]),
        _force_loop=force_loop,
    )
    # Prefix potentials are [2] or [3, 2], terminal potential is 1.
    # Every observed transition satisfies u_i + logPF_i - u_next = 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    torch.testing.assert_close(logpf.grad, torch.zeros_like(logpf))
    torch.testing.assert_close(termination.grad, torch.zeros_like(termination))


def test_unbalanced_single_transition_gradient():
    logpf = torch.tensor([[0.0]], requires_grad=True)
    loss = compute_modified_subtb_core(
        logpf, torch.tensor([[-2.0]]), torch.tensor([1.0]), torch.tensor([[True]])
    )
    # 2 + 0 - 1 = 1, so loss = 1 and derivative = 2.
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert logpf.grad.item() == pytest.approx(2.0)


def test_eos_terminated_trace_has_reward_at_the_stopping_prefix():
    loss = compute_modified_subtb_core(
        torch.tensor([[-0.5, -0.5]], requires_grad=True),
        torch.tensor([[-2.0, -0.5]], requires_grad=True),
        torch.tensor([1.0]),
        torch.tensor([[True, True]]),
        terminal_is_eos=torch.tensor([True]),
    )
    # Prefix potentials 2, 1.5, terminal 1 balance both edges separately.
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("entrypoint", ["gfn_subtb", "modified_subtb"])
def test_subtb_entrypoint_preserves_eos_terminal_boundary(entrypoint):
    from types import SimpleNamespace

    from synthstats.train.objectives.gfn_losses import gfn_subtb_loss
    from synthstats.train.objectives.trajectory_balance import compute_modified_subtb_loss

    pf = torch.tensor([[-0.5, -0.5]])
    config = SimpleNamespace(
        _eos_logprobs=torch.tensor([[-2.0, -0.5]]), _terminal_is_eos=torch.tensor([True])
    )
    fn = gfn_subtb_loss if entrypoint == "gfn_subtb" else compute_modified_subtb_loss
    loss, _ = fn(pf, pf, torch.ones(1, 2), config, loss_mask=torch.ones(1, 2))
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_modified_subtb_rejects_missing_eos():
    from synthstats.train.objectives.trajectory_balance import compute_modified_subtb_loss

    with pytest.raises(ValueError, match="requires aligned"):
        compute_modified_subtb_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), {})


@pytest.mark.parametrize("force_loop", [False, True])
def test_per_prefix_eos_log_rewards_enforce_two_terminal_tree_proportionality(force_loop):
    import math

    # Two-terminal tree with distinct stop rewards at s0 and s1:
    # - Immediate stop at root s0 has reward 4; pi(EOS | s0) = 2/3, pi(A | s0) = 1/3.
    # - After action A at s1, stopping has reward 2 and pi(EOS | s1) = 1.
    log_p_stop_s0 = math.log(2.0 / 3.0)
    log_p_a_s0 = math.log(1.0 / 3.0)
    log_p_stop_s1 = 0.0
    log_r_s0 = math.log(4.0)
    log_r_s1 = math.log(2.0)

    log_probs = torch.tensor(
        [
            [log_p_stop_s0, 0.0],
            [log_p_a_s0, log_p_stop_s1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    eos_logprobs = torch.tensor(
        [
            [log_p_stop_s0, 0.0],
            [log_p_stop_s0, log_p_stop_s1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    eos_log_rewards = torch.tensor(
        [
            [log_r_s0, log_r_s1],
            [log_r_s0, log_r_s1],
        ],
        dtype=torch.float32,
    )
    log_rewards = torch.tensor([log_r_s0, log_r_s1], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

    loss = compute_modified_subtb_core(
        log_probs,
        eos_logprobs,
        log_rewards,
        mask,
        subtb_lambda=0.9,
        _force_loop=force_loop,
        terminal_is_eos=torch.tensor([True, True]),
        eos_log_rewards=eos_log_rewards,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert log_probs.grad is not None
    assert eos_logprobs.grad is not None
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        eos_logprobs.grad, torch.zeros_like(eos_logprobs), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("force_loop", [False, True])
def test_subtb_retains_cross_observation_subtrajectories(force_loop):
    log_probs = torch.tensor([[1.0, 99.0, 2.0]], dtype=torch.float32, requires_grad=True)
    eos_logprobs = torch.tensor([[0.0, 99.0, 0.0]], dtype=torch.float32)
    log_rewards = torch.tensor([0.0], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    loss = compute_modified_subtb_core(
        log_probs,
        eos_logprobs,
        log_rewards,
        mask,
        subtb_lambda=1.0,
        _force_loop=force_loop,
    )
    assert loss.item() == pytest.approx(14.0 / 3.0, rel=1e-6)
    loss.backward()
    assert log_probs.grad is not None
    torch.testing.assert_close(
        log_probs.grad,
        torch.tensor([[8.0 / 3.0, 0.0, 10.0 / 3.0]], dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("force_loop", [False, True])
def test_subtb_large_residual_has_nonzero_gradient(force_loop):
    log_probs = torch.tensor([[201.0]], dtype=torch.float32, requires_grad=True)
    eos_logprobs = torch.tensor([[0.0]], dtype=torch.float32)
    log_rewards = torch.tensor([0.0], dtype=torch.float32)
    mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss = compute_modified_subtb_core(
        log_probs,
        eos_logprobs,
        log_rewards,
        mask,
        max_residual=100.0,
        _force_loop=force_loop,
    )
    assert loss.item() == pytest.approx(201.0**2, rel=1e-5)
    loss.backward()
    assert log_probs.grad is not None
    assert log_probs.grad.item() == pytest.approx(402.0, rel=1e-5)
