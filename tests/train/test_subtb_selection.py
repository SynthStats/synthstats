"""A requested objective cannot silently change during training."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from synthstats.cli.train import validate_train_config
from synthstats.train.config_validation import validate_gfn_algorithm_config
from synthstats.train.objectives.gfn_losses import gfn_subtb_loss


def test_cli_subtb_requires_worker_eos_producer():
    with pytest.raises(ValueError, match="EOS"):
        validate_train_config(
            OmegaConf.create({"backend": "vllm", "objective": {"loss_type": "subtb"}})
        )


def test_skyrl_subtb_requires_worker_eos_producer():
    with pytest.raises(ValueError, match="EOS"):
        validate_gfn_algorithm_config(
            {"algorithm": {"policy_loss_type": "gfn_subtb", "advantage_estimator": "tb_identity"}}
        )


def test_missing_eos_never_falls_back_to_tb():
    lp = torch.tensor([[-1.0]], requires_grad=True)
    with pytest.raises(ValueError, match="EOS"):
        gfn_subtb_loss(lp, lp.detach(), torch.zeros_like(lp), SimpleNamespace())
