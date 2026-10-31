from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from synthstats.envs.registration import register_environments
from synthstats.integrations.skyrl import registry as _registry_module
from synthstats.integrations.skyrl.registry import (
    SKYRL_AVAILABLE,
    _import_sync_registries,
    _register_tb_policy_losses,
    register_all,
    sync_registries_if_ray,
)


@pytest.fixture(autouse=True)
def _reset_registry_guard():
    """Reset the idempotent guard so each test starts fresh."""
    _registry_module._REGISTERED = False
    yield
    _registry_module._REGISTERED = False


class TestRegisterAll:
    def test_no_error_when_skyrl_missing(self):
        if SKYRL_AVAILABLE:
            pytest.skip("SkyRL installed; can't test missing-import path")
        register_all()

    def test_logs_warning_when_skyrl_missing(self, caplog):
        if SKYRL_AVAILABLE:
            pytest.skip("SkyRL installed; can't test missing-import path")
        with caplog.at_level(logging.WARNING, logger="synthstats.integrations.skyrl.registry"):
            register_all()
        assert any("SkyRL not available" in r.message for r in caplog.records)

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_policy_loss")
    def test_calls_register_gfn_losses_when_available(self, mock_rpl):
        with (
            patch(
                "synthstats.integrations.skyrl.registry.register_gfn_losses",
                create=True,
            ),
            patch("synthstats.integrations.skyrl.registry._register_tb_policy_losses"),
        ):
            # need to re-import since register_all uses a lazy import
            from synthstats.integrations.skyrl import registry

            with patch.object(registry, "SKYRL_AVAILABLE", True):
                with patch(
                    "synthstats.integrations.skyrl.loss.register_gfn_losses"
                ) as mock_gfn_actual:
                    # patch the lazy import inside register_all
                    mock_gfn_actual.return_value = True
                    with patch.dict(
                        "sys.modules",
                        {
                            "synthstats.integrations.skyrl.loss": MagicMock(
                                register_gfn_losses=mock_gfn_actual
                            )
                        },
                    ):
                        registry.register_all()
                        mock_gfn_actual.assert_called_once()


class TestRegisterEnvironments:
    def test_registers_env_ids(self) -> None:
        registrations: list[dict[str, str]] = []

        def _fake_register(*, id: str, entry_point: str) -> None:
            registrations.append({"id": id, "entry_point": entry_point})

        with patch.dict(
            "sys.modules",
            {"gymnasium": SimpleNamespace(register=_fake_register)},
        ):
            register_environments()

        assert registrations == [
            {
                "id": "synthstats-task-runtime-v0",
                "entry_point": "synthstats.envs.task_runtime_env:TaskRuntimeEnv",
            },
        ]


class TestRegisterTBPolicyLosses:
    def test_no_error_when_skyrl_missing(self):
        if SKYRL_AVAILABLE:
            pytest.skip("SkyRL installed; can't test missing-import path")
        _register_tb_policy_losses()

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_policy_loss")
    def test_registers_both_tb_policy_losses(self, mock_rpl):
        mock_decorator = MagicMock()
        mock_rpl.return_value = mock_decorator

        _register_tb_policy_losses()

        assert mock_rpl.call_count == 2
        call_args = [c.args[0] for c in mock_rpl.call_args_list]
        assert "trajectory_balance" in call_args
        assert "modified_subtb" in call_args

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_policy_loss")
    def test_duplicate_registration_raises(self, mock_rpl):
        """_register_tb_policy_losses propagates errors; idempotency is at register_all() level."""
        mock_rpl.return_value = MagicMock(side_effect=RuntimeError("already registered"))

        with pytest.raises(RuntimeError, match="already registered"):
            _register_tb_policy_losses()

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_policy_loss", None)
    def test_noop_when_register_policy_loss_is_none(self):
        # register_policy_loss can be None if imports failed
        _register_tb_policy_losses()  # should not raise


class TestSyncRegistriesIfRay:
    def test_no_error_when_skyrl_missing(self):
        if SKYRL_AVAILABLE:
            pytest.skip("SkyRL installed; can't test missing-import path")
        sync_registries_if_ray()

    def test_no_error_when_ray_missing(self, caplog):
        if SKYRL_AVAILABLE:
            pytest.skip("SkyRL installed; can't test missing-import path")
        with caplog.at_level(logging.WARNING, logger="synthstats.integrations.skyrl.registry"):
            sync_registries_if_ray()

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    def test_warns_if_ray_not_initialized(self, caplog):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with caplog.at_level(logging.WARNING, logger="synthstats.integrations.skyrl.registry"):
                sync_registries_if_ray()

        assert any("Ray not initialized" in r.message for r in caplog.records)

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    def test_warns_if_ray_import_fails(self, caplog):
        with patch.dict("sys.modules", {"ray": None}):
            with caplog.at_level(logging.WARNING, logger="synthstats.integrations.skyrl.registry"):
                sync_registries_if_ray()

        # when sys.modules["ray"] = None, import ray raises ImportError
        assert any("Ray not available" in r.message for r in caplog.records)

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_all")
    def test_calls_register_all_when_ray_initialized(self, mock_register_all):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with patch(
                "synthstats.integrations.skyrl.registry._import_sync_registries",
                return_value=None,
            ):
                sync_registries_if_ray()

        mock_register_all.assert_called_once()

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_all")
    def test_calls_skyrl_sync_registries_after_register_all(self, mock_register_all):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_sync = MagicMock()

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with patch(
                "synthstats.integrations.skyrl.registry._import_sync_registries",
                return_value=mock_sync,
            ):
                sync_registries_if_ray()

        mock_register_all.assert_called_once()
        mock_sync.assert_called_once()

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_all")
    def test_logs_when_sync_registries_unavailable(self, mock_register_all, caplog):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with patch(
                "synthstats.integrations.skyrl.registry._import_sync_registries",
                return_value=None,
            ):
                with caplog.at_level(
                    logging.DEBUG,
                    logger="synthstats.integrations.skyrl.registry",
                ):
                    sync_registries_if_ray()

        assert any("stored locally" in r.message for r in caplog.records)

    @patch("synthstats.integrations.skyrl.registry.SKYRL_AVAILABLE", True)
    @patch("synthstats.integrations.skyrl.registry.register_all", side_effect=RuntimeError("boom"))
    def test_catches_exceptions_during_sync(self, mock_register_all, caplog):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True

        with patch.dict("sys.modules", {"ray": mock_ray}):
            with caplog.at_level(logging.WARNING, logger="synthstats.integrations.skyrl.registry"):
                sync_registries_if_ray()

        assert any("Registry sync failed" in r.message for r in caplog.records)


class TestImportSyncRegistries:
    def test_returns_none_when_skyrl_missing(self):
        if SKYRL_AVAILABLE:
            pytest.skip("SkyRL installed; can't test missing-import path")
        assert _import_sync_registries() is None

    def test_returns_callable_or_none(self):
        result = _import_sync_registries()
        assert result is None or callable(result)


class TestPackageExports:
    def test_package_reexports_registry_surface(self):
        import synthstats.integrations.skyrl as pkg

        assert hasattr(pkg, "SKYRL_AVAILABLE")
        assert hasattr(pkg, "register_all")
        assert hasattr(pkg, "sync_registries_if_ray")
