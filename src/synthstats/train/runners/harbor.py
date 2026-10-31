"""Harbor runner helpers."""

from __future__ import annotations

HARBOR_GENERATOR_CONFIG_KEYS: tuple[str, ...] = (
    "agent_import_path",
    "agent_model_name",
    "sandbox_provider",
    "inference_port",
    "llm_api_base",
    "agent_timeout_sec",
    "collect_rollout_details",
)


def extract_harbor_generator_kwargs(generator_cfg: object) -> dict[str, object]:
    """Project Harbor generator settings out of a dict/object config."""
    harbor_kwargs: dict[str, object] = {}
    for key in HARBOR_GENERATOR_CONFIG_KEYS:
        if hasattr(generator_cfg, key):
            value = getattr(generator_cfg, key)
        elif isinstance(generator_cfg, dict):
            value = generator_cfg.get(key)
        else:
            value = None
        if value is not None:
            harbor_kwargs[key] = value
    return harbor_kwargs


__all__ = [
    "HARBOR_GENERATOR_CONFIG_KEYS",
    "extract_harbor_generator_kwargs",
]
