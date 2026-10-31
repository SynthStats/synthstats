from __future__ import annotations

from synthstats.train.objectives.registration import register_named_callables


def _sample_fn() -> None:
    return None


def _other_sample_fn() -> None:
    return None


class TestRegisterNamedCallables:
    def test_registers_every_named_callable_in_order(self):
        seen: list[tuple[str, object]] = []

        def fake_register(name: str):
            def decorator(fn):
                seen.append((name, fn))
                return fn

            return decorator

        registered = register_named_callables(
            fake_register,
            {
                "first": _sample_fn,
                "second": _other_sample_fn,
            },
        )

        assert registered == ("first", "second")
        assert seen == [
            ("first", _sample_fn),
            ("second", _other_sample_fn),
        ]
