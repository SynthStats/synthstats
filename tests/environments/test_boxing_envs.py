from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# query_parsing (pure logic, no dependencies)
# ---------------------------------------------------------------------------


class TestParseQueryValue:
    def test_int_parse(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        assert parse_query_int("school=3", "school") == 3

    def test_float_parse(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_float

        assert parse_query_float("temp=36.5", "temp") == pytest.approx(36.5)

    def test_missing_key_returns_none(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        assert parse_query_int("age=5", "school") is None

    def test_empty_query_returns_none(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        assert parse_query_int("", "school") is None

    def test_key_present_but_no_value(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        # "school=" at end of string, split()[0] raises IndexError
        assert parse_query_int("school=", "school") is None

    def test_bad_cast_returns_none(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        assert parse_query_int("school=abc", "school") is None

    def test_multiple_keys_picks_first(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        assert parse_query_int("school=2 school=7", "school") == 2

    def test_value_with_trailing_text(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_float

        assert parse_query_float("temp=36.5 units=C", "temp") == pytest.approx(36.5)

    def test_negative_int(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_int

        assert parse_query_int("offset=-3", "offset") == -3

    def test_custom_cast(self):
        from synthstats.tasks.statistical_modeling.envs.query_parsing import parse_query_value

        result = parse_query_value("flag=TRUE", "flag", lambda s: s.lower() == "true")
        assert result is True


# ---------------------------------------------------------------------------
# EightSchoolsEnv
# ---------------------------------------------------------------------------


class TestEightSchoolsEnv:
    def test_creation_defaults(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        assert len(env.KNOWN_SES) == 8
        assert len(env.theta) == 8
        assert len(env.observed_effects) == 8

    def test_reset_sets_parameters(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset(seed=42)
        # after reset, mu and tau should be sampled (not default zeros)
        # tau is half-normal, so >= 0
        assert env.tau >= 0

    def test_deterministic_with_seed(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env1 = EightSchoolsEnv()
        env1.reset(seed=99)

        env2 = EightSchoolsEnv()
        env2.reset(seed=99)

        assert env1.mu == env2.mu
        assert env1.tau == env2.tau
        assert env1.theta == env2.theta
        assert env1.observed_effects == env2.observed_effects

    def test_query_valid_school(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset(seed=42)
        result = env.query("school=1")
        assert "effect=" in result
        assert "se=" in result

    def test_query_all_schools(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset(seed=42)
        for school_id in range(1, 9):
            result = env.query(f"school={school_id}")
            assert "effect=" in result
            se_val = float(result.split("se=")[1])
            assert se_val == env.KNOWN_SES[school_id - 1]

    def test_query_invalid_format(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset(seed=42)
        result = env.query("garbage")
        assert "Invalid" in result

    def test_query_out_of_range(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset(seed=42)
        assert "Invalid" in env.query("school=0")
        assert "Invalid" in env.query("school=9")

    def test_query_returns_correct_se(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset(seed=42)
        # school 5 has se=9.0
        result = env.query("school=5")
        assert "se=9.0" in result

    def test_reset_without_seed(self):
        from synthstats.tasks.statistical_modeling.envs.eight_schools_env import EightSchoolsEnv

        env = EightSchoolsEnv()
        env.reset()  # should not raise


# ---------------------------------------------------------------------------
# SurgicalEnv
# ---------------------------------------------------------------------------


class TestSurgicalEnv:
    def test_creation_defaults(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        assert env.n_hospitals == 12
        assert len(env.DEFAULT_EXPOSURES) == 12

    def test_reset_populates_data(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset(seed=42)
        assert len(env.deaths) == 12
        assert len(env.hazards) == 12
        assert len(env.log_hazards) == 12
        assert len(env.exposures) == 12
        assert all(d >= 0 for d in env.deaths)
        assert all(h > 0 for h in env.hazards)

    def test_deterministic_with_seed(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env1 = SurgicalEnv()
        env1.reset(seed=77)

        env2 = SurgicalEnv()
        env2.reset(seed=77)

        assert env1.deaths == env2.deaths
        assert env1.exposures == env2.exposures
        assert env1.log_hazards == env2.log_hazards

    def test_query_valid_hospital(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset(seed=42)
        result = env.query("hospital=1")
        assert "deaths=" in result
        assert "exposure=" in result

    def test_query_all_hospitals(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset(seed=42)
        for hid in range(1, 13):
            result = env.query(f"hospital={hid}")
            assert "deaths=" in result
            deaths_str = result.split("deaths=")[1].split(",")[0]
            assert int(deaths_str) >= 0

    def test_query_invalid_format(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset(seed=42)
        result = env.query("garbage")
        assert "Invalid" in result

    def test_query_out_of_range(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset(seed=42)
        assert "Invalid" in env.query("hospital=0")
        assert "Invalid" in env.query("hospital=13")

    def test_poisson_small_lambda(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.rng.seed(42)
        # Knuth algorithm path (lam < 30)
        samples = [env._poisson(5.0) for _ in range(100)]
        mean = sum(samples) / len(samples)
        # should be roughly 5 (within reason for 100 samples)
        assert 2.0 < mean < 8.0

    def test_poisson_large_lambda(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.rng.seed(42)
        # normal approximation path (lam >= 30)
        samples = [env._poisson(50.0) for _ in range(200)]
        mean = sum(samples) / len(samples)
        assert 40.0 < mean < 60.0

    def test_exposures_randomized_around_defaults(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset(seed=42)
        for i, default_exp in enumerate(env.DEFAULT_EXPOSURES):
            # each exposure should be within 20% of default (clamped at 100)
            assert env.exposures[i] >= 100
            assert env.exposures[i] <= default_exp * 1.2 + 1

    def test_reset_without_seed(self):
        from synthstats.tasks.statistical_modeling.envs.surgical_env import SurgicalEnv

        env = SurgicalEnv()
        env.reset()  # should not raise
