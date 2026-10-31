class TestTaskProtocol:
    def test_task_protocol_has_name_attribute(self):
        from synthstats.core.task import Task

        assert "name" in Task.__annotations__

    def test_task_protocol_has_reset_method(self):
        from synthstats.core.task import Task

        assert hasattr(Task, "reset")

    def test_task_protocol_has_observe_method(self):
        from synthstats.core.task import Task

        assert hasattr(Task, "observe")

    def test_task_protocol_has_step_method(self):
        from synthstats.core.task import Task

        assert hasattr(Task, "step")


class TestPolicyProtocol:
    def test_policy_protocol_has_sampling_methods(self):
        from synthstats.core.policy import Policy

        assert hasattr(Policy, "sample_with_eos")
        assert hasattr(Policy, "score_action")
        assert hasattr(Policy, "score_action_with_eos")

    def test_policy_protocol_does_not_have_generate(self):
        from synthstats.core.policy import Policy

        # generate/logprobs live on GenerativePolicy, not Policy
        members = [m for m in dir(Policy) if not m.startswith("_")]
        assert "generate" not in members
        assert "logprobs" not in members

    def test_generative_policy_extends_policy(self):
        from synthstats.core.policy import GenerativePolicy

        assert hasattr(GenerativePolicy, "generate")
        assert hasattr(GenerativePolicy, "logprobs")
        # also has sampling methods via inheritance
        assert hasattr(GenerativePolicy, "sample_with_eos")
        assert hasattr(GenerativePolicy, "score_action")


class TestExecutorProtocol:
    def test_executor_protocol_has_name_attribute(self):
        from synthstats.core.executor import Executor

        assert "name" in Executor.__annotations__

    def test_executor_protocol_has_execute_method(self):
        from synthstats.core.executor import Executor

        assert hasattr(Executor, "execute")

    def test_tool_result_creation(self):
        from synthstats.core.executor import ToolResult

        result = ToolResult(output="success", success=True, error=None)
        assert result.output == "success"
        assert result.success is True
        assert result.error is None

    def test_tool_result_with_error(self):
        from synthstats.core.executor import ToolResult

        result = ToolResult(output="", success=False, error="timeout")
        assert result.success is False
        assert result.error == "timeout"


class TestJudgeProtocol:
    def test_judge_protocol_has_score_method(self):
        from synthstats.core.judge import Judge

        assert hasattr(Judge, "score")
