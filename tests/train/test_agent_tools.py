from synthstats.train.agent_tools import SYNTHSTATS_TOOLS


def test_agent_tool_schemas_cover_expected_functions() -> None:
    assert len(SYNTHSTATS_TOOLS) == 3
    names = [tool["function"]["name"] for tool in SYNTHSTATS_TOOLS]
    assert names == ["query", "compute_eig", "submit_program"]


def test_agent_tool_schemas_have_openai_function_shape() -> None:
    for tool in SYNTHSTATS_TOOLS:
        assert tool["type"] == "function"
        function = tool["function"]
        assert "name" in function
        assert "description" in function
        assert "parameters" in function
