import json

import pytest

from synthstats.core.types import Action, FinalAnswer, Program, ToolCall
from synthstats.runtime.codecs import ActionCodec, NativeToolCallingCodec, ParseError


@pytest.fixture
def codec():
    return NativeToolCallingCodec()


class TestParse:
    def test_valid_tool_call(self, codec):
        text = '<tool_call>{"name": "query", "arguments": {"query": "age=5"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"query": "age=5"}

    def test_tool_call_raw_field(self, codec):
        text = '<tool_call>{"name": "query", "arguments": {}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.raw

    def test_tool_call_missing_arguments_defaults_to_empty_dict(self, codec):
        text = '<tool_call>{"name": "query"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input == {}

    def test_submit_tool_call(self, codec):
        text = (
            '<tool_call>{"name": "submit", "arguments": '
            '{"code": "import pymc as pm", "language": "pymc"}}</tool_call>'
        )
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == "import pymc as pm"
        assert action.language == "pymc"

    def test_submit_program_tool_call_alias(self, codec):
        text = '<tool_call>{"name": "submit_program", "arguments": {"code": "x = 1"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == "x = 1"

    def test_submit_defaults_language_to_pymc(self, codec):
        text = '<tool_call>{"name": "submit", "arguments": {"code": "x = 1"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.language == "pymc"

    def test_final_answer(self, codec):
        text = '<tool_call>{"name": "final_answer", "arguments": {"text": "42"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "42"

    def test_parse_error_no_markers(self, codec):
        with pytest.raises(ParseError):
            codec.parse("just some plain text with no markers")

    def test_tool_call_checked_first(self, codec):
        text = '<tool_call>{"name": "query", "arguments": {}}</tool_call> extra text'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"

    def test_tool_call_complex_nested_input(self, codec):
        payload = {
            "name": "compute_eig",
            "arguments": {"experiments": [1, 2, 3], "config": {"n": 100}},
        }
        text = f"<tool_call>{json.dumps(payload)}</tool_call>"
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "compute_eig"
        assert action.input["experiments"] == [1, 2, 3]

    def test_tool_call_embedded_in_surrounding_text(self, codec):
        text = (
            'some preamble <tool_call>{"name": "query", "arguments": {"x": 1}}'
            "</tool_call> trailing text"
        )
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"

    def test_tool_call_id_extracted(self, codec):
        text = '<tool_call>{"name": "query", "arguments": {"x": 1}, "id": "call_123"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.tool_call_id == "call_123"

    def test_tool_call_id_from_tool_call_id_field(self, codec):
        text = '<tool_call>{"name": "query", "arguments": {}, "tool_call_id": "tc_456"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.tool_call_id == "tc_456"

    def test_submit_multiline_code(self, codec):
        code = "import pymc as pm\nwith pm.Model() as m:\n    x = pm.Normal('x', 0, 1)"
        text = (
            '<tool_call>{"name": "submit", "arguments": '
            f'{{"code": {json.dumps(code)}}}}}</tool_call>'
        )
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == code

    def test_json_fallback_tool_format(self, codec):
        text = '```json\n{"tool": "query", "input": {"x": 1}}\n```'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"


class TestRender:
    def test_render_tool_call(self, codec):
        action = ToolCall(name="query", input={"age": 5}, raw="")
        result = codec.render(action)
        assert "<tool_call>" in result
        assert "</tool_call>" in result
        data = json.loads(result.replace("<tool_call>", "").replace("</tool_call>", ""))
        assert data["name"] == "query"
        assert data["arguments"] == {"age": 5}

    def test_render_program(self, codec):
        action = Program(code="import pymc as pm", language="pymc")
        result = codec.render(action)
        assert "<tool_call>" in result
        data = json.loads(result.replace("<tool_call>", "").replace("</tool_call>", ""))
        assert data["name"] == "submit_program"
        assert data["arguments"]["code"] == "import pymc as pm"
        assert data["arguments"]["language"] == "pymc"

    def test_render_final_answer(self, codec):
        result = codec.render(FinalAnswer(text="the answer is 42"))
        data = json.loads(result.replace("<tool_call>", "").replace("</tool_call>", ""))
        assert data["name"] == "final_answer"
        assert data["arguments"]["text"] == "the answer is 42"

    def test_render_unsupported_raises(self, codec):
        class _CustomAction(Action):
            pass

        with pytest.raises(ValueError):
            codec.render(_CustomAction())


class TestRoundTrip:
    def test_tool_call_round_trip(self, codec):
        original = ToolCall(name="query", input={"age": 7}, raw="")
        rendered = codec.render(original)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, ToolCall)
        assert parsed.name == original.name
        assert parsed.input == original.input

    def test_program_round_trip(self, codec):
        original = Program(code="x = pm.Normal('x', 0, 1)")
        rendered = codec.render(original)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, Program)
        assert parsed.code == original.code
        assert parsed.language == original.language

    def test_tool_call_complex_input_round_trip(self, codec):
        original = ToolCall(
            name="compute_eig",
            input={"experiments": [1, 2, 3], "config": {"n": 100, "flag": True}},
            raw="",
        )
        rendered = codec.render(original)
        parsed = codec.parse(rendered)
        assert parsed.name == original.name
        assert parsed.input == original.input

    def test_program_multiline_round_trip(self, codec):
        code = "import pymc as pm\nwith pm.Model() as m:\n    mu = pm.Normal('mu', 0, 1)"
        original = Program(code=code)
        rendered = codec.render(original)
        parsed = codec.parse(rendered)
        assert parsed.code == code


class TestActionCodecProtocol:
    def test_native_codec_is_action_codec(self):
        codec = NativeToolCallingCodec()
        assert isinstance(codec, ActionCodec)

    def test_format_action_spec_returns_string(self):
        from synthstats.runtime.codecs import ToolSpec

        codec = NativeToolCallingCodec()
        spec = codec.format_action_spec(
            [
                ToolSpec(name="query", description="Run experiment", parameters={"type": "object"}),
            ]
        )
        assert isinstance(spec, str)
        assert "query" in spec
        assert "Run experiment" in spec


class TestToolSchemaRegistry:
    def test_get_tool_schemas_returns_list(self):
        from synthstats.tasks.statistical_modeling.tools import get_tool_schemas

        schemas = get_tool_schemas("dugongs")
        assert isinstance(schemas, list)
        assert len(schemas) == 2

    def test_schemas_have_query_and_submit(self):
        from synthstats.tasks.statistical_modeling.tools import get_tool_schemas

        schemas = get_tool_schemas("dugongs")
        names = {s["function"]["name"] for s in schemas}
        assert names == {"query", "submit"}

    def test_env_specific_description(self):
        from synthstats.tasks.statistical_modeling.tools import get_tool_schemas

        schemas = get_tool_schemas("dugongs")
        query_schema = next(s for s in schemas if s["function"]["name"] == "query")
        description = query_schema["function"]["parameters"]["properties"]["query"]["description"]
        assert "dugong" in description

    def test_tool_schemas_to_toolspecs(self):
        from synthstats.tasks.statistical_modeling.tools import (
            get_tool_schemas,
            tool_schemas_to_toolspecs,
        )

        schemas = get_tool_schemas("dugongs")
        specs = tool_schemas_to_toolspecs(schemas)
        assert len(specs) == 2
        assert specs[0].name == "query"
        assert specs[1].name == "submit"
