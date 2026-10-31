import json

import pytest

from synthstats.core.types import FinalAnswer, Program, ToolCall
from synthstats.runtime.codecs import (
    ActionCodec,
    JSONToolCodec,
    NativeToolCallingCodec,
    ParseError,
    ToolSpec,
    XMLToolCodec,
)

# --- ActionCodec protocol conformance ---


class TestActionCodecProtocol:
    def test_json_codec_is_action_codec(self):
        assert isinstance(JSONToolCodec(), ActionCodec)

    def test_xml_codec_is_action_codec(self):
        assert isinstance(XMLToolCodec(), ActionCodec)

    def test_native_codec_is_action_codec(self):
        assert isinstance(NativeToolCallingCodec(), ActionCodec)


# --- JSONToolCodec ---


class TestJSONToolCodecExtended:
    @pytest.fixture
    def codec(self):
        return JSONToolCodec()

    # parse: program action
    def test_parse_program_from_code_block(self, codec):
        text = '```json\n{"program": "x = 1", "language": "pymc"}\n```'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == "x = 1"
        assert action.language == "pymc"

    def test_parse_program_default_language(self, codec):
        text = '{"program": "import pymc"}'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.language == "pymc"

    # parse: error path
    def test_parse_error_on_garbage(self, codec):
        with pytest.raises(ParseError, match="No valid JSON"):
            codec.parse("just plain text, no JSON here")

    def test_parse_error_on_json_without_action_keys(self, codec):
        with pytest.raises(ParseError):
            codec.parse('{"foo": "bar"}')

    # parse: inline JSON (no code block)
    def test_parse_tool_call_inline(self, codec):
        text = 'I will run: {"tool": "query", "input": {"experiment": "exp1"}} now.'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"experiment": "exp1"}

    # _find_json_objects edge cases
    def test_find_json_nested_braces(self, codec):
        text = '{"tool": "query", "input": {"nested": {"deep": true}}}'
        objects = JSONToolCodec._find_json_objects(text)
        assert len(objects) == 1
        parsed = json.loads(objects[0])
        assert parsed["input"]["nested"]["deep"] is True

    def test_find_json_escaped_quotes(self, codec):
        text = r'{"tool": "query", "input": {"q": "say \"hello\""}}'
        objects = JSONToolCodec._find_json_objects(text)
        assert len(objects) == 1

    def test_find_json_multiple_objects(self, codec):
        text = 'first: {"a": 1} second: {"b": 2}'
        objects = JSONToolCodec._find_json_objects(text)
        assert len(objects) == 2

    def test_find_json_braces_in_string(self, codec):
        text = '{"tool": "query", "input": {"text": "use {braces} here"}}'
        objects = JSONToolCodec._find_json_objects(text)
        assert len(objects) == 1
        parsed = json.loads(objects[0])
        assert parsed["input"]["text"] == "use {braces} here"

    def test_find_json_no_objects(self, codec):
        assert JSONToolCodec._find_json_objects("no json here") == []

    # render: all action types
    def test_render_tool_call(self, codec):
        action = ToolCall(name="query", input={"q": "test"}, raw="")
        rendered = codec.render(action)
        data = json.loads(rendered)
        assert data["tool"] == "query"
        assert data["input"] == {"q": "test"}

    def test_render_final_answer(self, codec):
        action = FinalAnswer(text="the answer is 42")
        rendered = codec.render(action)
        data = json.loads(rendered)
        assert data["answer"] == "the answer is 42"

    def test_render_program(self, codec):
        action = Program(code="x = 1", language="numpyro")
        rendered = codec.render(action)
        data = json.loads(rendered)
        assert data["program"] == "x = 1"
        assert data["language"] == "numpyro"

    def test_render_unknown_type_raises(self, codec):
        with pytest.raises(ValueError, match="Unknown action type"):
            codec.render("not an action")

    # format_action_spec
    def test_format_action_spec(self, codec):
        tools = [
            ToolSpec(name="query", description="Run experiment", parameters={"type": "object"}),
            ToolSpec(name="submit", description="Submit code", parameters={}),
        ]
        spec = codec.format_action_spec(tools)
        assert "query" in spec
        assert "Run experiment" in spec
        assert "submit" in spec
        assert "```json" in spec

    def test_format_action_spec_empty_params_omitted(self, codec):
        tools = [ToolSpec(name="noop", description="Does nothing", parameters={})]
        spec = codec.format_action_spec(tools)
        assert "noop" in spec
        # empty dict is falsy, so parameters line should not appear
        assert "Parameters" not in spec

    # code block with language tag
    def test_parse_code_block_with_json_tag(self, codec):
        text = '```json\n{"answer": "yes"}\n```'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "yes"

    # code block takes precedence over inline
    def test_code_block_preferred_over_inline(self, codec):
        text = 'Here is inline: {"answer": "wrong"}\n```json\n{"answer": "right"}\n```'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "right"

    # tool call with empty input
    def test_parse_tool_call_no_input_key(self, codec):
        text = '{"tool": "query"}'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input == {}


# --- XMLToolCodec ---


class TestXMLToolCodecExtended:
    @pytest.fixture
    def codec(self):
        return XMLToolCodec()

    # parse: program with and without language
    def test_parse_program_with_language(self, codec):
        text = '<program language="numpyro">import numpyro</program>'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == "import numpyro"
        assert action.language == "numpyro"

    def test_parse_program_default_language(self, codec):
        text = "<program>x = 1</program>"
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.language == "pymc"

    # parse: tool with invalid JSON body
    def test_parse_tool_invalid_json_body(self, codec):
        text = '<tool name="query">not valid json</tool>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input["_parse_error"] is True
        assert action.input["_raw"] == "not valid json"

    # parse: tool with empty body
    def test_parse_tool_empty_body(self, codec):
        text = '<tool name="reset"></tool>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "reset"
        assert action.input == {}

    # parse error
    def test_parse_error_no_xml_action(self, codec):
        with pytest.raises(ParseError, match="No valid XML"):
            codec.parse("just some text without any XML actions")

    # render all types
    def test_render_tool_call(self, codec):
        action = ToolCall(name="query", input={"q": "test"}, raw="")
        rendered = codec.render(action)
        assert '<tool name="query">' in rendered
        assert "</tool>" in rendered

    def test_render_tool_call_empty_input(self, codec):
        action = ToolCall(name="reset", input={}, raw="")
        rendered = codec.render(action)
        assert '<tool name="reset"></tool>' == rendered

    def test_render_final_answer(self, codec):
        action = FinalAnswer(text="42")
        rendered = codec.render(action)
        assert rendered == "<answer>42</answer>"

    def test_render_program(self, codec):
        action = Program(code="x = 1", language="pymc")
        rendered = codec.render(action)
        assert '<program language="pymc">' in rendered
        assert "x = 1" in rendered

    def test_render_unknown_type_raises(self, codec):
        with pytest.raises(ValueError, match="Unknown action type"):
            codec.render("not an action")

    # format_action_spec
    def test_format_action_spec(self, codec):
        tools = [ToolSpec(name="query", description="Run query", parameters={"type": "object"})]
        spec = codec.format_action_spec(tools)
        assert "query" in spec
        assert "Run query" in spec
        assert "XML" in spec or "xml" in spec.lower()

    # tool name with single quotes
    def test_parse_tool_single_quotes(self, codec):
        text = '<tool name=\'query\'>{"q": "test"}</tool>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"

    # parse priority: tool > answer > program
    def test_tool_takes_priority_over_answer(self, codec):
        text = '<tool name="query">{"q": "t"}</tool><answer>42</answer>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)

    # multiline program body (whitespace preserved)
    def test_program_preserves_whitespace(self, codec):
        code = "def f():\n    return 1\n"
        text = f'<program language="pymc">{code}</program>'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == code


# --- NativeToolCallingCodec extended ---


class TestNativeToolCallingCodecExtended:
    @pytest.fixture
    def codec(self):
        return NativeToolCallingCodec()

    # string arguments that are valid JSON
    def test_string_arguments_valid_json_dict(self, codec):
        text = '<tool_call>{"name": "query", "arguments": "{\\"q\\": \\"test\\"}"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"q": "test"}

    # string arguments that are NOT valid JSON
    def test_string_arguments_not_json(self, codec):
        text = '<tool_call>{"name": "query", "arguments": "plain text"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input == {"value": "plain text"}

    # arguments as a non-dict parsed JSON (e.g., a list)
    def test_string_arguments_json_non_dict(self, codec):
        text = '<tool_call>{"name": "query", "arguments": "[1, 2, 3]"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input == {"value": "[1, 2, 3]"}

    # null arguments
    def test_null_arguments(self, codec):
        text = '<tool_call>{"name": "query", "arguments": null}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input == {}

    # final_answer without text key
    def test_final_answer_fallback_keys(self, codec):
        text = '<tool_call>{"name": "final_answer", "arguments": {"value": "42"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "42"

    # submit_program defaults
    def test_submit_program_default_language(self, codec):
        text = '<tool_call>{"name": "submit_program", "arguments": {"code": "x=1"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.language == "pymc"

    # empty name returns None from _parse_native_call -> falls through
    def test_empty_name_falls_through(self, codec):
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        with pytest.raises(ParseError):
            codec.parse(text)

    # render roundtrip
    def test_render_roundtrip_tool_call(self, codec):
        action = ToolCall(name="query", input={"q": "test"}, raw="")
        rendered = codec.render(action)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, ToolCall)
        assert parsed.name == "query"
        assert parsed.input == {"q": "test"}

    def test_render_roundtrip_final_answer(self, codec):
        action = FinalAnswer(text="hello world")
        rendered = codec.render(action)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, FinalAnswer)
        assert parsed.text == "hello world"

    def test_render_roundtrip_program(self, codec):
        action = Program(code="x = 1", language="numpyro")
        rendered = codec.render(action)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, Program)
        assert parsed.code == "x = 1"
        assert parsed.language == "numpyro"

    def test_render_unknown_raises(self, codec):
        with pytest.raises(ValueError, match="Unknown action type"):
            codec.render("bad")

    # bare native JSON (not wrapped in <tool_call> tags)
    def test_bare_native_json_final_answer(self, codec):
        text = '{"name": "final_answer", "arguments": {"text": "done"}}'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "done"

    # multiple tool_call tags: first valid wins
    def test_multiple_tool_call_tags(self, codec):
        text = (
            '<tool_call>{"name": "query", "arguments": {"q": "first"}}</tool_call>\n'
            '<tool_call>{"name": "query", "arguments": {"q": "second"}}</tool_call>'
        )
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.input == {"q": "first"}

    # invalid JSON in tool_call tag: skipped, falls through
    def test_invalid_json_in_tag_skipped(self, codec):
        text = (
            "<tool_call>not json</tool_call>\n"
            '<tool_call>{"name": "query", "arguments": {}}</tool_call>'
        )
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
