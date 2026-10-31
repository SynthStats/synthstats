import pytest

from synthstats.core.types import FinalAnswer, Program, ToolCall
from synthstats.runtime.codecs import (
    JSONToolCodec,
    NativeToolCallingCodec,
    ParseError,
    ToolSpec,
    XMLToolCodec,
)


class TestNativeToolCallingCodec:
    @pytest.fixture
    def codec(self):
        return NativeToolCallingCodec()

    def test_parse_tool_call_tag(self, codec):
        text = '<tool_call>{"name": "query", "arguments": {"query": "test"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"query": "test"}

    def test_parse_final_answer(self, codec):
        text = '<tool_call>{"name": "final_answer", "arguments": {"text": "42"}}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "42"

    def test_parse_submit_program(self, codec):
        text = (
            '<tool_call>{"name": "submit_program", '
            '"arguments": {"code": "x=1", "language": "pymc"}}</tool_call>'
        )
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.code == "x=1"
        assert action.language == "pymc"

    def test_parse_with_surrounding_text(self, codec):
        text = (
            "I will query now.\n"
            '<tool_call>{"name": "query", "arguments": {"query": "growth"}}</tool_call>\n'
            "Done."
        )
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"

    def test_fallback_to_json(self, codec):
        text = '```json\n{"tool": "query", "input": {"query": "test"}}\n```'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"

    def test_parse_bare_native_json(self, codec):
        text = '{"name": "query", "arguments": {"query": "test"}}'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"

    def test_render_tool_call(self, codec):
        action = ToolCall(name="query", input={"query": "test"}, raw="")
        rendered = codec.render(action)
        assert "<tool_call>" in rendered
        assert '"name": "query"' in rendered

    def test_render_final_answer(self, codec):
        action = FinalAnswer(text="42")
        rendered = codec.render(action)
        assert "final_answer" in rendered

    def test_render_program(self, codec):
        action = Program(code="x=1", language="pymc")
        rendered = codec.render(action)
        assert "submit_program" in rendered

    def test_format_action_spec(self, codec):
        tools = [ToolSpec(name="query", description="Run query", parameters={"type": "object"})]
        spec = codec.format_action_spec(tools)
        assert "query" in spec
        assert "tool_call" in spec

    def test_parse_error_no_action(self, codec):
        with pytest.raises(ParseError):
            codec.parse("just some random text with no actions")

    def test_string_arguments(self, codec):
        text = '<tool_call>{"name": "query", "arguments": "test_query"}</tool_call>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"


class TestJSONToolCodecBasic:
    def test_parse_code_block(self):
        codec = JSONToolCodec()
        text = '```json\n{"tool": "query", "input": {"q": "test"}}\n```'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)

    def test_parse_inline(self):
        codec = JSONToolCodec()
        text = 'Here is the action: {"answer": "42"}'
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)


class TestXMLToolCodecBasic:
    def test_parse_tool(self):
        codec = XMLToolCodec()
        text = '<tool name="query">{"q": "test"}</tool>'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)

    def test_parse_answer(self):
        codec = XMLToolCodec()
        text = "<answer>42</answer>"
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
