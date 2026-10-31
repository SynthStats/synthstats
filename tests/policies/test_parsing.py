from synthstats.policies.parsing import parse_action_dict


class TestParseActionDict:
    def test_parses_tool_call_with_reasoning_prefix(self):
        text = (
            "<think>check the data first</think>"
            '<tool_call>{"name":"query","arguments":{"q":"age=5"}}</tool_call>'
        )

        action = parse_action_dict(text)

        assert action["name"] == "query"
        assert action["arguments"] == {"q": "age=5"}

    def test_parses_json_after_reasoning_without_destructive_strip(self):
        text = '<think>{"scratch": true}</think>{"type": "answer", "payload": "42"}'

        action = parse_action_dict(text)

        assert action == {"type": "answer", "payload": "42"}

    def test_parses_submit_program_tag(self):
        text = (
            "<think>ready to submit</think>"
            '<submit_program language="haskell">model :: Meas Int</submit_program>'
        )

        action = parse_action_dict(text)

        assert action["type"] == "submit_program"
        assert action["payload"] == "model :: Meas Int"
        assert action["language"] == "haskell"

    def test_fallback_answer_uses_cleaned_runtime_text(self):
        text = "<think>private reasoning</think>final visible answer"

        action = parse_action_dict(text)

        assert action == {"type": "answer", "payload": "final visible answer"}

    def test_hidden_json_inside_think_does_not_override_visible_answer(self):
        text = '<think>{"type": "submit_program", "payload": "hidden"}</think>final visible answer'

        action = parse_action_dict(text)

        assert action == {"type": "answer", "payload": "final visible answer"}

    def test_explicit_submit_program_tag_inside_think_still_executes(self):
        text = "<think><submit_program>hidden()</submit_program></think>visible"

        action = parse_action_dict(text)

        assert action == {
            "type": "submit_program",
            "payload": "hidden()",
            "language": "pymc",
        }

    def test_malformed_tool_call_does_not_block_visible_json_recovery(self):
        text = '<tool_call>{not-json}</tool_call>{"type":"answer","payload":"ok"}'

        action = parse_action_dict(text)

        assert action == {"type": "answer", "payload": "ok"}
