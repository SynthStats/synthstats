from synthstats.core.program_extraction import extract_from_tool_call, extract_program_text


def test_submit_program_tags():
    text = "blah <submit_program>x = 1</submit_program> blah"
    assert extract_program_text(text) == "x = 1"


def test_python_code_fence():
    text = "some text\n```python\ny = 2\n```\nmore"
    assert extract_program_text(text) == "y = 2"


def test_submit_program_preferred_over_fence():
    text = "<submit_program>a = 1</submit_program>\n```python\nb = 2\n```"
    assert extract_program_text(text) == "a = 1"


def test_no_match_returns_none():
    assert extract_program_text("just plain text") is None
    assert extract_program_text("") is None


def test_strips_whitespace():
    text = "<submit_program>  \n  code  \n  </submit_program>"
    assert extract_program_text(text) == "code"


def test_submit_program_with_language_attribute():
    text = '<submit_program language="haskell">module Main where\nmain = print 1</submit_program>'
    assert extract_program_text(text) == "module Main where\nmain = print 1"


def test_submit_program_with_python_language():
    text = '<submit_program language="python">x = 1</submit_program>'
    assert extract_program_text(text) == "x = 1"


def test_submit_program_with_language_preferred_over_fence():
    text = (
        '<submit_program language="haskell">haskell code</submit_program>\n'
        "```python\npython code\n```"
    )
    assert extract_program_text(text) == "haskell code"


def test_tool_call_submit_name():
    """Tool schema uses 'submit'; extractor must accept it."""
    text = '{"name": "submit", "arguments": {"code": "x = 1"}}'
    assert extract_from_tool_call(text) == "x = 1"


def test_tool_call_submit_program_name():
    text = '{"name": "submit_program", "arguments": {"code": "y = 2"}}'
    assert extract_from_tool_call(text) == "y = 2"


def test_tool_call_submit_via_extract_program_text():
    """extract_program_text finds 'submit' tool calls."""
    text = '{"name": "submit", "arguments": {"code": "z = 3"}}'
    assert extract_program_text(text) == "z = 3"
