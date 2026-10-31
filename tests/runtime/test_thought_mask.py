"""Tests for thought mask module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from synthstats.core.think_mask import (
    ThinkBlockSpan,
    create_think_mask,
    create_think_mask_simple,
    find_think_blocks,
)


class TestFindThinkBlocks:
    def test_single_block(self):
        text = "Hello <think>reasoning here</think> world"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "reasoning here"
        assert text[blocks[0].start_char : blocks[0].end_char] == "<think>reasoning here</think>"

    def test_multiple_blocks(self):
        text = "<think>first</think> middle <think>second</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].content == "first"
        assert blocks[1].content == "second"

    def test_no_blocks(self):
        blocks = find_think_blocks("No thinking here")
        assert len(blocks) == 0

    def test_unclosed_block(self):
        text = "Start <think>never closed"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "never closed"
        assert blocks[0].end_char == len(text)

    def test_empty_block(self):
        blocks = find_think_blocks("Empty <think></think> block")
        assert len(blocks) == 1
        assert blocks[0].content == ""

    def test_multiline_block(self):
        text = "<think>\nStep 1\nStep 2\n</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert "Step 1" in blocks[0].content
        assert "Step 2" in blocks[0].content

    def test_nested_tags_parse_without_error(self):
        text = "<think>outer <think>inner</think> more</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) >= 1

    def test_only_opening_tag(self):
        text = "Prefix <think>all the rest is thinking"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "all the rest is thinking"

    def test_case_sensitivity(self):
        blocks = find_think_blocks("<THINK>not detected</THINK>")
        assert len(blocks) == 0

    def test_whitespace_in_tags(self):
        blocks = find_think_blocks("< think>not valid</ think>")
        assert len(blocks) == 0

    def test_adjacent_blocks(self):
        text = "<think>first</think><think>second</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].content == "first"
        assert blocks[1].content == "second"

    def test_special_characters_in_content(self):
        text = "<think>def foo():\n    return 42 < 100</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert "42 < 100" in blocks[0].content

    def test_empty_text(self):
        assert len(find_think_blocks("")) == 0

    def test_only_think_tags(self):
        text = "<think>everything</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "everything"
        assert blocks[0].start_char == 0
        assert blocks[0].end_char == len(text)


class TestCreateThinkMaskSimple:
    def test_removes_think_content(self):
        cleaned, spans = create_think_mask_simple("Hello <think>hidden</think> world")
        assert cleaned == "Hello  world"
        assert len(spans) == 1

    def test_multiple_removals(self):
        cleaned, spans = create_think_mask_simple("<think>a</think>b<think>c</think>d")
        assert cleaned == "bd"
        assert len(spans) == 2

    def test_no_blocks_unchanged(self):
        text = "No thinking here"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == text
        assert len(spans) == 0

    def test_empty_text(self):
        cleaned, spans = create_think_mask_simple("")
        assert cleaned == ""
        assert len(spans) == 0

    def test_unclosed_block_removed(self):
        cleaned, spans = create_think_mask_simple("Start <think>all rest")
        assert cleaned == "Start "
        assert len(spans) == 1

    def test_spans_are_original_positions(self):
        text = "AB<think>XY</think>CD"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == "ABCD"
        start, end = spans[0]
        assert text[start:end] == "<think>XY</think>"

    def test_multiline_removal(self):
        text = "Before\n<think>\nLine1\nLine2\n</think>\nAfter"
        cleaned, spans = create_think_mask_simple(text)
        assert "Line1" not in cleaned
        assert "Line2" not in cleaned
        assert "Before" in cleaned
        assert "After" in cleaned


@dataclass
class MockTokenizer:
    vocab: dict[str, int]
    reverse_vocab: dict[int, str]

    @classmethod
    def from_vocab(cls, words: list[str]) -> MockTokenizer:
        vocab = {w: i for i, w in enumerate(words)}
        reverse_vocab = {i: w for w, i in vocab.items()}
        return cls(vocab=vocab, reverse_vocab=reverse_vocab)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.reverse_vocab.get(tid, "") for tid in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab.get(c, 0) for c in text if c in self.vocab]


class TestCreateThinkMask:
    @pytest.fixture
    def char_tokenizer(self) -> MockTokenizer:
        chars = list("abcdefghijklmnopqrstuvwxyz<>/ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        return MockTokenizer.from_vocab(chars)

    def test_all_outside_think(self, char_tokenizer: MockTokenizer):
        text = "hello world"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert len(mask) == len(token_ids)
        assert all(mask)

    def test_all_inside_think(self, char_tokenizer: MockTokenizer):
        text = "<think>hidden</think>"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert len(mask) == len(token_ids)
        assert not any(mask)

    def test_mixed_mask(self, char_tokenizer: MockTokenizer):
        text = "hi<think>x</think>bye"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert len(mask) == len(token_ids)
        assert mask[0] is True
        assert mask[1] is True
        assert mask[-3] is True
        assert mask[-2] is True
        assert mask[-1] is True

    def test_empty_text(self, char_tokenizer: MockTokenizer):
        assert create_think_mask("", [], char_tokenizer) == []

    def test_multiple_blocks(self, char_tokenizer: MockTokenizer):
        text = "a<think>x</think>b<think>y</think>c"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert sum(mask) == 3

    def test_unclosed_block(self, char_tokenizer: MockTokenizer):
        text = "ab<think>rest"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert mask[0] is True
        assert mask[1] is True
        assert not any(mask[2:])


class TestThinkBlockSpan:
    def test_span_creation(self):
        span = ThinkBlockSpan(start_char=10, end_char=50, content="test")
        assert span.start_char == 10
        assert span.end_char == 50
        assert span.content == "test"

    def test_span_equality(self):
        assert ThinkBlockSpan(0, 10, "test") == ThinkBlockSpan(0, 10, "test")


class TestEdgeCases:
    def test_think_in_code_block(self):
        blocks = find_think_blocks("```python\n<think>code thinking</think>\n```")
        assert isinstance(blocks, list)
        assert len(blocks) == 1

    def test_html_like_tags_not_matched(self):
        blocks = find_think_blocks("<div>content</div> <span>more</span>")
        assert len(blocks) == 0

    def test_think_with_attributes_not_matched(self):
        blocks = find_think_blocks('<think id="1">content</think>')
        assert isinstance(blocks, list)

    def test_very_long_content(self):
        text = f"<think>{'x' * 10000}</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert len(blocks[0].content) == 10000

    def test_unicode_content(self):
        text = "<think>Thinking in 日本語 and émojis 🤔</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert "日本語" in blocks[0].content
        assert "🤔" in blocks[0].content


class TestThinkMaskE2EWithRealTokenizer:
    @pytest.fixture
    def real_tokenizer(self):
        pytest.importorskip("transformers", exc_type=ImportError)
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("gpt2")

    def test_real_tokenizer_no_think_blocks(self, real_tokenizer):
        text = "Hello world, this is a test."
        token_ids = real_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, real_tokenizer)

        assert len(mask) == len(token_ids)
        assert all(mask)

    def test_real_tokenizer_with_think_block(self, real_tokenizer):
        text = "Start <think>internal reasoning here</think> end."
        token_ids = real_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, real_tokenizer)

        assert len(mask) == len(token_ids)
        assert not all(mask)
        assert any(mask)

    def test_real_tokenizer_think_at_end(self, real_tokenizer):
        text = "Visible text <think>thinking..."
        token_ids = real_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, real_tokenizer)

        assert len(mask) == len(token_ids)
        assert mask[0] is True
