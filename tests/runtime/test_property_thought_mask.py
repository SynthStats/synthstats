"""Property-based tests for thought mask module."""

from dataclasses import dataclass

from hypothesis import assume, given, settings
from hypothesis import strategies as st


@dataclass
class CharTokenizer:
    vocab: dict[str, int]
    reverse_vocab: dict[int, str]

    @classmethod
    def from_chars(cls, chars: str) -> "CharTokenizer":
        vocab = {c: i for i, c in enumerate(chars)}
        reverse_vocab = {i: c for c, i in vocab.items()}
        return cls(vocab=vocab, reverse_vocab=reverse_vocab)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.reverse_vocab.get(tid, "") for tid in token_ids)

    def encode(self, text: str) -> list[int]:
        return [self.vocab[c] for c in text if c in self.vocab]


TOKENIZER = CharTokenizer.from_chars(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ<>/0123456789 \n"
)


class TestThinkMaskInvariants:
    @given(
        prefix=st.text(min_size=0, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz "),
        content=st.text(min_size=0, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 "),
        suffix=st.text(min_size=0, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    @settings(max_examples=100)
    def test_mask_length_equals_token_length(self, prefix, content, suffix):
        from synthstats.core.think_mask import create_think_mask

        text = f"{prefix}<think>{content}</think>{suffix}"
        token_ids = TOKENIZER.encode(text)

        assume(len(token_ids) > 0)

        mask = create_think_mask(text, token_ids, TOKENIZER)

        assert len(mask) == len(token_ids)

    @given(
        text=st.text(min_size=1, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 \n"),
    )
    @settings(max_examples=100)
    def test_no_think_tags_all_true(self, text):
        from synthstats.core.think_mask import create_think_mask

        assume("<think>" not in text.lower())

        token_ids = TOKENIZER.encode(text)
        assume(len(token_ids) > 0)

        mask = create_think_mask(text, token_ids, TOKENIZER)

        assert all(mask)

    @given(
        content=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    @settings(max_examples=100)
    def test_only_think_block_all_false(self, content):
        from synthstats.core.think_mask import create_think_mask

        text = f"<think>{content}</think>"
        token_ids = TOKENIZER.encode(text)

        assume(len(token_ids) > 0)

        mask = create_think_mask(text, token_ids, TOKENIZER)

        assert not any(mask)

    @given(
        n_blocks=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_find_all_blocks(self, n_blocks):
        from synthstats.core.think_mask import find_think_blocks

        text = "".join(f"x<think>content{i}</think>" for i in range(n_blocks))
        blocks = find_think_blocks(text)

        assert len(blocks) == n_blocks

    @given(
        prefix=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz "),
        think_content=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz "),
        suffix=st.text(min_size=0, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    @settings(max_examples=100, deadline=None)
    def test_simple_mask_removes_think(self, prefix, think_content, suffix):
        from synthstats.core.think_mask import create_think_mask_simple

        text = f"{prefix}<think>{think_content}</think>{suffix}"
        cleaned, spans = create_think_mask_simple(text)

        assert cleaned == f"{prefix}{suffix}"
        assert len(spans) == 1

    @given(
        text=st.text(min_size=0, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    def test_empty_token_list_gives_empty_mask(self, text):
        from synthstats.core.think_mask import create_think_mask

        assert create_think_mask(text, [], TOKENIZER) == []

    @given(
        content=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    @settings(max_examples=50)
    def test_unclosed_think_masks_to_end(self, content):
        from synthstats.core.think_mask import find_think_blocks

        text = f"before<think>{content}"
        blocks = find_think_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].end_char == len(text)
