"""Detect and mask <think>...</think> blocks in LLM outputs.

Provides utilities for explicit visible-output masking when a caller opts
into masking reasoning spans instead of training over the full generated trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>(.*)$", re.DOTALL)


@dataclass
class ThinkBlockSpan:
    """Character span of a <think>...</think> block."""

    start_char: int
    end_char: int
    content: str


def find_think_blocks(text: str) -> list[ThinkBlockSpan]:
    """Return all <think>...</think> spans. Handles multiple blocks and unclosed tags."""
    if not text:
        return []

    blocks: list[ThinkBlockSpan] = []
    search_start = 0

    while search_start < len(text):
        match = _THINK_PATTERN.search(text, search_start)

        if match:
            blocks.append(
                ThinkBlockSpan(
                    start_char=match.start(),
                    end_char=match.end(),
                    content=match.group(1),
                )
            )
            search_start = match.end()
        else:
            unclosed = _UNCLOSED_THINK_PATTERN.search(text, search_start)
            if unclosed:
                blocks.append(
                    ThinkBlockSpan(
                        start_char=unclosed.start(),
                        end_char=len(text),
                        content=unclosed.group(1),
                    )
                )
            break

    return blocks


def create_think_mask_simple(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip <think> blocks from text. Returns (cleaned_text, removed_spans)."""
    blocks = find_think_blocks(text)

    if not blocks:
        return text, []

    spans: list[tuple[int, int]] = []
    cleaned_parts: list[str] = []
    last_end = 0

    for block in blocks:
        cleaned_parts.append(text[last_end : block.start_char])
        spans.append((block.start_char, block.end_char))
        last_end = block.end_char

    cleaned_parts.append(text[last_end:])

    return "".join(cleaned_parts), spans


def _token_overlaps_any_block(
    token_start: int,
    token_end: int,
    blocks: list[ThinkBlockSpan],
) -> bool:
    for block in blocks:
        if token_start < block.end_char and token_end > block.start_char:
            return True
    return False


def create_think_mask(
    text: str,
    token_ids: list[int],
    tokenizer: Any,
) -> list[bool]:
    """Per-token loss mask: True = include, False = inside a <think> block."""
    if not token_ids:
        return []

    blocks = find_think_blocks(text)

    if not blocks:
        return [True] * len(token_ids)

    if callable(tokenizer):
        try:
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = (
                encoded.get("offset_mapping")
                if isinstance(encoded, dict)
                else getattr(encoded, "offset_mapping", None)
            )
            encoded_ids = (
                encoded.get("input_ids")
                if isinstance(encoded, dict)
                else getattr(encoded, "input_ids", None)
            )
            if (
                offsets is not None
                and encoded_ids is not None
                and list(encoded_ids) == list(token_ids)
            ):
                return [
                    True
                    if start == end
                    else not _token_overlaps_any_block(int(start), int(end), blocks)
                    for start, end in offsets
                ]
        except Exception:
            pass

    mask: list[bool] = []
    char_pos = 0

    for token_id in token_ids:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_len = len(token_text)

        if token_len == 0:
            # zero-length special tokens (BOS, EOS) are structural; always include
            mask.append(True)
        else:
            token_start = char_pos
            token_end = char_pos + token_len
            mask.append(not _token_overlaps_any_block(token_start, token_end, blocks))
            char_pos += token_len

    return mask
