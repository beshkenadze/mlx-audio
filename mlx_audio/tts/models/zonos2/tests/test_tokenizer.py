"""Synthetic tests for the ZONOS2 UTF-8 byte tokenizer.

No GPU / weights: pure byte round-trip + special-token layout + the
NeMo-normalization fallback path.
"""

from __future__ import annotations

import builtins

import pytest

from mlx_audio.tts.models.zonos2.tokenizer import (
    BYTE_ID_END,
    BYTE_ID_START,
    SPECIAL_TOKEN_IDS,
    TEXT_VOCAB,
    ZONOS2Tokenizer,
)


@pytest.fixture()
def tokenizer() -> ZONOS2Tokenizer:
    return ZONOS2Tokenizer()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "ASCII 0123456789 ~!@#$%^&*()",
        "\t newline\n and tabs ",
        "你好，世界",  # CJK: "你好，世界"
        "café naïve résumé",  # accented Latin
        "emoji \U0001f600\U0001f680\U0001f44d mixed こんにちは",  # emoji + Japanese
    ],
)
def test_byte_round_trip(tokenizer: ZONOS2Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_wraps_with_bos_eos(tokenizer: ZONOS2Tokenizer) -> None:
    ids = tokenizer.encode("A")
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    # "A" == byte 65 -> id 65 + 192 == 257
    assert ids == [tokenizer.bos_id, 65 + BYTE_ID_START, tokenizer.eos_id]


def test_byte_ids_are_offset_by_legacy_symbol_block(
    tokenizer: ZONOS2Tokenizer,
) -> None:
    # Every UTF-8 byte maps into the contiguous [192, 448) block.
    for byte in range(256):
        ch = bytes([byte])
        ids = [tok for tok in tokenizer.encode(ch.decode("latin-1"))]
        body = ids[1:-1]  # strip BOS / EOS
        for tok in body:
            assert BYTE_ID_START <= tok < BYTE_ID_END


def test_special_ids_in_range_and_distinct_from_byte_ids() -> None:
    # All specials fit inside the text vocab and sit below the byte block.
    assert len(set(SPECIAL_TOKEN_IDS)) == len(SPECIAL_TOKEN_IDS)
    for special in SPECIAL_TOKEN_IDS:
        assert 0 <= special < TEXT_VOCAB
        assert special < BYTE_ID_START  # disjoint from byte ids [192, 448)


def test_vocab_layout_decomposes_to_519() -> None:
    # 519 = 192 legacy symbols + 256 bytes + 71 conditioning slots.
    assert BYTE_ID_START == 192
    assert BYTE_ID_END == 448
    assert BYTE_ID_END - BYTE_ID_START == 256
    assert TEXT_VOCAB == 519
    assert TEXT_VOCAB - BYTE_ID_END == 71


def test_decode_skips_special_and_padding_ids(tokenizer: ZONOS2Tokenizer) -> None:
    # Inject specials + the padding id (text_vocab) around the bytes for "Hi".
    encoded = tokenizer.encode("Hi")
    noisy = [TEXT_VOCAB, *SPECIAL_TOKEN_IDS, *encoded, TEXT_VOCAB, 500]
    assert tokenizer.decode(noisy) == "Hi"


def test_normalize_returns_str_passthrough_without_nemo(
    tokenizer: ZONOS2Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the optional-dependency import to fail so we exercise the fallback.
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object):
        if name.startswith("nemo_text_processing"):
            raise ImportError("nemo_text_processing is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    fresh = ZONOS2Tokenizer()
    assert fresh.normalization_enabled is False
    out = fresh.normalize("$5.32 on 1/2")
    assert isinstance(out, str)
    assert out == "$5.32 on 1/2"  # identity passthrough


def test_decode_truncated_multibyte_does_not_raise(
    tokenizer: ZONOS2Tokenizer,
) -> None:
    # A multi-byte char split mid-sequence (streaming / early stop) must
    # degrade gracefully instead of raising UnicodeDecodeError.
    truncated = tokenizer.encode("A你")[:3]  # BOS, 'A', first byte of CJK char
    out = tokenizer.decode(truncated)
    assert isinstance(out, str)
    assert out.startswith("A")
