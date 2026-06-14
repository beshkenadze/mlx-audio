"""ZONOS2 text tokenizer (UTF-8 byte vocab + special tokens).

Mirrors the upstream byte-level text tokenizer used by the ZONOS2 TTS prompt
builder. The base vocabulary is raw UTF-8 bytes; everything else is a fixed
block of special / conditioning slots, exactly matching the released checkpoint
layout (``text_vocab = 519``).

Source of truth (Zyphra/ZONOS2):
``python/zonos2/tts/prompt.py`` (``text_to_byte_ids``, ``SPECIAL_TOKEN_IDS``,
``LEGACY_SYMBOL_VOCAB_SIZE``, ``BYTE_VOCAB_SIZE``) and
``python/zonos2/tokenizer/textnorm.py`` (NeMo WFST text normalization).

Layout of the ``text_vocab = 519`` id space:

* ``0..191``  - legacy symbol ids. The first four are the named specials
  ``PAD=0, UNK=1, BOS=2, EOS=3``; the remainder are reserved.
* ``192..447`` - the 256 raw UTF-8 byte ids (byte ``b`` maps to ``b + 192``).
* ``448..518`` - conditioning bucket slots (speaking-rate / quality /
  speaker-background / accurate-mode). These are emitted by the prompt builder,
  not by plain text ``encode`` / ``decode``.
* ``519`` - the text padding id (``text_vocab`` itself); this is why the model
  output vocab adds ``text_vocab + 1``.

So ``519 = 192 (legacy symbols) + 256 (bytes) + 71 (conditioning slots)``.
"""

from __future__ import annotations

from typing import Callable, List, Optional

# --- special token ids (mirror upstream prompt.py SPECIAL_TOKEN_IDS) ---
PAD_ID: int = 0
UNK_ID: int = 1
BOS_ID: int = 2
EOS_ID: int = 3
SPECIAL_TOKEN_IDS: tuple[int, ...] = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)

# --- vocab block sizes (mirror upstream prompt.py) ---
LEGACY_SYMBOL_VOCAB_SIZE: int = 192
BYTE_VOCAB_SIZE: int = 256
BYTE_TEXT_VOCAB_SIZE: int = LEGACY_SYMBOL_VOCAB_SIZE + BYTE_VOCAB_SIZE  # 448

# First / last raw-byte id (bytes occupy a contiguous block above the symbols).
BYTE_ID_START: int = LEGACY_SYMBOL_VOCAB_SIZE  # 192
BYTE_ID_END: int = BYTE_TEXT_VOCAB_SIZE  # 448 (exclusive)

# Released-checkpoint text vocabulary size; the padding id equals this value.
TEXT_VOCAB: int = 519


def _load_nemo_normalizer() -> Optional[Callable[[str], str]]:
    """Return a ``text -> str`` English NeMo normalizer, or ``None``.

    Imports ``nemo_text_processing`` lazily so that ``pynini`` /
    ``nemo_text_processing`` are never a hard dependency. Any import or
    construction failure degrades gracefully to ``None`` (identity passthrough).
    The bundled normalizer is English-only; multilingual routing is left to the
    integration layer.
    """
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except Exception:  # noqa: BLE001 - optional dependency, any failure -> fallback
        return None

    try:
        normalizer = Normalizer(input_case="cased", lang="en")
    except Exception:  # noqa: BLE001 - construction may fail without pynini
        return None

    def _normalize(text: str) -> str:
        try:
            result = normalizer.normalize(text, punct_post_process=True)
        except Exception:  # noqa: BLE001 - normalization must never fail tokenization
            return text
        return result if isinstance(result, str) and result.strip() else text

    return _normalize


class ZONOS2Tokenizer:
    """UTF-8 byte tokenizer for ZONOS2 text columns.

    ``encode`` wraps the UTF-8 bytes of the input with ``BOS`` / ``EOS`` and
    offsets each byte by :data:`LEGACY_SYMBOL_VOCAB_SIZE`, matching upstream
    ``text_to_byte_ids``. ``decode`` is the exact inverse over the byte block and
    skips any special / conditioning / padding ids, guaranteeing
    ``decode(encode(s)) == s`` for every valid UTF-8 string.
    """

    pad_id: int = PAD_ID
    unk_id: int = UNK_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    text_vocab: int = TEXT_VOCAB
    byte_id_start: int = BYTE_ID_START
    byte_id_end: int = BYTE_ID_END

    def __init__(self) -> None:
        # Resolved once; ``None`` means identity passthrough.
        self._normalizer: Optional[Callable[[str], str]] = _load_nemo_normalizer()

    @property
    def normalization_enabled(self) -> bool:
        """True when a NeMo normalizer was loaded; False for identity fallback."""
        return self._normalizer is not None

    def normalize(self, text: str) -> str:
        """Written -> spoken text normalization (English).

        Uses ``nemo_text_processing`` (NeMo WFST) when importable, otherwise
        returns ``text`` unchanged. Never raises on normalizer failure.
        """
        normalizer = self._normalizer
        if normalizer is None:
            return text
        return normalizer(text)

    def encode(self, text: str) -> List[int]:
        """Encode text to ids: ``[BOS, *(byte + 192), EOS]``."""
        return [
            BOS_ID,
            *(byte + LEGACY_SYMBOL_VOCAB_SIZE for byte in text.encode("utf-8")),
            EOS_ID,
        ]

    def decode(self, ids: List[int]) -> str:
        """Decode ids back to text, dropping non-byte (special/pad) ids.

        Uses ``errors="replace"`` so a truncated multi-byte sequence (e.g. from
        early-stopped or streamed model output) degrades gracefully instead of
        raising; ``encode`` always emits complete UTF-8, so round-trips stay
        exact.
        """
        raw = bytes(
            token - LEGACY_SYMBOL_VOCAB_SIZE
            for token in ids
            if BYTE_ID_START <= token < BYTE_ID_END
        )
        return raw.decode("utf-8", errors="replace")
