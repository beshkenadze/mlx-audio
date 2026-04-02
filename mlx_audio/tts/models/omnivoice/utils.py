import mlx.core as mx


def create_voice_clone_prompt(ref_audio_path: str, ref_text: str = "") -> mx.array:
    """
    Encode a reference audio file into acoustic token prefix for voice cloning.

    NOTE: Full implementation blocked by HiggsAudioV2 encoder (Task 7.3).
    Currently returns a zero-token placeholder of shape [0, 8] (empty prefix).

    Args:
        ref_audio_path: Path to reference audio file (WAV, 24kHz preferred)
        ref_text: Optional transcript of reference audio (for alignment)

    Returns:
        mx.array of shape [T_ref, 8] — acoustic token prefix, int32
        Returns empty array [0, 8] until HiggsAudioV2 encoder is implemented.
    """
    # Placeholder: return empty prefix until HiggsAudioV2 encoder available
    return mx.zeros((0, 8), dtype=mx.int32)
