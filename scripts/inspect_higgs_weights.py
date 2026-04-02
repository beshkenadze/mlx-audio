#!/usr/bin/env python3
"""
Inspect HiggsAudioV2 tokenizer weights.

Usage:
    python scripts/inspect_higgs_weights.py /path/to/k2-fsa/OmniVoice

Save output to docs/higgs_audio_weights.md as the inspection deliverable.
"""

import sys
from pathlib import Path

import mlx.core as mx


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_higgs_weights.py <model_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    weights_path = model_path / "audio_tokenizer" / "model.safetensors"

    if not weights_path.exists():
        print(f"ERROR: Not found: {weights_path}")
        sys.exit(1)

    weights = mx.load(str(weights_path))
    print(f"Total keys: {len(weights)}\n")
    print(f"{'Key':<80}  {'Shape':<30}  Dtype")
    print("-" * 120)
    for k, v in sorted(weights.items()):
        print(f"{k:<80}  {str(v.shape):<30}  {v.dtype}")

    prefixes = sorted({k.split(".")[0] for k in weights})
    print(f"\nTop-level prefixes: {prefixes}")


if __name__ == "__main__":
    main()
