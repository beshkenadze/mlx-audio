import unittest

import mlx.core as mx


class TestHiggsAudioTokenizer(unittest.TestCase):
    def test_higgs_audio_instantiation(self):
        from mlx_audio.codec.models.higgs_audio import (
            HiggsAudioConfig,
            HiggsAudioTokenizer,
        )

        tokenizer = HiggsAudioTokenizer(HiggsAudioConfig())
        self.assertIsNotNone(tokenizer)

    def test_higgs_audio_decode_raises_not_implemented(self):
        from mlx_audio.codec.models.higgs_audio import (
            HiggsAudioConfig,
            HiggsAudioTokenizer,
        )

        tokenizer = HiggsAudioTokenizer(HiggsAudioConfig())
        tokens = mx.zeros((10, 8), dtype=mx.int32)
        with self.assertRaises(NotImplementedError):
            tokenizer.decode(tokens)

    def test_higgs_audio_from_pretrained_raises_not_implemented(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioTokenizer

        with self.assertRaises(NotImplementedError):
            HiggsAudioTokenizer.from_pretrained("/tmp")

    def test_higgs_audio_config_tokens_per_second(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioConfig

        cfg = HiggsAudioConfig()
        self.assertAlmostEqual(cfg.tokens_per_second, 75.0)


if __name__ == "__main__":
    unittest.main()
