import unittest

import mlx.core as mx


class TestHiggsAudioDAC(unittest.TestCase):
    def test_residual_unit_shape(self):
        from mlx_audio.codec.models.higgs_audio.dac import ResidualUnit

        model = ResidualUnit(64)
        x = mx.zeros((1, 100, 64))
        y = model(x)
        self.assertEqual(y.shape, (1, 100, 64))

    def test_encoder_block_downsamples(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticEncoderBlock

        model = AcousticEncoderBlock(64, 128, stride=8)
        x = mx.zeros((1, 800, 64))
        y = model(x)
        self.assertEqual(y.shape[1], 100)

    def test_acoustic_encoder_hop(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticEncoder

        model = AcousticEncoder()
        x = mx.zeros((1, 960, 1))
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 256))

    def test_acoustic_decoder_upsample(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticDecoder

        model = AcousticDecoder()
        x = mx.zeros((1, 1, 256))
        y = model(x)
        self.assertEqual(y.shape, (1, 960, 1))

    def test_rvq_decode_shape(self):
        from mlx_audio.codec.models.higgs_audio.dac import ResidualVectorQuantizer

        model = ResidualVectorQuantizer()
        codes = mx.zeros((1, 17, 8), dtype=mx.int32)
        y = model.decode(codes)
        self.assertEqual(y.shape, (1, 17, 1024))


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
