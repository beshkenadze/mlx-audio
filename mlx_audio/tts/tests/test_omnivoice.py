import unittest

import mlx.core as mx


class TestOmniVoiceConfig(unittest.TestCase):
    def test_parse_from_dict_minimal(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
                "sample_rate": 24000,
            }
        )
        self.assertEqual(cfg.audio_vocab_size, 1025)
        self.assertEqual(cfg.num_audio_codebook, 8)
        self.assertEqual(cfg.sample_rate, 24000)

    def test_unknown_keys_are_ignored(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig

        # Should not raise
        OmniVoiceConfig.from_dict({"model_type": "omnivoice", "future_key": 99})

    def test_higgs_audio_config(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig

        cfg = HiggsAudioConfig.from_dict(
            {
                "model_type": "higgs_audio_v2_tokenizer",
                "sample_rate": 24000,
                "codebook_size": 1024,
                "downsample_factor": 320,
            }
        )
        self.assertEqual(cfg.downsample_factor, 320)
        self.assertAlmostEqual(cfg.tokens_per_second, 75.0)


class TestOmniVoiceRegistration(unittest.TestCase):
    def test_model_type_registered(self):
        from mlx_audio.tts.utils import MODEL_REMAPPING

        self.assertIn("omnivoice", MODEL_REMAPPING)
        self.assertEqual(MODEL_REMAPPING["omnivoice"], "omnivoice")


if __name__ == "__main__":
    unittest.main()
