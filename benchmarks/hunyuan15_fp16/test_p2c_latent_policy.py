"""CPU-only P2C policy fixtures: preserve BF16 and legacy file behavior."""
import unittest

from p2c_latent_policy import latent_action


class P2CLatentPolicyTests(unittest.TestCase):
    def test_legacy_without_opt_in_still_saves(self):
        self.assertEqual(latent_action(True, False, {}), ("save", "latent.pt"))

    def test_legacy_explicit_load_unchanged(self):
        self.assertEqual(latent_action(True, True, {}), ("load", "latent.pt"))

    def test_native_opt_in_skips_unrequested_debug_io(self):
        self.assertEqual(latent_action(True, False, {"HUNYUAN15_P2C_ENABLE": "1"}), ("skip", None))

    def test_native_opt_in_explicit_debug_save(self):
        d = {
            "HUNYUAN15_P2C_ENABLE": "1",
            "HUNYUAN15_P2C_SAVE_DEBUG_LATENTS": "1",
            "HUNYUAN15_P2C_LATENT_PATH": "/tmp/hunyuan-run-1234.pt",
        }
        self.assertEqual(latent_action(True, False, d), ("save", "/tmp/hunyuan-run-1234.pt"))

    def test_native_opt_in_explicit_load(self):
        self.assertEqual(latent_action(True, True, {"HUNYUAN15_P2C_ENABLE":"1"}),
                         ("load", "latent.pt"))

    def test_native_opt_in_custom_load(self):
        d={"HUNYUAN15_P2C_ENABLE":"1","HUNYUAN15_P2C_LATENT_PATH":"/tmp/x.pt"}
        self.assertEqual(latent_action(True, True, d), ("load", "/tmp/x.pt"))

    def test_bf16_or_other_model_is_unchanged_even_with_opt_in(self):
        d={"HUNYUAN15_P2C_ENABLE":"1","HUNYUAN15_P2C_LATENT_PATH":"/tmp/x.pt"}
        self.assertEqual(latent_action(False, False, d), ("save", "latent.pt"))
        self.assertEqual(latent_action(False, True, d), ("load", "latent.pt"))


if __name__ == "__main__":
    unittest.main()
