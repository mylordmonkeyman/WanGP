"""CPU-only regression tests for native FP16 Hunyuan 1.5 VAE 384px default."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.hunyuan15_vae384_default import resolve_hunyuan15_vae384_default as resolve


class VAE:
    ffactor_spatial = 16
    ffactor_temporal = 4


NATIVE = {"hunyuan15_native_fp16": True}
STOCK = {
    "tile_sample_min_size": 256,
    "tile_latent_min_size": 16,
    "tile_sample_min_tsize": 64,
    "tile_latent_min_tsize": 16,
    "tile_overlap_factor": 0.25,
}


class DefaultVAE384Tests(unittest.TestCase):
    def pick(self, model=NATIVE, base="hunyuan_1_5_t2v", vae=None,
             original=STOCK, **kwargs):
        return resolve(model, base, vae or VAE(), original, **kwargs)

    def test_native_t2v_default_enabled_and_nonmutating(self):
        config, applied = self.pick()
        self.assertTrue(applied)
        self.assertEqual(config["tile_sample_min_size"], 384)
        self.assertEqual(config["tile_latent_min_size"], 24)
        self.assertEqual(config["tile_sample_min_tsize"], 64)
        self.assertEqual(config["tile_latent_min_tsize"], 16)
        self.assertEqual(config["tile_overlap_factor"], 0.25)
        self.assertEqual(STOCK["tile_sample_min_size"], 256)
        self.assertIsNot(config, STOCK)

    def test_native_i2v_also_eligible(self):
        config, applied = self.pick(base="hunyuan_1_5_i2v")
        self.assertTrue(applied)
        self.assertEqual(config["tile_sample_min_size"], 384)

    def test_cli_opt_out_restores_exact_stock_object(self):
        config, applied = self.pick(enabled=False)
        self.assertFalse(applied)
        self.assertIs(config, STOCK)

    def test_user_selected_preset_is_not_overridden(self):
        for preset in (1, 2, 3):
            with self.subTest(preset=preset):
                config, applied = self.pick(vae_config=preset)
                self.assertFalse(applied)
                self.assertIs(config, STOCK)

    def test_non_native_or_other_models_unchanged(self):
        for model, base in (({}, "hunyuan_1_5_t2v"),
                            ({"hunyuan15_native_fp16": False}, "hunyuan_1_5_i2v"),
                            (NATIVE, "ltx_video"), (NATIVE, "hunyuan_video")):
            with self.subTest(model=model, base=base):
                config, applied = self.pick(model=model, base=base)
                self.assertFalse(applied)
                self.assertIs(config, STOCK)

    def test_low_vram_auto_and_unusual_tiling_keep_original(self):
        for changed in (
            {"tile_sample_min_size":192, "tile_latent_min_size":12,
             "tile_sample_min_tsize":48, "tile_latent_min_tsize":12},
            {"tile_sample_min_size":128, "tile_latent_min_size":8,
             "tile_sample_min_tsize":32, "tile_latent_min_tsize":8},
            {"tile_overlap_factor":0.5},
            {"tile_latent_min_size":12},
        ):
            candidate = {**STOCK, **changed}
            config, applied = self.pick(original=candidate)
            self.assertFalse(applied)
            self.assertIs(config, candidate)

    def test_invalid_vae_compression_factors_do_not_override(self):
        vae = VAE()
        vae.ffactor_spatial = 8
        config, applied = self.pick(vae=vae)
        self.assertFalse(applied)
        self.assertIs(config, STOCK)

    def test_none_vae_tiles_remains_none(self):
        self.assertEqual(self.pick(original=None), (None, False))

    def test_cli_and_production_dispatch_wiring(self):
        cli = (ROOT / "shared/cli_args.py").read_text()
        wgp = (ROOT / "wgp.py").read_text()
        ast.parse(cli)
        ast.parse(wgp)
        self.assertIn('add("--hunyuan15-vae-384"', cli)
        self.assertIn("argparse.BooleanOptionalAction", cli)
        self.assertIn("args.hunyuan15_vae_384", wgp)
        self.assertIn("resolve_hunyuan15_vae384_default(", wgp)
        self.assertIn("HUNYUAN15_VAE384_DEFAULT", wgp)


if __name__ == "__main__":
    unittest.main()
