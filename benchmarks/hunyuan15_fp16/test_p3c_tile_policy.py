"""P3C native-FP16 384/512px VAE tiling safety and geometry tests."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.hunyuan15_p3c_tile_policy import resolve_custom_vae_tiles as resolve


class VAE:
    ffactor_spatial = 16
    ffactor_temporal = 4


NATIVE = {"hunyuan15_native_fp16": True}
BASE = {
    "tile_sample_min_size": 256,
    "tile_latent_min_size": 16,
    "tile_sample_min_tsize": 64,
    "tile_latent_min_tsize": 16,
    "tile_overlap_factor": 0.25,
}
KW = {"output_width": 832, "output_height": 480, "video_length": 33}


class P3CTilePolicyTests(unittest.TestCase):
    def run_policy(self, tile, *, model=NATIVE, base="hunyuan_1_5_t2v", original=BASE, vae=None, **extra):
        kwargs=dict(KW);kwargs.update(extra)
        return resolve(model, base, vae or VAE(), original, tile, **kwargs)

    def test_unchanged_reference(self):
        result, info=self.run_policy("")
        self.assertIs(result, BASE)
        self.assertIsNone(info)

    def test_384_six_source_derived_spatial_positions(self):
        res, info=self.run_policy("384")
        self.assertEqual(res["tile_sample_min_size"],384)
        self.assertEqual(res["tile_latent_min_size"],24)
        self.assertEqual(info["spatial_stride_latents"],18)
        self.assertEqual(info["spatial_decoder_positions_predicted"],6)
        self.assertEqual(res["tile_sample_min_tsize"],64)
        self.assertEqual(BASE["tile_sample_min_size"],256)

    def test_512_six_source_derived_spatial_positions(self):
        res,info=self.run_policy("512")
        self.assertEqual(res["tile_latent_min_size"],32)
        self.assertEqual(info["spatial_stride_latents"],24)
        self.assertEqual(info["spatial_decoder_positions_predicted"],6)

    def test_not_native_or_not_hunyuan_never_changes_settings(self):
        for model, base in (({}, "hunyuan_1_5_t2v"),
                            ({"hunyuan15_native_fp16":False}, "hunyuan_1_5_i2v"),
                            (NATIVE, "ltx_video")):
            res, info=self.run_policy("384",model=model,base=base)
            self.assertIs(res,BASE)
            self.assertIsNone(info)

    def test_bad_sizes_fail_closed(self):
        for value in ("256","192","abc","513","0384"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_policy(value)

    def test_invalid_overlap_and_baseline_mismatch_fail(self):
        for patch in ({"tile_overlap_factor":0.5},{"tile_sample_min_tsize":48}):
            with self.assertRaises(ValueError):
                self.run_policy("384",original={**BASE, **patch})

    def test_wrong_target_geometry_fails(self):
        with self.assertRaises(ValueError):
            self.run_policy("384",video_length=97)

    def test_runtime_compression_factor_guard(self):
        vae=VAE();vae.ffactor_temporal=8
        with self.assertRaises(ValueError):
            self.run_policy("384",vae=vae)


if __name__ == "__main__":
    unittest.main()
