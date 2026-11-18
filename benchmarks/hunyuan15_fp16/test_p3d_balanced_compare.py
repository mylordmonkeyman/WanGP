"""CPU-only regression fixtures for P3D balanced VAE report."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3d_balanced_compare as p3d


class P3DBalancedReportTests(TestCase):
    def test_abba_paths_memory_and_quality(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for name,tile in p3d.SESSIONS:
                base=root/f"{name}_{tile}"/"p3c"/f"spatial_{tile}"
                base.mkdir(parents=True)
                samples=[34.0]*5 if tile==256 else [33.0]*5
                data={
                    "p3c_spatial_tile_requested":tile,
                    "effective_mmgp_profile_requested":1,
                    "effective_vae_config_requested":0,
                    "resolved_attention":"sdpa",
                    "warmup_generations":1,"measured_generations":5,
                    "clips":[{"clip":f"{name}_{tile}_{n}.mp4","decoded_frames":33} for n in range(6)],
                    "request_e2e_samples_seconds":samples,
                    "phase_samples":[{"decode_and_copy":11.0}]*5,
                    "peak_vram_bytes":2**30,
                    "checkpoint_sha256":"constant-model",
                    "converted_lora_sha256":"constant-lora",
                    "effective_settings":{"seed":20261112},
                }
                (base/"baseline.json").write_text(json.dumps(data))
                (base/"generation.log").write_text("\n".join(
                    [p3d.TRACE_MARK+json.dumps({
                        "spatial_calls":15 if tile==256 else 6,
                        "tile_sample_min_size":tile,
                        "tile_sample_min_tsize":64,
                    })+"\n"+p3d.TILE_MARK+json.dumps({
                        "tile":{"tile_sample_min_size":tile},
                        "overridden":tile==384,
                    }) for _ in range(6)]))
                (root/f"nvidia_{name}_{tile}.csv").write_text(
                    "".join(f"2026/11/18 00:00:0{n}, 9000, 23510\n" for n in range(6)))
            with mock.patch.object(p3d,"frame_md5",side_effect=lambda p:
                                   ["baseline" if "_256_" in p else "candidate"]*33):
                with mock.patch.object(p3d,"ffmpeg_quality",side_effect=lambda a,b,m:
                                       49.0 if m=="psnr" else 0.995):
                    out=p3d.evaluate(root)
            self.assertEqual(out["order"],["a1","b1","b2","a2"])
            self.assertEqual(len(out["matched_clip_quality"]),10)
            self.assertTrue(out["fidelity_screen_pass"])
            self.assertEqual(out["minimum_recorded_free_gpu_mib"],23510)
            self.assertEqual(out["results"]["b1"]["decoder_calls"],6)
            self.assertEqual(out["results"]["a2"]["decoder_calls"],15)
            self.assertGreater(out["pooled_performance"]["improvement_percent"],0)

    def test_memory_monitor_missing_or_too_short_fails(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/"samples.csv"
            with self.assertRaises(ValueError):p3d.read_memory(p)
            p.write_text("2026/11/18 00:00:00, 8000, 24000\n")
            with self.assertRaises(ValueError):p3d.read_memory(p)
