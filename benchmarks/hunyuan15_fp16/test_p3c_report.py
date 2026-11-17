"""Regression: P3C report preserves baseline clips while stripping JSON payload.

The real P3C run 44027986619 completed all 18 videos but exposed a
KeyError('clips') during quality aggregation. This test is GPU-free.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import json, sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import p3c_custom_compare as p3c


class P3CReportTests(TestCase):
    def test_candidate_quality_reads_original_baseline_clips(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for tile,hot in ((256,34.0),(384,33.0),(512,33.5)):
                directory=root/f"arm_{tile}"/"p3c"/f"spatial_{tile}"
                directory.mkdir(parents=True)
                d={
                    "p3c_spatial_tile_requested":tile,
                    "effective_mmgp_profile_requested":1,
                    "effective_vae_config_requested":0,
                    "resolved_attention":"sdpa",
                    "warmup_generations":1,
                    "measured_generations":5,
                    "clips":[{"clip":f"synthetic_{tile}_{i}.mp4"} for i in range(6)],
                    "request_e2e_samples_seconds":[hot,hot+0.01,hot+0.02,hot-0.01,hot-0.02],
                    "phase_samples":[{"decode_and_copy":11}]*5,
                    "peak_vram_bytes":2**30,
                    "checkpoint_sha256":"same-checkpoint",
                    "converted_lora_sha256":"same-lora",
                    "effective_settings":{"same":True},
                }
                (directory/"baseline.json").write_text(json.dumps(d))
                lines=[]
                for _ in range(6):
                    lines.append(p3c.MARK+json.dumps({
                        "spatial_calls":15 if tile==256 else 6,
                        "tile_sample_min_size":tile,
                        "tile_sample_min_tsize":64,
                        "parallel_tiling":False,
                        "spatial_tiling":True,
                    }))
                    lines.append(p3c.SELECTED+json.dumps({
                        "tile":{"tile_sample_min_size":tile},
                        "overridden":tile!=256,
                    }))
                (directory/"generation.log").write_text("\n".join(lines)+"\n")
            def fake_md5(filename):
                return ["same" if "_256_" in filename else "different"]*33
            with mock.patch.object(p3c,"frame_md5",side_effect=fake_md5):
                with mock.patch.object(p3c,"_ffmpeg_quality",side_effect=lambda *a:48.0):
                    report=p3c.summarize(root)
            self.assertEqual(report["arms"]["256"]["actual_decoder_calls"],[15]*6)
            self.assertEqual(report["arms"]["384"]["actual_decoder_calls"],[6]*6)
            self.assertEqual(report["arms"]["512"]["actual_decoder_calls"],[6]*6)
            self.assertEqual([x["mismatched_decoded_h264_frames"]
                              for x in report["arms"]["384"]["frame_comparisons"]],[33]*5)
            self.assertEqual(report["arms"]["256"]["frame_comparisons"][0]["mismatched_decoded_h264_frames"],0)
            self.assertEqual(report["arms"]["384"]["frame_comparisons"][0]["psnr_h264_db"],48.0)
