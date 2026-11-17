"""CPU-only P0 invariants; never imports CUDA models or downloads assets."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import p0_hot_queue
from p0_statistics import compare_baseline_candidate, mad


class P0StatisticsTests(unittest.TestCase):
    def test_noise_gate_promotes_clear_improvement(self):
        a = [20.0, 20.2, 19.8, 20.1, 19.9, 20.0]
        b = [17.0, 17.2, 16.8, 17.1, 16.9, 17.0]
        report = compare_baseline_candidate(a, b, bootstrap=1000)
        self.assertEqual(report["noise_gate"], "conclusive_improvement")
        self.assertGreater(report["delta_over_2s"], 1.0)
        self.assertGreater(report["bootstrap_95ci_delta_s"][0], 0)

    def test_overlap_is_inconclusive(self):
        report = compare_baseline_candidate(
            [20.0, 20.4, 19.8, 20.2],
            [20.1, 20.3, 20.0, 20.2], bootstrap=1000)
        self.assertEqual(report["noise_gate"], "inconclusive_or_regression")

    def test_zero_noise_requires_more_runs(self):
        report = compare_baseline_candidate([20] * 4, [19] * 4, bootstrap=1000)
        self.assertTrue(report["requires_more_runs"])
        self.assertIsNone(report["delta_over_2s"])

    def test_minimum_sample_count(self):
        with self.assertRaises(ValueError):
            compare_baseline_candidate([20] * 3, [18] * 4)

    def test_mad(self):
        self.assertEqual(mad([1, 2, 3]), 1)


class P0PlanningTests(unittest.TestCase):
    def test_fixed_model_and_reference_sampling(self):
        s = p0_hot_queue.settings()
        self.assertEqual(s["model_type"], p0_hot_queue.EXPECTED_MODEL)
        self.assertEqual(s["num_inference_steps"], 4)
        self.assertEqual(s["guidance_scale"], 1.0)
        self.assertEqual(s["flow_shift"], 8.0)
        self.assertEqual(s["video_length"], 33)
        self.assertEqual(s["resolution"], "832x480")
        self.assertEqual(s["seed"], 20261112)
        self.assertEqual(len(s["activated_loras"]), 1)
        self.assertNotIn("distilled", s["model_type"])

    def test_one_warmup_plus_five_hot_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = p0_hot_queue.prepare(root, measured=5)
            cfg = json.loads((root / "config" / "wgp_config.json").read_text())
            self.assertEqual(len(values), 6)
            self.assertTrue(all(v == values[0] for v in values))
            self.assertEqual(cfg["attention_mode"], "sdpa")
            self.assertEqual(cfg["profile"], 0)
            self.assertEqual(cfg["compile"], "disabled")
            self.assertEqual(cfg["vae_config"], 0)

    def test_changed_prompt_sequence_changes_only_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            values = p0_hot_queue.prepare(Path(directory), 2, changed_prompt=True)
            a, b = values[0], values[-1]
            self.assertNotEqual(a["prompt"], b["prompt"])
            self.assertEqual({k: v for k, v in a.items() if k != "prompt"},
                             {k: v for k, v in b.items() if k != "prompt"})


if __name__ == "__main__":
    unittest.main()
