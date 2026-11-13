"""CPU-only adapter contract tests; no CUDA extension, model, or runner required.

The tests mock kernel imports/eligibility. They DO NOT prove GPU correctness,
package ABI compatibility, speed or LTX-2.5 video quality.
Run manually after reviewing the branch:
    python -m unittest benchmarks.ltx25_flash_v100.test_routes
"""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from shared import ltx25_v100_attention as pilot


class V100AttentionRoutes(unittest.TestCase):
    def tearDown(self):
        pilot.configure("off", 1024)

    def test_disabled_by_default_and_invalid_mode(self):
        pilot.configure("off", 1024)
        self.assertIsNone(pilot.attention_override())
        with self.assertRaises(ValueError):
            pilot.configure("unknown", 1024)
        pilot.configure("off", 1024)
        with self.assertRaises(ValueError):
            pilot.configure("off", -1)

    def test_environment_when_no_cli_override(self):
        with mock.patch.dict(os.environ, {
            pilot.ENV_BACKEND: "zhinianqin",
            pilot.ENV_MIN_TOKENS: "2048",
        }):
            pilot.configure()
            self.assertEqual(pilot.attention_override(), "ltx25_v100_zhinianqin")
            self.assertEqual(pilot.min_tokens(), 2048)
            pilot.configure("1cat", 64)
            self.assertEqual(pilot.backend(), "1cat")
            self.assertEqual(pilot.min_tokens(), 64)

    def test_cpu_or_mask_is_ineligible(self):
        pilot.configure("1cat", 0)
        q = torch.ones((1, 8, 2, 64), dtype=torch.float16)
        self.assertIsNone(pilot.try_attention(
            q, q, q, force_attention="ltx25_v100_1cat",
        ))
        self.assertIsNone(pilot.try_attention(
            q, q, q, force_attention="ltx25_v100_1cat",
            attention_mask=torch.ones((1, 8, 8), dtype=torch.bool),
        ))

    def test_1cat_dense_signature(self):
        pilot.configure("1cat", 0)
        q = torch.ones((1, 8, 2, 64), dtype=torch.float16)
        dense = mock.Mock(return_value=q.clone())
        with (
            mock.patch.object(pilot, "_eligible", return_value=True),
            mock.patch.object(
                pilot, "_load_backend",
                return_value=SimpleNamespace(flash_attn_func=dense),
            ),
        ):
            result = pilot.try_attention(
                q, q, q, force_attention="ltx25_v100_1cat", softmax_scale=0.125,
            )
        self.assertEqual(result.shape, q.shape)
        self.assertEqual(result.dtype, q.dtype)
        self.assertEqual(dense.call_args.args, (q, q, q))
        self.assertEqual(dense.call_args.kwargs["softmax_scale"], 0.125)
        self.assertFalse(dense.call_args.kwargs["causal"])

    def test_zhinianqin_varlen_keywords_and_packing(self):
        pilot.configure("zhinianqin", 0)
        q = torch.ones((2, 8, 2, 128), dtype=torch.float16)
        varlen = mock.Mock(side_effect=lambda **kwargs: kwargs["q"].clone())
        cu = torch.tensor([0, 8, 16], dtype=torch.int32)
        with (
            mock.patch.object(pilot, "_eligible", return_value=True),
            mock.patch.object(pilot, "_cu_seqlens", return_value=cu),
            mock.patch.object(
                pilot, "_load_backend",
                return_value=SimpleNamespace(flash_attn_varlen_func=varlen),
            ),
            mock.patch("torch.cuda.current_device", return_value=0),
        ):
            result = pilot.try_attention(
                q, q, q, force_attention="ltx25_v100_zhinianqin",
            )
        self.assertEqual(result.shape, q.shape)
        self.assertEqual(varlen.call_args.kwargs["q"].shape, (16, 2, 128))
        self.assertIs(varlen.call_args.kwargs["cu_seqlens_q"], cu)
        self.assertIs(varlen.call_args.kwargs["cu_seqlens_k"], cu)
        self.assertEqual(varlen.call_args.kwargs["max_seqlen_q"], 8)
        self.assertEqual(varlen.call_args.kwargs["max_seqlen_k"], 8)

    def test_selected_kernel_failure_is_not_silent_sdpa(self):
        pilot.configure("1cat", 0)
        q = torch.ones((1, 8, 2, 64), dtype=torch.float16)
        with (
            mock.patch.object(pilot, "_eligible", return_value=True),
            mock.patch.object(pilot, "_load_backend", side_effect=ImportError("missing")),
        ):
            with self.assertRaises(ImportError):
                pilot.try_attention(q, q, q, force_attention="ltx25_v100_1cat")


if __name__ == "__main__":
    unittest.main()
