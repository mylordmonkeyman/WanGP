#!/usr/bin/env python3
"""CPU-only fail-closed contract test; no torch, CUDA, checkpoints or runner."""
import os
import unittest

os.environ.pop("WAN2GP_LTX25_PROFILE", None)
from shared import ltx25_v100_profile as p


class ProfileOff(unittest.TestCase):
    def test_off_is_identity(self):
        self.assertFalse(p._ENABLED)
        with p.scope("ltx25.attention.sdpa_fused"):
            answer = 7
        self.assertEqual(answer, 7)

        def ordinary(x):
            return x + 9

        self.assertIs(p.profiled_convrot_forward(ordinary), ordinary)
        self.assertIs(p.profiled_attention_forward(ordinary), ordinary)
        self.assertEqual(p.profile_one_forward(lambda: 123), 123)
        self.assertFalse(p._TRACE_ACTIVE)
        self.assertFalse(p._EMITTED)


if __name__ == "__main__":
    unittest.main()
