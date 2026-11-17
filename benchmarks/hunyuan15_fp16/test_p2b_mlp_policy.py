"""CPU-only policy tests for unpromoted P2B MLP chunk-size sweep."""
import os
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.hunyuan15_p2b_mlp_policy import native_mlp_divide


class P2BChunkPolicyTests(unittest.TestCase):
    def test_stock_default_is_four(self):
        self.assertEqual(native_mlp_divide(True, {}),4)
        self.assertEqual(native_mlp_divide(False, {}),4)

    def test_explicit_choices_for_native_only(self):
        for value in ("4","2","1"):
            with self.subTest(value=value):
                self.assertEqual(native_mlp_divide(True,{"HUNYUAN15_P2B_MLP_DIVIDE":value}),int(value))
                self.assertEqual(native_mlp_divide(False,{"HUNYUAN15_P2B_MLP_DIVIDE":value}),4)

    def test_invalid_rejected_for_native(self):
        for value in ("0","3","5","2.0","foo"," 2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native_mlp_divide(True,{"HUNYUAN15_P2B_MLP_DIVIDE":value})

    def test_both_image_mlp_paths_are_gated(self):
        source=(ROOT/"models/hyvideo/modules/models.py").read_text()
        self.assertEqual(source.count('os.environ.get("HUNYUAN15_P2B_MLP_DIVIDE")'),2)
        self.assertEqual(source.count("self.img_mlp.apply_(img_modulated, divide=native_mlp_divide(True))"),2)
        self.assertEqual(source.count("self.img_mlp.apply_(img_modulated)"),2)


if __name__=="__main__":unittest.main()
