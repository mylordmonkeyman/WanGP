"""CPU-only regression: user-authorized native-FP16-only MMGP default 1."""
import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.hunyuan15_fp16_profile import resolve_profile

FINETUNES = [
    ("hunyuan15_v100_t2v_480_fp16_experiment.json", "hunyuan_1_5_t2v"),
    ("hunyuan15_v100_i2v_480_step_fp16_experiment.json", "hunyuan_1_5_i2v"),
    ("hunyuan15_v100_t2v_480_lightx2v_4step_fp16_experiment.json", "hunyuan_1_5_t2v"),
]


class NativeFP16ProfileTests(unittest.TestCase):
    def test_explicitly_configured_native_finetunes_default_to_1(self):
        for name, base in FINETUNES:
            with self.subTest(name=name):
                model = json.loads((ROOT/"finetunes"/name).read_text())["model"]
                self.assertTrue(model["hunyuan15_native_fp16"])
                self.assertEqual(model["default_mmgp_profile"], 1)
                self.assertEqual(resolve_profile(-1, -1, "video", base, model, 0), 1)

    def test_explicit_profile0_and_profile3_overrides(self):
        model={"hunyuan15_native_fp16":True, "default_mmgp_profile":1}
        for override in (0, 3):
            self.assertEqual(resolve_profile(override, -1, "video",
                                            "hunyuan_1_5_t2v", model, 0), override)

    def test_cli_profile_override_respected(self):
        model={"hunyuan15_native_fp16":True, "default_mmgp_profile":1}
        self.assertEqual(resolve_profile(-1, 0, "video", "hunyuan_1_5_t2v", model, 0), 0)
        self.assertEqual(resolve_profile(-1, 3, "video", "hunyuan_1_5_t2v", model, 0), 3)

    def test_bf16_hunyuan_unchanged(self):
        model={"hunyuan15_native_fp16":False, "default_mmgp_profile":1}
        self.assertEqual(resolve_profile(-1, -1, "video", "hunyuan_1_5_t2v", model, 0), 0)

    def test_untagged_hunyuan_and_other_models_unchanged(self):
        for base, md in [("hunyuan_1_5_t2v", {}),
                         ("ltx_video", {"hunyuan15_native_fp16":True, "default_mmgp_profile":1})]:
            self.assertEqual(resolve_profile(-1, -1, "video", base, md, 3), 3)

    def test_nonvideo_default_unchanged(self):
        model={"hunyuan15_native_fp16":True,"default_mmgp_profile":1}
        self.assertEqual(resolve_profile(-1, -1, "image", "hunyuan_1_5_t2v", model, 4), 4)

    def test_malformed_profile_rejected(self):
        model={"hunyuan15_native_fp16":True, "default_mmgp_profile":99}
        with self.assertRaises(ValueError):
            resolve_profile(-1, -1, "video", "hunyuan_1_5_t2v", model, 0)

    def test_model_load_and_hot_cache_use_same_model_aware_policy(self):
        tree=ast.parse((ROOT/"wgp.py").read_text())
        funcs={node.name:node for node in tree.body
               if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertIn("compute_profile", funcs)
        uses=[(fn.name, node) for fn in funcs.values()
              for node in ast.walk(fn)
              if isinstance(node, ast.Call) and
              isinstance(node.func,ast.Name) and node.func.id=="compute_profile"]
        for name in ("load_models", "generate_media"):
            matches=[node for fn,node in uses if fn==name]
            self.assertGreaterEqual(len(matches), 1, f"missing {name} model-aware profile")
            self.assertTrue(all(len(node.args)>=3 for node in matches),
                            f"{name} omitted model_type in memory-profile resolution")


if __name__=="__main__":
    unittest.main()
