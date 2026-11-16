"""CPU-only smoke checks for the opt-in Hunyuan 1.5 FP16 investigation."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "hunyuan15_checkpoint_audit.py"
spec = importlib.util.spec_from_file_location("hunyuan15_checkpoint_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CheckpointHeaderAuditTest(unittest.TestCase):
    def write_header(self, directory, filename, header):
        path = Path(directory) / filename
        raw = json.dumps(header).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 16)
        return path

    def test_dtype_difference_does_not_obscure_key_match(self):
        with tempfile.TemporaryDirectory() as directory:
            fp16 = self.write_header(directory, "comfy.safetensors", {
                "diffusion_model.double_blocks.0.weight": {
                    "dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]
                }
            })
            bf16 = self.write_header(directory, "wangp.safetensors", {
                "double_blocks.0.weight": {
                    "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]
                }
            })
            result = audit.compare(audit.read_header(fp16), audit.read_header(bf16))
            self.assertTrue(result["keys_and_shapes_match"])

    def test_shape_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            fp32 = self.write_header(directory, "official.safetensors", {
                "model.diffusion_model.linear.weight": {
                    "dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]
                }
            })
            bf16 = self.write_header(directory, "wangp.safetensors", {
                "linear.weight": {
                    "dtype": "BF16", "shape": [4, 2], "data_offsets": [0, 16]
                }
            })
            result = audit.compare(audit.read_header(fp32), audit.read_header(bf16))
            self.assertEqual(result["shape_mismatches"], 1)
            self.assertFalse(result["keys_and_shapes_match"])

    def test_truncated_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            truncated = Path(directory) / "bad.safetensors"
            truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaises(ValueError):
                audit.read_header(truncated)


class DtypePathStaticTest(unittest.TestCase):
    def test_changed_sources_compile(self):
        for name in (
            "models/hyvideo/hunyuan.py",
            "models/hyvideo/modules/models.py",
            "models/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py",
            "scripts/hunyuan15_checkpoint_audit.py",
        ):
            path = ROOT / name
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_experimental_model_definitions(self):
        for filename, expected_architecture in (
            ("hunyuan15_v100_i2v_480_step_fp16_experiment.json", "hunyuan_1_5_i2v"),
            ("hunyuan15_v100_t2v_480_fp16_experiment.json", "hunyuan_1_5_t2v"),
        ):
            data = json.loads((ROOT / "finetunes" / filename).read_text(encoding="utf-8"))
            model = data["model"]
            self.assertEqual(model["architecture"], expected_architecture)
            self.assertEqual(model["dtype"], "fp16")
            self.assertTrue(model["hunyuan15_native_fp16"])
            self.assertFalse(model["auto_quantize"])
            self.assertEqual(len(model["URLs"]), 1)
            self.assertIn("Comfy-Org/Hunyuan-Video-1.5", model["URLs"][0])
            self.assertTrue(model["URLs"][0].endswith("_fp16.safetensors"))

    def test_selected_model_precision_route(self):
        code = (ROOT / "wgp.py").read_text(encoding="utf-8")
        compile(code, "wgp.py", "exec")
        start = code.index("def get_transformer_dtype(")
        section = code[start:code.index("def get_settings_file_name(", start)]
        self.assertIn('get_model_def(model_type)', section)
        self.assertIn('get("hunyuan15_native_fp16", False)', section)
        self.assertIn("return torch.float16", section)

    def test_native_fp16_is_opt_in(self):
        loader = (ROOT / "models/hyvideo/hunyuan.py").read_text(encoding="utf-8")
        self.assertIn('model_def.get("hunyuan15_native_fp16", False)', loader)
        self.assertIn('requested_dtype != torch.float16', loader)
        blocks = (ROOT / "models/hyvideo/modules/models.py").read_text(encoding="utf-8")
        self.assertIn('getattr(self, "native_fp16", False)', blocks)
        pipe = (ROOT / "models/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py").read_text(encoding="utf-8")
        self.assertIn('self.target_dtype = torch.float16 if getattr(transformer, "native_fp16", False)', pipe)


if __name__ == "__main__":
    unittest.main()
