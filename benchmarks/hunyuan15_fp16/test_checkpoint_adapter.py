"""CPU-only synthetic tests for Comfy QKV layout adapter (no torch dependency)."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
module_path = ROOT / "models" / "hyvideo" / "checkpoint_formats.py"
spec = importlib.util.spec_from_file_location("hunyuan15_checkpoint_formats", module_path)
formats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formats)


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), 1)
        self.ndim = 2

    def chunk(self, n, dim=0):
        if n != 3 or dim != 0:
            raise ValueError("test only supports QKV output-axis splitting")
        if len(self.rows) % 3:
            raise ValueError("bad shape")
        width = len(self.rows) // 3
        return [FakeTensor(self.rows[j * width:(j + 1) * width]) for j in range(3)]


class CheckpointAdapterTest(unittest.TestCase):
    def test_split_stream_qkv_preserve_refiner(self):
        fused = FakeTensor([10, 11, 20, 21, 30, 31])
        refiner = FakeTensor([100, 200, 300])
        source = {
            "double_blocks.0.img_attn_qkv.weight": fused,
            "txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight": refiner,
            "time_r_in.mlp.0.weight": FakeTensor([2]),
        }
        out = formats.unfuse_hunyuan15_qkv(source)
        self.assertEqual(out["double_blocks.0.img_attn_q.weight"].rows, [10, 11])
        self.assertEqual(out["double_blocks.0.img_attn_k.weight"].rows, [20, 21])
        self.assertEqual(out["double_blocks.0.img_attn_v.weight"].rows, [30, 31])
        self.assertIs(out["txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight"], refiner)
        self.assertIs(out["time_r_in.mlp.0.weight"], source["time_r_in.mlp.0.weight"])
        self.assertNotIn("double_blocks.0.img_attn_qkv.weight", out)
        self.assertEqual(len(source), 3)  # no mutation of source mapping

    def test_native_unfused_keys_pass_through(self):
        sd = {"double_blocks.0.img_attn_q.weight": FakeTensor([1])}
        out = formats.unfuse_hunyuan15_qkv(sd)
        self.assertIs(out["double_blocks.0.img_attn_q.weight"], sd["double_blocks.0.img_attn_q.weight"])

    def test_conflicting_target_rejected(self):
        sd = {
            "double_blocks.0.txt_attn_qkv.bias": FakeTensor([1, 2, 3]),
            "double_blocks.0.txt_attn_q.bias": FakeTensor([99]),
        }
        with self.assertRaises(ValueError):
            formats.unfuse_hunyuan15_qkv(sd)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            formats.unfuse_hunyuan15_qkv({
                "double_blocks.0.img_attn_qkv.weight": FakeTensor([1, 2])
            })


if __name__ == "__main__":
    unittest.main()
