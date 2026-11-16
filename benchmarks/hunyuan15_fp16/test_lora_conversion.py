"""CPU-only, no torch required, tests for Hunyuan 1.5 LoRA key conversion."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from hunyuan15_lora_convert import plan, target_modules


def m(shape):
    return {"dtype": "BF16", "shape": list(shape)}


class HunyuanLoraConversionTests(unittest.TestCase):
    def reference(self):
        d = {}
        for k in "qkv":
            d[f"double_blocks.0.img_attn_{k}.weight"] = m((2, 4))
            d[f"double_blocks.0.img_attn_{k}.bias"] = m((2,))
        d["double_blocks.0.img_attn_proj.weight"] = m((2, 2))
        d["double_blocks.0.img_attn_proj.bias"] = m((2,))
        d["double_blocks.0.img_mlp.fc1.weight"] = m((4, 2))
        d["double_blocks.0.img_mlp.fc1.bias"] = m((4,))
        d["double_blocks.0.img_mod.linear.weight"] = m((6, 2))
        d["double_blocks.0.img_mod.linear.bias"] = m((6,))
        d["txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight"] = m((6, 2))
        d["txt_in.individual_token_refiner.blocks.0.self_attn_qkv.bias"] = m((6,))
        return d

    def source(self):
        return {
            "double_blocks.0.img_attn.qkv.lora_down.weight": m((1, 4)),
            "double_blocks.0.img_attn.qkv.lora_up.weight": m((6, 1)),
            "double_blocks.0.img_attn.qkv.diff_b": m((6,)),
            "double_blocks.0.img_attn.proj.lora_down.weight": m((1, 2)),
            "double_blocks.0.img_attn.proj.lora_up.weight": m((2, 1)),
            "double_blocks.0.img_attn.proj.diff_b": m((2,)),
            "double_blocks.0.img_mlp.0.diff_b": m((4,)),
            "double_blocks.0.img_mlp.0.lora_down.weight": m((1, 2)),
            "double_blocks.0.img_mlp.0.lora_up.weight": m((4, 1)),
            "double_blocks.0.img_mod.lin.diff_b": m((6,)),
            "double_blocks.0.img_mod.lin.lora_down.weight": m((1, 2)),
            "double_blocks.0.img_mod.lin.lora_up.weight": m((6, 1)),
            "txt_in.individual_token_refiner.blocks.0.self_attn.qkv.lora_down.weight": m((1, 2)),
            "txt_in.individual_token_refiner.blocks.0.self_attn.qkv.lora_up.weight": m((6, 1)),
        }

    def test_qkv_up_bias_split_down_replicated(self):
        mapping, shapes, info = plan(self.source(), self.reference())
        self.assertEqual(mapping["double_blocks.0.img_attn.qkv.lora_down.weight"][0], tuple(
            f"double_blocks.0.img_attn_{x}.lora_down.weight" for x in "qkv"
        ))
        self.assertEqual(shapes["double_blocks.0.img_attn_q.lora_up.weight"], (2, 1))
        self.assertEqual(shapes["double_blocks.0.img_attn_v.diff_b"], (2,))
        self.assertEqual(shapes["double_blocks.0.img_attn_k.lora_down.weight"], (1, 4))
        self.assertEqual(info["input_tensors"], 14)
        self.assertEqual(info["output_tensors"], 20)

    def test_non_qkv_names_and_refiner_remains_fused(self):
        self.assertEqual(target_modules("double_blocks.1.txt_attn.proj"), ("double_blocks.1.txt_attn_proj",))
        self.assertEqual(target_modules("double_blocks.1.txt_attn.qkv"), (
            "double_blocks.1.txt_attn_q","double_blocks.1.txt_attn_k","double_blocks.1.txt_attn_v",
        ))
        self.assertEqual(target_modules("double_blocks.1.txt_mlp.2"), ("double_blocks.1.txt_mlp.fc2",))
        self.assertEqual(target_modules("double_blocks.1.txt_mod.lin"), ("double_blocks.1.txt_mod.linear",))
        self.assertEqual(target_modules("txt_in.individual_token_refiner.blocks.0.mlp.0"), (
            "txt_in.individual_token_refiner.blocks.0.mlp.fc1",))
        self.assertEqual(target_modules("txt_in.individual_token_refiner.blocks.0.self_attn.qkv"), (
            "txt_in.individual_token_refiner.blocks.0.self_attn_qkv",))
        self.assertEqual(target_modules("byt5_in.layernorm"), ("byt5_in.layernorm",))
        self.assertEqual(target_modules("time_in.in_layer"), ("time_in.mlp.0",))
        self.assertEqual(target_modules("time_in.out_layer"), ("time_in.mlp.2",))
        self.assertEqual(target_modules("time_r_in.in_layer"), ("time_r_in.mlp.0",))
        self.assertEqual(target_modules("vector_in.in_layer"), ("vector_in.in_layer",))

    def test_reject_missing_model_target(self):
        source = self.source()
        model = self.reference()
        del model["double_blocks.0.img_attn_k.weight"]
        with self.assertRaisesRegex(ValueError, "Missing target"):
            plan(source, model)

    def test_reject_wrong_qkv_shape(self):
        source = self.source()
        source["double_blocks.0.img_attn.qkv.lora_up.weight"] = m((7, 1))
        with self.assertRaisesRegex(ValueError, "cannot split"):
            plan(source, self.reference())

    def test_reject_incomplete_lora_pair(self):
        source = self.source()
        del source["double_blocks.0.img_attn.proj.lora_up.weight"]
        with self.assertRaisesRegex(ValueError, "Incomplete LoRA pair"):
            plan(source, self.reference())

    def test_reject_unknown_extra(self):
        source = self.source()
        source["unrelated.dummy.data"] = m((4,))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            plan(source, self.reference())


if __name__ == "__main__":
    unittest.main()
