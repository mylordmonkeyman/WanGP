"""Isolated opt-in Hunyuan 1.5 native-FP16 P2B image MLP chunk policy.

Stock MMDoubleStreamBlock applies 4 serial chunks to the image MLP.
The opt-in only applies to blocks explicitly marked native_fp16 and is
not enabled as a default for any Hunyuan or BF16 model.
"""
import os


def native_mlp_divide(native_fp16: bool, env=None) -> int:
    if not native_fp16:
        return 4
    if env is None:
        env = os.environ
    requested = env.get("HUNYUAN15_P2B_MLP_DIVIDE")
    if requested is None or requested == "":
        return 4
    if requested not in ("4", "2", "1"):
        raise ValueError("HUNYUAN15_P2B_MLP_DIVIDE must be 4, 2, or 1")
    return int(requested)
