# Hunyuan 1.5 checkpoint layout adapters for experimental WanGP-native FP16.
# Do not apply to LoRA tensors or quantized checkpoints.

def unfuse_hunyuan15_qkv(state_dict):
    """Unpack Comfy-Org image/text QKV tensors into WanGP's split layout.

    Comfy combines img_attn_q/k/v and txt_attn_q/k/v along output dimension 0.
    The individual token refiner's self_attn_qkv stays *fused* in WanGP too,
    and must NOT be changed. Tensor.chunk creates views without copying
    multi-GB checkpoint data. Works for standard FP16 and original FP32 keys.
    """
    expanded = {}
    unpacked = 0
    for name, tensor in state_dict.items():
        if not (".img_attn_qkv." in name or ".txt_attn_qkv." in name):
            if name in expanded:
                raise ValueError(f"Duplicate Hunyuan key: {name}")
            expanded[name] = tensor
            continue

        if tensor.ndim < 1 or tensor.shape[0] % 3:
            raise ValueError(f"Bad fused QKV tensor shape: {name} {tuple(tensor.shape)}")
        names = tuple(
            name.replace("_attn_qkv.", f"_attn_{part}.")
            for part in ("q", "k", "v")
        )
        if len(set(names)) != 3 or any(key in state_dict or key in expanded for key in names):
            raise ValueError(f"Ambiguous fused QKV mapping: {name}")
        parts = tensor.chunk(3, dim=0)
        if len(parts) != 3 or any(part.shape[0] * 3 != tensor.shape[0] for part in parts):
            raise ValueError(f"QKV split failed for {name}")
        for key, part in zip(names, parts):
            expanded[key] = part
        unpacked += 1
    if unpacked:
        print(f"Hunyuan 1.5 experimental loader: unpacked {unpacked} Comfy QKV tensors")
    return expanded
