"""Small model-specific WanGP MMGP profile policy.

Profile 1 is the user-authorized *default* for explicit Hunyuan 1.5
native-FP16 finetunes. Never change unrelated BF16 models. Explicit
per-generation or CLI memory-profile overrides always win.
"""


def resolve_profile(override_profile, force_profile_no, output_type,
                    base_model_type, model_def, global_default):
    if override_profile != -1:
        return override_profile
    if force_profile_no >= 0:
        return force_profile_no
    if (output_type == "video"
            and base_model_type in ("hunyuan_1_5_t2v", "hunyuan_1_5_i2v")
            and model_def.get("hunyuan15_native_fp16", False)):
        explicit_default = model_def.get("default_mmgp_profile")
        if explicit_default is not None:
            value = int(explicit_default)
            if value not in (0, 1, 2, 3, 4, 5):
                raise ValueError("Unsupported native Hunyuan 1.5 default MMGP profile")
            return value
    return global_default
