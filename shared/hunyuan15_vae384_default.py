"""Default 384px VAE spatial tiling for native-FP16 Hunyuan Video 1.5.

Only replace the auto-selected 256px/64-frame preset on the native-FP16
T2V/I2V variants. Other models, lower-memory presets, explicit VAE presets,
and a user opt-out retain their original VAE settings.
"""


def resolve_hunyuan15_vae384_default(model_def, base_model_type, vae, original,
                                     *, enabled=True, vae_config=0):
    """Return (VAE tile config, applied). Never mutate the source config."""
    if not enabled or vae_config != 0:
        return original, False
    if (base_model_type not in ("hunyuan_1_5_t2v", "hunyuan_1_5_i2v")
            or not model_def.get("hunyuan15_native_fp16", False)
            or not isinstance(original, dict)):
        return original, False

    # Preserve all other memory tiers and user-selected VAE presets. Requiring
    # the full stock geometry also prevents partially changed tile settings.
    expected = {
        "tile_sample_min_size": 256,
        "tile_latent_min_size": 16,
        "tile_sample_min_tsize": 64,
        "tile_latent_min_tsize": 16,
        "tile_overlap_factor": 0.25,
    }
    if any(original.get(key) != value for key, value in expected.items()):
        return original, False
    if getattr(vae, "ffactor_spatial", None) != 16 or getattr(vae, "ffactor_temporal", None) != 4:
        return original, False

    # 384 / 16 = 24 latent pixels; at overlap 0.25, stride=18 is integral.
    updated = dict(original)
    updated["tile_sample_min_size"] = 384
    updated["tile_latent_min_size"] = 24
    return updated, True
