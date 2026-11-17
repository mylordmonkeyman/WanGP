"""Opt-in VAE geometry for native-FP16 Hunyuan 1.5 P3C experiments.

This policy cannot change ordinary model defaults.  The stock 256px arm
is left completely unmodified; 384 and 512px are valid trial overrides.
"""


def resolve_custom_vae_tiles(model_def, base_model_type, vae, original,
                             requested_spatial, *, output_width, output_height,
                             video_length):
    if not requested_spatial:
        return original, None
    if (base_model_type not in ("hunyuan_1_5_t2v", "hunyuan_1_5_i2v") or
            not model_def.get("hunyuan15_native_fp16", False)):
        return original, None
    if not isinstance(original, dict):
        raise ValueError("P3C requires the existing Hunyuan VAE tiling dictionary")
    try:
        spatial = int(requested_spatial)
    except (TypeError, ValueError):
        raise ValueError("P3C spatial tile must be 384 or 512") from None
    if str(spatial) != str(requested_spatial) or spatial not in (384, 512):
        raise ValueError("P3C spatial tile must be exactly 384 or 512")
    s_factor = int(vae.ffactor_spatial)
    t_factor = int(vae.ffactor_temporal)
    if s_factor != 16 or t_factor != 4:
        raise ValueError("P3C unexpected VAE compression factors")
    if original.get("tile_sample_min_size") != 256 or original.get("tile_sample_min_tsize") != 64:
        raise ValueError("P3C expected the unchanged auto/preset-1 256px/64-frame baseline")
    if original.get("tile_overlap_factor") != 0.25:
        raise ValueError("P3C expected overlap factor 0.25")
    if output_width != 832 or output_height != 480 or video_length != 33:
        raise ValueError("P3C geometry restricted to the fixed 832x480x33 test")
    latent_tile = spatial // s_factor
    temporal_tile = 64 // t_factor
    overlap = 0.25
    if (spatial % s_factor or 64 % t_factor or
            not float(latent_tile * overlap).is_integer() or
            not float(temporal_tile * overlap).is_integer()):
        raise ValueError("P3C tile geometry violates integer-overlap requirement")
    stride = int(latent_tile * (1 - overlap))
    latent_h = output_height // s_factor
    latent_w = output_width // s_factor
    count_h = len(range(0, latent_h, stride))
    count_w = len(range(0, latent_w, stride))
    candidate = dict(original)
    candidate.update({
        "tile_sample_min_size": spatial,
        "tile_latent_min_size": latent_tile,
        "tile_sample_min_tsize": 64,
        "tile_latent_min_tsize": temporal_tile,
        "tile_overlap_factor": overlap,
    })
    info = {
        "spatial_tile_px": spatial, "spatial_latent_tile": latent_tile,
        "spatial_stride_latents": stride,
        "spatial_decoder_positions_predicted": count_h * count_w,
        "temporal_latent_tile": temporal_tile,
        "temporal_decoder_tiles_predicted": 1,
        "note": "Predicted from source loops, not instrumented runtime calls",
    }
    return candidate, info
