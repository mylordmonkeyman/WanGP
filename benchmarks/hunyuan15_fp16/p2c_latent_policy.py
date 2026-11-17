"""Pure P2C latent-save policy; the live pipeline is unchanged unless opted in."""
from __future__ import annotations


def latent_action(native_hunyuan15_fp16: bool, load_latent: bool, environ):
    """Return (load|save|skip, filename). Never skip an explicit load request.

    The defaults of BF16 Hunyuan, other models, and non-opted FP16 are
    unchanged. P2C_ENABLE alone skips an unnecessary per-generation debug
    write only for native Hunyuan1.5 FP16.
    """
    opted_in = native_hunyuan15_fp16 and environ.get("HUNYUAN15_P2C_ENABLE", "0") == "1"
    if not opted_in:
        return ("load" if load_latent else "save"), "latent.pt"
    target = environ.get("HUNYUAN15_P2C_LATENT_PATH", "").strip() or "latent.pt"
    if load_latent:
        return "load", target
    if environ.get("HUNYUAN15_P2C_SAVE_DEBUG_LATENTS", "0") == "1":
        return "save", target
    return "skip", None
