# Copyright (c) 2026 - Experimental WanGP LTX-2.5 SM70 attention adapters.
"""Opt-in, inference-only attention routes for a Tesla V100.

This is deliberately separate from the standard WanGP attention modes:
the LTX Attention class opts in explicitly, and unsupported calls go to SDPA.
Nothing is imported or compiled from either external project until an eligible
call selects it. Neither backend is installed automatically.

Backends:
  1cat       -> flash_attn_v100.flash_attn_func (dense BMHD)
  zhinianqin -> vllm_flash_attn.flash_attn_varlen_func (packed THD)

The zhinianqin vLLM wrapper uses a different argument *order* than the canonical
flash-attn varlen API, so always use keyword arguments.
"""
from __future__ import annotations

from functools import lru_cache
import importlib
import os

import torch


ENV_BACKEND = "WANGP_LTX25_V100_ATTENTION"
ENV_MIN_TOKENS = "WANGP_LTX25_V100_MIN_TOKENS"
MODES = ("off", "1cat", "zhinianqin")
PREFIX = "ltx25_v100_"

_configured_backend: str | None = None
_configured_min_tokens: int | None = None
_announced: set[str] = set()


def _validate_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in MODES:
        raise ValueError(f"Unsupported LTX25 V100 attention backend {mode!r}; choose {', '.join(MODES)}")
    return value


def _validate_min_tokens(value: int | str) -> int:
    count = int(value)
    if count < 0:
        raise ValueError("LTX25 V100 attention min-tokens must be >= 0")
    return count


def configure(backend: str | None = None, min_tokens: int | None = None) -> None:
    """Set process-launch settings; None reads an environment override.

    Call from wgp.py after parsing CLI flags. Importing this module alone
    does not activate either experimental backend.
    """
    global _configured_backend, _configured_min_tokens
    _configured_backend = _validate_mode(
        os.environ.get(ENV_BACKEND, "off") if backend is None else backend
    )
    _configured_min_tokens = _validate_min_tokens(
        os.environ.get(ENV_MIN_TOKENS, "1024") if min_tokens is None else min_tokens
    )


def backend() -> str:
    return _configured_backend if _configured_backend is not None else _validate_mode(
        os.environ.get(ENV_BACKEND, "off")
    )


def min_tokens() -> int:
    return _configured_min_tokens if _configured_min_tokens is not None else _validate_min_tokens(
        os.environ.get(ENV_MIN_TOKENS, "1024")
    )


def attention_override() -> str | None:
    """Only LTX Attention passes this override to shared pay_attention."""
    selected = backend()
    return None if selected == "off" else PREFIX + selected


@lru_cache(maxsize=2)
def _load_backend(selected: str):
    import_name = "flash_attn_v100" if selected == "1cat" else "vllm_flash_attn"
    try:
        module = importlib.import_module(import_name)
    except ImportError as exc:
        raise RuntimeError(
            f"LTX25 V100 attention backend {selected!r} was requested but "
            f"{import_name!r} cannot be imported. Install the corresponding "
            "CUDA extension in the WanGP Python environment (without replacing "
            "WanGP's normal flash_attn), or set --ltx25-v100-attention=off."
        ) from exc
    return module


@lru_cache(maxsize=64)
def _cu_seqlens(device_index: int, batch: int, sequence: int) -> torch.Tensor:
    """Tiny cached device-side int32 prefix lengths; no GPU -> CPU readback."""
    return torch.arange(
        0,
        (batch + 1) * sequence,
        sequence,
        device=torch.device("cuda", device_index),
        dtype=torch.int32,
    )


def _eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int],
    q_lens,
    k_lens,
) -> bool:
    # Pilot is deliberately restricted to unmasked, dense self-attention.
    # Audio/text cross-attention, arbitrary masks, padded batches, GQA, and
    # capture keep the original SDPA behavior until separately validated.
    if attention_mask is not None or dropout_p != 0.0 or causal:
        return False
    if window_size != (-1, -1) or q_lens is not None or k_lens is not None:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.shape[-1] not in (64, 128) or q.shape[1] < min_tokens():
        return False
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        return False
    if torch.cuda.get_device_capability(q.device) != (7, 0):
        return False
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return False
    # Neither experimental package is certified here for graph capture.
    if torch.cuda.is_current_stream_capturing():
        return False
    return True


def try_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    force_attention: str,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    q_lens=None,
    k_lens=None,
) -> torch.Tensor | None:
    """Return the accelerated result, or None for *ineligible* calls.

    Missing dependencies and CUDA kernel errors raise, intentionally: an opted-in
    benchmark must never silently report SDPA numbers as the selected backend.
    """
    if not force_attention.startswith(PREFIX):
        raise ValueError(f"Invalid LTX25 V100 attention route: {force_attention!r}")
    selected = _validate_mode(force_attention[len(PREFIX):])
    if selected == "off":
        return None
    if not _eligible(
        q, k, v, attention_mask=attention_mask, dropout_p=dropout_p,
        causal=causal, window_size=window_size, q_lens=q_lens, k_lens=k_lens,
    ):
        return None

    module = _load_backend(selected)
    batch, tokens, heads, dim = q.shape
    if selected == "1cat":
        # The standalone CUDA package (NOT the whole 1Cat-vLLM server).
        # Its public wrapper handles any BHMD copies itself; benchmark those.
        out = module.flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=softmax_scale,
            causal=False, window_size=(-1, -1),
        )
    else:
        # vllm_flash_attn exports a vLLM-specific varlen signature. Use explicit
        # keywords and pack BMHD -> THD. This reshape may copy for strided heads:
        # its cost belongs in end-to-end benchmarks.
        device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
        cu = _cu_seqlens(device_index, batch, tokens)
        out = module.flash_attn_varlen_func(
            q=q.reshape(batch * tokens, heads, dim),
            k=k.reshape(batch * tokens, heads, dim),
            v=v.reshape(batch * tokens, heads, dim),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
        ).reshape(batch, tokens, heads, dim)

    if out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"LTX25 V100 backend {selected!r} returned {tuple(out.shape)} / {out.dtype}; "
            f"expected {tuple(q.shape)} / {q.dtype}"
        )
    if selected not in _announced:
        print(f"[WanGP][LTX25] Using V100 attention backend {selected} for eligible unmasked FP16 self-attention")
        _announced.add(selected)
    return out
