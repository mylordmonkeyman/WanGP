# Default SM70 INT8 ConvRot acceleration in this WanGP fork

**Scope:** eligible qint8 ConvRot linear layers on an NVIDIA V100 (SM70).
This is a **source-code default** on `perf/ltx25-convrot-sm70` for
previously validated matrix-shape families. It does not automatically deploy
changes to an existing server checkout or affect other Git branches.

## Defaults and overrides

| Setting | Value with no environment override | Override |
|---|---|---|
| `WAN2GP_LTX25_CUBLAS_E_MODE` | `fp32rot` (set at WanGP startup) | `off` disables the adapter entirely; explicit `fp32rot` is strict; `auto` is normalized to the validated `fp32rot` selector |
| `WAN2GP_LTX25_E_CAST` | `vector4` | `torch` restores PyTorch INT8-to-FP16 conversion within Candidate E |
| `WAN2GP_LTX25_E_WORKSPACE` | `1` | `0` disables the reusable activation/output workspace |
| `LTX25_E_WORKSPACE_MAX_MIB` | `96` | Clamp persistent workspace limit up to 256 MiB |
| `LTX25_CUBLAS_E_LIB` | Unset | Path to a precompiled compatible SM70 cuBLAS extension |
| `WAN2GP_LTX25_E_AUTO_BUILD` | `1` | `0` disables local native compilation (auto will fall back if no prebuilt library) |

In automatic mode, the adapter first checks eligible N×K shape, rotation group
256, supported dtype, contiguous original INT8 weight, FP32 row scales,
on-device weights/activations, non-compiled execution, available VRAM cap, and
**exact SM70 device capability**. It skips ineligible linears and uses WanGP's
preexisting dispatcher, including CPU, fusion, unsupported layer shapes,
CUDA graphs, and other GPUs. No global float16 cast is performed: the
preexisting BF16 computation remains in place outside eligible linears.

## Automatic native library

When the first eligible linear is encountered:

1. If `LTX25_CUBLAS_E_LIB` is set, load exactly that file.
2. Otherwise, find `nvcc` locally, compile the repository's
   `benchmarks/ltx25_convrot/ltx25_cublas_e.cu` for `sm_70` with cuBLAS,
   and cache the shared object under `~/.cache/wan2gp/sm70-convrot/`
   (or under `XDG_CACHE_HOME`).
3. Cache key includes the CUDA source digest and PyTorch CUDA version.
   A filesystem lock and atomic rename avoid concurrent partial libraries.
4. When the default is **implicitly** selected, missing compiler/source/
   library loads log once and fall back to WanGP's original linear **before**
   any Candidate E operation is enqueued. If a caller explicitly requested
   `fp32rot`, failures remain strict to aid diagnosis.

There is no download, model conversion, or persistent second weight set.
Original checkpoint INT8 weights and FP32 scales remain under MMGP control.
The FP16 weight conversion is temporary **per linear**, and the bounded
default workspace contains only activation, accumulator and row metadata
buffers. On nondefault streams or oversized/reentrant operations, it uses
ephemeral buffers. No CUDA synchronization is performed inside the hot
linear dispatch.


### Why the implicit default selects `fp32rot`, not `auto`

The first direct no-environment `auto` attempt passed CUDA and generation
but **failed** exact video equality on all 49 frames and executed 14,736
optimized linear calls instead of the 16,368 from the validated path.
The failure reproduced. A same-config run with the concrete
`WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot` selection restored **16,368**
calls, the 4,434 FP32 exceptional rows, and exact 49-frame equality
at 95.7 seconds. See:

- Failed auto run: https://github.com/mylordmonkeyman/WanGP/actions/runs/43473397869
- Failed repeat: https://github.com/mylordmonkeyman/WanGP/actions/runs/43474671547
- Passing controlled explicit-mode run:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43475491051

Therefore, `wgp.py` and the ConvRot qtype module now insert the **proven
`fp32rot` selector into the process environment at startup** when there
is no user override. Explicit `off` and empty-string opt-outs remain
honored. The implicit-default marker allows safe fallback when native
compilation is missing, without suppressing errors from an explicit
`fp32rot` request. This selection is empirically grounded; the precise
cause of the earlier differing `auto` execution graph is not yet proven.

## Reverting

Before launching WanGP, set:

```bash
export WAN2GP_LTX25_CUBLAS_E_MODE=off
```

That restores the previous WanGP dispatch for ConvRot. Alternatively,
keep SM70 acceleration but use old staging choices:

```bash
export WAN2GP_LTX25_E_CAST=torch
export WAN2GP_LTX25_E_WORKSPACE=0
```

Unset the variables to restore the new defaults.

## Evidence and limitations

The optimized vector4/scratch combination passed the original Candidate E
numerical preflights, eight BF16/FP16 cast-output comparisons, bounded
scratch-lifetime tests, and full-video exact decoded-frame equality for
**all 49 frames** against the known-good Candidate E clip. A fresh
B–A–B–A timing screen found **95.6/96.3 s** for the combined candidate
versus **96.6/97.5 s** for the previous PyTorch-cast/no-scratch E path.

Full-video quality remains **not exhaustively validated** across many prompts,
seeds, LoRAs, MMGP memory profiles, and hardware/software environments.
Candidate E changes arithmetic from the original BF16 R0, even when it
matches earlier optimized video output. The source default is restricted to
validated shape families on the V100 and is reversible. The code does not
change other attention backends, default model precision, or checkpoint files.

Evidence and command-line test details:
`benchmarks/ltx25_convrot/GPU_RANGE_GUARD.md`.
