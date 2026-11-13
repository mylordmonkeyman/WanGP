# Default SM70 INT8 ConvRot acceleration in this WanGP fork

**Scope:** eligible qint8 ConvRot linear layers on an NVIDIA V100 (SM70).
This is a **source-code default** on `perf/ltx25-convrot-sm70` for
previously validated matrix-shape families. It does not automatically deploy
changes to an existing server checkout or affect other Git branches.

## Defaults and overrides

| Setting | Value with no environment override | Override |
|---|---|---|
| `WAN2GP_LTX25_CUBLAS_E_MODE` | `fp32rot` (set at WanGP startup) | `off` disables the adapter entirely; explicit `fp32rot` is strict; `auto` is normalized to the validated `fp32rot` selector |
| `WAN2GP_LTX25_E_CAST` | `vector4` | `vector8` opts into exact packed-eight conversion (pilot); `torch` restores PyTorch INT8-to-FP16 conversion within Candidate E |
| `WAN2GP_LTX25_HADAMARD` | `radix4` | `torch` restores the historical FP32 H256 rotation; `flat_mm` is a separately tested torch.mm alternative |
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
The promoted radix-4 FP32 rotation uses the same H4-Kronecker transform as
the previous torch FP32 implementation, but changes accumulation order and
therefore is **not pixel-exact**. The previous rotation is available via
`WAN2GP_LTX25_HADAMARD=torch` without disabling the acceleration adapter.

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

To revert only the Hadamard change while preserving the validated V100 ConvRot
cuBLAS, vector4 cast, and bounded scratch optimizations:

```bash
export WAN2GP_LTX25_HADAMARD=torch
```

For full ConvRot rollback, before launching WanGP, set:

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

### Validated no-env default-on acceptance

The corrected no-env default passed the complete V100 workflow at
https://github.com/mylordmonkeyman/WanGP/actions/runs/43476591579 .
It generated the same 49 decoded frames as the validated Candidate E
reference at **96.1 seconds**, with **16,368** optimized calls,
**4,434** device-side FP32 fallback rows and **16,360** scratch leases.
Workspace residency peaked at ~95.5 MiB, no persistent FP16 weight copy
was cached, and the native extension was resolved without an explicit
library path. The prior standalone `auto` mode is never used for
default selection, because its two full-video equality gates failed.

The exact tested source was preserved in:
`archive/ltx25-convrot-sm70-default-promoted-validated-20261113`
at `54a888d600ce896e0b40ff00c75fd8ee46f0249d`.

**Deployment caveat:** GitHub commits and V100 Actions tests do not
modify an existing separate installation of WanGP. Update the actual
server checkout to this branch (preserving local changes) and restart
its WanGP process to activate these source-code defaults. An existing
long-running Python process continues using previously imported modules
until restarted.

### Strict-repeat verification

The second no-env default-on generation also **PASSED** all the
reference-count and decoded-video gates:
https://github.com/mylordmonkeyman/WanGP/actions/runs/43477341297 .
It completed in **95.9 s**, with exactly **16,368** optimized linears,
**4,434** FP32 fallback rows, **16,360** workspace leases, and
**49/49 identical decoded frames**, matching the first 96.1 s no-env
default result. The strict-reference snapshot is
`archive/ltx25-convrot-sm70-default-strict-validated-20261113`
at commit `ae6051e99e173bd81375065ebf30f135b0f87881`.

### Radix-4 promotion: user-reviewed video, numerical caveat

On November 13, 2026, the user visually reviewed the earlier radix-4 T2V
experiment and their own longer I2V comparison, reporting the outputs looked
essentially identical. The user specifically requested default promotion and
did **not** want near-miss quality decisions made from PSNR or exact-frame
mismatch without visual inspection.

The user-provided I2V comparison had 241 frames and reported approximately
36.73 dB mean per-frame PSNR (lowest 30.66 dB); original run wall times were
697 s for the historical torch H256 and 567 s for radix-4, i.e. 130 s (18.7%)
less time. These are wall-clock measurements, **not isolated GPU-inference
benchmarks**: model loading, downloads, CUDA warmup and run order might affect
the result. The older synthetic full-adapter BF16/FP16 preflight passed and the
early 49-frame T2V test produced nonidentical decoded pixels. This is not
universal quality equivalence and does not establish that every seed, LoRA,
resolution or model supports the same speedup.

The default is now `WAN2GP_LTX25_HADAMARD=radix4` for eligible SM70
Candidate E linears; `torch` is an explicit rollback. The test workflow keeps
**strict decoded-pixel equality** for historical torch-based comparisons but
treats the radix-4 decoded mismatch count as a diagnostic. For radix-4, the
generation test must still pass numerical preflights, create a complete
decodable video, use the expected optimized linears, and respect bounded
workspace/no-persistent-weight-copy constraints. Pixel mismatch must not be
misrepresented as either exact equality or an automatic quality regression.

### Opt-in packed-vector8 INT8-to-FP16 cast pilot (not default)

A synthetic Tesla V100 SM70 five-shape CUDA-event sweep compared the current
validated vector4 weight-conversion kernel with aligned packed8 and packed16
loads/stores. The packed8/256 variant converted the *same signed INT8 weights*
to the *same FP16 bits* and improved isolated conversion throughput by
approximately 1.29× (2048×2048), 1.43× (2048×4096), 1.47× (4096×4096),
and 1.51× (both 64 Mi-element shapes). Packed16 was slower. The best launch
geometry varied marginally between 128/256/512 threads; 256 is the practical
candidate. Results: GitHub Actions run
https://github.com/mylordmonkeyman/WanGP/actions/runs/43509629041 .

Experimental invocation, on a freshly started WanGP process with the revised
native extension:

```bash
export WAN2GP_LTX25_E_CAST=vector8
```

No persistent FP16 master weight is introduced. The exact original INT8
checkpoint remains authoritative, the cast output is transient, and the
BF16 activation path outside eligible linears is unchanged. The V100
complete-adapter BF16/FP16 numerical preflight passed, and a 49-frame
1024×576 torch-H256 video matched all 49 decoded frames of the strict
reference (https://github.com/mylordmonkeyman/WanGP/actions/runs/43510285511).
The isolated conversion speedup **does not guarantee any full-generation
gain**: a single vector8 video took 98.4 seconds versus 95.8 seconds in
an earlier non-paired vector4 run, so generation-order/clock/host variance
is unresolved. An independent A–B–B–A workflow
`ltx25-cast8-abba.yml` is provided to measure end-to-end gains.
Only promote to default if controlled generation benchmarking supports it.

Restore validated weight conversion without disabling radix-4:

```bash
export WAN2GP_LTX25_E_CAST=vector4
```
