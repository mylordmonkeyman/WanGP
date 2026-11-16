# Cold-forward MMGP optimization evidence — 2026-11-15

Work branch: `perf/ltx25-v100-mmgp-transfer-20261115`. Archived tiled work: `archive/ltx25-v100-tiled-epilogue-20261115` at `8788b047`. Stable `main` remains `523cb6f8` and the tiled epilogue is still default OFF.

## First vs second transformer forward

[Verified 73f T2V/I2V run 43763757911](https://github.com/mylordmonkeyman/WanGP/actions/runs/43763757911): 1024×576, 8 steps, same seed/checkpoint/attention/MMGP profile and default Candidate E, original I2V bar image. Both decoded video/audio match pinned stable hashes exactly. Temporary runner instrumentation restored production source byte-for-byte.

| T2V measurement | Stage 1 first | Stage 1 second | Stage 2 first | Stage 2 second |
| --- | ---: | ---: | ---: | ---: |
| Transformer CUDA span (s) | 4.002 | 2.516 | 12.216 | 6.316 |
| MMGP block-loader CPU sum (s) | 1.295 | 0.455 | **5.697** | **0.383** |
| GPU H2D event-duration sum (s) | 1.717 | 1.580 | 1.711 | 1.576 |
| H2D bytes (GiB) | 18.00 | 16.56 | 18.00 | 16.56 |
| GPU consumer-wait event sum (ms) | 27.7 | 0.65 | 22.9 | 0.64 |

I2V independently reproduced Stage 2 CPU loader 5.663 s first / 0.380 s second, similar H2D GPU durations, and ~26 ms first GPU consumer waits.

**Interpretation:** The cold Stage 2 discrepancy appears primarily related to CPU-side MMGP block loading (allocation, parameter installation, cache management or other host work), not increased GPU copy duration alone. These CPU/GPU intervals overlap and profiling perturbs behavior: none is a verified recoverable wall saving.

## Gated next steps

- First-vs-warm CUDA traces: 1st and 2nd forward of each stage, both T2V and I2V, classify actual GEMM, attention, cast, H2D and other kernel work independently.
- Decompose MMGP block-loader host time into move/allocate/transfer enqueue, parameter installation, prefetch futures, unload and cache checks. Keep outputs bit-identical.
- Prototype only an evidenced bottleneck behind a new flag, then whole-process A–B–B–A with output fidelity and VRAM gates. No second persistent expanded weight set or changes to main/defaults.

## First/second CUDA kernel evidence

[Paired 73f T2V/I2V profiler run 43763934932](https://github.com/mylordmonkeyman/WanGP/actions/runs/43763934932) **PASS**, four sampled forwards per mode, decoded video/audio bit-identical. The *I2V* Stage 2 pair is the cleaner comparable kernel attribution:

| I2V Stage 2 GPU event category | First forward | Second forward |
| --- | ---: | ---: |
| GEMM/GEMV kernel durations (sum) | 2.311 s | 2.300 s |
| Attention kernel durations (sum) | 1.059 s | 1.051 s |
| H2D and other copy durations (sum) | **3.669 s** | **1.606 s** |
| H2D memcpy event count | **8,174** | **46** |
| Total GPU kernel and memory event-duration sum | 9.725 s | 7.742 s |

The first forward has essentially the same main math kernel duration as the second; the large change is transfer fragmentation and accompanying CPU-side loading. CUDA event sums include concurrent streams and must **not** be subtracted from wall time.

**Trace defect:** The T2V Stage 1 second-forward exported trace contains one absurd multi-billion-millisecond cast event. This sample is invalid and excluded from conclusions. The T2V Stage 2 warm trace also differs sharply in captured kernel counts, so do not treat it as reliable for direct first/second computational cost subtraction. Use the clean I2V Stage 2 pair and independently measured MMGP per-forward CUDA events.

## Reversible first implementation under test

`WAN2GP_LTX25_STAGE2_KEEP_CUDA_CACHE=1` in `models/ltx2/ltx_pipelines/distilled.py` skips **only** the one CUDA caching-allocator purge just after spatial upscaling; the existing `gc.collect()` and `torch.cuda.synchronize()` remain. Flag defaults to off. Model weights and numerical kernels are untouched, and it does not create a second persistent FP16 weight set. The hypothesis is that retaining free allocator blocks reduces the large cold Stage2 CPU block-load cost, but it may not: MMGP auto-preload can also release model blocks to make room for the upscaler.

[73f T2V/I2V A–B–B–A end-to-end run 43765110798](https://github.com/mylordmonkeyman/WanGP/actions/runs/43765110798) tests eight fresh processes and checks every decoded video/audio hash against the stable main. Do not promote unless output parity, memory safety and a reproducible whole-video saving are demonstrated.

## Completed A–B–B–A cache-retention experiment (73f)

[Eight-generation run 43765110798](https://github.com/mylordmonkeyman/WanGP/actions/runs/43765110798) **PASS**. Each arm uses one fresh 73-frame 1024×576, 8-step bar generation; no CUDA profiler in timing runs, same Candidate E coverage (27,360) and disabled tiled epilogue (zero tiled calls). All eight videos' decoded video/audio SHA256 match pinned stable `main`.

| Mode | A1 original | B1 keep cache | B2 keep cache | A2 original | A mean | B mean | B speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | 112.267 s | 111.822 s | 112.077 s | 112.077 s | 112.172 s | 111.9495 s | +0.1984% |
| I2V | 113.440 s | 113.282 s | 114.290 s | 114.133 s | 113.7865 s | 113.7860 s | +0.0004% |

**Decision:** Leave `WAN2GP_LTX25_STAGE2_KEEP_CUDA_CACHE` default **off**. Marginal T2V change and zero I2V change are within run-to-run scatter and do not justify promotion. This narrows the bottleneck: simply retaining free CUDA allocator cache blocks is not sufficient to remove cold host block-installation overhead.

## Completed MMGP per-block CPU breakdown

[Profiling run 43765627202](https://github.com/mylordmonkeyman/WanGP/actions/runs/43765627202) **PASS**, T2V only, decoded outputs bit-identical. Temporary instrumentation overlays restored bytes after process. In each per-forward table, component timing sums are **nested** and not additive; the sections measure synchronous host time spent invoking Python work and enqueuing GPU work, not exclusive GPU wall.

| MMGP CPU section (summed, T2V) | Stage 1 first | Stage 1 second | Stage 2 first | Stage 2 second |
| --- | ---: | ---: | ---: | ---: |
| All `gpu_load_blocks` calls | 1,310 ms | 384 ms | **5,840 ms** | 642 ms |
| `copy_block` parameter installation / LoRA (within above) | 353 ms | 15 ms | **5,018 ms** | **27 ms** |
| `_move_block` allocation + fused H2D copy enqueue (nested) | 325 ms | 326 ms | 291 ms | 551 ms |
| `_load_pipelined` (nested) | 544 ms | 368 ms | 383 ms | 623 ms |
| Unload preceding block | 8 ms | — | 7 ms | — |
| MMGP `empty_cache_if_needed` checks | 1 ms | — | 3 ms | — |
| Recorded `copy_block` invocations | 51 | 46 | **99** | 46 |

The cold Stage 2 forward executes many **preload=True** parameter installations in addition to ordinary recurrent block loads: this is the dominant measured CPU-side section. The stage2 first cold 99 `copy_block` calls incur 5.02 seconds vs 27 milliseconds for 46 warm calls. Largest individual 64-block and base-weight loads are ~100–338 ms. The 5.02 seconds is not necessarily recoverable, since some work overlaps GPU compute and weights must be transferred/installed somewhere.

**Revised next target:** Profile or change the cold `copy_block` parameter-installation/preload path, not a generic global allocator cache purge. Test whether the cold passes create many small unpinned per-tensor copies and whether grouping/staging or first-pass preload placement can cut launch and CPU dispatch overhead without a second persistent weight set or increased peak VRAM. Start with a strictly opt-in, bounded change and repeat end-to-end A–B–B–A before default promotion.

## Candidate P1 — bounded packed MMGP transformer cold preloads

**Date:** 2026-11-15. **Branch:** `perf/ltx25-v100-mmgp-transfer-20261115`. **Code commit:** `bc6ea56fd2f85131a2fef96867fc57ba052dfcac`. **Default: OFF**, enabled only by `WAN2GP_LTX25_MMGP_PACKED_PRELOAD=1`.

**Rationale:** Earlier detailed T2V Stage2 first forward spent ~5.018 seconds in 99 `copy_block` calls, compared with 0.027 seconds and 46 calls in the second; I2V CUDA Stage2 cold trace counted 8,174 copy events versus 46 warm. Instead of altering CUDA math or extending GPU weight residence, P1 targets the per-tensor pageable loading path used when MMGP bulk pinned sources are unavailable.

**Implementation:** In `mmgp/offload.py`, for `model_id=="transformer"`, `preload=True` and a non-base block only, P1:
- Prefers unmodified MMGP bulk source runs when already pinned; never replaces those.
- Reuses the existing MMGP `_block_items` / `_ring_layout` / `_ring_view` machinery to flatten eligible pageable tensor subclass leaves with existing GPU alignment/metadata.
- Packs at most 512 MiB per block into two alternating, capacity-governed pinned **CPU** buffers; exact bytes are copied via the already-used `_copy_runs` routine, then a **single** asynchronous host-to-GPU copy fills the *final* GPU allocation whose views become the block's installed weight tensors. No additional persistent GPU weight set or FP16-expanded weights.
- Uses CUDA transfer-completion events to protect pinned scratch reuse; may introduce CPU waits, so measured wall time is required. If layout is unsupported, source is already pinned, block exceeds cap, or pinned budget refuses a slot, original parameter-loading code is used.
- Leaves all other models, ordinary streamed blocks, math/attention kernels, quantization, existing flag defaults, Stage2 cache cleanup, and current MMGP VRAM preload policy untouched.

**Experiment:** [V100 73-frame T2V+I2V full-process ABBA run 43779437456](https://github.com/mylordmonkeyman/WanGP/actions/runs/43779437456), 8 separate processes, identical original 1024×576 bar prompt/image, seed 20261112, 8 steps, same model, Candidate E+radix4 and Stage2 FP16, tiled epilogue off, Stage2 CUDA cache toggle off. Every arm must match pinned decoded video/audio reference and maintain zero persistent FP16 weight copies; B must report `packed_blocks>0` to demonstrate that the code executed. **No claim of speedup until the run completes.**

**Decision gate:** Significant repeatable whole-process gain and exact fidelity before considering promotion, with 241-frame ABBA validation afterward. If pinned RAM is refused or performance is worse, keep the flag off, preserve the negative result, and investigate alternative bounded staging or prefetch placement. Stable `main` untouched.

## P1 73-frame qualification result — PASS, modest but reproducible within this ABBA

[Run 43779437456](https://github.com/mylordmonkeyman/WanGP/actions/runs/43779437456), commit `f4bf5e0e`, completed 2026-11-16 UTC. Eight whole-process runs, 1024×576, 73 frames, 8 steps, same bar prompt/image, checkpoint and seed. Both B arms packed **48** cold transformer preload blocks totaling **18,549,947,392 bytes** T2V (18.55 GB), with no bounded/pinned-budget skips. Note this is **cumulative bytes copied**, not peak additional memory: only two bounded reusable pinned CPU scratch buffers are allocated and no second persistent GPU weight set is created.

| Mode | A1 baseline | B1 packed | B2 packed | A2 baseline | A mean | B mean | improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | 112.656s | 111.023s | 110.967s | 111.631s | 112.1435s | 110.995s | **+1.0241%** |
| I2V | 114.791s | 112.975s | 113.337s | 114.232s | 114.5115s | 113.156s | **+1.1837%** |

**Fidelity:** All eight arms have identical decoded video **and** audio hashes to the pinned stable-main reference; Candidate E 27,360 calls, 384 native fallbacks, tiled epilogue zero, and same exceptional-row counts in each mode. No persistent FP16 weight copies. This is a small but encouraging measured wall-clock gain; **do not promote** on 73-frame results alone. It may include run noise; 241-frame validation and additional memory/VRAM measurement are required.

**Next:** 241-frame T2V+I2V ABBA automatically launched at [run 43784862151](https://github.com/mylordmonkeyman/WanGP/actions/runs/43784862151), commit `48a1c162`, following this success. Review fidelity, absolute times and VRAM behavior before further changes. If gains remain marginal, pursue targeted CPU timing of packed `_copy_runs` vs original `_move_block`, and test 1-vs-2 staging slots or asynchronous prepack scheduling; do not increase VRAM residency or change math.

## P1 241-frame qualification — PASS, speedup shrinks with duration

[Run 43784862151](https://github.com/mylordmonkeyman/WanGP/actions/runs/43784862151), 2026-11-16 UTC, **PASS**: 8 independent 241-frame, 1024×576, 8-step processes, same pinned model/image/prompt/seed. B arms each packed 48 cold transformer blocks, ~18.55 GB cumulative, with zero skipped pinned/bounded/budget cases. Decoded video and audio SHA256 **identical** for every A/B arm; Candidate E 45,216 calls, 1,824 native fallbacks, zero tiled epilogue calls, matching exceptional-row counts (T2V 5,851; I2V 5,578). Zero persistent FP16 weight copies. These checks do **not** constitute an instrumented peak-VRAM comparison; that remains outstanding.

| Mode | A1 original | B1 packed | B2 packed | A2 original | A mean | B mean | B improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | 223.766s | 221.920s | 222.210s | 223.815s | **223.7905s** | **222.065s** | **+0.7710%** |
| I2V | 225.328s | 224.880s | 225.326s | 224.924s | **225.126s** | **225.103s** | **+0.0102%** |

**Interpretation:** The 73f ~1% gain does not scale with frames, especially for I2V. P1 improves some one-time cold installation overhead, not the dominant repeated GEMM/attention work. 241f T2V saves 1.73s on average; I2V saves 0.023s (noise). Neither result justifies promoting P1 to default. **Keep default off.** Existing implementation remains opt-in for further experiments.

**Next experiments:** Separate CPU packing time (`_copy_runs`), pinned scratch waits, transfer enqueue and GPU consumer waits, and compare against the existing MMGP pageable path. Check peak allocated/reserved VRAM and pinned host RAM; avoid adding a second GPU weight set. Consider prepacking on a bounded worker overlapping upstream computation or optimizing non-preload streamed blocks only if traces show meaningful opportunity. Do not promote P1 to main.

## P1 73-frame phase attribution — measured, not a new default

[Profiling run 43791198999](https://github.com/mylordmonkeyman/WanGP/actions/runs/43791198999) **PASS** at commit `90705eec`, eight fresh-process 73f 1024×576 8-step T2V/I2V A–B–B–A generations. Every decoded video/audio output matched pinned stable main. Candidate E 27,360 calls, native fallbacks 384, tiled calls zero per generation. B packed 48 Stage2 cold blocks (~18.55 GB cumulative copied), with no skips. The separately enabled `WAN2GP_LTX25_MMGP_PACKED_PROFILE=1` instrumentation measured the B arms only.

| Mode | A1 | B1 | B2 | A2 | A mean | B mean | B saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | 111.960s | 111.371s | 111.378s | 111.213s | 111.5865s | 111.3745s | 0.1900% |
| I2V | 113.681s | 112.880s | 111.938s | 113.238s | 113.4595s | 112.4090s | 0.9259% |

| B-arm Stage2 CPU phase, mean of four | Time |
| --- | ---: |
| Host packing `_copy_runs` | 1,481.0 ms |
| Parameter view installation | 450.5 ms |
| Pinned scratch allocation | 300.7 ms |
| GPU transfer enqueue (host time) | 153.6 ms |
| Scratch-slot wait | 0.92 ms |

The B-arm Stage2 CUDA allocator high-water was 19,789,3xx,xxx bytes allocated (~18.43 GiB), 20,635,975,680 bytes reserved (~19.22 GiB); this is **not** a matched A/B peak-VRAM comparison. CPU phase timers are host-side and may overlap GPU work; summing them does not establish recoverable end-to-end time. Pinned host-RAM peak remains unmeasured.

**Decision:** Keep P1 and profiling flags OFF by default. Packing, not slot waits, dominates the measured B CPU path. Before another implementation, obtain matched baseline allocator high-water and per-block `_copy_runs` breakdown; then evaluate a bounded asynchronous pack/preparation experiment or reduced-copy layout with the same final GPU storage. Preserve 73f/241f fidelity, A–B–B–A, and peak-VRAM gates. No promotion to main.

## P2 — size-adaptive host copy workers (2026-11-15, default OFF)

[CPU-only correctness microbenchmark 43802615356](https://github.com/mylordmonkeyman/WanGP/actions/runs/43802615356) finished PASS on the V100 runner at commit `a0f0ba5b`. For 384 MiB copied to pinned host memory, median of three warmed trials:

| Source-run size | 1 thread | 4 threads | 8 threads | Microbench observation |
| --- | ---: | ---: | ---: | --- |
| 1 MiB | 152.525 ms | 111.545 ms | 369.349 ms | 8-worker fanout counterproductive |
| 4 MiB | 109.165 ms | **45.398 ms** | 56.319 ms | 4 best sampled |
| 16 MiB | 47.859 ms | 36.828 ms | **22.959 ms** | 8 best sampled |
| 64 MiB | 48.403 ms | 18.190 ms | **15.894 ms** | 8 best sampled |

These synthetic measurements do not directly predict whole-video speedup. The preceding P1 pipeline instrumentation reported ~1.48s in _copy_runs host packing, with ~0.45s installing tensor views, ~0.30s pinned allocation and negligible scratch-slot waits; a faster CPU copy strategy is plausible but may overlap GPU compute.

**Implementation:** [commit e3a6e31d](https://github.com/mylordmonkeyman/WanGP/commit/e3a6e31d1aaed8b1ee6387b0446a12b0e5d59957) changes only the opt-in `_copy_runs(..., adaptive_workers=True)` call from the cold transformer packed-preload path. `WAN2GP_LTX25_MMGP_PACKED_ADAPTIVE8=1` must be used alongside `WAN2GP_LTX25_MMGP_PACKED_PRELOAD=1`. Copy operations with individual contiguous run length <4 MiB use one main-thread memmove; 4–16 MiB use four workers; >=16 MiB use eight workers, with a max-eight thread pool. With the flag off, existing P1 three-worker behavior is exactly retained. MMGP ring/pool `_copy_runs` and all unpacked/production code remain unmodified in behavior. No new GPU buffers or quantization changes.

**Full video gate:** [73-frame T2V/I2V ABBA workflow 43803583176](https://github.com/mylordmonkeyman/WanGP/actions/runs/43803583176), A=three-worker P1 packed preload, B=size-adaptive P2 packed preload, same 1024x576 image/prompt, seed 20261112, steps, checkpoint and 73-frame pinned decoded video/audio SHA-256. All 8 processes must pass exact fidelity and zero persistent FP16 copies. This isolates threading policy rather than attributing the entire P1 change to P2. Static checks PASS; GPU benchmark queued at documentation time. Do not infer a speedup until it completes.

**Follow-up gated 241f:** `.github/workflows/ltx25-mmgp-adaptive-pack-241f-v100.yml` and `benchmarks/ltx25_convrot/benchmark_mmgp_adaptive_241f_abba.py` are prepared with manual dispatch or `benchmarks/ltx25_convrot/mmgp_adaptive_241f_launch.signal`; do **not** launch unless a meaningful 73f gain is confirmed. 241f compares 3-worker vs adaptive8 *both packed*, not stock; use independent A–B–B–A and per-mode decoded hashes. If P2 is noise/negative, retain flags disabled and reconsider parallel CPU staging, block layout or non-overlapped host work.

**Isolation:** No edits to stable `main` or archive, no second persistent GPU weight copy. In particular, other workflows triggered by edits to `mmgp/offload.py` may perform redundant regression benches; inspect their run status to avoid concurrent use of the sole V100.


## P2 adaptive host packing — 241-frame qualification (2026-11-16)

[GitHub Actions run 43814831364](https://github.com/mylordmonkeyman/WanGP/actions/runs/43814831364) **PASS**, static and V100 benchmark jobs. Eight fresh-process 241-frame 1024×576, eight-step T2V/I2V generations, A–B–B–A per mode, comparing P1 three-worker packed preload (A) against P2 adaptive eight-worker packed preload (B). This is **not** a comparison to stock main. The exact decoded video and audio hashes matched the reference in all eight cases; Candidate E 45,216 calls, native fallback 1,824, tiled epilogue zero, matching exceptional-row counts. Each run packed 48 cold blocks (~18.55 GB cumulative), with zero bounded, pin-budget, or pinned-source skips. No second persistent GPU weight set.

| Mode | A1 P1 legacy3 | B1 P2 adaptive8 | B2 P2 adaptive8 | A2 P1 legacy3 | A mean | B mean | B speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | 222.256s | 221.969s | 221.000s | 222.632s | 222.444s | 221.4845s | **0.4313%** |
| I2V | 223.739s | 224.212s | 223.570s | 224.479s | 224.109s | 223.891s | **0.0973%** |

**Decision:** P2 passes correctness but its 73f gains (0.70% T2V / 0.88% I2V) shrink to 0.43% T2V / 0.10% I2V at 241f; I2V difference is consistent with noise. Keep both `WAN2GP_LTX25_MMGP_PACKED_PRELOAD` and `WAN2GP_LTX25_MMGP_PACKED_ADAPTIVE8` default **OFF**; do not promote to main. The host-copy microbench improvement does not translate into a large whole-process gain. No additional full-video run is warranted for P2 alone. A matched baseline-vs-P2 peak VRAM comparison remains unmeasured; prior P1 profile reported ~18.43 GiB allocated / 19.22 GiB reserved in optimized Stage 2, but that is not a matched P2 memory comparison.

**Next research gate:** Identify truly non-overlapped CPU-side MMGP block installation and GPU consumer stalls with per-block timeline traces before implementing more packing changes. Prioritize reducing parameter-view installation / Python dispatch or moving bounded preparation earlier only if measured on the critical path. Require explicit peak VRAM and pinned-host-RAM comparison for any new candidate. No persistent duplicate GPU weights, no default changes, and no overlapping V100 workflows.

## P3 — Matched CPU/GPU critical-path and memory attribution (2026-11-16)

**Study launched; no claimed speedup or result until successful GPU artifact analysis.** [GitHub Actions run 43899122726](https://github.com/mylordmonkeyman/WanGP/actions/runs/43899122726), development commit `cc64cb826dcb3fa714139093f72192981ca73b7f`, runs four fresh 73f 1024×576, 8-step bar generations: `t2v-baseline`, `t2v-p1p2`, `i2v-baseline`, and `i2v-p1p2`. Baseline has `WAN2GP_LTX25_MMGP_PACKED_PRELOAD=0`, `WAN2GP_LTX25_MMGP_PACKED_ADAPTIVE8=0`; experimental arm enables **both**. The selective Stage2 FP16, Candidate E both-stage `(4096,2048)`, radix-4, SDPA, MMGP profile 0, seed, source model, image/prompt, and tiled-epilogue **off** are unchanged. Both flags still default OFF, and the code of stable `main` is unchanged.

**Instrumented evidence captured:** for every `gpu_load_blocks` call, record host CPU start/stop timestamps, host dispatch duration, and events on the GPU consumer stream immediately before/after CPU loading; retain original per-forward transfer-event durations/bytes, consumer wait-event durations, prefetch-future host time, and host section timers. Record the CUDA allocator allocated/reserved high-water and MMGP pinned RAM total/`staging` from samples taken after each block load, then compare A/B under **the same tracing overhead**. Require `stage2:1` and `stage2:2` forward tags, valid CUDA event completion, and exact decoded video/audio hashes against pinned `main`. Temporary `mmgp/offload.py` and `distilled.py` overlays are compiled, checked and restored byte-exact; the hosted static job has passed.

**Interpretation limits:** the GPU consumer-stream interval between two marker events includes *all previously queued GPU compute and transfer dependencies*, and thus is **not** by itself GPU idle or recoverable time. CPU per-block time and GPU marker spans overlap. The pinned-memory snapshot is a sampled maximum, not a continuous allocator high-water. Profiled whole-process wall times are not production speed measurements. Before P4, identify exposed CPU submission gaps vs actual transfer wait using per-block timelines; only prototype a material, demonstrably non-overlapped bottleneck. No second persistent GPU weights or promotion to main.

## P3 completed matched attribution — 2026-11-16, run 43899122726

[Four 73f real T2V/I2V profiling runs 43899122726](https://github.com/mylordmonkeyman/WanGP/actions/runs/43899122726), commit `cc64cb826`, **PASS** (static and V100). Baseline vs P1+P2 under identical temporary diagnostic instrumentation. Both 73f T2V and I2V decoded video/audio hashes match pinned main; Candidate E 27,360 calls, tiled epilogue zero, no additional persistent GPU weight sets. Raw event and memory JSON artifact: [14240308572](https://github.com/mylordmonkeyman/WanGP/actions/runs/43899122726/artifacts/14240308572).

| Measurement (first Stage2 forward) | T2V baseline | T2V P1+P2 | I2V baseline | I2V P1+P2 |
| --- | ---: | ---: | ---: | ---: |
| Host block loader CPU sum | 5,755.5 ms | 4,136.1 ms | 5,941.1 ms | 4,126.0 ms |
| Preload=True CPU work, 49 calls | 4,962.2 ms | 3,348.4 ms | 5,085.5 ms | 3,309.0 ms |
| CPU `copy_block_params_and_lora` *nested section*, 99 calls | 4,965.5 ms | 1,779.6 ms | 5,092.4 ms | 1,759.3 ms |
| Consumer-stream event-gap sum during block loads | 5,457.5 ms | 3,839.6 ms | 5,587.6 ms | 3,819.3 ms |
| GPU first-forward elapsed event span | 12,319.2 ms | 10,718.0 ms | 12,314.7 ms | 10,491.9 ms |
| Warm Stage2 GPU forward elapsed event span (second) | 6,333.4 ms | 6,280.0 ms | 6,227.5 ms | 6,189.4 ms |

The **preload-only CPU/gap improvement**, approximately 1.61s T2V / 1.78s I2V, accounts for nearly the entire difference. Per-block medians fell from 95.5ms to 56.2ms (T2V) and 96.9ms to 55.4ms (I2V). Some individual optimized blocks remained long (~205–212ms), so the packed-copy and view-installation path may still have headroom. The `copy_block` nested section fell by more than this because P1 shifts work into packing prior to normal installation; do not interpret nested timing difference as recoverable wall savings.

**Matched memory, sampled across block loads:**

| Measure | Baseline | P1+P2 | Difference |
| --- | ---: | ---: | ---: |
| Sampled max CUDA memory allocated, T2V | 20,127,448,392 B | 20,127,112,008 B | −336,384 B |
| Sampled max CUDA reserved, T2V and I2V | 21,307,064,320 B | 21,206,401,024 B | −100,663,296 B (~96 MiB) |
| MMGP pinned `staging` RAM, both modes | 201,326,592 B | 922,746,880 B | **+721,420,288 B (~688 MiB)** |
| MMGP total pinned RAM, both modes | 1,328,828,416 B | 2,083,803,136 B | **+754,974,720 B (~720 MiB)** |

Total staging is bounded by P1's two scratch slots and other MMGP staging reservations, not by 18.55 GB cumulative transferred bytes. This is a sampled high-water, not proof of global process peak. GPU allocated maxima are essentially equal and GPU reserved modestly lower, but high-water comparisons do not establish GPU safety on other workloads.

**Critical-path interpretation:** Since GPU first-forward span falls ~1.60s T2V/~1.82s I2V in parallel with the host preload reduction, a substantial part of cold CPU submission is visible in the first-transformer-forward GPU event span. Nevertheless CUDA events straddling host work include earlier queued GPU kernels, and first-forward GPU-span differences are not interchangeable with whole-process speedup. Earlier non-profiled, repeated P1/P2 video benchmark results of only ~0–1% remain the authoritative net-latency evidence. P3 is one run per variant/mode, not a fresh ABBA speed qualification.

**Decision:** P3 confirms that loading and preloading are meaningful first-forward bottlenecks, *not* that more memcpy worker tuning will improve overall end-to-end time. Candidate P4 should prioritize bounded preparation/installation ahead of consumer execution or reduced per-tensor Python/view work with a strictly opt-in design. However, the extra ~720 MiB pinned RAM from P1/P2 is a real resource tradeoff; any P4 attempt should quantify and preferably reduce it, and must not duplicate persistent GPU weights. Require 73f and 241f non-instrumented matched ABBA T2V/I2V and exact decoded outputs before promotion. Leave P1/P2 and P4 off by default; stable main remains untouched.

## P1+P2 combined versus main-equivalent — uninstrumented full-process ABBA (2026-11-16)

**Test launched; results pending.** [GitHub Actions run 43903780497](https://github.com/mylordmonkeyman/WanGP/actions/runs/43903780497) at SHA `0b963f4135088af4207f0e6e1b80826c3415f34a`; static checks PASS, single V100 runner executing 73-frame benchmark, then 241-frame. This experiment directly checks combined improvement instead of adding percentages from separate P1 and P2 comparisons.

- Fresh subprocess control A (`A1,A2`): `WAN2GP_LTX25_MMGP_PACKED_PRELOAD=0`, `WAN2GP_LTX25_MMGP_PACKED_ADAPTIVE8=0`. Stable-`main`-**equivalent settings** running on the development checkout, not a separate checkout of actual main.
- Fresh subprocess optimized B (`B1,B2`): **both P1 and P2** enabled (`PACKED_PRELOAD=1`, `PACKED_ADAPTIVE8=1`); actual packed blocks verified. No additional persistent GPU weight sets.
- Mode order: T2V ABBA and I2V ABBA at 73f, followed by T2V ABBA and I2V ABBA at 241f. Same 1024×576 Distilled bar prompt/image, checkpoint, 8 diffusion steps, seed 20261112, Candidate E both-stage `(4096,2048)`, radix-4, selective Stage2 FP16, SDPA and MMGP profile 0. Tiled epilogue, Stage2 allocator-retention and all additional MMGP profilers OFF. Source `main` remains `523cb6f8`.
- Timing is per actual full-generation subprocess wall time, **not synthetic memcpy or CUDA profiling timing**. Required decoded video/audio SHA256: pinned main baseline for 73f, A1 and repeat A2 for 241f; unchanged Candidate E call counts and zero persistent expanded FP16 weight copies. Save each arm and aggregate A/B mean, percent gain, raw logs and quality status. No merging or default promotion based on unverified results.

Await job completion to quantify whether combined P1+P2 is worthwhile.


## Production promotion and verified combined results (2026-11-16)

The uninstrumented [combined 73f/241f ABBA run 43903780497](https://github.com/mylordmonkeyman/WanGP/actions/runs/43903780497) completed **SUCCESS**. Full-process mean A (main-equivalent flags OFF) versus B (P1+P2 ON):

| Mode | Frames | A mean wall s | B mean wall s | Improvement | Decoded video/audio |
| --- | ---: | ---: | ---: | ---: | --- |
| T2V | 73 | 112.2105 | 109.9820 | 1.9860% | 4/4 exact |
| I2V | 73 | 113.3895 | 112.0235 | 1.2047% | 4/4 exact |
| T2V | 241 | 222.6335 | 221.7650 | 0.3901% | 4/4 exact |
| I2V | 241 | 225.1505 | 224.2965 | 0.3793% | 4/4 exact |

All 16 arms matched decoded video and audio exactly. Each optimized run packed 48 cold Stage 2 blocks, ~18.55 GB cumulative bytes, zero pin-budget skips, with unchanged Candidate E call counts and zero tiled epilogue calls. The P3 critical-path run 43899122726 also passed; Stage2 first-forward consumer-stream gap sums fell 5.46→3.84 s (T2V), 5.59→3.82 s (I2V), but event spans include queued GPU work and are not directly recoverable wall time. Sampled peak GPU allocated was approximately equal, while pinned-host peak increased from ~1.33 to ~2.08 GB (decimal bytes).

**Explicit user-approved promotion:** [PR #1](https://github.com/mylordmonkeyman/WanGP/pull/1) was squash-merged to stable `main` as [`57cf1c4eec3bcba4d20a32a9161cbc94eb2de5c1`](https://github.com/mylordmonkeyman/WanGP/commit/57cf1c4eec3bcba4d20a32a9161cbc94eb2de5c1). The clean production commit changes **only `mmgp/offload.py`** (+90/-3), enables P1 and P2 by default, removes experimental P1 profiling, and adds no workflows, instrumentation, benchmark scripts, or alternative persistent GPU weights. Both optimizations remain reversible: `WAN2GP_LTX25_MMGP_PACKED_PRELOAD=0` disables P1/P2; `WAN2GP_LTX25_MMGP_PACKED_ADAPTIVE8=0` retains P1 but disables P2. The original archival branch remains unchanged. The experimental development branch intentionally retains explicit opt-in flags and measurement harnesses for isolated P4 studies; **production main has the default-on behavior**.

**Next research:** On this development branch, investigate a default-off P4 that reduces remaining cold Stage2 CPU install/preload cost and/or the ~720 MiB incremental pinned host memory without a second persistent GPU weight set. Gate on V100 microbench, memory and decoded fidelity, followed by matched 73f and 241f T2V/I2V ABBA; do not repeat the completed P1+P2 qualification.

## P4 experiment selection — 2026-11-16 follow-through

The completed P3 profiler and uninstrumented combined ABBA establish a narrow opportunity, not a reason to rerun P1/P2. Cold Stage2 first-forward host preload work drops ~1.6–1.8 s under P1+P2, but 241f whole-process improvements are only 0.3901% T2V and 0.3793% I2V. Additional pinned host RAM is ~720 MiB; sampled GPU allocated memory is essentially unchanged. P4 must optimize *remaining* installation/preparation without adding persistent GPU weights or making the pinned-host tradeoff worse.

**Preferred bounded P4 probe:** first isolate scratch-slot capacity and lifetime. Measure P1+P2 with one versus two pinned scratch slots, recording per-block slot waits, allocations, CUDA consumer dependencies, pinned-host sampled peak, and the first Stage2 forward. A one-slot variant is worth pursuing only if it materially lowers pinned RAM while avoiding new serialization. The earlier P1-only slot-wait measurement (~0.92 ms total) is encouraging but is not proof for adaptive8 or all 241f shapes. Implement this as a new default-off flag, never silently change production defaults. Reject immediately if it causes unsafe slot reuse, excessive waiting, or memory regressions.

**Alternative if slot reduction fails:** instrument parameter-view creation and installation per preload block; prototype fewer Python-level per-tensor operations or bounded preparation before consumer execution only when traces identify a measurable non-overlapped section. Preserve tensor identity, strides, storage ownership and stream completion; do not pre-stage a second GPU weight set. First run CPU/GPU correctness microbench and sampled memory comparisons, then full uninstrumented 73f and 241f T2V/I2V A–B–B–A with exact decoded video/audio hashes and no concurrent V100 runs. No P4 implementation or performance claim is made by this planning entry.


## P4 single-slot pinned staging candidate — 2026-11-16

Development-only implementation [101f7bbd](https://github.com/mylordmonkeyman/WanGP/commit/101f7bbd2a3b1088773d56cca2c2007d13d75c21) adds `WAN2GP_LTX25_MMGP_PACKED_SINGLE_SLOT=1`, default **OFF**, effective only with packed preload enabled. The existing two-slot staging is unchanged by default; P4 allocates one bounded pinned scratch buffer and synchronizes the prior per-slot CUDA completion event **before** reusing that buffer. It does not create another GPU weight set or change the Candidate E/Radix4/FP16 numerical path. A single slot can serialize host packing with DMA, so memory savings and performance must be measured, not assumed.

Matched 73f T2V/I2V A–B–B–A harness [a3d35922](https://github.com/mylordmonkeyman/WanGP/commit/a3d35922bf7fdda63cab384e6ffba38f51f2e016) compares P1+P2 **two-slot A** with P1+P2 **one-slot B** (all other flags identical), checks exact decoded video/audio against pinned references, Candidate E call counts, no tiled epilogue and zero persistent FP16 weight copies. Workflow [P4 73f](https://github.com/mylordmonkeyman/WanGP/actions/runs/43936642632) was launched, but results are **pending**; do not claim a speedup or run 241f until 73f correctness and a useful memory/performance tradeoff are demonstrated. P3's ~720 MiB pinned RAM increase motivates the test but is not a prediction of one-slot savings. A matched sampled pinned-host and GPU high-water comparison is still required before any promotion.

**Runner coordination:** The offload-source commit also triggered existing P1 73f performance/profile workflows [43936573597](https://github.com/mylordmonkeyman/WanGP/actions/runs/43936573597) and [43936573627](https://github.com/mylordmonkeyman/WanGP/actions/runs/43936573627). Those are redundant for P4 and occupy/queue for the same single V100 runner. Do not start further V100 tests while any of these jobs remain active. Existing production `main` and the tiled archive remain unchanged.
