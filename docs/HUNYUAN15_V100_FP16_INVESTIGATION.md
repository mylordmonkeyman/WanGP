# Hunyuan Video 1.5 native FP16 on a Tesla V100 (research branch)

Status: **experimental, code path implemented but no 8.3B checkpoint has yet been loaded or generated with this branch**. Keep upstream/main defaults unchanged until gates pass.

Branch: `research/hunyuan15-v100-native-fp16-20261116` (based on main, separate from LTX-2.5 work).

## Goal and hardware

Make WanGP run Hunyuan Video 1.5 480p T2V and I2V with **native FP16 transformer weights and FP16 intermediate computation** on a Tesla V100 32 GB (SM70). Preserve stable FP32 computations where deliberately locked for precision (time/guidance embeddings, optionally final layers), VAE FP32, and existing MMGP offloading. This is not a demand to force *every* operation into FP16.

Volta lacks native BF16 compute; FP16 Tensor Cores are supported. BF16 checkpoints cannot be assumed to give native FP16 computations merely because the weights were renamed.

## What the code audit found (2026-11-16)

- `models/hyvideo/hunyuan.py::HunyuanVideoSampler.from_pretrained` previously ignored its dtype argument and forced `precision="bf16"` / `dtype=torch.bfloat16`. This is the first blocker.
- `models/hyvideo/modules/models.py` contained seven unconditional `.to(torch.bfloat16)` calls for activation/modulation/cache paths.
- `models/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py` fixed its target dtype to BF16 and explicitly cast latents and text embeddings to BF16.
- MMGP's `offload.load_model_data` can convert floating parameters into a requested `default_dtype` and throws for unresolved missing keys. It has limited common-prefix fallback; it is **not** a general guarantee that Diffusers/ComfyUI checkpoint layouts match WanGP.
- Existing WanGP Hunyuan 1.5 defaults point at DeepBeepMeep BF16 or Quanto BF16/INT8 files. Those defaults are not changed here.
- `HunyuanVideoAudioPipeline` retains legacy BF16 handling because ordinary Hunyuan 1.5 T2V/I2V uses `HunyuanVideoPipeline`. Avatar/audio and 1.5 upsampler variants require their own separate tests.

## Checkpoint-source decision

| Candidate | Approx. model size | Native dtype / expected key format | Assessment |
| --- | ---: | --- | --- |
| DeepBeepMeep WanGP 480p BF16 | 16.7 GB | BF16 / known WanGP keys | **Reference** to compare key names, shapes, and output quality; not a direct native-FP16 source |
| Comfy-Org 480p FP16 | 16.7 GB | FP16 / ComfyUI single-file format | **Preferred source.** 216 image/text QKV tensors require an audited unfusion; after splitting all 1,793 expected WanGP keys and shapes match |
| Tencent original `transformer/480p_*/diffusion_pytorch_model.safetensors` | 33.3 GB | **Verified FP32** (1,797 tensors) / original Diffusers format | **Fallback only**; 1,793 matching WanGP tensors + four unused extra keys. Existing Comfy FP16 matches direct FP32->FP16 rounding in sampled weights |
| DeepBeepMeep Quanto BF16 INT8 | ~8.5 GB | BF16/INT8 | Do **not** mislabel as FP16; evaluate FP16 Quantization only after full-FP16 correctness is established |

Source URLs:
- https://huggingface.co/Comfy-Org/Hunyuan-Video-1.5/tree/main/split_files/diffusion_models
- https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer
- https://huggingface.co/DeepBeepMeep/HunyuanVideo1.5/tree/main

**Recommendation (updated after automated metadata/weight probes):** Use Comfy-Org FP16 directly with an *in-memory, view-based* unfusion of image/text QKV projections, rather than downloading 33.3 GB of Tencent FP32 for conversion. After unfusing 216 QKV tensors, its 1,793 expected tensor keys and shapes precisely match the WanGP 480p I2V step-distilled reference, plus four extra `time_r_in.mlp` keys (also present in Tencent FP32). Small matched weight samples (3,808 values across 5 tensors) are **bit-identical** to Tencent's directly rounded FP32→FP16 weights. This does **not** establish that all ~8B model weights are identical. Do not download the Tencent FP32 unless a subsequent numerical or visual-quality test reveals a reason.

## Experimental implementation

- A model-definition setting, `"hunyuan15_native_fp16": true`, activates FP16 **only** for Hunyuan 1.5; the transformer dtype must also resolve to `torch.float16` (set `"dtype": "fp16"` in the model subtree).
- The loader creates a meta-device FP16 transformer and asks MMGP to load/cast weights into FP16.
- Model double/single blocks and the inference pipeline select FP16 activation/latent/prompt dtypes only for the opt-in model.
- All code without the flag continues to use the prior BF16 path.
- The model loader uses `preprocess_sd=unfuse_hunyuan15_qkv` in experimental mode; it splits Comfy-Org's 216 fused image/text projections into Q, K, V tensor views, while preserving the refiner's `self_attn_qkv` tensors. It rejects on-the-fly quantization and export in this mode.
- `wgp.get_transformer_dtype` explicitly resolves the FP16 model-def opt-in, even if the machine's default dtype policy would otherwise select BF16.
- This is a **prototype**, not a statement of runtime correctness or fidelity. An FP16-only checkpoint in the wrong key layout will fail loading.

The branch includes two **research-only, selectable finetune definitions** in `finetunes/` (stock `defaults/` untouched):
- `hunyuan15_v100_i2v_480_step_fp16_experiment.json` — direct Comfy-Org 480p I2V step-distilled FP16, 8 steps.
- `hunyuan15_v100_t2v_480_fp16_experiment.json` — direct Comfy-Org 480p T2V FP16, initially 30 steps; the T2V checkpoint layout and FP16 runtime remain separately unverified.

Both set `"dtype":"fp16"`, `"hunyuan15_native_fp16":true`, and `"auto_quantize":false` and point to real Comfy-Org URLs. Select the **unquantized** model mode, not Quanto INT8. Their published filenames are not newly created WanGP-quantized files: they are the original Comfy FP16 checkpoints, adapted only in memory on load.

## Published research measurements (CPU-only; no full checkpoint download)

- Initial header audit: [GitHub Actions 43920843463](https://github.com/mylordmonkeyman/WanGP/actions/runs/43920843463). Comfy 1,365 F16 tensors, WanGP 1,793 BF16, Tencent 1,797 F32.
- Correctly scoped QKV audit: [GitHub Actions 43921412288](https://github.com/mylordmonkeyman/WanGP/actions/runs/43921412288). Split 216 QKV tensors; 1,793 / 1,793 WanGP tensor keys and shapes match. Four extra unused `time_r_in.mlp` names are also found in the Tencent reference.
- FP32 provenance sample: [GitHub Actions 43921937836](https://github.com/mylordmonkeyman/WanGP/actions/runs/43921937836). All 3,808 sampled Comfy FP16 values (5 common small tensors) **bitwise equal** to directly rounded Tencent FP32 values. This is a sample, **not** proof for every tensor or the fused QKV value ordering.
- Source and synthetic layout-adapter checks: [GitHub Actions 43921737820](https://github.com/mylordmonkeyman/WanGP/actions/runs/43921737820) succeeded. These tests are not GPU inference.

## Stage 0: inspect source checkpoint without allocating 16-33 GB

Download/select the exact *same model variant* in both sources (for example, the official 480p I2V step-distilled version; never compare I2V with T2V or CFG-distilled with step-distilled).

```bash
python3 scripts/hunyuan15_checkpoint_audit.py \
  /path/to/hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors \
  --reference /path/to/hunyuan_video_1.5_i2v_480_step_distilled_bf16.safetensors \
  --require-key-match
```

The tool reads only the JSON header; it reports stored dtypes, key names, shape mismatches, and common prefix differences. A match **does not validate numerical behavior or tensor provenance**. If keys mismatch, record the exact differences and prepare an audited mapping; never silently ignore missing tensors or reassign mismatched shapes. To inspect Tencent FP32, substitute its `diffusion_pytorch_model.safetensors` as the candidate.

## Stage 1: load and numeric checks (V100)

1. Run CPU-only static tests with `python3 -m unittest discover -s benchmarks/hunyuan15_fp16 -p 'test_*.py' -v`. These check source syntax, opt-in guard and metadata comparison; **they are not GPU inference tests**.
2. On the V100, record `torch.cuda.get_device_capability()`, checkpoint path, MMGP profile, allocated/peak VRAM, dtype inventory of transformer modules and sample inputs, and the exact commit SHA.
3. Load 480p I2V step-distilled FP16 with 8 steps at 33 frames; start with no LoRA/caching, FP32 VAE, and a conservative memory profile. If that succeeds, repeat at 97 frames.
4. Run 480p T2V FP16 separately; initially use a normal/CFG-distilled base and documented sampling schedule. Test LightX2V 4-step LoRA only after a clean base-model result.
5. For each denoise step, check model output, latent and intermediate residuals for NaN/Inf and unusually large maxima. FP16's narrower exponent range is the primary fidelity risk. Compare matched seeds, settings, output hashes and visual frames to a valid BF16/FP32 reference on compatible hardware. In particular inspect faces, hands, and flicker.
6. Only after all finite/quality gates pass, consider making the path selectable in normal UI and evaluating native FP16 Quanto/INT8 or other quantizations. Preserve the BF16 fallback and explicit opt-in until repeated verification.

## Expected acceptance criteria

- Exact tensor key and shape coverage; no missing parameters and no unchecked transforms.
- No BF16 transformer activations remain in the opt-in path; intentional FP32 math/locks are documented.
- No NaN/Inf or material FP16 overflow across repeated real V100 T2V/I2V tests.
- Same-seed quality is acceptably close to validated reference; no new face anomalies or other visible artifacts.
- Report complete wall-clock latency, peak GPU and host RAM, and decoding time for each model/precision combination, without claiming speed improvements from microbenchmarks alone.
- `main`, stock Hunyuan versions, and existing LTX branches remain unchanged.

## Open risks / limitations

- Comfy-Org's **header layout has been reconciled** by unfusing image/text QKV, but the live checkpoint, QKV value order, transformer numerical stability and generated videos remain unverified.
- Tencent's reference checkpoint has been header-verified as F32. No 33.3 GB Tencent checkpoint conversion is planned unless Comfy proves inadequate.
- Existing `lock_layers_dtypes` deliberately preserves FP32 numerics in some layers; native FP16 means FP16 transformer backbone, not eliminating sensible mixed precision.
- The V100 has not been exercised with the new code; no accuracy, compatibility, or wall-time result has been claimed.
- MMGP `save_model` currently assumes BF16 for some Hunyuan nonquantized filenames; any eventual FP32->FP16 exporter must explicitly produce and verify an FP16-named, WanGP-keyed result rather than relying on that default.
