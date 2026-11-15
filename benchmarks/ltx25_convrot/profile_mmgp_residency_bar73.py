#!/usr/bin/env python3
"""Profiling-only 73f T2V/I2V MMGP critical-path and residency evidence.

Temporarily patch a disposable GitHub runner checkout, not the repository source
or stable production code. Restore exact source bytes after each run. CUDA event
pairs are asynchronous until process exit; profiled elapsed wall is not a speed
comparison. Every video must match pinned stable decoded audio/video hashes.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import traceback

import selective_fp16_t2v_smoke as base
import t2v73_241_original_stock_fourway as four
import distilled_bar73_stage_shape_fourway as stage_shape
from profile_gemm_t2v_i2v import PROMPT, INPUT_IMAGE, check_png
from verify_cleanup_bar73_bitexact import KNOWN

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LTX25_MMGP_PROBE_ROOT",
          "/cache/ltx25/results/mmgp-probe-" + os.environ.get("GITHUB_RUN_ID", "manual"))).resolve()
CHECKPOINT = base.CHECKPOINTS / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
SOURCES = ("mmgp/offload.py", "models/ltx2/ltx_pipelines/distilled.py")


def require(test, why):
    if not test:
        raise RuntimeError(why)


def splice(source, before, after, count=1):
    actual=source.count(before)
    if actual!=count:
        raise RuntimeError("MMGP probe overlay source drift: " + repr(before[:110])
                           + " expected="+str(count)+" actual="+str(actual))
    return source.replace(before, after)


@contextmanager
def temporary_mmgp_overlay():
    """Patch exact pinned text anchors only and restore even if generation fails."""
    originals={name:(ROOT/name).read_bytes() for name in SOURCES}
    try:
        o=originals[SOURCES[0]].decode("utf-8")
        o=splice(o, "import torch\n", "import torch\nfrom benchmarks.ltx25_convrot import mmgp_residency_probe as mmgp_probe\n")
        target="consumer.wait_event(transfer if isinstance(transfer, torch.cuda.Event) else memory_pressure.wait(transfer))"
        o=splice(o,target,"mmgp_probe.consumer_wait(consumer, transfer, entry_name, memory_pressure.wait)",count=2)
        o=splice(o,
            '            if staged is None:\n'
            '                for offset, source in sources:\n'
            '                    buffer[offset:offset + source.numel()].copy_(source, non_blocking=True)\n'
            '            else:\n'
            '                buffer[:nbytes].copy_(self.ring.buffer[region[0]:region[0] + nbytes], non_blocking=True)\n',
            '            mmgp_copy_start = mmgp_probe.copy_start(stream)\n'
            '            if staged is None:\n'
            '                for offset, source in sources:\n'
            '                    buffer[offset:offset + source.numel()].copy_(source, non_blocking=True)\n'
            '            else:\n'
            '                buffer[:nbytes].copy_(self.ring.buffer[region[0]:region[0] + nbytes], non_blocking=True)\n'
            '            mmgp_probe.copy_stop(stream, mmgp_copy_start, entry_name, nbytes)\n')
        o=splice(o,
            '    def gpu_load_blocks(self, model_id, blocks_name, preload = False):\n'
            '        # cl = clock.start()',
            '    def gpu_load_blocks(self, model_id, blocks_name, preload = False):\n'
            '        mmgp_load_start = mmgp_probe.block_load_start()\n'
            '        # cl = clock.start()')
        o=splice(o,
            '        # cl.stop()\n'
            '        # print(f"load time: {cl.format_time_gap()}")',
            '        mmgp_probe.block_load_stop(mmgp_load_start, model_id, blocks_name, preload)\n'
            '        # cl.stop()\n'
            '        # print(f"load time: {cl.format_time_gap()}")')
        o=splice(o,
            '        self.preloaded_blocks_per_model[model_id] = preloaded_blocks\n'
            '        # what the budget leaves beyond the base',
            '        self.preloaded_blocks_per_model[model_id] = preloaded_blocks\n'
            '        mmgp_probe.record_plan(model_id, len(preloaded_blocks), preload_total,\n'
            '                               self.shuttle_sizes[model_id], window,\n'
            '                               sum(1 for key in self.blocks_of_modules_sizes if key.startswith(model_id + "/")))\n'
            '        # what the budget leaves beyond the base')
        d=originals[SOURCES[1]].decode("utf-8")
        d=splice(d,
            'from shared.kernels import ltx25_cublas_sm70 as candidate_e\n',
            'from shared.kernels import ltx25_cublas_sm70 as candidate_e\n'
            'from benchmarks.ltx25_convrot import mmgp_residency_probe as mmgp_probe\n')
        d=splice(d,
            '        return fp16_gemm_experiment.call_transformer(self._module, *args, **kwargs)',
            '        return mmgp_probe.forward_call(fp16_gemm_experiment.call_transformer, self._module, *args, **kwargs)')
        compile(o, str(ROOT/SOURCES[0]), "exec")
        compile(d, str(ROOT/SOURCES[1]), "exec")
        (ROOT/SOURCES[0]).write_text(o)
        (ROOT/SOURCES[1]).write_text(d)
        yield
    finally:
        for name,data in originals.items():
            (ROOT/name).write_bytes(data)
        require(all((ROOT/name).read_bytes()==data for name,data in originals.items()),
                "Failed to restore original source after MMGP probe")


def run(mode, template):
    name=mode+"-mmgp-profile"
    settings=four.settings(73)
    if mode=="i2v":
        settings.update(image_start=str(INPUT_IMAGE),image_prompt_type="S")
    settings_file=OUT/"settings"/(name+".json")
    settings_file.write_text(json.dumps(settings,indent=2)+"\n")
    cfg,target=base.config_for(name,template)
    env=four.optimized_env("stage2_fp16",name)
    for key in ("WAN2GP_LTX25_E_TEST_4096X2048",
                "WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
                "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1",
                "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2",
                "WAN2GP_LTX25_PIPELINE_PROFILE",
                "WAN2GP_LTX25_PROFILE_STAGES",
                "LTX25_PIPELINE_TRACE_ATTRIBUTION"):
        env.pop(key,None)
    env["WAN2GP_LTX25_E_EPILOGUE_TILED"]="0"
    env["WAN2GP_LTX25_E_EPILOGUE_TILED_MAX_M"]="128"
    profile_file=OUT/"results"/(name+"-mmgp.json")
    env["LTX25_MMGP_PROBE_JSON"]=str(profile_file)
    cmd=[base.PY,"wgp.py","--offline","--config",str(cfg),"--process",
         str(settings_file),"--output-dir",str(target),"--profile","0",
         "--attention","sdpa","--no-prevent-power-throttling","--verbose","1"]
    dry=base.command_run(cmd+["--dry-run"],env=env,timeout=300,
                         log=OUT/"logs"/(name+"-dryrun.log"))
    require(dry["returncode"]==0, name+": dry-run failed")
    base.gpu_free()
    print("MMGP_PROBE_BEGIN",name,flush=True)
    with temporary_mmgp_overlay():
        timed=base.command_run(cmd,env=env,timeout=3600,
                               log=OUT/"logs"/(name+"-generation.log"),check=False)
    require(timed["returncode"]==0,f"{name}: failed: {timed['log_tail']}")
    require(profile_file.exists(),name+": no MMGP profile emitted")
    profile=json.loads(profile_file.read_text())
    require(profile.get("status")=="PASS",name+": GPU interval report incomplete: "+repr(profile.get("errors")))
    logs=Path(timed["log"]).read_text(errors="replace")
    stat=four.read_candidate_stats(logs)
    require(stat and stat.get("used")==27360 and stat.get("mode")=="fp32rot"
            and stat.get("rotation_mode")=="radix4"
            and stat.get("active_4096x2048_phases")==["stage1","stage2"]
            and stat.get("epilogue_tiled_calls")==0,
            f"{name}: original Candidate E settings changed: {stat}")
    fp16=stage_shape.audit_stage2(OUT/"results"/(name+"-fp16.json"),"stage2_fp16")
    require(fp16["persistent_fp16_weight_copies"]==0,"Persistent weight duplication")
    clips=[p for p in target.rglob("*.mp4") if p.stat().st_size>1024]
    require(len(clips)==1,name+": output video missing or ambiguous")
    decoded=four.probe_video(clips[0],73)
    match={"video":decoded["decoded_video_sha256"]==KNOWN[mode]["video"],
           "audio":decoded["decoded_audio_sha256"]==KNOWN[mode]["audio"]}
    row={"mode":mode,"status":"PASS" if all(match.values()) else "FIDELITY_MISMATCH",
         "profiled_process_wall_seconds_not_comparable":timed["wall_seconds"],
         "candidate_e_calls":stat["used"],"tiled_calls":stat["epilogue_tiled_calls"],
         "decoded_matches_pinned_main":match,
         "mmgp_probe_summary": {"by_phase":profile["by_phase"],
                                "preload_plans":profile["preload_plans"],
                                "transformer_forwards":profile["transformer_forwards"]},
         "profile_path":str(profile_file)}
    (OUT/"results"/(name+"-summary.json")).write_text(json.dumps(row,indent=2)+"\n")
    print("MMGP_PROBE_CASE",json.dumps(row,sort_keys=True),flush=True)
    require(all(match.values()),name+": baseline output fidelity changed")
    return row


def main():
    require(not OUT.exists(),"Existing output root: "+str(OUT))
    for folder in ("settings","config","videos","results","logs"):
        (OUT/folder).mkdir(parents=True,exist_ok=False)
    check_png()
    require(CHECKPOINT.is_file(),"ConvRot checkpoint missing")
    base.OUT=OUT
    base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    base.FRAMES,base.STEPS,base.SEED=73,8,20261112
    base.RESOLUTION,base.PROMPT="1024x576",PROMPT
    four.OUT=OUT
    base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70",
        "-shared","-Xcompiler","-fPIC",
        "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas",
        "-o",str(base.EXTENSION)],timeout=300,log=OUT/"logs"/"compile.log")
    require(base.EXTENSION.exists(),"Missing CUDA kernel")
    cfg=base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        "stable_main_sha":"523cb6f8cf3aeca2daa38a61a3d7e81c01c61c43",
        "branch_sha":os.environ.get("GITHUB_SHA"),"checkpoint":str(CHECKPOINT),
        "frames":73,"resolution":"1024x576","steps":8,"seed":20261112,
        "mode_order":["t2v","i2v"],"mmgp_profile":0,
        "attention":"sdpa","no_persistent_weight_copy":True,
        "timings":"CUDA events and CPU enqueue intervals; profiled whole-process wall is NOT a production speed comparison",
        "baseline_uninstrumented_run":"43746795037",
        "code_overlay":"temporary, restored after each complete subprocess"
    },indent=2)+"\n")
    cases=[]
    for mode in ("t2v","i2v"):
        cases.append(run(mode,cfg))
        (OUT/"partial.json").write_text(json.dumps(cases,indent=2)+"\n")
    (OUT/"summary.json").write_text(json.dumps({"status":"PASS","cases":cases},indent=2)+"\n")
    print("LTX25_MMGP_RESIDENCY_PROBE_PASS",flush=True)


if __name__=="__main__":
    main()
