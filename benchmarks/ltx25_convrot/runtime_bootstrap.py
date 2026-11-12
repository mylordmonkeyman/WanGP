#!/usr/bin/env python3
"""Bootstrap a bounded, isolated LTX-2.5 WanGP Python runtime on V100.

Only changes the runner's dedicated /cache virtualenv, never the host WanGP
installation. Preserve torch 2.10+cu128 and Volta SM70. Track failures
individually; do not silently claim a full pipeline import has succeeded.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VENV = Path("/cache/venvs/ltx25-torch210-cu128")
OUT = Path("/cache/ltx25/results")
CONSTRAINTS = Path("/cache/ltx25/torch-constraints.txt")
CONSTRAINTS.parent.mkdir(parents=True, exist_ok=True)
CONSTRAINTS.write_text("torch==2.10.0+cu128\n" "torchaudio==2.10.0+cu128\n")
PYTHON = str(VENV / "bin/python")

GROUPS = [
    ("hub_compat", ["huggingface-hub==0.36.0"]),
    ("model_runtime", [
        "psutil", "diffusers==0.36.0", "transformers==4.54.0",
        "accelerate==1.9.0", "einops", "omegaconf", "hydra-core",
        "torchdiffeq", "rotary-embedding-torch>=0.5.3",
        "sentencepiece", "tiktoken", "fugashi", "unidic-lite",
    ]),
    ("media", [
        "pillow", "imageio", "imageio-ffmpeg", "av", "soundfile",
        "ffmpeg-python", "moviepy==1.0.3", "scipy", "librosa==0.11.0",
        "num2words==0.5.14", "ftfy", "pyyaml", "opencv-python-headless",
    ]),
    ("wan_ui", [
        "gradio==5.29.0", "gradio_rangeslider",
        "tqdm", "requests", "pydantic==2.11.10",
        "loguru", "orjson==3.11.6", "peft==0.17.0",
        "timm", "open_clip_torch",
    ]),
]

KNOWN_MISSING = {
    "psutil": "psutil",
    "onnxruntime": "onnxruntime>=1.22.0",
    "markdown": "Markdown",
    "easydict": "easydict",
    "matplotlib": "matplotlib>=3.9,<4",
    "facexlib": "facexlib==0.3.0",
    "misaki": "misaki",
    "vector_quantize_pytorch": "vector-quantize-pytorch==1.27.19",
    "kornia": "kornia",
    "webdataset": "webdataset",
    "open_clip": "open_clip_torch",
    "scikit_image": "scikit-image",
    "einshape": "einshape==1.0",
    "sageattention": "sageattention",
    "flash_attn_interface": "flash-attn",
    "lightning": "lightning",
    "pytorch_lightning": "pytorch-lightning",
    "toml": "toml",
    "tomli": "tomli",
    "s3tokenizer": "s3tokenizer",
    "ddgs": "ddgs>=9.16.0",
    "pypdf": "pypdf>=6.0.0",
    "dashscope": "dashscope",
    "stringzilla": "stringzilla==4.0.14",
    "wetext": "wetext==0.1.2",
    "spacy_pkuseg": "spacy_pkuseg",
    "piexif": "piexif",
    "iopath": "iopath>=0.1.10",
    "audio_separator": "audio-separator==0.36.1",
    "insightface": "insightface==0.7.3",
    "keyring": "keyring==25.7.0",
    "resvg_py": "resvg-py==0.5.0",
    "pyannote": "pyannote.audio==3.3.2",
    "nvidia_ml_py": "nvidia-ml-py",
    "faster_whisper": "faster-whisper",
    "chumpy": "chumpy",
    "smplfitter": "smplfitter",
    "mistral_common": "mistral-common",
    "mcp": "mcp==1.30.0",
    "ffmpeg": "ffmpeg-python",
    "webrtcvad": "webrtcvad",
    "sounddevice": "sounddevice>=0.4.0",
    "mutagen": "mutagen",
    "pyloudnorm": "pyloudnorm",
    "torchcodec": "torchcodec==0.10.0",
    "xatlas": "xatlas>=0.0.11",
    "jax": "jax",
    "taichi": "taichi",
    "loguru": "loguru",
    "gradio_rangeslider": "gradio_rangeslider",
    "mido": "mido==1.3.3",
    "mir_eval": "mir_eval==0.8.2",
    "diffusers": "diffusers==0.36.0",
    "transformers": "transformers==4.54.0",
    "accelerate": "accelerate==1.9.0",
    "torchaudio": "torchaudio==2.10.0+cu128",
    "torchvision": "torchvision==0.25.0+cu128",
    "einops": "einops",
    "omegaconf": "omegaconf",
    "hydra": "hydra-core",
    "torchdiffeq": "torchdiffeq",
    "rotary_embedding_torch": "rotary-embedding-torch",
    "sentencepiece": "sentencepiece",
    "librosa": "librosa==0.11.0",
    "soundfile": "soundfile",
    "av": "av",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "imageio": "imageio",
    "moviepy": "moviepy==1.0.3",
    "gradio": "gradio==5.29.0",
    "gradio_rangeslider": "gradio_rangeslider",
    "num2words": "num2words==0.5.14",
    "scipy": "scipy",
    "safetensors": "safetensors==0.7.0",
    "optimum": "optimum-quanto==0.2.7",
    "timm": "timm",
    "open_clip": "open_clip_torch",
    "peft": "peft==0.17.0",
    "mcp": "mcp==1.30.0",
    "orjson": "orjson==3.11.6",
    "apprise": "apprise==1.12.0",
    "sounddevice": "sounddevice",
    "pyloudnorm": "pyloudnorm",
    "mutagen": "mutagen",
    "speechbrain": "speechbrain==1.0.3",
    "transformers": "transformers==4.54.0",
    "decord": "decord",
    "mido": "mido==1.3.3",
    "symusic": "symusic==0.6.0",
    "pretty_midi": "pretty_midi==0.2.11.post0",
    "mir_eval": "mir_eval==0.8.2",
    "pyloudnorm": "pyloudnorm",
    "einshape": "einshape==1.0",
    "mcp": "mcp==1.30.0",
    "ffmpeg": "ffmpeg-python",
    "requests": "requests",
    "skimage": "scikit-image>=0.26",
    "segment_anything": "segment-anything",
    "xatlas": "xatlas",
    "fast_simplification": "fast-simplification",
    "numba": "numba",
    "pydantic": "pydantic==2.11.10",
    "ftfy": "ftfy",
    "sentence_transformers": "sentence-transformers",
    "tokenizers": "tokenizers",
    "huggingface_hub": "huggingface-hub==0.36.0",
    "gguf": "gguf==0.17.1",
    "pygame": "pygame",
    "xxhash": "xxhash",
    "s3tokenizer": "s3tokenizer",
    "munch": "munch",
    "aiofiles": "aiofiles",
    "apprise": "apprise==1.12.0",
    "keyring": "keyring==25.7.0",
    "git": "GitPython==3.1.45",
    "pydub": "pydub",
    "pyannote": "pyannote.audio==3.3.2",
    "rembg": "rembg==2.0.65",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python-headless",
    "fast_simplification": "fast-simplification",
    "xatlas": "xatlas",
    "conformer": "conformer==0.3.2",
    "openai_whisper": "openai-whisper==20250625",
    "whisper": "openai-whisper==20250625",
    "spacy": "spacy",
    "fugashi": "fugashi",
    "unidic_lite": "unidic-lite",
    "tiktoken": "tiktoken",
    "optimum": "optimum-quanto==0.2.7",
    "omegaconf": "omegaconf",
    "omegaconf": "omegaconf",
    "torchcodec": "torchcodec==0.10.0",
    "pycocotools": "pycocotools",
    "sherpa_onnx": "sherpa-onnx==1.13.2",
}

PROBES = [
    "mmgp.offload",
    "models.ltx2.ltx_core.model.transformer",
    "models.ltx2.ltx_pipelines.distilled",
    "models.ltx2.ltx2",
    "wgp_cli_help",
    "wgp_cli_dryrun",
]


def run(args, timeout=540, env=None):
    begun = time.monotonic()
    try:
        done = subprocess.run(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, env=env)
        return {"returncode": done.returncode, "seconds": round(time.monotonic()-begun, 1),
                "tail": done.stdout[-4000:]}
    except subprocess.TimeoutExpired as err:
        return {"returncode": 124, "seconds": round(time.monotonic()-begun, 1),
                "tail": f"Timed out after {timeout}s"}
    except Exception as err:
        return {"returncode": -1, "seconds": round(time.monotonic()-begun, 1),
                "tail": f"{type(err).__name__}: {err}"}


def pip_install(packages, pyindex=None):
    cmd = [PYTHON, "-m", "pip", "install", "--disable-pip-version-check",
           "--no-input", "--timeout", "45", "--retries", "2",
           "--prefer-binary", "-c", str(CONSTRAINTS)]
    if pyindex:
        cmd += ["--index-url", pyindex]
    else:
        cmd += ["--index-url", "https://pypi.org/simple"]
    return run(cmd + packages)


def probe_import(module):
    if module == "wgp_cli_help":
        # Headless --help exits after import/CLI option registration,
        # without loading checkpoints, generating, or starting a web server.
        return run([PYTHON, "wgp.py", "--help"], timeout=120)
    if module == "wgp_cli_dryrun":
        from wgp_baseline import SETTINGS
        baseline_root = Path("/cache/ltx25")
        params = baseline_root / "baseline_49frame_settings.json"
        params.write_text(json.dumps(SETTINGS, indent=2) + "\n")
        cfg = baseline_root / "config-baseline"
        cfg.mkdir(parents=True, exist_ok=True)
        # A prior dry-run left an invalid FP8 option in the isolated config.
        # WanGP accepts 'pytorch', not 'disabled', for precise FP8 fallback.
        config_file = cfg / "wgp_config.json"
        if config_file.is_file():
            config = json.loads(config_file.read_text())
            config["fp8_kernels"] = "pytorch"
            config["int8_kernels"] = "disabled"
            config_file.write_text(json.dumps(config, indent=2) + "\n")
        command = [PYTHON, "wgp.py", "--offline", "--config", str(cfg),
                   "--process", str(params),
                   "--output-dir", str(baseline_root / "videos" / "r0_baseline"),
                   "--profile", "0", "--attention", "sdpa",
                   "--no-prevent-power-throttling", "--dry-run"]
        return run(command, timeout=180)
    script = "import importlib; importlib.import_module(" + repr(module) + ");print('IMPORT_OK')"
    return run([PYTHON, "-c", script], timeout=100)


def torch_health():
    return run([PYTHON, "-c",
                "import torch,torchaudio;"
                "assert torch.__version__.startswith('2.10.0+cu128'),torch.__version__;"
                "assert torchaudio.__version__.startswith('2.10.0+cu128'),torchaudio.__version__;"
                "assert torch.cuda.is_available();"
                "assert torch.cuda.get_device_capability()==(7,0);"
                "assert 'sm_70' in torch.cuda.get_arch_list();"
                "print('SM70_TORCH_OK',torch.__version__,torchaudio.__version__)"],
               timeout=70)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {"sha": sha, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "profile": "isolated-sm70-runtime", "model_weights_loaded": False,
              "steps": [], "probes": []}
    try:
        first = torch_health()
        report["initial_torch"] = first
        if first["returncode"]:
            # the torch wheel was verified in prior smoke; torchaudio may not
            # yet be installed. Try the paired cu128 wheel, never replace torch.
            tor = pip_install(["torchaudio==2.10.0+cu128"],
                              "https://download.pytorch.org/whl/cu128")
            report["steps"].append({"group": "torch_audio", **tor})
            if tor["returncode"]:
                report["status"] = "TORCHAUDIO_BLOCKED"
                return
        # Install a torchvision wheel paired to the exact CUDA 12.8 torch.
        # The image/video media path imports torchvision at module import.
        vision = pip_install(["torchvision==0.25.0+cu128"],
                             "https://download.pytorch.org/whl/cu128")
        report["steps"].append({"group": "torch_vision", **vision})
        print("RUNTIME_INSTALL " + json.dumps({"name":"torch_vision",**vision})[:1800], flush=True)
        if vision["returncode"]:
            report["status"] = "TORCHVISION_BLOCKED"
            return

        for name, packages in GROUPS:
            result = pip_install(packages)
            report["steps"].append({"group": name, **result})
            print("RUNTIME_INSTALL " + json.dumps({"name":name,**result})[:4500], flush=True)
            if result["returncode"]:
                report["status"] = "INSTALL_BLOCKED"
                # Do not pretend subsequent probes are impossible; run them
                # so the exact module blocker is still recorded.
                break

        tried = set()
        previous = {}
        for attempt in range(40):
            results = []
            for module in PROBES:
                # Skip previously passed imports; they do not need to rerun
                # merely because another package was installed.
                if previous.get(module, {}).get("status") == "PASS":
                    results.append(previous[module])
                    continue
                result = probe_import(module)
                data = {"name": module, "status": "PASS" if not result["returncode"] else "ERROR",
                        "tail": result["tail"][-1100:]}
                results.append(data)
                previous[module] = data
            report["probes"] = results
            print("RUNTIME_PROBES " + json.dumps(results)[:9000], flush=True)
            missing = None
            for row in results:
                if row["status"] == "PASS":
                    continue
                matches = re.findall(r"No module named ['\"]([^'\"]+)['\"]", row["tail"])
                for entry in matches:
                    root = entry.split(".")[0]
                    if root in KNOWN_MISSING and root not in tried:
                        missing = root
                        break
                if missing: break
            if not missing: break
            tried.add(missing)
            package = KNOWN_MISSING[missing]
            idx = "https://download.pytorch.org/whl/cu128" if missing in {"torchaudio", "torchvision"} else None
            inst = pip_install([package], idx)
            report["steps"].append({"group":"missing_import:"+missing, **inst})
            print("RUNTIME_AUTO_DEP " + json.dumps({"module":missing,"package":package,**inst})[:2400],flush=True)
            if inst["returncode"]:
                break

        # Recheck *all* imports after dependency installation: adding a
        # package can alter another dependency, so cached PASS values are not
        # sufficient for the final validation gate.
        final_results = []
        for module in PROBES:
            result = probe_import(module)
            final_results.append({
                "name": module,
                "status": "PASS" if not result["returncode"] else "ERROR",
                "tail": result["tail"][-1100:],
            })
        report["probes"] = final_results
        health = torch_health()
        report["torch_after"] = health
        success = all(p["status"]=="PASS" for p in report["probes"])
        report["status"] = "PASS" if success and health["returncode"] == 0 else "BLOCKED"
    except Exception as err:
        report["status"] = "ERROR"
        report["error"] = str(err)[:800]
    finally:
        target = OUT / f"runtime_bootstrap_{sha[:12]}.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print("RUNTIME_BOOTSTRAP_STATUS " + report.get("status", "UNKNOWN"), flush=True)
        print("RUNTIME_BOOTSTRAP_REPORT " + str(target), flush=True)
        if report.get("probes"):
            print("RUNTIME_FINAL_PROBES " + json.dumps(report["probes"])[:13000], flush=True)


if __name__=="__main__":
    main()
