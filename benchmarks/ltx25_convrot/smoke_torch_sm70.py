"""Phase 0 V100 PyTorch smoke test. No model download or generation."""
import json
import platform
import torch

def main():
    assert torch.__version__.startswith("2.10.0"), torch.__version__
    assert torch.cuda.is_available(), "CUDA not available to PyTorch"
    assert torch.cuda.device_count() == 1, "Runner should see only the V100"
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) == (7, 0), (props.major, props.minor)
    arches = torch.cuda.get_arch_list()
    assert "sm_70" in arches, f"PyTorch wheel missing SM70: {arches}"

    torch.manual_seed(42)
    a = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    out = a @ b
    ref = a.float() @ b.float()
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)

    # V100 has no native BF16 GEMM. Validate BF16 interface + FP16 internal route.
    bf16_acts = torch.randn((64, 64), device="cuda", dtype=torch.float32).bfloat16()
    fp16_result = bf16_acts.to(torch.float16) @ b
    bf16_result = fp16_result.to(torch.bfloat16)
    assert bf16_result.dtype is torch.bfloat16
    assert torch.isfinite(bf16_result).all().item()
    torch.cuda.synchronize()

    info = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "arch_list": arches,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_count": torch.cuda.device_count(),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "platform": platform.platform(),
        "meaning": "PyTorch SM70 math works; NOT ConvRot accuracy or benchmark evidence",
    }
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
