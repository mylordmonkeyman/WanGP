#!/usr/bin/env python3
"""Check BF16 bit-identical tiled epilogues at observed 241-frame audio row count."""
import bench_epilogue_tiling as test

# 241f Stage1/Stage2 audio length was 251 in a previous true-GPU trace.
# Exact whole-video dimensions / calls are checked separately in the ABBA.
test.CASES = [
    (251, 2048, 2048, "safe", True),
    (251, 4096, 2048, "safe", False),
    (251, 16384, 4096, "safe", True),
    (251, 16384, 4096, "one_slow", True),
    (251, 16384, 4096, "some_slow", False),
]
if __name__ == "__main__":
    test.main()
    print("EPILOGUE_TILED_241F_MICROBENCH_PASS", flush=True)
