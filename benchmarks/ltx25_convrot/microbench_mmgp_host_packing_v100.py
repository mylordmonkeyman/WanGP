#!/usr/bin/env python3
"""Bounded CPU-only packing microbench for the cold LTX-2.5 MMGP preload path.

Measures memcpy from representative pageable tensor leaves to a single pinned
CPU destination. Does not load model weights, allocate GPU weights, or alter
runtime behavior. Reports medians and checks every copied byte.
"""
import ctypes
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch

MIB = 1 << 20
TOTAL = 384 * MIB
SIZES = (1, 4, 16, 64)  # MiB per source leaf
TRIALS = 3

def pack(dst, runs, workers, pool):
    pending = []
    for src, nbytes, offset in runs:
        if workers == 1:
            ctypes.memmove(dst + offset, src, nbytes)
        else:
            chunk = (nbytes + workers - 1) // workers
            for start in range(0, nbytes, chunk):
                pending.append(pool.submit(ctypes.memmove, dst + offset + start,
                                           src + start, min(chunk, nbytes - start)))
    for task in pending:
        task.result()

def main():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
    result = []
    for size_mib in SIZES:
        count = TOTAL // (size_mib * MIB)
        leaves = [torch.full((size_mib * MIB,), i % 251, dtype=torch.uint8)
                  for i in range(count)]
        destination = torch.empty(TOTAL, dtype=torch.uint8, pin_memory=True)
        runs = [(leaf.data_ptr(), leaf.numel(), i * size_mib * MIB)
                for i, leaf in enumerate(leaves)]
        for workers in (1, 2, 4, 8):
            times = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for trial in range(TRIALS + 1):
                    begin = time.perf_counter()
                    pack(destination.data_ptr(), runs, workers, pool)
                    elapsed = 1000 * (time.perf_counter() - begin)
                    # Check all bytes, not merely the first and last byte.
                    for i, leaf in enumerate(leaves):
                        segment = destination.narrow(0, i * leaf.numel(), leaf.numel())
                        assert torch.equal(segment, leaf), (size_mib, workers, trial, i)
                    if trial:
                        times.append(round(elapsed, 3))
            result.append({"leaf_mib": size_mib, "workers": workers,
                           "median_pack_ms": statistics.median(times),
                           "trials_ms": times, "verified": True})
            print("LTX25_PACK_MICROBENCH_CASE", json.dumps(result[-1]), flush=True)
        del leaves, destination
    print("LTX25_PACK_MICROBENCH_PASS", json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
