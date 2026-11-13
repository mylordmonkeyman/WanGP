# Copyright (c) 2026. Experimental LTX-2.5 V100 ConvRot workspace.
"""Opt-in reusable CUDA scratch, not a weight cache.

Only stores *activation, accumulator, and row metadata* tensors. Does NOT
retain any FP16 copy of a model weight: q.to(float16) is transient per
linear, and original INT8 checkpoint storage stays under MMGP's control.

This is a bounded experiment. A single pool per GPU owns a single CUDA
stream. If a call uses another stream, an oversized shape, a concurrent
Python caller, or insufficient pool capacity, the caller uses its normal
ephemeral allocations instead. The pool is never used across streams.

Releasing a lease means all dependent CUDA operations, including the cast
from FP32 c32 to the returned BF16/FP16 output, have been *enqueued* on the
pool's stream. FIFO CUDA stream ordering (not host synchronization)
protects the next acquisition of the same memory on that same stream.
"""
from __future__ import annotations

import atexit
import json
import os
import threading

import torch

ENABLED = os.environ.get("WAN2GP_LTX25_E_WORKSPACE", "") == "1"
# Constrain persistent scratch to 96 MiB / GPU by default. This does not
# include transient FP16 weight conversion, FP32 rotated activations or
# PyTorch's own unrelated allocator cache.
LIMIT_BYTES = max(0, min(256, int(os.environ.get("LTX25_E_WORKSPACE_MAX_MIB", "96")))) * (1 << 20)
_POOLS = {}
_LOCK = threading.RLock()
_STATS = {"acquired": 0, "miss_oversize": 0, "miss_stream": 0,
          "miss_reentrant": 0, "growths": 0, "max_persistent_bytes": 0,
          "enabled": ENABLED, "persistent_weight_copies": 0}


class _Pool:
    def __init__(self, device, stream_id):
        self.device = device
        self.stream_id = stream_id
        self.lock = threading.Lock()
        self.x16 = None
        self.c32 = None
        self.exponents = None
        self.slow_rows = None
        self.size_x = 0
        self.size_c = 0
        self.size_rows = 0

    def bytes_held(self):
        return 2 * self.size_x + 4 * self.size_c + 8 * self.size_rows

    def ensure(self, m, n, k):
        target_x = max(self.size_x, m * k)
        target_c = max(self.size_c, m * n)
        target_rows = max(self.size_rows, m)
        target_bytes = 2 * target_x + 4 * target_c + 8 * target_rows
        if target_bytes > LIMIT_BYTES:
            return False
        # Create new tensors before releasing old. No copy needed: the next
        # operation fully writes every slice it reads. Called under the lease
        # lock; no live prior use from another Python call in this pool.
        if target_x > self.size_x:
            self.x16 = torch.empty((target_x,), dtype=torch.float16, device=self.device)
            self.size_x = target_x
            _STATS["growths"] += 1
        if target_c > self.size_c:
            self.c32 = torch.empty((target_c,), dtype=torch.float32, device=self.device)
            self.size_c = target_c
            _STATS["growths"] += 1
        if target_rows > self.size_rows:
            self.exponents = torch.empty((target_rows,), dtype=torch.int32, device=self.device)
            self.slow_rows = torch.empty((target_rows,), dtype=torch.int32, device=self.device)
            self.size_rows = target_rows
            _STATS["growths"] += 1
        _STATS["max_persistent_bytes"] = max(_STATS["max_persistent_bytes"], self.bytes_held())
        return True


class Lease:
    def __init__(self, pool, m, n, k):
        self._pool = pool
        self.released = False
        self.x16 = pool.x16[:m*k].view(m, k)
        self.c32 = pool.c32[:m*n].view(m, n)
        self.exponents = pool.exponents[:m]
        self.slow_rows = pool.slow_rows[:m]

    def release(self):
        if not self.released:
            self.released = True
            self._pool.lock.release()


def try_acquire(m, n, k, device, stream):
    """Return a lease or None, with no GPU-to-host scalar synchronization."""
    if not ENABLED or not LIMIT_BYTES:
        return None
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    # Explicitly reject nondefault streams for this first research version.
    stream_id = stream.cuda_stream
    if stream_id != torch.cuda.default_stream(device).cuda_stream:
        _STATS["miss_stream"] += 1
        return None
    with _LOCK:
        pool = _POOLS.get(index)
        if pool is None:
            pool = _Pool(device, stream_id)
            _POOLS[index] = pool
        elif pool.stream_id != stream_id:
            _STATS["miss_stream"] += 1
            return None
    if not pool.lock.acquire(blocking=False):
        _STATS["miss_reentrant"] += 1
        return None
    try:
        if not pool.ensure(m, n, k):
            _STATS["miss_oversize"] += 1
            pool.lock.release()
            return None
        _STATS["acquired"] += 1
        return Lease(pool, m, n, k)
    except BaseException:
        pool.lock.release()
        raise


def _summary():
    if ENABLED:
        print("[LTX25_WORKSPACE] " + json.dumps(_STATS, sort_keys=True), flush=True)


if ENABLED:
    atexit.register(_summary)
