"""CPU-only robust performance decision helpers for Hunyuan P0/P1 experiments."""
from __future__ import annotations

import math
import random
import statistics


def mad(values):
    if not values:
        raise ValueError("empty measurement sample")
    mid = statistics.median(values)
    return statistics.median(abs(v - mid) for v in values)


def compare_baseline_candidate(baseline, candidate, *, bootstrap=10000, seed=20261112):
    """Positive Δ means candidate is faster. A/B/B/A ordering is enforced by caller.

    A conclusive result needs Δ > 2*S and a 95% bootstrap CI entirely
    above zero; this utility intentionally does not grant a quality gate.
    """
    a, b = list(baseline), list(candidate)
    if len(a) < 4 or len(b) < 4:
        raise ValueError("at least four valid hot observations per arm required")
    if bootstrap < 200:
        raise ValueError("insufficient bootstrap repetitions")
    med_a, med_b = statistics.median(a), statistics.median(b)
    delta = med_a - med_b
    s = 1.4826 * math.hypot(mad(a), mad(b))
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap):
        draw_a = statistics.median(rng.choices(a, k=len(a)))
        draw_b = statistics.median(rng.choices(b, k=len(b)))
        draws.append(draw_a - draw_b)
    draws.sort()
    lower = draws[int(0.025 * (bootstrap - 1))]
    upper = draws[int(0.975 * (bootstrap - 1))]
    conclusive = s > 0 and delta > 2 * s and lower > 0
    return {
        "baseline_median_s": med_a,
        "candidate_median_s": med_b,
        "delta_s": delta,
        "improvement_percent": (100 * delta / med_a) if med_a else None,
        "mad_baseline_s": mad(a),
        "mad_candidate_s": mad(b),
        "robust_spread_s": s,
        "delta_over_2s": delta / (2 * s) if s else None,
        "bootstrap_95ci_delta_s": [lower, upper],
        "noise_gate": "conclusive_improvement" if conclusive else "inconclusive_or_regression",
        "requires_more_runs": s == 0,
    }
