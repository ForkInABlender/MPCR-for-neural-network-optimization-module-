"""Small reproducible CPU numerical benchmark; not a hardware certification tool."""
import time
import numpy as np
from .kernels import dense_matmul

def benchmark_matmul(size=512, repeats=3, seed=1234):
    if size < 1 or repeats < 1:
        raise ValueError("size and repeats must be positive")
    rng = np.random.default_rng(seed)
    a = rng.random((size, size), dtype=np.float64)
    b = rng.random((size, size), dtype=np.float64)
    dense_matmul(a, b)  # warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        dense_matmul(a, b)
        samples.append(time.perf_counter() - start)
    return {
        "size": size,
        "repeats": repeats,
        "seconds": samples,
        "median_seconds": float(np.median(samples)),
        "gflops_estimate": float((2 * size**3) / np.median(samples) / 1e9),
    }
