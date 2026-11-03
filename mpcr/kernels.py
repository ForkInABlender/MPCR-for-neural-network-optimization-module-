#Dylan Kenneth Eliot

"""Numerical kernels with lazy imports and optional PyTorch acceleration."""
def _numpy():
    try:
        import numpy
        return numpy
    except ImportError as exc:
        raise RuntimeError("Numerical kernels require NumPy") from exc

def available_devices():
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return tuple(devices)

def dense_matmul(a, b, out=None):
    np = _numpy()
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape[1] != bb.shape[0]:
        raise ValueError("matmul expects compatible 2-D arrays")
    result = np.matmul(aa, bb)
    if out is not None:
        np.copyto(out, result)
        return out
    return result

def sparse_matmul(a, b):
    try:
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError("Sparse kernels require SciPy") from exc
    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    return a @ b

def gpu_matmul(a, b, device=None):
    np = _numpy()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU backend requires optional dependency: torch") from exc
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    ta, tb = torch.as_tensor(np.asarray(a), device=device), torch.as_tensor(np.asarray(b), device=device)
    return (ta @ tb).detach().cpu().numpy()

def auto_matmul(a, b):
    devices = available_devices()
    if "cuda" in devices or "mps" in devices:
        return gpu_matmul(a, b)
    return dense_matmul(a, b)

def _matmul_row_tile(a, b, row_range):
    r0, r1 = row_range
    return r0, r1, a[r0:r1] @ b

def tiled_matmul(a, b, runtime, tile_rows=256):
    """Compute dense matmul in row tiles using MPCR's thread executor."""
    np = _numpy()
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape[1] != bb.shape[0]:
        raise ValueError("matmul expects compatible 2-D arrays")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    ranges = tuple((i, min(i + tile_rows, aa.shape[0])) for i in range(0, aa.shape[0], tile_rows))
    out = np.empty((aa.shape[0], bb.shape[1]), dtype=np.result_type(aa, bb))
    if not ranges:
        return out
    futures = [runtime.submit(_matmul_row_tile, aa, bb, rr, backend="thread") for rr in ranges]
    for future in futures:
        r0, r1, block = future.result()
        out[r0:r1] = block
    return out
