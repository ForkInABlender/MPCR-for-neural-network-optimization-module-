"""Optional NimPy extension adapter; always retains a Python fallback."""
try:
    import mpcr_nim as _nim
except Exception:
    _nim = None

def nim_available():
    return _nim is not None

def nim_add(a, b):
    if _nim is not None and hasattr(_nim, "add_f64"):
        return _nim.add_f64(float(a), float(b))
    return float(a) + float(b)
