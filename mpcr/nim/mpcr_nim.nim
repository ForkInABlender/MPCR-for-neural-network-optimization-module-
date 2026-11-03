# Optional native extension. Build with Nim + NimPy installed:
# nim c --app:lib --threads:on --out:mpcr_nim.so mpcr/nim/mpcr_nim.nim

# MPCR Nim native Python extension
# Nim 2.4.x + NimPy 0.2.1
# Python 3.10 ABI

import nimpy

proc nim_add(a: float64, b: float64): float64 {.exportpy.} =
  a + b

proc nim_available(): bool {.exportpy.} =
  true
