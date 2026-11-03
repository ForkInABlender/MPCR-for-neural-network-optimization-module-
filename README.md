# MPCR — Multidimensional Porous Computational Runtime

MPCR is an experimental Python runtime for dependency-aware workloads across CPU
threads, CPU processes, SciPy/NumPy numerical kernels, and an optional PyTorch GPU
backend. Its architecture uses M-theory-inspired execution layers and a
Menger-sponge-inspired recursive topology as engineering metaphors.

These are architectural inspirations, not claims that M-theory describes computer
hardware or that fractal subdivision inherently improves performance. Performance
must be measured against a baseline.

## Features

- Task DAG with dependency validation and parallel execution by ready wave.
- Thread and process executors.
- Recursive Menger-sponge cell generation (20 retained cells per 3-D subdivision).
- Balanced array partitioning, with sponge-derived region ordering metadata.
- NumPy dense matrix multiplication and SciPy sparse matrix multiplication.
- Optional PyTorch device discovery and GPU matrix multiplication.
- Optional NimPy extension hook with a pure-Python fallback.
- Runtime diagnostics and a small benchmark utility.

## Install

```bash
python -m pip install -e .
# Optional GPU backend:
python -m pip install -e ".[gpu]"
```

### Optional Nim backend

Install Nim and Nimble, then install NimPy:

```bash
nimble install nimpy
```

Compile the extension from the project root:

```bash
nim c --app:lib --threads:on --out:mpcr_nim.so mpcr/nim/mpcr_nim.nim
```

The extension is optional. `mpcr.nim_backend` detects it and falls back to Python
when it is not present. The Nim source uses NimPy's `exportpy` interface. A Nim
compiler and NimPy were not available in the build environment used for this
package, so the extension source is supplied but not represented as compiled or
tested.

## Quick start

```python
import numpy as np
from mpcr import MPCRRuntime

runtime = MPCRRuntime(thread_workers=4, process_workers=2)

a = np.random.default_rng(1).random((512, 512))
b = np.random.default_rng(2).random((512, 512))

c = runtime.matmul(a, b, backend="auto")
print(c.shape, runtime.diagnostics())

runtime.shutdown()
```

## Task graph

Functions submitted to the process backend must be importable top-level functions
and their arguments/results must be pickleable. Lambdas and nested functions are
suitable for the thread backend, not generally for process workers.

```python
from mpcr import MPCRRuntime, TaskGraph

def square(x):
    return x * x

graph = TaskGraph()
graph.add("a", square, 4)
graph.add("b", square, 5)
graph.add("sum", lambda x, y: x + y, deps=("a", "b"))

with MPCRRuntime() as runtime:
    result = runtime.run_graph(graph, backend="thread")
    print(result["sum"])  # 41
```

## Architecture

- Orchestration layer: validates and schedules task dependencies.
- Topology layer: generates 20-way Menger-sponge subdivision metadata.
- Execution layer: threads, processes, or GPU-capable numerical kernels.
- Numerical layer: NumPy/SciPy and optional PyTorch.
- Native layer: optional NimPy extension.

## Important boundaries

- This runtime cannot change the infrastructure or inference scheduler of a hosted
  language model. It accelerates only workloads explicitly executed through MPCR.
- GPU availability is detected at runtime. If PyTorch is installed but no supported
  accelerator is available, `backend="auto"` uses CPU.
- Python threads do not guarantee speedup for CPU-bound pure-Python code because of
  the CPython GIL. Native NumPy/SciPy operations may release the GIL.
- Process execution has serialization and startup costs; use it for sufficiently
  coarse independent tasks.
- Recursive subdivision can make performance worse when tasks are too small.


## 0.2 additions

- `SharedNDArray` provides owner/attacher shared-memory transport for contiguous,
  non-object NumPy arrays. The owner controls lifetime and unlinks on context exit.
- `recursive_ranges()` creates recursive bisection leaves.
- `sponge_tiles()` creates a complete non-overlapping 2-D tiling with sponge-derived
  ordering metadata.
- `MPCRRuntime.tiled_matmul()` runs row-tiled dense multiplication on the thread pool.
- `MPCRRuntime.choose_backend()` offers a conservative, inspectable heuristic.
- Numerical imports are lazy, so task orchestration can be imported without eagerly
  loading SciPy or NumPy kernels.

