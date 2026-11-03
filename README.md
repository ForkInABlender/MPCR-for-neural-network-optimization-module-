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

To install, git clone this repository and then copy into your package distribution folder or use as is directly from the repository folder. Or if you're feeling extra clever, put it into a path that is sys.path and sys.modules compatible.

no pip, conda,or wheel installer required. 

## Nimble & nimpy for nim_backend.py

This is no longer optional even for Android OS running python3.10.11 via termux.apk with proot of Ubuntu 22.04 image installed from makers of andronix.apk;

Download the version compatible with your device CPU: https://github.com/nim-lang/nimble/releases/tag/v0.26.2

if you don't know what CPU you have, run ``uname -a`` in the terminal.

To install, you must put nimble on /root. from /root/nimble, run ``/root/nimble install nimpy``.
Let the install work then crash, and rerun. Then copy the path for nim compiler into MPCR/nim_backend.py


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

