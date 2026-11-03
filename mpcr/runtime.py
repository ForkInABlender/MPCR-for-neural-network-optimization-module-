"""MPCR orchestration and CPU/GPU execution runtime."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import AbstractContextManager
import os
from .kernels import dense_matmul, sparse_matmul, gpu_matmul, auto_matmul, available_devices, tiled_matmul
from .topology import partition_ranges, sponge_region_order, recursive_ranges, sponge_tiles
from .nim_backend import nim_available
from .shared import SharedNDArray

class MPCRRuntime(AbstractContextManager):
    def __init__(self, thread_workers=None, process_workers=None,
                 enable_processes=True, enable_threads=True):
        cpu_count = os.cpu_count() or 1
        self.thread_workers = max(1, thread_workers or cpu_count)
        self.process_workers = max(1, process_workers or cpu_count)
        self.enable_threads = enable_threads
        self.enable_processes = enable_processes
        self._threads = ThreadPoolExecutor(
            max_workers=self.thread_workers, thread_name_prefix="mpcr-thread"
        ) if enable_threads else None
        self._processes = ProcessPoolExecutor(
            max_workers=self.process_workers
        ) if enable_processes else None
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("MPCRRuntime is shut down")

    def submit(self, fn, *args, backend="thread", **kwargs):
        self._check_open()
        if backend == "thread":
            if self._threads is None:
                raise RuntimeError("thread executor is disabled")
            return self._threads.submit(fn, *args, **kwargs)
        if backend == "process":
            if self._processes is None:
                raise RuntimeError("process executor is disabled")
            return self._processes.submit(fn, *args, **kwargs)
        raise ValueError("backend must be 'thread' or 'process'")

    def map(self, fn, iterable, backend="thread", chunksize=1):
        self._check_open()
        items = list(iterable)
        if backend == "thread":
            if self._threads is None:
                raise RuntimeError("thread executor is disabled")
            return list(self._threads.map(fn, items))
        if backend == "process":
            if self._processes is None:
                raise RuntimeError("process executor is disabled")
            return list(self._processes.map(fn, items, chunksize=chunksize))
        raise ValueError("backend must be 'thread' or 'process'")

    def run_graph(self, graph, backend="thread"):
        self._check_open()
        graph.validate()
        results, pending = {}, set(graph.tasks)
        while pending:
            ready = sorted(
                tid for tid in pending
                if set(graph.tasks[tid].deps).issubset(results)
            )
            if not ready:
                raise RuntimeError("graph deadlock or dependency cycle")
            futures = {}
            for tid in ready:
                task = graph.tasks[tid]
                dep_values = [results[d] for d in task.deps]
                futures[tid] = self.submit(
                    _invoke_with_dependencies, task.fn, task.args, task.kwargs,
                    dep_values, backend=backend
                )
            for tid, future in futures.items():
                results[tid] = future.result()
                pending.remove(tid)
        return results

    def partition(self, length, parts, depth=1):
        ranges = partition_ranges(length, parts)
        order = sponge_region_order(len(ranges), depth)
        return tuple(
            {"index": i, "start": start, "stop": stop, "region": order[i]}
            for i, (start, stop) in enumerate(ranges)
        )

    def matmul(self, a, b, backend="auto", device=None):
        self._check_open()
        if backend == "numpy":
            return dense_matmul(a, b)
        if backend == "scipy":
            return sparse_matmul(a, b)
        if backend == "gpu":
            return gpu_matmul(a, b, device=device)
        if backend == "auto":
            return auto_matmul(a, b)
        raise ValueError("backend must be auto, numpy, scipy, or gpu")


    def recursive_partition(self, length, max_leaf_size, min_leaf_size=1):
        """Return recursively bisected work ranges with sponge region labels."""
        ranges = recursive_ranges(length, max_leaf_size, min_leaf_size)
        regions = sponge_region_order(len(ranges), depth=1)
        return tuple(
            {"index": i, "start": a, "stop": b, "region": regions[i]}
            for i, (a, b) in enumerate(ranges)
        )

    def sponge_tiles(self, rows, cols, tile_rows=256, tile_cols=256, depth=1):
        return sponge_tiles(rows, cols, tile_rows, tile_cols, depth)

    def tiled_matmul(self, a, b, tile_rows=256):
        self._check_open()
        return tiled_matmul(a, b, self, tile_rows=tile_rows)

    def shared_array(self, array):
        """Create an owner-managed shared-memory copy of an array."""
        self._check_open()
        return SharedNDArray(array)

    def choose_backend(self, task_kind="numeric", estimated_bytes=0,
                       independent_tasks=1, gpu_preferred=False):
        """Transparent heuristic; callers can always override the result.

        It is intentionally conservative: process pools are not selected for
        arrays because serialization or shared-memory setup may dominate.
        """
        self._check_open()
        if gpu_preferred and ("cuda" in available_devices() or "mps" in available_devices()):
            return "gpu"
        if task_kind in ("python_cpu", "cpu_bound") and independent_tasks > 1 and self._processes:
            return "process"
        if task_kind in ("native_numeric", "numpy", "scipy") and self._threads:
            return "thread"
        if estimated_bytes >= 64 * 1024 * 1024 and self._threads:
            return "thread"
        return "sequential"

    def diagnostics(self):
        return {
            "thread_workers": self.thread_workers if self._threads else 0,
            "process_workers": self.process_workers if self._processes else 0,
            "cpu_count": os.cpu_count() or 1,
            "devices": available_devices(),
            "shared_memory": True,
            "scheduling_policy": "explicit task hints; conservative heuristic",
            "nim_extension": nim_available(),
            "closed": self._closed,
        }

    def shutdown(self, wait=True, cancel_futures=False):
        if self._closed:
            return
        self._closed = True
        if self._threads:
            self._threads.shutdown(wait=wait, cancel_futures=cancel_futures)
        if self._processes:
            self._processes.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

def _invoke_with_dependencies(fn, args, kwargs, dep_values):
    # Dependencies are appended positionally after explicitly supplied arguments.
    return fn(*args, *dep_values, **kwargs)
