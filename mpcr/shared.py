"""Shared-memory transport for NumPy arrays across process boundaries."""
from dataclasses import dataclass
from multiprocessing import shared_memory
import numpy as np

@dataclass(frozen=True)
class SharedArraySpec:
    name: str
    shape: tuple
    dtype: str
    nbytes: int

class SharedNDArray:
    """Owns a shared-memory block containing a contiguous NumPy array.

    The owner must remain alive until all attached workers have finished.
    Attached views should be closed before the owner is unlinked.
    """
    def __init__(self, array, name=None):
        source = np.ascontiguousarray(array)
        if source.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot be shared safely")
        self._shm = shared_memory.SharedMemory(
            create=True, size=max(1, source.nbytes), name=name
        )
        self._owner = True
        self._closed = False
        self.spec = SharedArraySpec(
            self._shm.name, tuple(source.shape), source.dtype.str, source.nbytes
        )
        self.array = np.ndarray(source.shape, dtype=source.dtype, buffer=self._shm.buf)
        self.array[...] = source

    @classmethod
    def attach(cls, spec):
        obj = cls.__new__(cls)
        obj._shm = shared_memory.SharedMemory(name=spec.name, create=False)
        obj._owner = False
        obj._closed = False
        obj.spec = spec
        obj.array = np.ndarray(
            spec.shape, dtype=np.dtype(spec.dtype), buffer=obj._shm.buf
        )
        return obj

    def close(self):
        if not self._closed:
            self.array = None
            self._shm.close()
            self._closed = True

    def unlink(self):
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        if self._owner:
            self.unlink()
        return False
