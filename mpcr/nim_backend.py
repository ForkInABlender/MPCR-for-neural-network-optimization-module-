#Dylan Kenneth Eliot

"""MPCR Nim native extension loader."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "mpcr" / "nim" / "mpcr_nim.nim"
LIBRARY = ROOT / "mpcr" / "nim" / "mpcr_nim.so"

STAGED_LIBRARY = (
    Path("/tmp/mpcr_runtime/mpcr/nim/mpcr_nim.so")
)


def compile_nim():
    """Compile the Nim source into a shared Python extension."""
    command = [
        "/root/.nimble/pkgs2/nim-2.4.0-e4e638a20c4650cc6b463f8b19f5009db3a916b5/bin/nim",
        "c",
        "--forceBuild:on",
        "-d:release",
        "--app:lib",
        "--threads:on",
        "--mm:orc",
        "--out:" + str(LIBRARY),
        str(SOURCE),
    ]

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Nim compilation failed:\n" + result.stdout
        )

    if not LIBRARY.is_file():
        raise RuntimeError("Nim compiler did not produce the .so file")

    return result.stdout


def load_nim():
    """Stage and import the compiled Nim extension."""
    STAGED_LIBRARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        str(LIBRARY),
        str(STAGED_LIBRARY),
    )

    spec = importlib.util.spec_from_file_location(
        "mpcr_nim",
        str(STAGED_LIBRARY),
    )

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load mpcr_nim.so")

    module = importlib.util.module_from_spec(spec)
    sys.modules["mpcr_nim"] = module
    spec.loader.exec_module(module)

    return module


# Compile and import immediately.
build_output = compile_nim()
_nim = load_nim()

# Export every Python-visible callable from the shared library.
for _name in dir(_nim):
    if not (_name.startswith("__") and _name.endswith("__")):
        _value = getattr(_nim, _name)

        if callable(_value):
            globals()[_name] = _value


def has_nim():
    return True


def nim_functions():
    return [
        name
        for name in dir(_nim)
        if not (name.startswith("__") and name.endswith("__"))
        and callable(getattr(_nim, name))
    ]


__all__ = nim_functions() + [
    "has_nim",
    "nim_available",
    "nim_functions",
    "compile_nim",
    "build_output",
]
