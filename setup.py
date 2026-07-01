"""setuptools shim — declares the simdq C extension.

The rest of the package metadata (name, version, deps, scripts, package data)
lives in pyproject.toml. This file exists solely because PEP 621 declarative
build configuration cannot describe C extensions; we keep the shim minimal
and let setuptools merge the two sources.

The simdq kernels have SIMD paths for x86 (AVX2 / AVX-512) and ARM (NEON on
aarch64). Other hosts are skipped rather than failing the whole install —
code that touches simdq will raise ImportError, but pure-Python docuverse
still installs. OpenMP is required on every SIMD-enabled host.

Override with DOCUVERSE_SIMDQ=1 to force the build, or DOCUVERSE_SIMDQ=0 to
force skipping.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys

from setuptools import Extension, setup

# setuptools requires sources/include_dirs to be RELATIVE paths from the
# directory holding setup.py, not absolute paths.
NATIVE = "docuverse/engines/retrieval/simdq/_native"


def _brew_prefix(pkg: str) -> str | None:
    try:
        out = subprocess.run(
            ["brew", "--prefix", pkg],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return out if out and os.path.isdir(out) else None


def _build_simdq_extension() -> Extension | None:
    force = os.environ.get("DOCUVERSE_SIMDQ")
    if force == "0":
        sys.stderr.write("[simdq setup.py] DOCUVERSE_SIMDQ=0: skipping native build.\n")
        return None

    system = platform.system()
    machine = platform.machine().lower()
    is_x86 = machine in ("x86_64", "amd64", "i386", "i686")
    is_arm64 = machine in ("arm64", "aarch64")

    if not (is_x86 or is_arm64) and force != "1":
        sys.stderr.write(
            f"[simdq setup.py] Host arch {machine!r} has no simdq SIMD path "
            f"(supported: x86 AVX2/AVX-512, arm64 NEON). Skipping native "
            f"build. Set DOCUVERSE_SIMDQ=1 to force.\n"
        )
        return None

    compile_args = ["-O3", "-march=native"]
    link_args: list[str] = []
    include_dirs = [f"{NATIVE}/include"]
    library_dirs: list[str] = []
    libraries: list[str] = []

    if system == "Darwin":
        # Apple clang doesn't accept -fopenmp directly; it needs -Xpreprocessor
        # -fopenmp for the compile step and explicit -lomp for the link step,
        # plus libomp's include/lib paths from Homebrew.
        libomp = _brew_prefix("libomp")
        if libomp is None:
            sys.stderr.write(
                "[simdq setup.py] libomp not found (brew install libomp). "
                "Skipping native build. Set DOCUVERSE_SIMDQ=1 after installing "
                "libomp to force.\n"
            )
            return None
        compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{libomp}/include"]
        link_args += [f"-L{libomp}/lib", "-lomp"]
        library_dirs.append(f"{libomp}/lib")
        libraries.append("omp")
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    return Extension(
        name="docuverse.engines.retrieval.simdq._simdq_native",
        sources=[f"{NATIVE}/bindings/module.c"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )


ext = _build_simdq_extension()
setup(ext_modules=[ext] if ext is not None else [])
