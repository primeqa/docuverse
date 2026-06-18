"""setuptools shim — declares the simdq C extension.

The rest of the package metadata (name, version, deps, scripts, package data)
lives in pyproject.toml. This file exists solely because PEP 621 declarative
build configuration cannot describe C extensions; we keep the shim minimal
and let setuptools merge the two sources.

The extension is built unconditionally during `pip install -e .` /
`pip install .`. If a target host lacks AVX2 the build will fail at compile
time with an explicit error from <immintrin.h>; we make no attempt to ship
a scalar-only fallback in v1.
"""
from __future__ import annotations

import platform
import sys

from setuptools import Extension, setup

# setuptools requires sources/include_dirs to be RELATIVE paths from the
# directory holding setup.py, not absolute paths.
NATIVE = "docuverse/engines/retrieval/simdq/_native"

extra_compile_args = ["-O3", "-march=native", "-fopenmp"]
extra_link_args = ["-fopenmp"]

# macOS: clang's libomp is not on the default linker path; users on Mac who
# want simdq must `brew install libomp` and pass CPPFLAGS/LDFLAGS themselves.
# We do not paper over this here — the extension is Linux-first in v1.
if platform.system() == "Darwin":
    sys.stderr.write(
        "[simdq setup.py] Building on macOS. Ensure libomp is installed "
        "and CPPFLAGS/LDFLAGS reach the compiler; v1 is Linux-first.\n"
    )

simdq_ext = Extension(
    name="docuverse.engines.retrieval.simdq._simdq_native",
    sources=[f"{NATIVE}/bindings/module.c"],
    include_dirs=[f"{NATIVE}/include"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(ext_modules=[simdq_ext])
