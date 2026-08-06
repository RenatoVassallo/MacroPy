"""X-13ARIMA-SEATS binary bundled with MacroPy.

MacroPy ships the US Census Bureau's ``x13as`` executable per platform
under ``MacroPy/bin/<system>-<machine>/`` so downstream projects (for
example NowForecasting's seasonal-adjustment preprocess) do not depend on
a system-wide install. The macOS arm64 binary is compiled from the
official ASCII Fortran source (v1.1 build 62, makefile.gf) with Homebrew
gfortran; other platforms can be added by dropping a binary into the
matching subdirectory.
"""

from __future__ import annotations

import os
import platform
import stat
from pathlib import Path

_BIN_DIR = Path(__file__).resolve().parent / "bin"


def _platform_tag() -> str:
    system = platform.system().lower()      # darwin / linux / windows
    machine = platform.machine().lower()    # arm64 / x86_64 / amd64 / aarch64
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def x13_path() -> Path:
    """Absolute path to the bundled ``x13as`` for this platform.

    Raises ``FileNotFoundError`` with build instructions when no binary is
    bundled for the current platform, so callers can fall back to a system
    install cleanly.
    """
    exe = "x13as.exe" if os.name == "nt" else "x13as"
    path = _BIN_DIR / _platform_tag() / exe
    if not path.exists():
        bundled = sorted(p.name for p in _BIN_DIR.glob("*")) if _BIN_DIR.exists() else []
        raise FileNotFoundError(
            f"MacroPy bundles no x13as for {_platform_tag()} (expected {path}; "
            f"bundled platforms: {bundled or 'none'}). Run scripts/build_x13.sh "
            "from the MacroPy repo to build and install one for this platform, "
            "or set the X13PATH environment variable to an existing binary."
        )
    if os.name != "nt" and not os.access(path, os.X_OK):
        path.chmod(path.stat().st_mode
                   | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
