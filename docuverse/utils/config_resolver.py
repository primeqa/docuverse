"""6-tier config-file resolver shared across engines.

The repo historically hard-codes ``config/<file>.json`` paths, which only
work when the user runs from a checkout. This resolver lets engines look up
config files in a stable order so that:

1. Operators can override anything in ``$DOCUVERSE_HOME`` or ``~/.docuverse``
   without touching the checkout.
2. The legacy flat ``config/<file>.json`` layout keeps working — old
   checkouts and downstream code that hard-codes paths are unaffected.
3. A new ``config/<kind>/<file>`` layout is supported for users who want
   to organize their configs by category (servers/, engines/, data_formats/).
4. ``pip install docuverse`` (no checkout) still finds shipped defaults
   via packaged data.

Search order (first match wins):

    1. ``$DOCUVERSE_HOME/<rel_path>``                (operator override)
    2. ``~/.docuverse/<rel_path>``                   (per-user override)
    3. ``./config/<rel_path>``                       (new categorized layout)
    4. ``./config/<basename>``                       (legacy flat layout)
    5. ``importlib.resources.files("docuverse")
         / "config_defaults" / <rel_path>``          (packaged shipped defaults)
    6. → raise FileNotFoundError listing what was tried.

When tier 4 (legacy fallback) fires, a one-shot ``DeprecationWarning`` is
emitted so users know to migrate to the new layout when convenient.
"""
from __future__ import annotations

import os
import warnings
from importlib import resources
from pathlib import Path

# Track which legacy paths we've already warned about so each is reported once.
_LEGACY_WARNED: set[str] = set()


def _candidate_dirs() -> list[tuple[str, Path]]:
    """Return the (label, base-dir) tiers, in search order, that always run."""
    tiers: list[tuple[str, Path]] = []
    docuverse_home = os.environ.get("DOCUVERSE_HOME")
    if docuverse_home:
        tiers.append(("DOCUVERSE_HOME", Path(docuverse_home)))
    tiers.append(("~/.docuverse", Path.home() / ".docuverse"))
    tiers.append(("./config (new layout)", Path.cwd() / "config"))
    return tiers


def _packaged_defaults_dir() -> Path | None:
    """Path to ``docuverse/config_defaults/`` if it exists, else None.

    The directory is shipped as package data; if a deployment doesn't include
    it (older wheel, etc.) we silently skip this tier rather than raising.
    """
    try:
        base = resources.files("docuverse") / "config_defaults"
        if base.is_dir():
            return Path(str(base))
    except (ModuleNotFoundError, AttributeError):
        pass
    return None


def resolve(rel_path: str) -> str:
    """Resolve ``rel_path`` against the tier list; return an absolute path.

    ``rel_path`` is interpreted relative to a config base dir (e.g.
    ``"servers/milvus_servers.json"`` or just ``"milvus_servers.json"``).
    The basename alone is also tried in the legacy-flat tier so that callers
    passing ``"servers/milvus_servers.json"`` still find a legacy
    ``config/milvus_servers.json``.

    Raises ``FileNotFoundError`` listing every path tried.
    """
    tried: list[str] = []

    # Tiers 1–3: explicit base-dir candidates.
    for label, base in _candidate_dirs():
        candidate = base / rel_path
        tried.append(f"{label}: {candidate}")
        if candidate.is_file():
            return str(candidate.resolve())

    # Tier 4: legacy flat layout — try the basename in ./config/.
    legacy = Path.cwd() / "config" / Path(rel_path).name
    tried.append(f"./config (legacy flat): {legacy}")
    if legacy.is_file():
        if str(legacy) not in _LEGACY_WARNED:
            _LEGACY_WARNED.add(str(legacy))
            warnings.warn(
                f"Found {legacy} via the legacy flat config/ layout. "
                f"Move it to config/{rel_path} to silence this warning. "
                f"See config/README.md for the new layout.",
                DeprecationWarning,
                stacklevel=2,
            )
        return str(legacy.resolve())

    # Tier 5: packaged defaults shipped with the wheel.
    packaged = _packaged_defaults_dir()
    if packaged is not None:
        candidate = packaged / rel_path
        tried.append(f"packaged defaults: {candidate}")
        if candidate.is_file():
            return str(candidate.resolve())

    # Tier 6: fail with everything we tried.
    raise FileNotFoundError(
        f"Could not locate config file {rel_path!r}. Tried (in order):\n  "
        + "\n  ".join(tried)
        + "\nSet $DOCUVERSE_HOME or place the file at one of the locations above."
    )


def resolve_optional(rel_path: str) -> str | None:
    """Like :func:`resolve` but returns ``None`` instead of raising."""
    try:
        return resolve(rel_path)
    except FileNotFoundError:
        return None


__all__ = ["resolve", "resolve_optional"]
