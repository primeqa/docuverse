"""Tests for the 6-tier config resolver.

The resolver MUST be deterministic across the tier order: $DOCUVERSE_HOME,
~/.docuverse, ./config/<rel>, ./config/<basename> (legacy), packaged defaults.
"""
from __future__ import annotations

import os
import warnings

import pytest

from docuverse.utils import config_resolver
from docuverse.utils.config_resolver import resolve, resolve_optional


@pytest.fixture(autouse=True)
def _isolate_cwd_and_env(tmp_path, monkeypatch):
    """Each test runs in a fresh tmp_path with no env-vars or warning state."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCUVERSE_HOME", raising=False)
    monkeypatch.setattr(config_resolver, "_LEGACY_WARNED", set())
    # Force ~/.docuverse to point at an empty tmp dir so per-user files don't leak in.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    return tmp_path


def _touch(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_new_layout_found_under_cwd_config(tmp_path):
    _touch(tmp_path / "config" / "servers" / "milvus_servers.json", "{}")
    resolved = resolve("servers/milvus_servers.json")
    assert resolved.endswith("config/servers/milvus_servers.json")


def test_legacy_flat_layout_resolves_with_deprecation(tmp_path):
    _touch(tmp_path / "config" / "milvus_servers.json", "{}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve("servers/milvus_servers.json")
    assert resolved.endswith("config/milvus_servers.json")
    assert any(
        issubclass(w.category, DeprecationWarning) and "legacy" in str(w.message).lower()
        for w in caught
    ), "legacy fallback must emit a DeprecationWarning"


def test_legacy_warning_only_fires_once_per_path(tmp_path):
    _touch(tmp_path / "config" / "elastic_servers.json", "{}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve("servers/elastic_servers.json")
        resolve("servers/elastic_servers.json")
    legacy_warnings = [w for w in caught if "legacy" in str(w.message).lower()]
    assert len(legacy_warnings) == 1, (
        f"expected 1 legacy warning, got {len(legacy_warnings)}"
    )


def test_new_layout_wins_over_legacy(tmp_path):
    _touch(tmp_path / "config" / "servers" / "milvus_servers.json", "new")
    _touch(tmp_path / "config" / "milvus_servers.json", "legacy")
    resolved = resolve("servers/milvus_servers.json")
    assert "servers/milvus_servers.json" in resolved
    with open(resolved) as f:
        assert f.read() == "new", "new categorized layout must win over legacy"


def test_docuverse_home_wins_over_cwd_config(tmp_path, monkeypatch):
    home = tmp_path / "operator_home"
    _touch(home / "servers" / "milvus_servers.json", "from-DOCUVERSE_HOME")
    _touch(tmp_path / "config" / "servers" / "milvus_servers.json", "from-cwd")
    monkeypatch.setenv("DOCUVERSE_HOME", str(home))
    resolved = resolve("servers/milvus_servers.json")
    with open(resolved) as f:
        assert f.read() == "from-DOCUVERSE_HOME"


def test_user_home_wins_over_cwd_config(tmp_path, monkeypatch):
    user_dir = tmp_path / "fake_home" / ".docuverse"
    _touch(user_dir / "servers" / "milvus_servers.json", "from-user-home")
    _touch(tmp_path / "config" / "servers" / "milvus_servers.json", "from-cwd")
    resolved = resolve("servers/milvus_servers.json")
    with open(resolved) as f:
        assert f.read() == "from-user-home"


def test_docuverse_home_wins_over_user_home(tmp_path, monkeypatch):
    operator = tmp_path / "operator_home"
    _touch(operator / "servers" / "milvus_servers.json", "operator")
    _touch(
        tmp_path / "fake_home" / ".docuverse" / "servers" / "milvus_servers.json",
        "user",
    )
    monkeypatch.setenv("DOCUVERSE_HOME", str(operator))
    resolved = resolve("servers/milvus_servers.json")
    with open(resolved) as f:
        assert f.read() == "operator"


def test_missing_file_raises_with_listing():
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve("servers/nonexistent.json")
    msg = str(excinfo.value)
    # Every tier we tried should appear in the error so the user can fix it.
    assert "DOCUVERSE_HOME" in msg or "~/.docuverse" in msg
    assert "config" in msg
    assert "nonexistent.json" in msg


def test_resolve_optional_returns_none_on_miss():
    assert resolve_optional("never/exists.json") is None


def test_resolve_optional_returns_path_on_hit(tmp_path):
    _touch(tmp_path / "config" / "engines" / "milvus_default_config.yaml", "k: v")
    assert resolve_optional("engines/milvus_default_config.yaml") is not None


def test_basename_only_finds_file_in_legacy_tier(tmp_path):
    """Bare basename like ``"milvus_servers.json"`` should resolve via legacy fallback."""
    _touch(tmp_path / "config" / "milvus_servers.json", "{}")
    resolved = resolve("milvus_servers.json")
    assert resolved.endswith("config/milvus_servers.json")
