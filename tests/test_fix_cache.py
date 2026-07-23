"""Tests for the fix cache."""

from aero_forge.cache.fix_cache import FixCache


def test_cache_round_trip(tmp_path):
    cache = FixCache(path=tmp_path / "cache.json", enabled=True)
    assert cache.get("error text", "code") is None
    cache.set("error text", "code", "fixed code")
    assert cache.get("error text", "code") == "fixed code"


def test_cache_disabled(tmp_path):
    cache = FixCache(path=tmp_path / "cache.json", enabled=False)
    cache.set("error", "code", "fix")
    assert cache.get("error", "code") is None


def test_cache_clear(tmp_path):
    cache = FixCache(path=tmp_path / "cache.json", enabled=True)
    cache.set("error", "code", "fix")
    cache.clear()
    assert cache.get("error", "code") is None
    assert not (tmp_path / "cache.json").is_file()
