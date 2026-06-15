"""
cache.py - Content-addressed disk + in-memory cache for AnalysisResult objects.

Cache key = SHA-256 of file bytes (or URL string).
Disk storage = JSON-serialised AnalysisResult under ~/.cache/docling_skill/.

TYPOGRAPHY RULE: Never output the Unicode character U+2500 ("─").
Always use the ASCII hyphen "-" for dividers, separators, and dashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_skill.analyze import AnalysisResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CACHE_DIR_ENV = "DOCLING_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "docling_skill"
_MAX_DISK_ENTRIES = 1000
_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

# In-memory LRU (simple dict, bounded)
_MEM_CACHE: dict[str, tuple[Any, float]] = {}
_MEM_MAX = 64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached(source: str) -> "AnalysisResult | None":
    """
    Return a cached AnalysisResult for *source*, or None on miss/expiry.

    Checks memory first, then disk.
    """
    key = _cache_key(source)

    # Memory
    if key in _MEM_CACHE:
        result, ts = _MEM_CACHE[key]
        if time.time() - ts < _TTL_SECONDS:
            return result
        del _MEM_CACHE[key]

    # Disk
    disk_path = _disk_path(key)
    if disk_path.exists():
        try:
            raw = json.loads(disk_path.read_text(encoding="utf-8"))
            if time.time() - raw.get("_cached_at", 0) > _TTL_SECONDS:
                disk_path.unlink(missing_ok=True)
                return None
            result = _deserialise(raw)
            _mem_put(key, result)
            return result
        except Exception:
            disk_path.unlink(missing_ok=True)

    return None


def put_cached(source: str, result: "AnalysisResult") -> None:
    """Store *result* in memory and disk cache."""
    key = _cache_key(source)
    _mem_put(key, result)

    disk_path = _disk_path(key)
    disk_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw = result.to_dict()
        raw["_cached_at"] = time.time()
        # Strip heavy doc object (not serialisable)
        raw.pop("document", None)
        disk_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # Cache write failure is non-fatal

    _evict_old_disk_entries()


def invalidate(source: str) -> None:
    """Remove a single entry from both caches."""
    key = _cache_key(source)
    _MEM_CACHE.pop(key, None)
    p = _disk_path(key)
    if p.exists():
        p.unlink(missing_ok=True)


def clear_all() -> int:
    """Wipe the entire disk cache. Returns number of files deleted."""
    _MEM_CACHE.clear()
    cache_dir = _get_cache_dir()
    deleted = 0
    for f in cache_dir.glob("*.json"):
        f.unlink(missing_ok=True)
        deleted += 1
    return deleted


def cache_stats() -> dict:
    """Return cache statistics (memory entries, disk entries, dir)."""
    cache_dir = _get_cache_dir()
    disk_entries = list(cache_dir.glob("*.json"))
    return {
        "mem_entries": len(_MEM_CACHE),
        "disk_entries": len(disk_entries),
        "cache_dir": str(cache_dir),
        "ttl_days": _TTL_SECONDS // 86400,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _cache_key(source: str) -> str:
    """SHA-256 of file bytes (local) or URL string."""
    p = Path(source)
    if p.exists():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(source.encode()).hexdigest()


def _get_cache_dir() -> Path:
    d = Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _disk_path(key: str) -> Path:
    return _get_cache_dir() / f"{key}.json"


def _mem_put(key: str, result: Any) -> None:
    if len(_MEM_CACHE) >= _MEM_MAX:
        # Evict oldest
        oldest = min(_MEM_CACHE, key=lambda k: _MEM_CACHE[k][1])
        del _MEM_CACHE[oldest]
    _MEM_CACHE[key] = (result, time.time())


def _evict_old_disk_entries() -> None:
    cache_dir = _get_cache_dir()
    entries = sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if len(entries) > _MAX_DISK_ENTRIES:
        for old in entries[: len(entries) - _MAX_DISK_ENTRIES]:
            old.unlink(missing_ok=True)


def _deserialise(raw: dict) -> "AnalysisResult":
    from docling_skill.analyze import AnalysisResult, PageInfo

    pages = [PageInfo(**p) for p in raw.get("pages", [])]
    raw.pop("_cached_at", None)
    raw.pop("document", None)
    raw["pages"] = pages
    return AnalysisResult(**raw)
