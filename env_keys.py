"""
Central .env.keys loader
=========================
Reads KEY=VALUE pairs from  .env.keys  in the repo root and exposes them
via ``get_key()``.

Lookup priority for any secret:
  1. .env.keys file  (repo root, next to this file)
  2. OS environment variable

The file intentionally uses a non-standard name (.env.keys, not .env) so it
is never auto-loaded by python-dotenv or similar tools — secrets are only
injected when explicitly requested through this module.

Keys currently used in this project
-------------------------------------
  Gemini_API_key      → Google Gemini API key
  T212_key            → Trading 212 API key
  OPENAI_API_KEY      → OpenAI API key
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# .env.keys lives in the same directory as this file (the repo root)
_ENV_KEYS_PATH = Path(__file__).resolve().parent / ".env.keys"


@lru_cache(maxsize=1)
def _load() -> dict[str, str]:
    """Parse .env.keys → {KEY: value}. Cached after first call."""
    if not _ENV_KEYS_PATH.is_file():
        return {}
    result: dict[str, str] = {}
    for line in _ENV_KEYS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                result[k] = v
    return result


def get_key(*names: str) -> str | None:
    """
    Return the first non-empty value found across the given key names.

    Checks .env.keys first, then os.environ, for each name in order.

    Usage examples::

        get_key("Gemini_API_key", "GOOGLE_API_KEY")
        get_key("T212_key", "TRADING212_API_KEY")
        get_key("OPENAI_API_KEY")
    """
    store = _load()
    for name in names:
        val = store.get(name) or os.environ.get(name)
        if val:
            return val
    return None
