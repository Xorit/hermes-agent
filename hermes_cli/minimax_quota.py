"""MiniMax Token Plan quota fetching and caching for CLI status bar.

Fetches from GET <base>/v1/api/openplatform/coding_plan/remains where base
is the API host extracted from the provider's inference_base_url
(e.g. https://api.minimax.io/anthropic → https://api.minimax.io).

The *_usage_count fields in the API response are REMAINING tokens (not used),
so: used = total - remaining.

Response shape (model_remains[] entry for "MiniMax-M*"):
{
  "model_remains": [
    {
      "model_name": "MiniMax-M2.7",
      "total_usage_count": 1000000,
      "remaining_usage_count": 670000,   <-- REMAINING, not used!
      "end_time_ms": 1745534400000,      <-- 5h window reset (ms UTC)
      "weekly_end_time_ms": 1746057600000 <-- weekly reset (ms UTC)
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Thread
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level cache — written by background thread, read by TUI render path.
# Single dict assignment is atomic in CPython; no lock needed.
_quota_cache: Optional[Dict[str, Any]] = None
_quota_cache_ts: float = 0.0
QUOTA_CACHE_TTL: float = 60.0  # seconds before stale refresh is triggered


@dataclass
class MiniMaxQuota:
    """Parsed MiniMax Token Plan quota data."""
    used_percent: int           # 0-100, depleting toward 0 (100 = fully used)
    reset_time_utc: str         # "Apr 24 20:00 UTC" — 5h window reset
    weekly_reset_utc: str       # "Apr 28 00:00 UTC" — weekly reset
    error: Optional[str] = None


def _build_quota_url(inference_base_url: str) -> str:
    """Strip /anthropic suffix from inference base URL and return the quota API URL."""
    base = inference_base_url.rstrip("/")
    # Remove trailing suffixes to get the API host.
    # Order matters: try longer suffixes first.
    for suffix in ("/v1/anthropic", "/anthropic"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}/v1/api/openplatform/coding_plan/remains"


def _fetch_minimax_quota(api_key: str, inference_base_url: str) -> Dict[str, Any]:
    """Call the MiniMax Token Plan quota endpoint.

    Args:
        api_key: Bearer token for the MiniMax API.
        inference_base_url: The provider's inference base URL
            (e.g. https://api.minimax.io/anthropic).

    Returns:
        A dict with keys: used_percent (int), reset_time_utc (str),
        weekly_reset_utc (str), error (Optional[str]).
    """
    url = _build_quota_url(inference_base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=15.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    model_remains = data.get("model_remains") or []
    entry: Optional[Dict[str, Any]] = None
    for item in model_remains:
        model_name = str(item.get("model_name") or "")
        if re.match(r"^MiniMax-M\d", model_name):
            entry = item
            break

    if not entry:
        return {"error": "MiniMax-M* entry not found in model_remains"}

    total = int(entry.get("total_usage_count") or 0)
    remaining = int(entry.get("remaining_usage_count") or 0)
    if total <= 0:
        return {"error": "invalid total_usage_count"}

    # *_usage_count fields are REMAINING tokens (not used)
    used = total - remaining
    used_percent = min(100, max(0, round((used / total) * 100)))

    end_ms = int(entry.get("end_time_ms") or 0)
    weekly_end_ms = int(entry.get("weekly_end_time_ms") or 0)

    def _format_ms(ms: int) -> str:
        if not ms:
            return "??"
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%b %d %H:%M UTC")

    return {
        "used_percent": used_percent,
        "reset_time_utc": _format_ms(end_ms),
        "weekly_reset_utc": _format_ms(weekly_end_ms),
        "error": None,
    }


def _background_refresh(api_key: str, inference_base_url: str) -> None:
    """Fire-and-forget background thread entry point. Updates module cache."""
    global _quota_cache, _quota_cache_ts
    try:
        _quota_cache = _fetch_minimax_quota(api_key, inference_base_url)
        _quota_cache_ts = time.time()
    except Exception as exc:
        logger.debug("MiniMax quota fetch failed: %s", exc)
        # Leave existing cache intact on error — last known value is better than nothing


def refresh_quota_async(api_key: str, inference_base_url: str) -> None:
    """Spawn a daemon thread to refresh quota if cache is stale.

    Thread-safe: TUI render path only reads _quota_cache (atomic dict read).
    Background thread writes _quota_cache (atomic dict write). No lock needed.
    """
    global _quota_cache, _quota_cache_ts
    if _quota_cache is None or (time.time() - _quota_cache_ts) > QUOTA_CACHE_TTL:
        t = Thread(
            target=_background_refresh,
            args=(api_key, inference_base_url),
            daemon=True,
            name="minimax-quota-refresh",
        )
        t.start()


def get_cached_quota() -> Optional[Dict[str, Any]]:
    """Return the cached quota dict, or None if not yet fetched."""
    return _quota_cache


def _inject_minimax_quota(snapshot: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    """Inject MiniMax Token Plan quota into the status bar snapshot.

    Called automatically via the ``on_status_bar_snapshot`` plugin hook.
    """
    provider = runtime.get("provider", "")
    if provider not in ("minimax", "minimax-cn"):
        return
    try:
        refresh_quota_async(
            runtime.get("api_key") or "",
            runtime.get("base_url") or "",
        )
        snapshot["minimax_quota"] = get_cached_quota()
    except Exception:
        snapshot["minimax_quota"] = None


# Auto-register the hook so minimax quota is always active when imported
try:
    from hermes_cli.plugins import register_status_bar_snapshot_hook
    register_status_bar_snapshot_hook(_inject_minimax_quota)
except Exception as exc:
    logger.debug("Auto-registration of minimax quota hook failed (plugins system unavailable): %s", exc)
