"""MiniMax Token Plan quota fetching and caching for CLI status bar.

Fetches from GET <base>/v1/api/openplatform/coding_plan/remains where base
is the API host extracted from the provider's inference_base_url
(e.g. https://api.minimax.io/anthropic → https://api.minimax.io).

Actual API response shape (model_remains[] entry for "MiniMax-M*"):
{
  "model_name": "MiniMax-M*",
  "current_interval_total_count": 4500,
  "current_interval_usage_count": 4191,   <-- USED tokens (NOT remaining)
  "end_time": 1777093200000,              <-- interval reset (ms UTC)
  "current_weekly_total_count": 45000,
  "current_weekly_usage_count": 32014,   <-- USED tokens (NOT remaining)
  "weekly_end_time": 1777248000000,      <-- weekly reset (ms UTC)
  "remains_time": 13723205,               <-- remaining SECONDS in interval
  "weekly_remains_time": 168523205        <-- remaining SECONDS in week
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
_pending_start: float = 0.0  # Track when pending state was set


@dataclass
class MiniMaxQuota:
    """Parsed MiniMax Token Plan quota data."""
    used_percent: int           # 0-100, depleting toward 0 (100 = fully used)
    reset_time_utc: str         # "Apr 22 21:00 UTC" — 5h window reset
    weekly_reset_utc: str       # "Apr 27 00:00 UTC" — weekly reset
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
        # MiniMax API uses "MiniMax-M*" as a wildcard entry covering all M-series models
        if re.match(r"^MiniMax-M", model_name) and "*" in model_name:
            entry = item
            break

    if not entry:
        return {"error": "MiniMax-M* entry not found in model_remains"}

    # Correct field names from the actual API response:
    # current_interval_usage_count = USED tokens (NOT remaining)
    # current_interval_total_count = total tokens for the interval
    total = int(entry.get("current_interval_total_count") or 0)
    used = int(entry.get("current_interval_usage_count") or 0)
    if total <= 0:
        return {"error": "invalid current_interval_total_count"}

    # used_percent depletes toward 0 (100 = fully exhausted)
    used_percent = min(100, max(0, round((used / total) * 100)))

    end_ms = int(entry.get("end_time") or 0)
    weekly_end_ms = int(entry.get("weekly_end_time") or 0)

    def _format_ms(ms: int) -> str:
        if not ms:
            return "??"
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%b %d %H:%M UTC")

    # Weekly usage — also depletes toward 0
    weekly_total = int(entry.get("current_weekly_total_count") or 0)
    weekly_used = int(entry.get("current_weekly_usage_count") or 0)
    weekly_used_percent = min(100, max(0, round((weekly_used / weekly_total) * 100))) if weekly_total > 0 else 0

    return {
        "used_percent": used_percent,
        "reset_time_utc": _format_ms(end_ms),
        "weekly_used_percent": weekly_used_percent,
        "weekly_reset_utc": _format_ms(weekly_end_ms),
        "error": None,
    }


def _clear_quota_cache() -> None:
    """Clear the module-level quota cache and pending state."""
    global _quota_cache, _quota_cache_ts, _pending_start
    _quota_cache = None
    _quota_cache_ts = 0.0
    _pending_start = 0.0


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
    global _quota_cache, _quota_cache_ts, _pending_start
    if _quota_cache is None or (time.time() - _quota_cache_ts) > QUOTA_CACHE_TTL or (_quota_cache.get("pending") and (time.time() - _pending_start) > 10):
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


def _canonical_minimax_provider(provider: Any, base_url: Any = "", api_key: Any = "") -> str:
    """Return a canonical provider id, inferring MiniMax from aliases, host, or API key.

    Status-bar snapshots can be rendered before the runtime provider has been
    fully canonicalized (for example while it is still ``auto``).  The MiniMax
    quota endpoint is tied to the direct MiniMax hosts, so the base URL is a
    reliable fallback signal when the provider string is still an alias or raw
    pre-resolution value.

    The API key is a last-resort signal: MiniMax keys are typically 60+ chars
    and start with ``ey`` (JWT).  We only use key pattern matching when both
    ``provider`` and ``base_url`` are uninformative.
    """
    normalized = str(provider or "").strip().lower()
    try:
        from hermes_cli.models import normalize_provider
        normalized = normalize_provider(normalized)
    except Exception:
        normalized = {
            "minimax-china": "minimax-cn",
            "minimax_cn": "minimax-cn",
        }.get(normalized, normalized)

    if normalized in ("minimax", "minimax-cn"):
        return normalized

    try:
        host = urllib.parse.urlparse(str(base_url or "")).netloc.lower()
    except Exception:
        host = ""
    if host == "api.minimax.io" or host.endswith(".minimax.io"):
        return "minimax"
    if host == "api.minimaxi.com" or host.endswith(".minimaxi.com"):
        return "minimax-cn"

    # Last resort: probe the API key pattern.
    # MiniMax keys are JWT tokens (eyJ... format), typically 60+ chars.
    key = str(api_key or "").strip()
    if len(key) >= 60 and key.startswith("ey"):
        # Heuristic: long JWT-style token is typical for MiniMax
        return "minimax"

    return normalized


def _inject_minimax_quota(snapshot: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    """Inject MiniMax Token Plan quota into the status bar snapshot.

    Called automatically via the ``on_status_bar_snapshot`` plugin hook.
    """
    global _quota_cache, _pending_start

    base_url = runtime.get("base_url") or ""
    api_key = runtime.get("api_key") or ""
    provider = _canonical_minimax_provider(runtime.get("provider", ""), base_url, api_key)

    if provider not in ("minimax", "minimax-cn"):
        # Provider switched away — clear stale cache so there is no cross-contamination
        # when switching back to MiniMax within the same session.
        _clear_quota_cache()
        return

    try:
        refresh_quota_async(api_key, base_url)
        # Always read from the live module cache so the snapshot always has the
        # most recent data.  Snapshot dicts are created fresh on each render, so
        # copying the cache contents here is safe and ensures the snapshot always
        # reflects the current quota state (pending → real data after background
        # fetch completes).
        cached = get_cached_quota()
        if cached is None:
            _quota_cache = {"pending": True}
            _pending_start = time.time()
        # Reset _pending_start once we have real (non-pending) data so the module
        # state is clean after the transition.
        elif not cached.get("pending"):
            _pending_start = 0.0
        snapshot["minimax_quota"] = dict(_quota_cache) if _quota_cache else None
    except Exception:
        snapshot["minimax_quota"] = None


# Auto-register the hook so minimax quota is always active when imported
try:
    from hermes_cli.plugins import register_status_bar_snapshot_hook
    register_status_bar_snapshot_hook(_inject_minimax_quota)
except Exception as exc:
    logger.debug("Auto-registration of minimax quota hook failed (plugins system unavailable): %s", exc)
