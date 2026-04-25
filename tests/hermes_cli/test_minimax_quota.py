"""Tests for hermes_cli.minimax_quota module."""

import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from hermes_cli.minimax_quota import (
    _build_quota_url,
    _canonical_minimax_provider,
    _fetch_minimax_quota,
    _inject_minimax_quota,
    get_cached_quota,
    refresh_quota_async,
    _quota_cache,
    _quota_cache_ts,
    QUOTA_CACHE_TTL,
    MiniMaxQuota,
)


# =============================================================================
# URL construction
# =============================================================================

class TestBuildQuotaUrl:
    def test_strips_anthropic_suffix(self):
        url = _build_quota_url("https://api.minimax.io/anthropic")
        assert url == "https://api.minimax.io/v1/api/openplatform/coding_plan/remains"

    def test_strips_v1_anthropic_suffix(self):
        url = _build_quota_url("https://api.minimax.io/v1/anthropic")
        assert url == "https://api.minimax.io/v1/api/openplatform/coding_plan/remains"

    def test_handles_minimax_cn(self):
        url = _build_quota_url("https://api.minimaxi.com/anthropic")
        assert url == "https://api.minimaxi.com/v1/api/openplatform/coding_plan/remains"


# =============================================================================
# Response parsing
# =============================================================================

class TestFetchMinimaxQuota:
    def _mock_response(self, payload: dict, status: int = 200):
        """Return a context manager that mocks urllib.request.urlopen."""
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.status = status
        return patch("urllib.request.urlopen", return_value=mock_resp)

    def test_parses_m2_7_entry(self):
        # MiniMax API returns "MiniMax-M*" as the wildcard entry
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M*",
                    "current_interval_total_count": 1_000_000,
                    "current_interval_usage_count": 330_000,  # USED, not remaining
                    "end_time": 1777060800000,   # 2026-04-24 20:00 UTC
                    "weekly_end_time": 1777248000000,  # 2026-04-27 00:00 UTC
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["error"] is None
        assert result["used_percent"] == 33  # 330K / 1M * 100 = 33
        assert result["reset_time_utc"] == "Apr 24 20:00 UTC"
        assert result["weekly_reset_utc"] == "Apr 27 00:00 UTC"

    def test_used_percent_is_100_when_exhausted(self):
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M*",
                    "current_interval_total_count": 1_000_000,
                    "current_interval_usage_count": 1_000_000,  # fully used
                    "end_time": 1777060800000,
                    "weekly_end_time": 1777248000000,
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["used_percent"] == 100
        assert result["error"] is None

    def test_used_percent_is_0_when_unused(self):
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M*",
                    "current_interval_total_count": 1_000_000,
                    "current_interval_usage_count": 0,  # unused
                    "end_time": 1777060800000,
                    "weekly_end_time": 1777248000000,
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["used_percent"] == 0
        assert result["error"] is None

    def test_returns_error_when_no_mstar_entry(self):
        payload = {"model_remains": [{"model_name": "some-other-model"}]}
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["error"] == "MiniMax-M* entry not found in model_remains"

    def test_returns_error_when_empty_model_remains(self):
        payload = {"model_remains": []}
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["error"] == "MiniMax-M* entry not found in model_remains"

    def test_returns_error_when_invalid_total(self):
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M*",
                    "current_interval_total_count": 0,
                    "current_interval_usage_count": 0,
                    "end_time": 0,
                    "weekly_end_time": 0,
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["error"] == "invalid current_interval_total_count"


# =============================================================================
# Cache behavior
# =============================================================================

class TestCacheBehavior:
    def teardown_method(self):
        # Reset module-level cache between tests
        import hermes_cli.minimax_quota as mq
        mq._quota_cache = None
        mq._quota_cache_ts = 0.0

    def test_get_cached_quota_returns_none_initially(self):
        assert get_cached_quota() is None

    def test_refresh_quota_async_stores_result(self):
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M*",
                    "current_interval_total_count": 1_000_000,
                    "current_interval_usage_count": 500_000,
                    "end_time": 1777060800000,
                    "weekly_end_time": 1777248000000,
                }
            ]
        }
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            refresh_quota_async("fake-token", "https://api.minimax.io/anthropic")
            time.sleep(0.1)  # give daemon thread time to write

        cached = get_cached_quota()
        assert cached is not None
        assert cached["used_percent"] == 50

    def test_refresh_quota_async_noop_when_cache_fresh(self):
        import hermes_cli.minimax_quota as mq
        # Pre-populate cache
        mq._quota_cache = {"used_percent": 42, "error": None}
        mq._quota_cache_ts = time.time()

        with patch("urllib.request.urlopen") as mock_urlopen:
            refresh_quota_async("fake-token", "https://api.minimax.io/anthropic")
            time.sleep(0.05)

        # Should not have called the API (cache is fresh)
        mock_urlopen.assert_not_called()

    def test_get_cached_quota_returns_stale_value_on_error(self):
        import hermes_cli.minimax_quota as mq
        mq._quota_cache = {"used_percent": 42, "error": None}
        mq._quota_cache_ts = time.time()

        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            refresh_quota_async("fake-token", "https://api.minimax.io/anthropic")
            time.sleep(0.1)

        # Stale value should still be in cache
        assert get_cached_quota()["used_percent"] == 42

# =============================================================================
# Status-bar hook provider gating
# =============================================================================

class TestStatusBarHook:
    def teardown_method(self):
        import hermes_cli.minimax_quota as mq
        mq._quota_cache = None
        mq._quota_cache_ts = 0.0
        mq._pending_start = 0.0

    def test_canonical_provider_handles_aliases(self):
        assert _canonical_minimax_provider("minimax") == "minimax"
        assert _canonical_minimax_provider("minimax-cn") == "minimax-cn"
        assert _canonical_minimax_provider("minimax-china") == "minimax-cn"
        assert _canonical_minimax_provider("minimax_cn") == "minimax-cn"

    def test_canonical_provider_infers_from_global_base_url_when_provider_auto(self):
        assert (
            _canonical_minimax_provider("auto", "https://api.minimax.io/anthropic")
            == "minimax"
        )

    def test_canonical_provider_infers_from_cn_base_url_when_provider_auto(self):
        assert (
            _canonical_minimax_provider("auto", "https://api.minimaxi.com/anthropic")
            == "minimax-cn"
        )

    def test_injects_pending_snapshot_when_provider_is_alias(self):
        snapshot = {}
        runtime = {
            "provider": "minimax-china",
            "api_key": "fake-token",
            "base_url": "https://api.minimaxi.com/anthropic",
        }
        with patch("hermes_cli.minimax_quota.refresh_quota_async") as mock_refresh:
            _inject_minimax_quota(snapshot, runtime)

        mock_refresh.assert_called_once_with("fake-token", "https://api.minimaxi.com/anthropic")
        assert snapshot["minimax_quota"] == {"pending": True}
        assert get_cached_quota() == {"pending": True}

    def test_injects_when_provider_auto_but_base_url_is_minimax(self):
        snapshot = {}
        runtime = {
            "provider": "auto",
            "api_key": "fake-token",
            "base_url": "https://api.minimax.io/anthropic",
        }
        with patch("hermes_cli.minimax_quota.refresh_quota_async") as mock_refresh:
            _inject_minimax_quota(snapshot, runtime)

        mock_refresh.assert_called_once_with("fake-token", "https://api.minimax.io/anthropic")
        assert snapshot["minimax_quota"] == {"pending": True}
        assert get_cached_quota() == {"pending": True}

    def test_does_not_inject_for_non_minimax_provider_or_host(self):
        snapshot = {}
        runtime = {
            "provider": "openai",
            "api_key": "fake-token",
            "base_url": "https://api.openai.com/v1",
        }
        with patch("hermes_cli.minimax_quota.refresh_quota_async") as mock_refresh:
            _inject_minimax_quota(snapshot, runtime)

        mock_refresh.assert_not_called()
        assert "minimax_quota" not in snapshot
