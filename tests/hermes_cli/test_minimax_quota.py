"""Tests for hermes_cli.minimax_quota module."""

import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from hermes_cli.minimax_quota import (
    _build_quota_url,
    _fetch_minimax_quota,
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
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M2.7",
                    "total_usage_count": 1_000_000,
                    "remaining_usage_count": 670_000,
                    "end_time_ms": 1777060800000,   # 2026-04-24 20:00 UTC
                    "weekly_end_time_ms": 1777248000000,  # 2026-04-27 00:00 UTC
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["error"] is None
        assert result["used_percent"] == 33  # (1M - 670K) / 1M * 100 = 33
        assert result["reset_time_utc"] == "Apr 24 20:00 UTC"
        assert result["weekly_reset_utc"] == "Apr 27 00:00 UTC"

    def test_used_percent_is_100_when_exhausted(self):
        payload = {
            "model_remains": [
                {
                    "model_name": "MiniMax-M2.7",
                    "total_usage_count": 1_000_000,
                    "remaining_usage_count": 0,
                    "end_time_ms": 1777060800000,
                    "weekly_end_time_ms": 1777248000000,
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
                    "model_name": "MiniMax-M2.7",
                    "total_usage_count": 1_000_000,
                    "remaining_usage_count": 1_000_000,
                    "end_time_ms": 1777060800000,
                    "weekly_end_time_ms": 1777248000000,
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["used_percent"] == 0
        assert result["error"] is None

    def test_returns_error_when_no_m2_entry(self):
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
                    "model_name": "MiniMax-M2.7",
                    "total_usage_count": 0,
                    "remaining_usage_count": 0,
                    "end_time_ms": 0,
                    "weekly_end_time_ms": 0,
                }
            ]
        }
        with self._mock_response(payload):
            result = _fetch_minimax_quota("fake-token", "https://api.minimax.io/anthropic")

        assert result["error"] == "invalid total_usage_count"


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
                    "model_name": "MiniMax-M2.7",
                    "total_usage_count": 1_000_000,
                    "remaining_usage_count": 500_000,
                    "end_time_ms": 1777060800000,
                    "weekly_end_time_ms": 1777248000000,
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
