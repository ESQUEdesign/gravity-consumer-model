"""
Financial Modeling Prep (FMP) data loader.

Pre-parsed financial data for public companies. Complements SEC EDGAR
with cleaner financial ratios and sector screening.

API documentation:
    https://financialmodelingprep.com/api/v3/

Free tier: 250 calls/day. API key required.

Endpoint patterns:
    GET /api/v3/search?query={q}&apikey={key}
    GET /api/v3/profile/{ticker}?apikey={key}
    GET /api/v3/key-metrics-ttm/{ticker}?apikey={key}
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUEST_TIMEOUT = 30  # seconds
_RATE_LIMIT = 1.0  # minimum seconds between requests (free tier is rate-limited)

# Profile fields to extract
_PROFILE_FIELDS = (
    "companyName", "mktCap", "sector", "industry",
    "fullTimeEmployees", "price", "description", "symbol",
    "exchange", "country", "city", "state",
)

# Key metrics TTM fields to extract
_METRICS_FIELDS = (
    "revenuePerShareTTM", "netIncomePerShareTTM", "peRatioTTM",
    "debtToEquityTTM", "currentRatioTTM", "roeTTM",
    "grossProfitMarginTTM", "operatingProfitMarginTTM",
    "returnOnAssetsTTM", "dividendYieldTTM",
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FinancialModelingPrepLoader:
    """Load company financial data from the Financial Modeling Prep API.

    Provides company search, profile lookups, and key financial metrics.
    Designed to benchmark retail-sector public companies against market
    conditions in a target trade area.

    Parameters
    ----------
    api_key : str or None
        FMP API key. If None, all requests will return empty results.
    timeout : int
        HTTP request timeout in seconds.
    rate_limit : float
        Minimum seconds between consecutive API requests.

    Examples
    --------
    >>> loader = FinancialModelingPrepLoader(api_key="your_key_here")
    >>> results = loader.search_company("Target")
    >>> profile = loader.get_profile("TGT")
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = _REQUEST_TIMEOUT,
        rate_limit: float = _RATE_LIMIT,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._rate_limit = rate_limit
        self._session = requests.Session()
        self._last_request_time: float = 0.0
        # In-memory caches keyed by relevant identifiers
        self._cache: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if necessary to respect the rate limit."""
        if self._rate_limit <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def _get_json(self, endpoint: str, params: Optional[dict] = None) -> object:
        """Make a GET request and return parsed JSON.

        Parameters
        ----------
        endpoint : str
            URL path relative to BASE_URL (e.g. ``/search``).
        params : dict or None
            Additional query parameters. ``apikey`` is added automatically.

        Returns
        -------
        list or dict or None
            Parsed JSON response, or None on failure.
        """
        if not self.api_key:
            logger.warning("FMP API key not set. Returning empty results.")
            return None

        url = f"{self.BASE_URL}{endpoint}"
        if params is None:
            params = {}
        params["apikey"] = self.api_key

        self._throttle()
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("FMP API request failed for %s: %s", endpoint, exc)
            return None

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("Failed to parse FMP JSON for %s: %s", endpoint, exc)
            return None

        # FMP returns error messages as dicts with "Error Message" key
        if isinstance(data, dict) and "Error Message" in data:
            logger.warning(
                "FMP API error for %s: %s", endpoint, data["Error Message"]
            )
            return None

        return data

    def _safe_float(self, val, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def _safe_int(self, val, default: int = 0) -> int:
        """Safely convert a value to int."""
        try:
            return int(float(val)) if val is not None else default
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_company(self, query: str) -> list[dict]:
        """Search for companies by name or partial match.

        Parameters
        ----------
        query : str
            Company name or search term (e.g. ``"Walmart"``).

        Returns
        -------
        list[dict]
            Each dict contains ``symbol``, ``name``, ``currency``, and
            ``exchangeShortName``. Returns empty list on failure.
        """
        cache_key = f"search:{query.lower()}"
        if cache_key in self._cache:
            logger.debug("FMP cache hit for search '%s'", query)
            return self._cache[cache_key]

        logger.info("FMP: searching for '%s'", query)
        data = self._get_json("/search", params={"query": query})

        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            results.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", ""),
                "currency": item.get("currency", ""),
                "exchangeShortName": item.get("exchangeShortName", ""),
            })

        self._cache[cache_key] = results
        logger.info("FMP: search '%s' returned %d results", query, len(results))
        return results

    def get_profile(self, ticker: str) -> dict:
        """Fetch a company profile by ticker symbol.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. ``"TGT"`` for Target).

        Returns
        -------
        dict
            Company profile with keys: ``companyName``, ``mktCap``,
            ``sector``, ``industry``, ``fullTimeEmployees``, ``price``,
            ``description``, ``symbol``, ``exchange``, ``country``,
            ``city``, ``state``. Returns empty dict on failure.
        """
        ticker = ticker.upper()
        cache_key = f"profile:{ticker}"
        if cache_key in self._cache:
            logger.debug("FMP cache hit for profile '%s'", ticker)
            return self._cache[cache_key]

        logger.info("FMP: fetching profile for %s", ticker)
        data = self._get_json(f"/profile/{ticker}")

        if not isinstance(data, list) or not data:
            return {}

        raw = data[0]
        result = {}
        for field in _PROFILE_FIELDS:
            result[field] = raw.get(field, None)

        self._cache[cache_key] = result
        logger.info("FMP: profile loaded for %s", ticker)
        return result

    def get_key_metrics(self, ticker: str) -> dict:
        """Fetch trailing twelve month (TTM) key metrics for a company.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. ``"TGT"``).

        Returns
        -------
        dict
            Financial metrics including ``revenuePerShareTTM``,
            ``netIncomePerShareTTM``, ``peRatioTTM``, ``debtToEquityTTM``,
            ``currentRatioTTM``, ``roeTTM``, ``grossProfitMarginTTM``,
            ``operatingProfitMarginTTM``, ``returnOnAssetsTTM``,
            ``dividendYieldTTM``. Returns empty dict on failure.
        """
        ticker = ticker.upper()
        cache_key = f"metrics:{ticker}"
        if cache_key in self._cache:
            logger.debug("FMP cache hit for metrics '%s'", ticker)
            return self._cache[cache_key]

        logger.info("FMP: fetching key metrics TTM for %s", ticker)
        data = self._get_json(f"/key-metrics-ttm/{ticker}")

        if not isinstance(data, list) or not data:
            return {}

        raw = data[0]
        result = {}
        for field in _METRICS_FIELDS:
            result[field] = self._safe_float(raw.get(field), default=None)

        self._cache[cache_key] = result
        logger.info("FMP: key metrics loaded for %s", ticker)
        return result

    def benchmark_retail(self, brand_names: list[str]) -> list[dict]:
        """Search and retrieve profile + metrics for a list of brand names.

        For each brand name, searches FMP for a matching company, then
        fetches the profile and key metrics for the best match.

        Parameters
        ----------
        brand_names : list[str]
            List of brand/company names to look up (e.g.
            ``["Walmart", "Target", "Costco"]``).

        Returns
        -------
        list[dict]
            Each dict is a merged profile + metrics for a found company.
            Companies that are not found or fail to load are silently
            skipped. Returns empty list if none are found.
        """
        if not self.api_key:
            logger.warning("FMP API key not set. Cannot benchmark retail.")
            return []

        benchmarks = []
        for name in brand_names:
            search_results = self.search_company(name)
            if not search_results:
                logger.warning(
                    "FMP: no search results for brand '%s', skipping", name
                )
                continue

            # Pick the first match (best relevance)
            ticker = search_results[0].get("symbol", "")
            if not ticker:
                logger.warning(
                    "FMP: no ticker symbol in search result for '%s'", name
                )
                continue

            profile = self.get_profile(ticker)
            metrics = self.get_key_metrics(ticker)

            if not profile and not metrics:
                logger.warning(
                    "FMP: no profile or metrics for %s (%s)", name, ticker
                )
                continue

            combined = {
                "search_name": name,
                "ticker": ticker,
                **profile,
                **metrics,
            }
            benchmarks.append(combined)

        logger.info(
            "FMP: benchmarked %d/%d brands successfully",
            len(benchmarks), len(brand_names),
        )
        return benchmarks

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("FMP data cache cleared.")

    def __repr__(self) -> str:
        has_key = "set" if self.api_key else "unset"
        return (
            f"FinancialModelingPrepLoader(api_key={has_key}, "
            f"cached_items={len(self._cache)})"
        )
