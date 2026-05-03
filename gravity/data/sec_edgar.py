"""SEC EDGAR public company financial data loader.

Fetches XBRL company facts for public retailers from the SEC EDGAR API.
Free API, no key required.  Must include a descriptive User-Agent header.
Rate limit: 10 requests/second (we default to ~7/sec for safety).

API documentation:
    https://www.sec.gov/edgar/sec-api-documentation

Endpoint pattern:
    https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://data.sec.gov"
_REQUEST_TIMEOUT = 30
_DEFAULT_RATE_LIMIT = 0.15  # ~7 requests/sec (well under 10/sec ceiling)

_USER_AGENT = "GravityConsumerModel/2.0 contact@example.com"

# CIK mapping for major retail and food-service public companies
_RETAIL_CIKS: dict[str, str] = {
    "walmart": "0000104169",
    "target": "0000027419",
    "costco": "0000909832",
    "kroger": "0000056873",
    "home depot": "0000354950",
    "lowes": "0000060667",
    "dollar general": "0000034067",
    "dollar tree": "0000935703",
    "walgreens": "0001618921",
    "cvs": "0000064803",
    "best buy": "0000764478",
    "tjx": "0000109198",
    "ross stores": "0000745732",
    "nordstrom": "0000072333",
    "macys": "0000794367",
    "gap": "0000039911",
    "autozone": "0000866787",
    "oreilly": "0000898173",
    "tractor supply": "0000916365",
    "publix": "0000081061",
    "albertsons": "0001646972",
    "mcdonalds": "0000063908",
    "starbucks": "0000829224",
    "chipotle": "0001058090",
    "yum brands": "0001041061",
    "dominos": "0001286681",
    "darden": "0000940944",
    "wendys": "0000818479",
    "dicks sporting goods": "0001089063",
    "foot locker": "0000850209",
    "ulta beauty": "0001337996",
    "bath body works": "0000701985",
    "williams sonoma": "0000945114",
    "five below": "0001177702",
    "burlington": "0001579298",
    "ollies": "0001721000",
    "advance auto parts": "0001158449",
    "sherwin williams": "0000089800",
    "genuine parts": "0000040987",
    "carmax": "0001170010",
    "wayfair": "0001616533",
    "chewy": "0001766502",
}

# XBRL concepts we try to extract (us-gaap taxonomy)
_REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]
_NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "ProfitLoss",
]
_GROSS_PROFIT_CONCEPTS = [
    "GrossProfit",
]
_OPERATING_INCOME_CONCEPTS = [
    "OperatingIncomeLoss",
]
_STORE_COUNT_CONCEPTS = [
    "NumberOfStores",
    "NumberOfRestaurants",
    "NumberOfOperatingSegmentStores",
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SECEdgarLoader:
    """Load public company financials from the SEC EDGAR XBRL API.

    Parameters
    ----------
    rate_limit : float
        Minimum seconds between consecutive API requests (default 0.15).
    timeout : int
        HTTP request timeout in seconds.

    Examples
    --------
    >>> loader = SECEdgarLoader()
    >>> facts = loader.get_retailer_financials("walmart")
    >>> print(facts.get("revenue"))
    """

    BASE_URL = _BASE_URL
    HEADERS = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }

    def __init__(
        self,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self._rate_limit = rate_limit
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._last_request: float = 0.0
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    @staticmethod
    def _normalize_brand(name: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", name.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _resolve_cik(self, brand_name: str) -> Optional[str]:
        """Find CIK for a brand name using fuzzy substring matching."""
        norm = self._normalize_brand(brand_name)
        if not norm:
            return None

        # Exact match first
        if norm in _RETAIL_CIKS:
            return _RETAIL_CIKS[norm]

        # Substring match: brand is substring of key or key is substring of brand
        for key, cik in _RETAIL_CIKS.items():
            if norm in key or key in norm:
                return cik

        logger.warning("SEC EDGAR: no CIK found for brand '%s'", brand_name)
        return None

    @staticmethod
    def _extract_annual_value(
        facts: dict,
        concepts: list[str],
        taxonomy: str = "us-gaap",
    ) -> Optional[float]:
        """Extract the most recent 10-K annual value for a list of concept names.

        Walks through the concepts in priority order, looking in
        ``facts[taxonomy][concept]["units"]["USD"]`` for 10-K filings,
        then returns the value with the latest ``end`` date.
        """
        tax_data = facts.get(taxonomy, {})
        for concept in concepts:
            concept_data = tax_data.get(concept)
            if not concept_data:
                continue

            units = concept_data.get("units", {})
            # Try USD first, then pure (for ratios / counts)
            entries = units.get("USD") or units.get("pure") or units.get("shares")
            if not entries:
                continue

            # Filter to 10-K filings only
            annual = [
                e for e in entries
                if e.get("form") == "10-K" and e.get("val") is not None
            ]
            if not annual:
                continue

            # Sort by end date descending and return most recent
            annual.sort(key=lambda e: e.get("end", ""), reverse=True)
            try:
                return float(annual[0]["val"])
            except (TypeError, ValueError, KeyError):
                continue

        return None

    @staticmethod
    def _extract_store_count(facts: dict) -> Optional[int]:
        """Extract store/restaurant count from XBRL facts."""
        tax_data = facts.get("us-gaap", {})
        for concept in _STORE_COUNT_CONCEPTS:
            concept_data = tax_data.get(concept)
            if not concept_data:
                continue

            units = concept_data.get("units", {})
            # Store counts are typically in "stores" or pure units
            for unit_key in ("stores", "pure", "Store", "USD"):
                entries = units.get(unit_key)
                if not entries:
                    continue

                annual = [
                    e for e in entries
                    if e.get("form") == "10-K" and e.get("val") is not None
                ]
                if not annual:
                    continue

                annual.sort(key=lambda e: e.get("end", ""), reverse=True)
                try:
                    return int(float(annual[0]["val"]))
                except (TypeError, ValueError, KeyError):
                    continue

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_company_facts(self, cik: str) -> dict:
        """Fetch all XBRL facts for a CIK number.

        Parameters
        ----------
        cik : str
            10-digit zero-padded CIK (e.g. "0000104169").

        Returns
        -------
        dict
            Raw XBRL company facts JSON, or empty dict on failure.
        """
        cik_padded = cik.zfill(10)
        cache_key = cik_padded

        if cache_key in self._cache:
            logger.debug("SEC EDGAR cache hit for CIK %s", cik_padded)
            return self._cache[cache_key]

        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        logger.info("Fetching SEC EDGAR company facts: %s", url)

        self._throttle()
        self._last_request = time.monotonic()

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("SEC EDGAR request failed for CIK %s: %s", cik_padded, exc)
            return {}
        except (ValueError, TypeError) as exc:
            logger.warning(
                "SEC EDGAR JSON parse error for CIK %s: %s", cik_padded, exc
            )
            return {}

        facts = data.get("facts", {})
        self._cache[cache_key] = facts
        entity = data.get("entityName", "Unknown")
        logger.info(
            "SEC EDGAR loaded facts for %s (CIK %s): %d taxonomies",
            entity, cik_padded, len(facts),
        )
        return facts

    def get_retailer_financials(self, brand_name: str) -> dict:
        """Look up a retail brand and return key financial metrics.

        Parameters
        ----------
        brand_name : str
            Common brand name (e.g. "Walmart", "Target", "Starbucks").

        Returns
        -------
        dict
            Keys: name, cik, revenue, net_income, gross_profit,
            gross_margin, operating_income, operating_margin,
            total_stores, year.  Values are None where unavailable.
            Returns empty dict if brand is not found.
        """
        cik = self._resolve_cik(brand_name)
        if not cik:
            return {}

        facts = self.get_company_facts(cik)
        if not facts:
            return {}

        revenue = self._extract_annual_value(facts, _REVENUE_CONCEPTS)
        net_income = self._extract_annual_value(facts, _NET_INCOME_CONCEPTS)
        gross_profit = self._extract_annual_value(facts, _GROSS_PROFIT_CONCEPTS)
        operating_income = self._extract_annual_value(
            facts, _OPERATING_INCOME_CONCEPTS
        )
        store_count = self._extract_store_count(facts)

        # Compute margins if components are available
        gross_margin = None
        if gross_profit is not None and revenue and revenue > 0:
            gross_margin = round(gross_profit / revenue, 4)

        operating_margin = None
        if operating_income is not None and revenue and revenue > 0:
            operating_margin = round(operating_income / revenue, 4)

        # Determine the reporting year from the most recent 10-K revenue entry
        year = self._extract_filing_year(facts)

        result = {
            "name": brand_name,
            "cik": cik,
            "revenue": revenue,
            "net_income": net_income,
            "gross_profit": gross_profit,
            "gross_margin": gross_margin,
            "operating_income": operating_income,
            "operating_margin": operating_margin,
            "total_stores": store_count,
            "year": year,
        }

        logger.info(
            "SEC EDGAR financials for %s: revenue=%s, net_income=%s, stores=%s",
            brand_name, revenue, net_income, store_count,
        )
        return result

    def _extract_filing_year(self, facts: dict) -> Optional[int]:
        """Get the fiscal year of the most recent 10-K revenue filing."""
        tax_data = facts.get("us-gaap", {})
        for concept in _REVENUE_CONCEPTS:
            concept_data = tax_data.get(concept)
            if not concept_data:
                continue
            entries = concept_data.get("units", {}).get("USD", [])
            annual = [e for e in entries if e.get("form") == "10-K"]
            if annual:
                annual.sort(key=lambda e: e.get("end", ""), reverse=True)
                end_date = annual[0].get("end", "")
                if len(end_date) >= 4:
                    try:
                        return int(end_date[:4])
                    except ValueError:
                        pass
        return None

    def benchmark_competitors(self, brand_names: list[str]) -> list[dict]:
        """Fetch financials for multiple brands for competitive benchmarking.

        Parameters
        ----------
        brand_names : list[str]
            List of brand names to look up.

        Returns
        -------
        list[dict]
            One dict per found public company (skips unknown brands).
        """
        results = []
        for name in brand_names:
            financials = self.get_retailer_financials(name)
            if financials:
                results.append(financials)
            else:
                logger.info(
                    "SEC EDGAR: skipping '%s' (not found or no data)", name
                )

        logger.info(
            "SEC EDGAR benchmark: %d/%d brands resolved",
            len(results), len(brand_names),
        )
        return results

    def clear_cache(self) -> None:
        """Clear the in-memory facts cache."""
        self._cache.clear()
        logger.debug("SEC EDGAR cache cleared.")

    def __repr__(self) -> str:
        return (
            f"SECEdgarLoader(rate_limit={self._rate_limit}, "
            f"cached_ciks={len(self._cache)})"
        )
