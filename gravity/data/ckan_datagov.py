"""
CKAN / data.gov dataset discovery loader.

Searches the data.gov catalog for business license, retail permit,
and commercial property datasets relevant to a geography. This is a
DISCOVERY tool -- it returns dataset metadata and download links,
not parsed data.

API documentation:
    https://catalog.data.gov/api/3/action/package_search

Endpoint pattern:
    GET https://catalog.data.gov/api/3/action/package_search?q={query}&rows={rows}
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUEST_TIMEOUT = 30  # seconds
_RATE_LIMIT = 0.5  # minimum seconds between requests
_DEFAULT_ROWS = 5  # default number of results per search

# Search query templates for business-related datasets
_QUERY_TEMPLATES = [
    "{state} business license",
    "{county} {state} commercial",
    "{state} retail permit",
    "{county} {state} business",
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CKANDataGovLoader:
    """Discover datasets on data.gov relevant to retail site analysis.

    Searches the CKAN-based data.gov catalog for business license,
    commercial property, and retail permit datasets. Returns metadata
    and download links -- does not parse the datasets themselves.

    No API key is required.

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds.
    rate_limit : float
        Minimum seconds between consecutive API requests.

    Examples
    --------
    >>> loader = CKANDataGovLoader()
    >>> datasets = loader.search_datasets("Texas business license")
    >>> for ds in datasets:
    ...     print(ds["title"], ds["url"])
    """

    BASE_URL = "https://catalog.data.gov/api/3/action"

    def __init__(
        self,
        timeout: int = _REQUEST_TIMEOUT,
        rate_limit: float = _RATE_LIMIT,
    ) -> None:
        self.timeout = timeout
        self._rate_limit = rate_limit
        self._session = requests.Session()
        self._last_request_time: float = 0.0
        # In-memory cache: keyed by (query, rows)
        self._cache: dict[tuple[str, int], list[dict]] = {}

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

    def _parse_dataset(self, pkg: dict) -> dict:
        """Extract a clean metadata dict from a CKAN package result.

        Parameters
        ----------
        pkg : dict
            Raw CKAN package dict from the search results.

        Returns
        -------
        dict
            Cleaned dataset metadata with keys: ``title``, ``notes``
            (description), ``url``, ``resources`` (list of dicts with
            ``url`` and ``format``), ``organization``, ``id``.
        """
        # Extract resources with their URLs and formats
        resources = []
        for res in pkg.get("resources", []):
            res_url = res.get("url", "")
            res_format = res.get("format", "").upper()
            if res_url:
                resources.append({
                    "url": res_url,
                    "format": res_format,
                })

        # Organization name
        org = pkg.get("organization", {})
        org_name = org.get("title", "") if isinstance(org, dict) else ""

        # Dataset URL on data.gov
        dataset_name = pkg.get("name", "")
        dataset_url = (
            f"https://catalog.data.gov/dataset/{dataset_name}"
            if dataset_name
            else pkg.get("url", "")
        )

        return {
            "id": pkg.get("id", ""),
            "title": pkg.get("title", "Untitled"),
            "notes": (pkg.get("notes", "") or "")[:500],  # truncate long descriptions
            "url": dataset_url,
            "resources": resources,
            "organization": org_name,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_datasets(self, query: str, rows: int = _DEFAULT_ROWS) -> list[dict]:
        """Search the data.gov catalog.

        Parameters
        ----------
        query : str
            Free-text search query.
        rows : int
            Maximum number of results to return (default 5).

        Returns
        -------
        list[dict]
            Each dict contains ``title``, ``notes``, ``url``,
            ``resources`` (list of ``{url, format}``), ``organization``,
            and ``id``. Returns empty list on failure.
        """
        cache_key = (query.lower().strip(), rows)
        if cache_key in self._cache:
            logger.debug("CKAN cache hit for query '%s'", query)
            return self._cache[cache_key]

        url = f"{self.BASE_URL}/package_search"
        params = {
            "q": query,
            "rows": rows,
        }

        logger.info("CKAN data.gov: searching '%s' (rows=%d)", query, rows)
        self._throttle()
        self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("CKAN data.gov request failed for '%s': %s", query, exc)
            return []

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("Failed to parse CKAN JSON for '%s': %s", query, exc)
            return []

        # CKAN wraps results: {"success": true, "result": {"results": [...]}}
        if not data.get("success", False):
            logger.warning(
                "CKAN data.gov returned success=false for '%s'", query
            )
            return []

        result_block = data.get("result", {})
        packages = result_block.get("results", [])

        results = [self._parse_dataset(pkg) for pkg in packages]

        self._cache[cache_key] = results
        logger.info(
            "CKAN data.gov: '%s' returned %d datasets", query, len(results)
        )
        return results

    def find_business_datasets(
        self,
        state_name: str,
        county_name: str = "",
    ) -> list[dict]:
        """Search for business license and permit datasets by geography.

        Runs multiple targeted searches and deduplicates results.

        Parameters
        ----------
        state_name : str
            Full state name (e.g. ``"Texas"``).
        county_name : str
            County name (e.g. ``"Travis"``). Optional; broadens search
            if provided.

        Returns
        -------
        list[dict]
            Deduplicated list of dataset metadata dicts. Returns empty
            list if no datasets are found.
        """
        seen_ids: set[str] = set()
        all_datasets: list[dict] = []

        for template in _QUERY_TEMPLATES:
            query = template.format(
                state=state_name,
                county=county_name if county_name else state_name,
            )
            datasets = self.search_datasets(query, rows=5)
            for ds in datasets:
                ds_id = ds.get("id", "")
                if ds_id and ds_id not in seen_ids:
                    seen_ids.add(ds_id)
                    all_datasets.append(ds)

        logger.info(
            "CKAN: found %d unique business datasets for %s %s",
            len(all_datasets), county_name, state_name,
        )
        return all_datasets

    def summarize_available(
        self,
        state_fips: str,
        county_fips: str,
        state_name: str,
        county_name: str,
    ) -> dict:
        """Summarize available data.gov datasets for a geography.

        Parameters
        ----------
        state_fips : str
            2-digit state FIPS code (for reference; not used in search).
        county_fips : str
            3-digit county FIPS code (for reference; not used in search).
        state_name : str
            Full state name (e.g. ``"Texas"``).
        county_name : str
            County name (e.g. ``"Travis County"``).

        Returns
        -------
        dict
            Summary with keys:
            - ``datasets_found`` (int): Total unique datasets found.
            - ``datasets`` (list[dict]): Each has ``title``, ``url``,
              ``format`` (first resource format), and ``description``.
            - ``state_fips`` (str): Echo of input for reference.
            - ``county_fips`` (str): Echo of input for reference.
            Returns dict with ``datasets_found: 0`` on failure.
        """
        # Strip common suffixes from county name for cleaner searches
        clean_county = county_name.replace(" County", "").replace(" Parish", "").strip()

        datasets = self.find_business_datasets(
            state_name=state_name,
            county_name=clean_county,
        )

        summary_datasets = []
        for ds in datasets:
            # Pick the first resource format for summary display
            resources = ds.get("resources", [])
            first_format = resources[0]["format"] if resources else "UNKNOWN"
            summary_datasets.append({
                "title": ds["title"],
                "url": ds["url"],
                "format": first_format,
                "description": ds.get("notes", "")[:200],
            })

        result = {
            "datasets_found": len(summary_datasets),
            "datasets": summary_datasets,
            "state_fips": state_fips,
            "county_fips": county_fips,
        }

        logger.info(
            "CKAN summary for %s, %s (FIPS %s%s): %d datasets",
            county_name, state_name, state_fips, county_fips,
            result["datasets_found"],
        )
        return result

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("CKAN data cache cleared.")

    def __repr__(self) -> str:
        return f"CKANDataGovLoader(cached_queries={len(self._cache)})"
