"""Live connector for the Indiana General Assembly MyIGA API.

The bundled corpus ships as static JSON so the pipeline runs without credentials.
This connector replaces it for live operation: point it at a session, pull the
bills, and hand the same document dicts to the same graph.

Getting a token: the MyIGA Developer Program issues free API tokens on request
by email to apitoken.request@iga.in.gov. Documentation lives at
https://docs.api.iga.in.gov/.

Two operational details the docs call out and this client handles:

* The API rate limits and returns HTTP 429. `_get` backs off exponentially and
  respects Retry-After when present.
* Bill text is versioned -- introduced, committee-reported, engrossed, enrolled.
  For compliance tracking you almost always want the *enrolled* version, since
  that is the text that became law. `fetch_session_bills` defaults to it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

from .. import __version__
from ..logging_setup import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.iga.in.gov"
USER_AGENT = f"LegisWatch/{__version__} (compliance obligation tracking)"


class IGAClient:
    def __init__(self, token: str | None = None, base_url: str = BASE_URL) -> None:
        if requests is None:
            raise RuntimeError("pip install requests")
        self.token = token or os.getenv("IGA_API_TOKEN")
        if not self.token:
            raise RuntimeError(
                "No IGA API token. Set IGA_API_TOKEN, or request one free from "
                "apitoken.request@iga.in.gov (see https://docs.api.iga.in.gov/)."
            )
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {self.token}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            }
        )

    def _get(self, path: str, *, max_retries: int = 5, **params: Any) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        delay = 1.0
        for attempt in range(max_retries):
            resp = self.session.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                log.warning(
                    "iga_rate_limited",
                    extra={"path": path, "attempt": attempt, "sleep_s": wait},
                )
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
                continue

            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"IGA API rate limit not cleared after {max_retries} attempts: {path}")

    # -- endpoints ---------------------------------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._get("/sessions").get("items", [])

    def list_bills(self, session: str) -> list[dict[str, Any]]:
        return self._get(f"/{session}/bills").get("items", [])

    def get_bill(self, session: str, bill_name: str) -> dict[str, Any]:
        return self._get(f"/{session}/bills/{bill_name}")

    def get_bill_versions(self, session: str, bill_name: str) -> list[dict[str, Any]]:
        return self._get(f"/{session}/bills/{bill_name}/versions").get("items", [])

    def get_version_text(self, version_link: str) -> str:
        payload = self._get(version_link)
        return payload.get("full_text") or payload.get("text") or ""

    # -- adapter to the pipeline's document shape --------------------------

    def fetch_session_bills(
        self,
        session: str,
        *,
        prefer_version: str = "enrolled",
        limit: int | None = None,
    ) -> Iterator[dict]:
        """Yield document dicts in the shape `run_pipeline` expects.

        Identical shape to the entries in data/corpus/corpus.json, so swapping
        the static corpus for live data is a one-line change at the call site.
        """
        bills = self.list_bills(session)
        if limit:
            bills = bills[:limit]

        for bill in bills:
            name = bill.get("billName") or bill.get("name")
            if not name:
                continue
            try:
                versions = self.get_bill_versions(session, name)
                if not versions:
                    continue
                chosen = next(
                    (
                        v
                        for v in versions
                        if prefer_version in (v.get("stageVerbose", "") or "").lower()
                    ),
                    versions[-1],
                )
                text = self.get_version_text(chosen.get("link", ""))
                if not text.strip():
                    continue

                yield {
                    "doc_id": f"{session}-{name}",
                    "citation": f"{name} ({session} session, {chosen.get('stageVerbose', 'unknown stage')})",
                    "title": bill.get("title") or bill.get("description") or name,
                    "enacting_act": name,
                    "effective_date": bill.get("effectiveDate"),
                    "source_url": f"https://iga.in.gov/legislative/{session}/bills/{name.lower().replace(' ', '')}/details",
                    "is_negative_control": False,
                    "text": text,
                }
            except Exception as e:  # one bad bill must not kill the run
                log.error("iga_bill_fetch_failed", extra={"bill": name, "error": str(e)[:300]})
                continue
