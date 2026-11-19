"""Small HTTP helper with retries, back-off and JSON caching."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bcn-flood-transport-prototype/0.1"})


class SourceUnavailable(RuntimeError):
    """Raised when an external data source cannot be reached."""


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             retries: int = 3, wait_s: float = 10, timeout: float = 120) -> dict | list:
    last = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                log.warning("%s -> %s (attempt %d/%d)", url, last, attempt + 1, retries)
                if "Daily API request limit" in r.text:
                    break  # waiting will not help today
                time.sleep(wait_s * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise SourceUnavailable(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
            return r.json()
        except requests.RequestException as exc:
            last = str(exc)
            log.warning("%s -> %s (attempt %d/%d)", url, exc, attempt + 1, retries)
            time.sleep(wait_s * (attempt + 1))
    raise SourceUnavailable(f"{url} unavailable: {last}")


def post_form(url: str, data: dict, retries: int = 2, wait_s: float = 10, timeout: float = 300) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = SESSION.post(url, data=data, timeout=timeout)
            if r.status_code >= 400:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(wait_s)
                continue
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = str(exc)
            time.sleep(wait_s)
    raise SourceUnavailable(f"{url} unavailable: {last}")


def cached_json(path: Path, fetch, refresh: bool = False):
    """Return JSON from `path` if present, otherwise call fetch() and store it."""
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data
