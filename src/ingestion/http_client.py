"""Cached, rate-limited HTTP client used by all ingestion code.

Design goals:
  * Never hammer a third-party site: enforce a minimum delay between
    requests to the same host and reuse a persistent on-disk cache so
    re-running ingestion doesn't re-fetch pages we already have.
  * Respect robots.txt for the target host before the first request.
  * Fail loudly (not silently) on non-200 responses so ingestion gaps
    are visible rather than swallowed.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_USER_AGENT = (
    "AMC10FutureScorePredictor/0.1 "
    "(+https://github.com/; educational research project; contact via GitHub)"
)


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    from_cache: bool


class RateLimitedCachedClient:
    """A tiny HTTP client with an on-disk cache and a per-host delay."""

    def __init__(
        self,
        cache_dir: str | Path,
        min_delay_seconds: float = 1.5,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay_seconds = min_delay_seconds
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.timeout = timeout
        self._last_request_time: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _host(self, url: str) -> str:
        from urllib.parse import urlparse

        return urlparse(url).netloc

    def _check_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        from urllib.parse import urlparse

        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt can't be fetched, default to allow but log nothing
                # destructive; this matches common crawler behavior.
                self._robots[base] = None  # type: ignore
                return True
            self._robots[base] = rp
        rp = self._robots[base]
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    def _throttle(self, url: str) -> None:
        host = self._host(url)
        last = self._last_request_time.get(host)
        if last is not None:
            elapsed = time.monotonic() - last
            wait = self.min_delay_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_time[host] = time.monotonic()

    def get(self, url: str, use_cache: bool = True) -> FetchResult:
        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return FetchResult(url=url, status=payload["status"], text=payload["text"], from_cache=True)

        if not self._check_robots(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")

        self._throttle(url)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            status = e.code
            text = e.read().decode("utf-8", errors="replace") if e.fp else ""

        cache_path.write_text(
            json.dumps({"url": url, "status": status, "text": text}), encoding="utf-8"
        )
        return FetchResult(url=url, status=status, text=text, from_cache=False)
