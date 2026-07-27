"""Benchmark dataset access over the HuggingFace datasets-server REST API.

This deliberately does not use the ``datasets`` library. Pulling it in would
drag ``pyarrow``, ``pandas`` and a long transitive tail into a package whose
only other dependencies are an HTTP client and a validator, and it would
download whole corpora to read the 200 rows a verification run actually
samples. The REST API returns exactly the requested slice as JSON.

Two endpoints are used, both verified live on 2026-07-26::

    GET /splits?dataset=<id>
        -> {"splits": [{"dataset", "config", "split"}], "pending": [], "failed": []}
    GET /rows?dataset=<id>&config=<c>&split=<s>&offset=<n>&length=<1..100>
        -> {"features": [...], "rows": [{"row_idx", "row", "truncated_cells"}],
            "num_rows_total": N}

Everything fetched is cached under ``<cache_dir>/datasets`` in an envelope that
records which spec and row range produced it. A verification run must be
reproducible weeks later and must not re-pay for network flakiness mid-run, so
a cache hit short-circuits the network entirely and ``offline=True`` refuses to
fall back to it.

Two HuggingFace behaviours are worth knowing before reading the error handling:

* ``/api/datasets/<id>`` answers 401 "Invalid username or password" for a gated
  repo *and* for one that does not exist, so it is useless as an existence
  check and is never called here. ``/splits`` is used instead -- but it too
  returns 401 for both cases, so the 401 message names both possibilities
  rather than guessing.
* Rows above the server's response-size limit come back with ``truncated_cells``
  populated and the offending fields cut short. Silently handing a truncated
  question to a grader would produce a wrong answer that looks like a model
  failure, so pages are re-fetched at a smaller page size until the truncation
  clears.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import httpx

from ..config import register_secret
from ..errors import DatasetError

__all__ = ["DatasetLoader", "DatasetSpec", "cache_info", "clear_cache"]

_log = logging.getLogger(__name__)

SERVER = "https://datasets-server.huggingface.co"
#: Hard server-side ceiling on ``length``; 101 is answered with HTTP 422.
MAX_PAGE = 100
#: Bumping this invalidates every cache entry, because it is part of the key.
CACHE_VERSION = 1
_ENVELOPE_KEY = "_llmverify"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One ``(dataset, config, split)`` triple plus its access conditions.

    ``config`` may be ``None``, in which case it is resolved from ``/splits``
    at load time. Resolution refuses to guess when several configs offer the
    named split: picking one silently would make a run's meaning depend on
    HuggingFace's ordering.
    """

    dataset: str
    config: str | None
    split: str
    gated: bool = False
    licence: str = "unknown"

    @property
    def hub_url(self) -> str:
        """Page where a gated dataset's terms are accepted."""
        return f"https://huggingface.co/datasets/{self.dataset}"

    def __str__(self) -> str:
        return f"{self.dataset}:{self.config or '<auto>'}/{self.split}"


class DatasetLoader:
    """Fetches and caches rows from the HuggingFace datasets-server.

    One instance is shared by every benchmark in a run so that the HTTP
    connection pool and the cache directory are shared too. It is deliberately
    not built on :class:`~llmverify.adapters._http.HttpClient`: that client is
    scoped to a provider, carries provider auth headers, and applies retry
    policy tuned for inference endpoints rather than for a dataset CDN.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        hf_token: str | None = None,
        timeout_s: float = 60.0,
        offline: bool = False,
        token_env: str = "HF_TOKEN",
    ) -> None:
        """Create a loader.

        ``token_env`` is the *name* of the environment variable ``hf_token``
        came from. It is carried only so that an authentication failure can
        tell the user which variable to set; the value itself is registered for
        redaction and never appears in a message.
        """
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.token_env = token_env
        self._token = hf_token
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None
        register_secret(hf_token)

    async def __aenter__(self) -> DatasetLoader:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -------------------------------------------------------------- discovery

    async def splits(self, dataset: str) -> list[dict[str, Any]]:
        """List the ``{dataset, config, split}`` triples a repo exposes.

        Cached like row data, which is what makes ``offline=True`` usable for a
        spec that left ``config`` unset.
        """
        key = _cache_key({"kind": "splits", "dataset": dataset})
        cached = _read_cache(self.cache_dir, key)
        if cached is not None:
            return list(cached.get("splits") or [])

        if self.offline:
            raise DatasetError(
                f"offline mode: no cached split list for {dataset!r}. Run once with network "
                f"access to populate {self.cache_dir / 'datasets'}, or pin the config and split "
                "explicitly so no lookup is needed."
            )

        status, payload = await self._get("splits", {"dataset": dataset})
        if status != 200:
            await self._fail(status, payload, dataset=dataset)

        splits = list(payload.get("splits") or []) if isinstance(payload, dict) else []
        if not splits:
            pending = payload.get("pending") or [] if isinstance(payload, dict) else []
            if pending:
                raise DatasetError(
                    f"{dataset!r} is still being indexed by the datasets-server "
                    f"({len(pending)} job(s) pending). Retry in a few minutes."
                )
            raise DatasetError(
                f"the datasets-server exposes no splits for {dataset!r}; it may have failed "
                f"to convert. See {DatasetSpec(dataset, None, '').hub_url}"
            )

        _write_cache(
            self.cache_dir, key, {"kind": "splits", "dataset": dataset}, {"splits": splits}
        )
        return splits

    # ------------------------------------------------------------------- rows

    async def rows(
        self, spec: DatasetSpec, *, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch ``limit`` rows starting at ``offset``, newest cache first.

        Returns the inner ``row`` mappings, not the server's row envelopes: the
        ``row_idx`` is implied by position and ``truncated_cells`` is resolved
        before the rows are returned rather than pushed onto callers.

        ``limit=None`` means "the whole split", which for a large corpus is a
        lot of requests. Benchmarks are expected to pass a real limit.
        """
        if offset < 0:
            raise DatasetError(f"offset must be non-negative, got {offset}")
        if limit is not None and limit < 0:
            raise DatasetError(f"limit must be non-negative or None, got {limit}")

        key_fields = {
            "kind": "rows",
            "dataset": spec.dataset,
            "config": spec.config,
            "split": spec.split,
            "offset": offset,
            "limit": limit,
        }
        key = _cache_key(key_fields)
        cached = _read_cache(self.cache_dir, key)
        if cached is not None:
            return list(cached.get("rows") or [])

        if self.offline:
            span = "all rows" if limit is None else f"{limit} rows"
            raise DatasetError(
                f"offline mode: no cache entry for {span} of {spec} at offset {offset}. "
                f"Expected {_cache_file(self.cache_dir, key)}. Re-run once with network access, "
                "or point --cache-dir at a directory populated elsewhere."
            )

        config = spec.config or await self._resolve_config(spec)

        collected: list[dict[str, Any]] = []
        truncated: list[int] = []
        total: int | None = None
        cursor = offset
        # Once a single-row page has come back truncated, this split simply has
        # oversized rows and shrinking every later page only multiplies requests.
        shrink = True

        while limit is None or len(collected) < limit:
            want = MAX_PAGE if limit is None else min(MAX_PAGE, limit - len(collected))
            entries, total, page_truncated, asked = await self._fetch_page(
                spec, config, cursor, want, shrink=shrink
            )
            if not entries:
                break
            if page_truncated and asked <= 1:
                shrink = False
            collected.extend(e.get("row") or {} for e in entries)
            truncated.extend(page_truncated)
            cursor += len(entries)
            if total is not None and cursor >= total:
                break
            # A short page means the split ran out -- but only when it is short
            # against the size actually requested, which truncation handling may
            # have reduced below ``want``.
            if len(entries) < asked:
                break

        if truncated:
            _log.warning(
                "%s: %d row(s) still have truncated cells at the minimum page size; "
                "their content is incomplete: %s",
                spec,
                len(truncated),
                truncated[:10],
            )

        meta = {
            **key_fields,
            "resolved_config": config,
            "gated": spec.gated,
            "licence": spec.licence,
            "num_rows_total": total,
            "returned": len(collected),
            "truncated_row_idx": truncated,
        }
        _write_cache(self.cache_dir, key, meta, {"rows": collected})
        return collected

    async def _resolve_config(self, spec: DatasetSpec) -> str:
        """Pick the config that owns ``spec.split``, or refuse if it is ambiguous."""
        entries = await self.splits(spec.dataset)
        matching = [e for e in entries if e.get("split") == spec.split]
        if not matching:
            available = ", ".join(sorted({f"{e.get('config')}/{e.get('split')}" for e in entries}))
            raise DatasetError(
                f"{spec.dataset!r} has no split named {spec.split!r}. "
                f"Available config/split pairs: {available}"
            )
        configs = sorted({str(e.get("config")) for e in matching})
        if len(configs) == 1:
            return configs[0]
        if "default" in configs:
            return "default"
        raise DatasetError(
            f"{spec.dataset!r} offers split {spec.split!r} under {len(configs)} configs "
            f"({', '.join(configs)}); name one explicitly so the run is reproducible."
        )

    async def _fetch_page(
        self, spec: DatasetSpec, config: str, offset: int, length: int, *, shrink: bool = True
    ) -> tuple[list[dict[str, Any]], int | None, list[int], int]:
        """Fetch one page, shrinking it until the server stops truncating cells.

        Returns the row envelopes, the split's total row count, the indices of
        rows that stayed truncated, and the page size finally used -- the caller
        needs that last value to tell a shrunk page from the end of the split.

        The server truncates on total response size, so halving the page size
        is what actually recovers the full cell content. A row that is still
        truncated on its own is genuinely oversized and is reported upward.
        """
        length = max(1, min(MAX_PAGE, length))
        while True:
            status, payload = await self._get(
                "rows",
                {
                    "dataset": spec.dataset,
                    "config": config,
                    "split": spec.split,
                    "offset": offset,
                    "length": length,
                },
            )
            if status != 200:
                await self._fail(status, payload, dataset=spec.dataset, spec=spec, config=config)

            entries = list(payload.get("rows") or []) if isinstance(payload, dict) else []
            total = payload.get("num_rows_total") if isinstance(payload, dict) else None
            damaged = [e for e in entries if e.get("truncated_cells")]
            if damaged and shrink and length > 1:
                length = max(1, length // 2)
                continue
            idx = [int(e.get("row_idx", -1)) for e in damaged]
            return entries, (int(total) if isinstance(total, int) else None), idx, length

    # ------------------------------------------------------------------- http

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"user-agent": "llmverify/0.1"}
            if self._token:
                headers["authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self._timeout_s, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    async def _get(self, path: str, params: dict[str, Any]) -> tuple[int, Any]:
        """GET a datasets-server endpoint, retrying only transient failures.

        The query string is built here rather than handed to httpx because the
        dataset id contains a ``/`` that must be percent-encoded; that is the
        form the API was verified against.
        """
        query = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
        url = f"{SERVER}/{path}?{query}"
        client = self._http()
        delay = 1.0

        for attempt in range(3):
            try:
                response = await client.get(url)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise DatasetError(
                        f"could not reach the HuggingFace datasets-server ({path}): {exc}"
                    ) from exc
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if response.status_code in _RETRY_STATUSES and attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue

            try:
                payload: Any = response.json()
            except ValueError:
                payload = {"error": response.text[:500]}
            return response.status_code, payload

        raise DatasetError(f"datasets-server request for {path} did not complete")

    async def _fail(
        self,
        status: int,
        payload: Any,
        *,
        dataset: str,
        spec: DatasetSpec | None = None,
        config: str | None = None,
    ) -> NoReturn:
        """Turn a non-200 into a :class:`DatasetError` a user can act on."""
        detail = ""
        if isinstance(payload, dict) and payload.get("error"):
            detail = f" Server said: {payload['error']}"

        if status in (401, 403):
            hub = f"https://huggingface.co/datasets/{dataset}"
            if self._token:
                cause = (
                    f"A token from ${self.token_env} was sent, so the account behind it has "
                    f"most likely not accepted the dataset terms yet -- do that at {hub}."
                )
            else:
                cause = (
                    f"No token was sent. Accept the dataset terms at {hub}, create a read token "
                    f"at https://huggingface.co/settings/tokens, and export it as "
                    f"${self.token_env}."
                )
            raise DatasetError(
                f"HuggingFace refused access to {dataset!r} (HTTP {status}). The datasets-server "
                f"answers identically for a gated dataset and for one that does not exist, so "
                f"also check the id for typos. {cause}{detail}"
            )

        if status == 404:
            hint = await self._describe_splits(dataset)
            target = f"config={config!r} split={spec.split!r}" if spec else "the requested split"
            raise DatasetError(
                f"{dataset!r} has no {target}. The dataset name may also be wrong. {hint}{detail}"
            )

        if status == 422:
            raise DatasetError(
                f"the datasets-server rejected the request for {dataset!r} (HTTP 422).{detail}"
            )

        raise DatasetError(f"datasets-server returned HTTP {status} for {dataset!r}.{detail}")

    async def _describe_splits(self, dataset: str) -> str:
        """Best-effort listing of valid config/split pairs, for a 404 message."""
        try:
            entries = await self.splits(dataset)
        except DatasetError:
            return "Its split list could not be read either."
        pairs = sorted({f"{e.get('config')}/{e.get('split')}" for e in entries})
        return f"Available config/split pairs: {', '.join(pairs)}."


# ------------------------------------------------------------------ cache i/o


def _cache_key(fields: dict[str, Any]) -> str:
    blob = json.dumps({**fields, "v": CACHE_VERSION}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_file(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / "datasets" / f"{key}.json"


def _read_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return a cached body, or ``None`` for a miss.

    A corrupt or half-written entry counts as a miss rather than an error: the
    cache is an optimisation and must never be able to fail a run.
    """
    path = _cache_file(cache_dir, key)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        _log.debug("discarding corrupt cache entry %s", path)
        return None
    if not isinstance(data, dict) or _ENVELOPE_KEY not in data:
        return None
    return data


def _write_cache(cache_dir: Path, key: str, meta: dict[str, Any], body: dict[str, Any]) -> None:
    """Write an entry atomically. Failure to cache is never fatal."""
    path = _cache_file(cache_dir, key)
    envelope = {
        _ENVELOPE_KEY: {
            "cache_version": CACHE_VERSION,
            "fetched_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
            "source": SERVER,
            **meta,
        },
        **body,
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        _log.debug("could not write cache entry %s: %s", path, exc)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def cache_info(cache_dir: Path) -> dict[str, Any]:
    """Summarise the dataset cache: entry count, size and age range.

    Backs ``llmverify cache``. Returns zeroes and ``None`` timestamps for a
    cache directory that does not exist yet, which is not an error.
    """
    root = Path(cache_dir) / "datasets"
    entries = 0
    total_bytes = 0
    oldest: float | None = None
    newest: float | None = None

    for path in root.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries += 1
        total_bytes += stat.st_size
        if oldest is None or stat.st_mtime < oldest:
            oldest = stat.st_mtime
        if newest is None or stat.st_mtime > newest:
            newest = stat.st_mtime

    return {
        "path": str(root),
        "entries": entries,
        "bytes": total_bytes,
        "oldest": _iso(oldest),
        "newest": _iso(newest),
    }


def clear_cache(cache_dir: Path) -> int:
    """Delete every dataset cache entry and return the bytes freed."""
    root = Path(cache_dir) / "datasets"
    freed = 0
    for path in list(root.glob("*.json")) + list(root.glob("*.tmp")):
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        freed += size
    with contextlib.suppress(OSError):
        root.rmdir()
    return freed


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat(timespec="seconds")
