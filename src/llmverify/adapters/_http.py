"""Shared HTTP plumbing for adapters.

Adapters do not talk to ``httpx`` directly. They go through :class:`HttpClient`,
which centralises three things that matter for verification specifically:

* **Timing.** Total wall time and, for streamed calls, time-to-first-token are
  captured at the transport layer, where they are actually accurate.
* **Error fidelity.** A 4xx is not an exception to be swallowed. The parameter-
  support probe *needs* the status and body to tell "rejected" from "ignored",
  so failures are returned as structured :class:`~llmverify.errors.ProviderError`
  instances with the body intact.
* **Retry discipline.** Only idempotent transport failures and 429/5xx are
  retried. A 400 is a result, not a flake, and retrying it would corrupt probes.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import AuthError, ProviderError, RateLimited

__all__ = ["HttpClient", "HttpResult"]

_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


@dataclass(slots=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    json: Any
    text: str
    total_s: float
    ttft_s: float | None = None
    #: Server-sent events, when the request was streamed.
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class HttpClient:
    """Async HTTP client with verification-friendly retry and timing.

    One instance per provider. Concurrency is bounded by a semaphore so that a
    fan-out of probes cannot exceed what the endpoint tolerates -- exceeding it
    produces 429s that look like capability failures.
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout_s: float = 180.0,
        connect_timeout_s: float = 20.0,
        max_concurrency: int = 4,
        max_retries: int = 3,
        verify_tls: bool = True,
        proxy: str | None = None,
        query_params: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._query_params = dict(query_params or {})
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            headers={"user-agent": "llmverify/0.1", **(headers or {})},
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            verify=verify_tls,
            proxy=proxy,
            follow_redirects=True,
            # A verifier must see exactly what the endpoint sent, so no
            # transparent decompression surprises: httpx handles gzip/br for us
            # but we keep the raw text around regardless.
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        retry: bool = True,
    ) -> HttpResult:
        """Perform a request, retrying only genuinely transient failures.

        Returns an :class:`HttpResult` for any HTTP status, including 4xx.
        Raises only when the transport itself failed after all retries.
        """
        merged_params = {**self._query_params, **(params or {})}
        attempts = self.max_retries + 1 if retry else 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            async with self._sem:
                started = time.perf_counter()
                try:
                    resp = await self._client.request(
                        method,
                        self._url(path),
                        json=json_body,
                        headers=headers,
                        params=merged_params or None,
                    )
                except (httpx.TransportError, httpx.StreamError) as exc:
                    last_exc = exc
                    if attempt + 1 >= attempts:
                        raise ProviderError(f"transport failure: {exc}") from exc
                    await self._backoff(attempt)
                    continue
                elapsed = time.perf_counter() - started

            if resp.status_code in _RETRY_STATUSES and attempt + 1 < attempts:
                await self._backoff(attempt, resp.headers.get("retry-after"))
                continue

            return HttpResult(
                status=resp.status_code,
                headers={k.lower(): v for k, v in resp.headers.items()},
                json=_safe_json(resp),
                text=resp.text,
                total_s=elapsed,
            )

        raise ProviderError(f"request failed after {attempts} attempts: {last_exc}")

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> HttpResult:
        """Perform a server-sent-events request and collect every event.

        Time-to-first-token is measured at the first event carrying content,
        not the first byte of the response -- providers emit role/ping frames
        first and counting those would flatter their latency.
        """
        import json as _json

        merged_params = {**self._query_params, **(params or {})}
        events: list[dict[str, Any]] = []
        ttft: float | None = None

        async with self._sem:
            started = time.perf_counter()
            try:
                async with self._client.stream(
                    method,
                    self._url(path),
                    json=json_body,
                    headers=headers,
                    params=merged_params or None,
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        return HttpResult(
                            status=resp.status_code,
                            headers={k.lower(): v for k, v in resp.headers.items()},
                            json=None,
                            text=body,
                            total_s=time.perf_counter() - started,
                        )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            event = _json.loads(payload)
                        except ValueError:
                            continue
                        events.append(event)
                        if ttft is None and _event_has_content(event):
                            ttft = time.perf_counter() - started
                    status = resp.status_code
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            except (httpx.TransportError, httpx.StreamError) as exc:
                raise ProviderError(f"stream failure: {exc}") from exc
            total = time.perf_counter() - started

        return HttpResult(
            status=status,
            headers=resp_headers,
            json=None,
            text="",
            total_s=total,
            ttft_s=ttft,
            events=events,
        )

    async def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        # Full jitter, capped. Deterministic seeding is deliberately *not* used
        # here: retry timing must not correlate across probes, or a provider
        # could use the pattern to recognise the verifier.
        delay = min(2.0 * (2**attempt), 20.0)
        await asyncio.sleep(random.uniform(0.0, delay))  # noqa: S311


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _event_has_content(event: dict[str, Any]) -> bool:
    """True when an SSE frame carries actual generated content."""
    etype = event.get("type", "")
    if etype in {"content_block_delta", "message_delta"}:
        return True
    if etype in {"message_start", "content_block_start", "ping", "response.created"}:
        return False
    for choice in event.get("choices") or ():
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
            return True
    if event.get("candidates"):
        return True
    return False


def raise_for_status(result: HttpResult, *, context: str) -> None:
    """Convert a failed :class:`HttpResult` into the right exception type."""
    if result.ok:
        return
    message = f"{context}: HTTP {result.status}"
    detail = (result.text or "")[:800]
    if result.status in (401, 403):
        raise AuthError(message, status=result.status, body=detail, headers=result.headers)
    if result.status == 429:
        retry_after = result.headers.get("retry-after")
        raise RateLimited(
            message,
            retry_after=float(retry_after) if _is_number(retry_after) else None,
            status=result.status,
            body=detail,
            headers=result.headers,
        )
    raise ProviderError(message, status=result.status, body=detail, headers=result.headers)


def _is_number(value: str | None) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True
