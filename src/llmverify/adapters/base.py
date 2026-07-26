"""The adapter contract.

An adapter is the only place in this package that knows a wire format. Probes
are written entirely against :mod:`llmverify.types`, so adding support for a new
provider protocol means writing one subclass and registering it -- no changes
anywhere else.

See ``docs/adapters.md`` for a worked example of a third-party adapter.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar

from ..config import ProviderConfig
from ..errors import UnsupportedCapability
from ..types import ApiFamily, ChatRequest, ChatResponse, Message, ParamSupport, ToolSpec
from ._http import HttpClient, HttpResult

__all__ = ["Adapter", "Capabilities", "ProbeOutcome"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What an adapter believes its protocol can express.

    These are *protocol* capabilities, not endpoint capabilities. Whether a
    given endpoint honours them is exactly what the api-surface probe measures,
    and the gap between the two is evidence.
    """

    logprobs: bool = False
    seed: bool = False
    tools: bool = True
    vision: bool = True
    structured_output: bool = True
    reasoning_effort: bool = False
    thinking_signature: bool = False
    count_tokens_endpoint: bool = False
    list_models_endpoint: bool = True
    streaming: bool = True


@dataclass(slots=True)
class ProbeOutcome:
    """Result of asking whether an endpoint honours one request parameter."""

    parameter: str
    support: ParamSupport
    status: int | None = None
    detail: str = ""
    body_excerpt: str = ""


class Adapter(abc.ABC):
    """Base class for all provider adapters."""

    #: Registry key, matched against ``ProviderConfig.api``.
    name: ClassVar[str]
    #: Which wire protocol this adapter speaks.
    family: ClassVar[ApiFamily]
    #: First-party endpoint, used when the config omits ``base_url``.
    default_base_url: ClassVar[str]
    #: How the API key is presented when the config does not override it.
    default_auth_scheme: ClassVar[str] = "bearer"
    #: Protocol-level capabilities.
    capabilities: ClassVar[Capabilities] = Capabilities()

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.api_key = config.resolve_api_key()
        self.http = HttpClient(
            base_url=config.base_url or self.default_base_url,
            headers=self._auth_headers(),
            timeout_s=config.timeout_s,
            connect_timeout_s=config.connect_timeout_s,
            max_concurrency=config.max_concurrency,
            max_retries=config.max_retries,
            verify_tls=config.verify_tls,
            proxy=config.proxy,
            query_params=self._auth_query_params(),
        )

    # ---------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> Adapter:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- auth

    def _auth_headers(self) -> dict[str, str]:
        scheme = self.config.auth_scheme or self.default_auth_scheme
        headers = dict(self.config.headers)
        if not self.api_key or scheme in ("none", "query"):
            return headers
        if scheme == "bearer":
            headers.setdefault("authorization", f"Bearer {self.api_key}")
        elif scheme == "x-api-key":
            headers.setdefault("x-api-key", self.api_key)
        return headers

    def _auth_query_params(self) -> dict[str, str]:
        scheme = self.config.auth_scheme or self.default_auth_scheme
        params = dict(self.config.query_params)
        if self.api_key and scheme == "query":
            params.setdefault("key", self.api_key)
        return params

    # ---------------------------------------------------------------- core API

    @abc.abstractmethod
    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Translate a :class:`ChatRequest` into this protocol's JSON body.

        Must omit, not null out, any field the request left as ``None`` --
        several probes depend on the difference.
        """

    @abc.abstractmethod
    def parse_response(self, result: HttpResult) -> ChatResponse:
        """Translate a successful HTTP result into a :class:`ChatResponse`."""

    @abc.abstractmethod
    def parse_stream(self, result: HttpResult) -> ChatResponse:
        """Assemble a :class:`ChatResponse` from collected SSE frames."""

    #: Path appended to ``base_url`` for completions.
    chat_path: ClassVar[str] = "/chat/completions"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a request and return the parsed response.

        Raises :class:`~llmverify.errors.ProviderError` on a non-2xx status.
        Probes that want to *inspect* failures should call :meth:`try_chat`.
        """
        response, error = await self.try_chat(request)
        if error is not None:
            raise error
        assert response is not None
        return response

    async def try_chat(
        self, request: ChatRequest
    ) -> tuple[ChatResponse | None, Exception | None]:
        """Like :meth:`chat`, but returns errors instead of raising them."""
        from ._http import raise_for_status

        payload = self.build_payload(request)
        headers = dict(request.extra_headers)
        try:
            if request.stream:
                result = await self.http.stream(
                    "POST", self.chat_path, json_body=payload, headers=headers or None
                )
            else:
                result = await self.http.request(
                    "POST", self.chat_path, json_body=payload, headers=headers or None
                )
        except Exception as exc:  # transport-level
            return None, exc

        if not result.ok:
            try:
                raise_for_status(result, context=f"{self.name} chat")
            except Exception as exc:
                return None, exc

        try:
            parsed = self.parse_stream(result) if request.stream else self.parse_response(result)
        except Exception as exc:
            return None, exc
        return parsed, None

    # ---------------------------------------------------------------- optional

    async def count_tokens(
        self, messages: tuple[Message, ...], tools: tuple[ToolSpec, ...] = ()
    ) -> int:
        """Return the provider's own token count for these messages.

        A dedicated tokenizer endpoint is the cleanest possible tokenizer
        fingerprint, because it removes generation from the measurement
        entirely. Adapters without one raise
        :class:`~llmverify.errors.UnsupportedCapability`.
        """
        raise UnsupportedCapability(f"{self.name} exposes no token counting endpoint")

    async def list_models(self) -> list[dict[str, Any]]:
        """Return the endpoint's model catalogue.

        The *shape* of these entries is itself a fingerprint of what software is
        serving the endpoint.
        """
        from ._http import raise_for_status

        result = await self.http.request("GET", "/models")
        raise_for_status(result, context=f"{self.name} list models")
        body = result.json or {}
        if isinstance(body, dict):
            data = body.get("data") or body.get("models") or []
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict)]
        if isinstance(body, list):
            return [m for m in body if isinstance(m, dict)]
        return []

    async def probe_parameter(
        self,
        request: ChatRequest,
        *,
        parameter: str,
        payload_patch: dict[str, Any],
        detect_effect: Any = None,
    ) -> ProbeOutcome:
        """Determine whether the endpoint accepts, rejects or silently drops a parameter.

        ``detect_effect`` is an optional callable taking the parsed response and
        returning ``True`` when the parameter demonstrably took effect. Without
        it, a 2xx can only be reported as ``ACCEPTED``; with it, a 2xx whose
        response shows no effect is correctly reported as ``IGNORED`` -- the
        signature of a LiteLLM front end running ``drop_params=True``.
        """
        patched = request.replace(extra_body={**request.extra_body, **payload_patch})
        response, error = await self.try_chat(patched)

        if error is not None:
            from ..errors import ProviderError

            status = getattr(error, "status", None)
            body = (getattr(error, "body", "") or "")[:400]
            if isinstance(error, ProviderError) and status and 400 <= status < 500:
                return ProbeOutcome(
                    parameter,
                    ParamSupport.REJECTED,
                    status=status,
                    detail=f"HTTP {status}",
                    body_excerpt=body,
                )
            return ProbeOutcome(
                parameter, ParamSupport.UNKNOWN, status=status, detail=str(error)[:200]
            )

        assert response is not None
        if detect_effect is not None:
            took_effect = detect_effect(response)
            if took_effect is False:
                return ProbeOutcome(
                    parameter,
                    ParamSupport.IGNORED,
                    status=response.http_status,
                    detail="accepted with 2xx but had no observable effect",
                )
            if took_effect is None:
                return ProbeOutcome(
                    parameter, ParamSupport.UNKNOWN, status=response.http_status
                )
        return ProbeOutcome(parameter, ParamSupport.ACCEPTED, status=response.http_status)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[Adapter]] = {}


def register_adapter(cls: type[Adapter]) -> type[Adapter]:
    """Register an adapter class under its ``name``. Usable as a decorator."""
    _REGISTRY[cls.name] = cls
    return cls


def _load_entry_points() -> None:
    """Discover adapters published by third-party packages."""
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group="llmverify.adapters")
    except TypeError:  # pragma: no cover - Python <3.10 signature
        eps = entry_points().get("llmverify.adapters", [])  # type: ignore[assignment]
    for ep in eps:
        if ep.name in _REGISTRY:
            continue
        try:
            cls = ep.load()
        except Exception:  # a broken plugin must not break the tool
            continue
        if isinstance(cls, type) and issubclass(cls, Adapter):
            _REGISTRY[cls.name] = cls


def available_adapters() -> dict[str, type[Adapter]]:
    """All registered adapters, including plugins."""
    from . import (  # noqa: F401
        anthropic,
        gemini,
        openai_compat,
    )

    _load_entry_points()
    return dict(_REGISTRY)


def get_adapter(config: ProviderConfig) -> Adapter:
    """Instantiate the adapter named by ``config.api``."""
    from ..errors import AdapterError

    registry = available_adapters()
    cls = registry.get(config.api)
    if cls is None:
        known = ", ".join(sorted(registry)) or "(none)"
        raise AdapterError(f"unknown adapter {config.api!r}; available: {known}")
    return cls(config)
