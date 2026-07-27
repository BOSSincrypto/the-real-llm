"""Provider adapters: the only code in this package that knows a wire format.

Declared explicitly rather than left as an implicit namespace package. A
namespace package is open for extension by anything else installed alongside
it, which would let an unrelated distribution drop a module into
``llmverify.adapters`` and have it looked up as though it shipped here. For a
tool whose output is an accusation about someone's honesty, the import path
that resolves its adapters should be closed.

Third-party adapters are still supported, and are registered through the
``llmverify.adapters`` entry-point group instead. See ``docs/adapters.md``.
"""

from __future__ import annotations

from .base import (
    Adapter,
    Capabilities,
    ProbeOutcome,
    available_adapters,
    get_adapter,
    register_adapter,
)

__all__ = [
    "Adapter",
    "Capabilities",
    "ProbeOutcome",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
