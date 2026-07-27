"""Fixtures shared by the suite.

Three things live here that every other module depends on.

**A running mock provider.** Started once per test that asks for it, on an
ephemeral port, so adapters exercise their real HTTP paths.

**A synthetic reference snapshot.** The bundled snapshot is the tool's own data
and is tested on its own terms in ``test_reference.py``; using it to drive probe
tests would couple every probe assertion to numbers that change whenever the
snapshot is refreshed. The snapshot built here holds two fictional models whose
token accounting, context window and benchmark scores are chosen to make each
probe's decision boundary reachable in a handful of requests.

**An offline benchmark.** Every real benchmark reads HuggingFace. No test may
touch the network, so probe tests run against ``mock_arithmetic``: six-digit
addition, generated from a fixed seed, with an exact answer and a question the
mock provider can actually solve. Its questions also paraphrase cleanly -- the
operands are protected regions, so they survive byte-identically -- which is
what lets the evasion tests compare a verbatim arm against a paraphrased one.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from llmverify.benchmarks.base import Benchmark, BenchmarkItem, register_benchmark
from llmverify.benchmarks.datasets import DatasetLoader
from llmverify.config import BudgetConfig, ProviderConfig, RunConfig
from llmverify.reference.schema import (
    BenchmarkScore,
    FamilySignature,
    ModelRecord,
    Pricing,
    ReferenceSnapshot,
    TokenAccounting,
)
from mockserver import MockConfig, MockProvider, Persona

#: Identifiers the synthetic snapshot knows about. Deliberately unlike any real
#: model id, so a test that accidentally resolves against the bundled snapshot
#: fails loudly instead of quietly comparing against real published numbers.
CLAIMED_MODEL = "mock-model-1"
OTHER_MODEL = "mock-other-2"

#: Token accounting for each. The two are far enough apart that the +/-2 token
#: tolerance cannot confuse them, which is what makes "the overhead matches a
#: different model" a testable claim.
CLAIMED_OVERHEAD = (286, 406)
OTHER_OVERHEAD = (675, 804)

#: Published accuracy for the offline benchmark. The benchmark probe widens the
#: null by 15 points when the reference score states no reasoning effort, so the
#: effective null is 75%: comfortably above a 5%-accurate endpoint and
#: comfortably below a perfect one.
REFERENCE_SCORE_PP = 90.0

#: Small enough that the long-context ladder stops after three rungs.
CONTEXT_WINDOW = 64_000


# --------------------------------------------------------------------------- #
# Offline benchmark
# --------------------------------------------------------------------------- #


@register_benchmark
class MockArithmetic(Benchmark):
    """Six-digit addition, generated locally and graded exactly.

    Registered at import time because the probe under test resolves benchmarks
    through the global registry, exactly as a plugin would.
    """

    name: ClassVar[str] = "mock_arithmetic"
    reference_key: ClassVar[str] = "mock_arithmetic"
    hf_dataset: ClassVar[str | None] = None
    hf_config: ClassVar[str | None] = None
    hf_split: ClassVar[str] = ""
    gated: ClassVar[bool] = False
    licence: ClassVar[str] = "n/a (generated locally)"
    discriminative: ClassVar[bool] = True
    score_spread: ClassVar[float] = 25.0
    description: ClassVar[str] = (
        "Six-digit addition generated from a fixed seed. Offline, exactly gradable, and "
        "used only by this test suite."
    )
    answer_instruction: ClassVar[str] = (
        'Put the final answer on its own last line, formatted exactly as "ANSWER: <integer>".'
    )

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Generate items. ``loader`` is accepted per the contract and unused."""
        return generate_items(limit if limit is not None else 200)


def generate_items(count: int) -> list[BenchmarkItem]:
    """``count`` addition items, identical for a given index on any machine."""
    items: list[BenchmarkItem] = []
    for index in range(max(0, count)):
        rng = random.Random(f"mock-arithmetic:{index}")
        left = rng.randrange(100_000, 999_999)
        right = rng.randrange(100_000, 999_999)
        items.append(
            BenchmarkItem(
                id=f"mock-arith-{index:04d}",
                question=f"Compute {left} + {right}.",
                answer=str(left + right),
                meta={"left": left, "right": right},
            )
        )
    return items


def known_item_texts(count: int = 200) -> tuple[str, ...]:
    """The lookup table a provider routing on recognisable inputs would hold."""
    return tuple(item.question for item in generate_items(count))


# --------------------------------------------------------------------------- #
# Reference snapshot
# --------------------------------------------------------------------------- #


def _score(model_score: float) -> BenchmarkScore:
    return BenchmarkScore(
        benchmark="mock_arithmetic",
        score=model_score,
        unit="percent",
        source="synthetic fixture",
        as_of=dt.date(2026, 7, 1),
        confidence="primary",
        n_items=200,
    )


def build_snapshot() -> ReferenceSnapshot:
    """Two fictional models, one signature per protocol this suite speaks."""
    return ReferenceSnapshot(
        schema_version=1,
        as_of=dt.date(2026, 7, 20),
        generated_by="test fixture",
        sources={"fixture": "tests/conftest.py"},
        families=(
            FamilySignature(
                family="openai",
                response_id_prefix="chatcmpl-",
                supports_logprobs=True,
                supports_seed=True,
                has_system_fingerprint=True,
                usage_keys=("prompt_tokens", "completion_tokens", "completion_tokens_details"),
                finish_reasons=("stop", "length", "tool_calls", "content_filter"),
            ),
            FamilySignature(
                family="anthropic",
                response_id_prefix="msg_",
                supports_logprobs=False,
                supports_seed=False,
                has_system_fingerprint=False,
                usage_keys=("input_tokens", "output_tokens", "cache_read_input_tokens"),
                finish_reasons=("end_turn", "max_tokens", "stop_sequence", "tool_use"),
            ),
            FamilySignature(family="gemini", has_system_fingerprint=False),
        ),
        models=(
            ModelRecord(
                id=CLAIMED_MODEL,
                vendor="mockvendor",
                family="openai",
                display_name="Mock Model 1",
                aliases=("mockvendor/mock-model-1",),
                released=dt.date(2026, 5, 1),
                context_window=CONTEXT_WINDOW,
                max_output_tokens=8192,
                reasoning=True,
                modalities=("text",),
                pricing=Pricing(input_per_mtok=1.0, output_per_mtok=4.0, source="fixture"),
                token_accounting=TokenAccounting(
                    tool_overhead_auto=CLAIMED_OVERHEAD[0],
                    tool_overhead_forced=CLAIMED_OVERHEAD[1],
                    source="fixture",
                    as_of=dt.date(2026, 7, 1),
                ),
                scores=(_score(REFERENCE_SCORE_PP),),
            ),
            ModelRecord(
                id=OTHER_MODEL,
                vendor="mockvendor",
                family="openai",
                display_name="Mock Model 2",
                context_window=32_000,
                reasoning=False,
                pricing=Pricing(input_per_mtok=0.2, output_per_mtok=0.8, source="fixture"),
                token_accounting=TokenAccounting(
                    tool_overhead_auto=OTHER_OVERHEAD[0],
                    tool_overhead_forced=OTHER_OVERHEAD[1],
                    source="fixture",
                    as_of=dt.date(2026, 7, 1),
                ),
                scores=(_score(40.0),),
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def snapshot() -> ReferenceSnapshot:
    return build_snapshot()


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    path.mkdir()
    return path


@pytest.fixture
def mock_provider() -> Iterator[MockProvider]:
    """An HONEST provider on a real port, torn down at the end of the test."""
    with MockProvider(MockConfig.for_persona(Persona.HONEST, model_id=CLAIMED_MODEL)) as server:
        yield server


@pytest.fixture
def make_provider() -> Iterator[Callable[..., MockProvider]]:
    """Factory for a provider with any persona, cleaned up at the end."""
    started: list[MockProvider] = []

    def factory(persona: Persona = Persona.HONEST, **overrides: Any) -> MockProvider:
        overrides.setdefault("model_id", CLAIMED_MODEL)
        server = MockProvider(MockConfig.for_persona(persona, **overrides))
        started.append(server)
        return server

    yield factory
    for server in started:
        server.close()


def openai_config(server: MockProvider, **overrides: Any) -> ProviderConfig:
    """A provider config pointed at ``server``'s OpenAI-compatible route."""
    data: dict[str, Any] = {
        "name": "mock-openai",
        "api": "openai",
        "base_url": server.openai_url,
        "model": CLAIMED_MODEL,
        "claimed_model": CLAIMED_MODEL,
        "max_concurrency": 4,
        "max_retries": 0,
        "timeout_s": 30.0,
        "connect_timeout_s": 5.0,
    }
    data.update(overrides)
    return ProviderConfig.model_validate(data)


def anthropic_config(server: MockProvider, **overrides: Any) -> ProviderConfig:
    """A provider config pointed at ``server``'s Messages route."""
    return openai_config(
        server,
        **{"name": "mock-anthropic", "api": "anthropic", "base_url": server.anthropic_url,
           **overrides},
    )


def gemini_config(server: MockProvider, **overrides: Any) -> ProviderConfig:
    """A provider config pointed at ``server``'s generateContent route."""
    return openai_config(
        server,
        **{"name": "mock-gemini", "api": "gemini", "base_url": server.gemini_url,
           "auth_scheme": "none", **overrides},
    )


def run_config(cache: Path, **overrides: Any) -> RunConfig:
    """A run config with generous ceilings and no network-backed benchmark."""
    data: dict[str, Any] = {
        "benchmarks": ("mock_arithmetic",),
        "budget": BudgetConfig(max_cost_usd=None, max_samples=600, max_wall_s=300.0),
        "cache_dir": cache,
        "seed": 20260726,
    }
    data.update(overrides)
    return RunConfig.model_validate(data)


@pytest.fixture
def provider_config(mock_provider: MockProvider) -> ProviderConfig:
    return openai_config(mock_provider)


@pytest.fixture
def run_settings(cache_dir: Path) -> RunConfig:
    return run_config(cache_dir)
