"""Orchestration: run the selected probes and turn what they saw into a verdict.

The runner owns four things nothing else does.

**Setup.** It resolves the adapter, the optional baseline adapter, the reference
snapshot and the claimed model's record, and seeds the :class:`~llmverify.probes.Budget`
with pricing so that a cost ceiling means something. A model the snapshot has
never heard of does not abort the run: metadata, api-surface, tokenizer and
capability probes all still say something useful about an unknown model, and
saying "unknown, here is what we could still measure" is more honest than
refusing to look.

**Ordering and isolation.** Probes run in the order the registry declares --
cheap and decisive first -- and any probe that reads or writes
:attr:`~llmverify.probes.ProbeContext.shared` runs alone. The shared dictionary
is how the metadata probe hands its warm-up response to everyone else and how
the benchmark probe hands per-item results to the evasion probe; none of that is
synchronised, and interleaving two probes that touch it would trade a correct
verdict for a few seconds.

**Short-circuiting, in one direction only.** After each layer the evidence so
far is aggregated. If it has already refuted the claim on at least
:data:`SHORT_CIRCUIT_FAMILIES` independent evidence families, the run stops --
that is what makes a blatant fake cost cents instead of dollars.

Accumulated *positive* evidence never stops a run, because the layers do not all
answer the same question. Layers 0 and 1 establish identity; layer 2 measures
whether that identity is being served intact. A quantized deployment of the
genuine model passes every identity check honestly and still fails the quality
ones, so stopping on "it really is Opus 5" would skip the probes that notice its
CJK output is corrupted and its million-token window is a fiction.

An early stop is a partial audit either way, so every probe that never ran is
recorded as skipped with the reason and the verdict carries a note. It must
never read like a clean bill of health.

**Containment.** Every probe is wrapped: a budget ceiling truncates the run
cleanly, an unsupported capability becomes ``UNSUPPORTED`` evidence, and anything
else becomes ``ERROR`` evidence with a redacted message while the run continues.
One broken probe cannot abort an audit, and adapters are closed even when the
caller cancels.

This module emits progress as plain data through a callback. It imports nothing
from ``rich`` and writes nothing to a terminal, so the same run can drive a
console, an HTML report, a test or a CI log without the runner knowing which.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import enum
import inspect
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .adapters.base import Adapter, get_adapter
from .config import ProviderConfig, RunConfig, redact
from .errors import BudgetExhausted, UnsupportedCapability
from .evidence import Evidence, EvidenceStatus, Verdict, VerdictReport, aggregate
from .probes import Budget, Probe, ProbeContext, select_probes
from .reference import ReferenceSnapshot, find_model, load_snapshot, snapshot_age_days
from .reference.schema import ModelRecord
from .results import ProbeTiming, RunResult

__all__ = [
    "GLOBAL_GRACE_S",
    "SHORT_CIRCUIT_FAMILIES",
    "STALE_SNAPSHOT_DAYS",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressState",
    "provider_unreachable",
    "verify_many",
    "verify_provider",
]

#: Age at which the snapshot's published numbers are called out as stale. Labs
#: ship models faster than this, and a threshold measured against a model that
#: has since been superseded is the quiet way to get a verdict wrong.
STALE_SNAPSHOT_DAYS: int = 30

#: Distinct evidence families that must have contributed before the run is
#: allowed to stop early. The aggregator already refuses to leave INCONCLUSIVE
#: on fewer than three contributing probes; requiring three *families* is
#: stricter, because five tokenizer measurements are one opinion, not five.
SHORT_CIRCUIT_FAMILIES: int = 3

#: Slack between the per-probe deadline and the whole-run deadline. Per-probe
#: timeouts should fire first, since those degrade into TRUNCATED evidence,
#: whereas the global one is a backstop for something stuck outside a probe.
GLOBAL_GRACE_S: float = 5.0

#: Probes that read benchmark datasets. When one of these is selected the runner
#: builds a single loader so that the connection pool and the on-disk cache are
#: shared; any other probe wanting one creates and closes its own, per the
#: contract in :func:`llmverify.probes.benchmark.dataset_loader`.
_DATASET_PROBES = frozenset({"benchmark", "evasion"})


class ProgressState(str, enum.Enum):
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One thing that happened, described without deciding how to show it."""

    probe: str
    layer: int
    state: ProgressState
    elapsed_s: float = 0.0
    message: str = ""
    #: Which provider this event belongs to, so that a single callback can
    #: follow :func:`verify_many` across several endpoints.
    provider: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


async def verify_provider(
    provider: ProviderConfig,
    run: RunConfig,
    *,
    snapshot: ReferenceSnapshot | None = None,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Verify one endpoint and return the full result.

    Raises only for problems that make a run meaningless before it starts: an
    unknown adapter, or an API key environment variable that is not set. Once
    the first probe has run, every failure is recorded as evidence instead, so
    the caller always gets a result it can show.
    """
    started_at = dt.datetime.now(dt.timezone.utc)
    warnings: list[str] = []

    snapshot = snapshot if snapshot is not None else _load_snapshot(warnings)
    reference = find_model(snapshot, provider.target_model) if snapshot else None
    _check_reference(provider, snapshot, reference, warnings)

    # Selection first: an unknown probe name is a usage error, and raising it
    # before an adapter exists means there is no socket to leak.
    probes = select_probes(run)
    adapter = get_adapter(provider)
    ctx = ProbeContext(
        adapter=adapter,
        provider=provider,
        run=run,
        budget=_budget_for(run, provider, reference),
        snapshot=snapshot,
        reference=reference,
    )

    baseline: Adapter | None = None
    loader: Any | None = None
    collector = _Collector()
    emit = _emitter(progress, provider.name)
    try:
        if run.baseline is not None:
            baseline = get_adapter(run.baseline)
            ctx.baseline = baseline
            ctx.baseline_provider = run.baseline

        loader = _dataset_loader(run, probes)
        if loader is not None:
            ctx.shared["dataset_loader"] = loader

        await asyncio.wait_for(
            _run_probes(collector, ctx, probes, run, emit),
            _global_timeout(run),
        )
    except asyncio.TimeoutError:
        collector.stop_reason = (
            f"the whole-run wall-clock ceiling of {run.budget.max_wall_s:.0f}s elapsed "
            "while no single probe had exceeded its own deadline; the run was cut off "
            "with the evidence gathered up to that point."
        )
    finally:
        await _close_quietly(adapter, baseline, loader)

    return _assemble(
        provider=provider,
        run=run,
        adapter=adapter,
        snapshot=snapshot,
        reference=reference,
        collector=collector,
        ctx=ctx,
        warnings=warnings,
        started_at=started_at,
        baseline_used=baseline is not None,
    )


async def verify_many(
    providers: Sequence[ProviderConfig],
    run: RunConfig,
    *,
    snapshot: ReferenceSnapshot | None = None,
    progress: ProgressCallback | None = None,
    concurrency: int = 1,
) -> list[RunResult]:
    """Verify several endpoints and return one result per provider, in order.

    Sequential by default, and that default is not laziness. The performance
    probe measures time-to-first-token and output rate; running two providers at
    once makes them contend for one event loop and one uplink, which shows up as
    the endpoints looking slower than they are. Raise ``concurrency`` only when
    no timing measurement in the run is being trusted.

    A provider that fails outright still yields a result carrying the error, so
    that one unreachable endpoint does not erase the comparison table.
    """
    if not providers:
        return []

    snapshot = snapshot if snapshot is not None else _load_snapshot([])
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(provider: ProviderConfig) -> RunResult:
        async with semaphore:
            try:
                return await verify_provider(
                    provider, run, snapshot=snapshot, progress=progress
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return _failed_result(provider, run, snapshot, exc)

    if concurrency <= 1:
        return [await one(provider) for provider in providers]
    return list(await asyncio.gather(*(one(provider) for provider in providers)))


def provider_unreachable(result: RunResult) -> bool:
    """Whether the run never got a usable answer out of the endpoint.

    Three questions, in order of how directly they settle it: did any request
    come back with a usage object, did any probe extract a finding that moved
    the posterior, and did anything fail at all. Only when the first two are no
    and the third is yes has nothing been learned -- the shape of a wrong base
    URL, a rejected key or a dead host.

    Deliberately not "some probe errored": an endpoint that refuses one
    capability and serves everything else is ordinary. Deliberately not "no
    evidence with status OK" either, because several probes emit a well-formed
    zero-weight finding after every one of their requests failed.
    """
    if result.input_tokens or result.output_tokens:
        return False
    if any(e.status is EvidenceStatus.OK and abs(e.llr) > 1e-9 for e in result.evidence):
        return False
    return any(e.status is EvidenceStatus.ERROR for e in result.evidence)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Slot:
    """One probe's output, reserved before the probe starts.

    Reserving the slot up front is what makes a whole-run timeout lossless: the
    evidence a cancelled batch had already produced is reachable from outside
    the coroutine that was cancelled.
    """

    probe: str
    layer: int
    evidence: list[Evidence] = field(default_factory=list)
    timing: ProbeTiming | None = None


@dataclass(slots=True)
class _Collector:
    slots: list[_Slot] = field(default_factory=list)
    stop_reason: str = ""

    def reserve(self, probe: Probe) -> _Slot:
        slot = _Slot(probe=probe.name, layer=probe.layer)
        self.slots.append(slot)
        return slot

    @property
    def evidence(self) -> list[Evidence]:
        return [e for slot in self.slots for e in slot.evidence]

    @property
    def timings(self) -> list[ProbeTiming]:
        return [slot.timing for slot in self.slots if slot.timing is not None]


def _emitter(progress: ProgressCallback | None, provider_name: str) -> Callable[..., None]:
    """Wrap the caller's callback so a broken reporter cannot fail the run."""

    def emit(
        probe: Probe,
        layer: int,
        state: ProgressState,
        *,
        elapsed_s: float = 0.0,
        message: str = "",
    ) -> None:
        if progress is None:
            return
        event = ProgressEvent(
            probe=probe.name,
            layer=layer,
            state=state,
            elapsed_s=elapsed_s,
            message=message,
            provider=provider_name,
        )
        with contextlib.suppress(Exception):
            progress(event)

    return emit


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


async def _run_probes(
    collector: _Collector,
    ctx: ProbeContext,
    probes: Sequence[Probe],
    run: RunConfig,
    emit: Callable[..., None],
) -> None:
    """Run every selected probe, layer by layer, stopping when it is settled."""
    remaining = list(probes)
    while remaining:
        layer = remaining[0].layer
        group = [p for p in remaining if p.layer == layer]
        remaining = remaining[len(group) :]

        stopped = await _run_layer(collector, ctx, group, emit)
        if stopped:
            collector.stop_reason = stopped
            break

        reason = _short_circuit_reason(collector, run, layer, len(remaining))
        if reason:
            collector.stop_reason = reason
            break

    for probe in remaining:
        slot = collector.reserve(probe)
        reason = collector.stop_reason or "the run stopped before this layer was reached."
        slot.evidence.append(
            Evidence(
                probe=probe.name,
                label="not_run",
                llr=0.0,
                family=probe.family,
                status=EvidenceStatus.SKIPPED,
                detail=f"not run: {reason}",
            )
        )
        slot.timing = ProbeTiming(probe.name, probe.layer, 0.0, 0, "skipped", reason)
        emit(probe, probe.layer, ProgressState.SKIPPED, message=reason)


async def _run_layer(
    collector: _Collector,
    ctx: ProbeContext,
    probes: Sequence[Probe],
    emit: Callable[..., None],
) -> str:
    """Run one layer, returning why the run must stop, or an empty string.

    Probes that touch shared state run one at a time, in declared order. Runs of
    consecutive probes that do not are executed together, bounded by the
    provider's own concurrency limit so that a fan-out does not turn into a wall
    of 429s that later reads as a capability failure.
    """
    index = 0
    while index < len(probes):
        if _touches_shared(probes[index]):
            probe = probes[index]
            index += 1
            stop = await _run_one(probe, ctx, collector.reserve(probe), emit)
        else:
            batch: list[tuple[Probe, _Slot]] = []
            while index < len(probes) and not _touches_shared(probes[index]):
                probe = probes[index]
                batch.append((probe, collector.reserve(probe)))
                index += 1
            stop = await _run_batch(batch, ctx, emit)

        if stop:
            return stop
    return ""


async def _run_batch(
    batch: Sequence[tuple[Probe, _Slot]],
    ctx: ProbeContext,
    emit: Callable[..., None],
) -> str:
    """Run independent probes concurrently, reporting the first reason to stop."""
    if len(batch) == 1:
        probe, slot = batch[0]
        return await _run_one(probe, ctx, slot, emit)

    limit = asyncio.Semaphore(max(1, ctx.provider.max_concurrency))

    async def guarded(probe: Probe, slot: _Slot) -> str:
        async with limit:
            return await _run_one(probe, ctx, slot, emit)

    outcomes = await asyncio.gather(*(guarded(p, s) for p, s in batch))
    return next((reason for reason in outcomes if reason), "")


async def _run_one(
    probe: Probe,
    ctx: ProbeContext,
    slot: _Slot,
    emit: Callable[..., None],
) -> str:
    """Run a single probe into its slot, returning why the run must stop or "".

    Nothing a probe does propagates out of here. The classification matters more
    than the containment: a budget ceiling is a clean stop, a refused capability
    is a normal finding, and only an unexpected failure is an error -- conflating
    them would let "this endpoint has no logprobs" read as a bug in the tool.
    """
    emit(probe, probe.layer, ProgressState.STARTED)

    try:
        applicable = probe.applicable(ctx)
    except Exception as exc:
        slot.evidence.append(_error_evidence(probe, exc, "deciding whether it applies"))
        slot.timing = ProbeTiming(probe.name, probe.layer, 0.0, 0, "error", _short(exc))
        emit(probe, probe.layer, ProgressState.FAILED, message=_short(exc))
        return ""

    if not applicable:
        reason = "this probe has nothing to measure on this endpoint or claimed model."
        slot.evidence.append(ctx.skipped(probe.name, reason, family=probe.family))
        slot.timing = ProbeTiming(probe.name, probe.layer, 0.0, 0, "skipped", None)
        emit(probe, probe.layer, ProgressState.SKIPPED, message=reason)
        return ""

    deadline = _remaining_wall(ctx.budget)
    try:
        ctx.budget.check()
        if deadline is not None and deadline <= 0:
            raise BudgetExhausted("the wall-clock ceiling was already reached")
    except BudgetExhausted as exc:
        slot.evidence.append(_truncated_evidence(probe, str(exc)))
        slot.timing = ProbeTiming(probe.name, probe.layer, 0.0, 0, "truncated", _short(exc))
        emit(probe, probe.layer, ProgressState.SKIPPED, message=str(exc))
        return f"the run stopped before {probe.name}: {redact(str(exc))}"

    started = time.perf_counter()
    samples_before = ctx.budget.samples
    stop = ""

    try:
        results = await asyncio.wait_for(probe.run(ctx), deadline)
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        detail = (
            f"the probe was still running after {elapsed:.1f}s, which is the whole "
            "remaining wall-clock allowance for this run."
        )
        results = [_truncated_evidence(probe, detail)]
        stop = f"the run stopped in {probe.name}: {detail}"
    except BudgetExhausted as exc:
        results = [_truncated_evidence(probe, str(exc))]
        stop = f"the run stopped in {probe.name}: {redact(str(exc))}"
    except UnsupportedCapability as exc:
        results = [ctx.unsupported(probe.name, redact(str(exc))[:300], family=probe.family)]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        results = [_error_evidence(probe, exc, "running")]

    elapsed = time.perf_counter() - started
    slot.evidence.extend(results)
    slot.timing = ProbeTiming(
        probe=probe.name,
        layer=probe.layer,
        duration_s=elapsed,
        requests=max(0, ctx.budget.samples - samples_before),
        status=_timing_status(results),
        error=next(
            (redact(e.detail)[:200] for e in results if e.status is EvidenceStatus.ERROR),
            None,
        ),
    )

    state = ProgressState.FAILED if slot.timing.status == "error" else ProgressState.FINISHED
    emit(probe, probe.layer, state, elapsed_s=elapsed, message=_summarise(results))
    return stop


def _summarise(results: Sequence[Evidence]) -> str:
    usable = [e for e in results if e.status is EvidenceStatus.OK]
    if not usable:
        statuses = sorted({e.status.value for e in results}) or ["nothing"]
        return "no usable evidence: " + ", ".join(statuses)
    bans = sum(e.bans for e in usable)
    return f"{len(usable)} finding(s), {bans:+.2f} bans"


def _timing_status(results: Sequence[Evidence]) -> str:
    """One word for how a probe went, worst outcome first.

    An error is reported ahead of a truncation and both ahead of success,
    because a run summary that hides the one probe that broke is worse than no
    summary at all.
    """
    statuses = {e.status for e in results}
    for status in (
        EvidenceStatus.ERROR,
        EvidenceStatus.TRUNCATED,
        EvidenceStatus.OK,
        EvidenceStatus.UNSUPPORTED,
        EvidenceStatus.SKIPPED,
    ):
        if status in statuses:
            return status.value
    return "empty"


def _error_evidence(probe: Probe, exc: BaseException, phase: str) -> Evidence:
    return Evidence(
        probe=probe.name,
        label="probe_error",
        llr=0.0,
        family=probe.family,
        status=EvidenceStatus.ERROR,
        detail=f"the probe failed while {phase}: {redact(str(exc))[:300]}",
        data={"error_type": type(exc).__name__},
    )


def _truncated_evidence(probe: Probe, detail: str) -> Evidence:
    return Evidence(
        probe=probe.name,
        label="budget",
        llr=0.0,
        family=probe.family,
        status=EvidenceStatus.TRUNCATED,
        detail=redact(detail)[:300],
    )


def _short(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}")[:200]


# --------------------------------------------------------------------------- #
# Shared-state isolation
# --------------------------------------------------------------------------- #

_SHARED_CACHE: dict[type[Probe], bool] = {}


def _touches_shared(probe: Probe) -> bool:
    """Whether a probe communicates through ``ProbeContext.shared``.

    Probes hand each other results through that dictionary -- the metadata
    probe's warm-up response, the benchmark probe's per-item grades -- and the
    handover is ordered rather than synchronised. Running two such probes
    together would be a race for no gain, since the ordering is already
    cheap-first.

    There is no flag on :class:`~llmverify.probes.Probe` declaring this, so the
    probe's whole defining module is inspected: the access is often in a
    module-level helper rather than in the class body. Anything that cannot be
    inspected -- a plugin shipped without source, a class built at runtime -- is
    treated as touching shared state, because being wrong in that direction
    costs a little time and being wrong in the other costs a correct verdict.
    """
    cls = type(probe)
    cached = _SHARED_CACHE.get(cls)
    if cached is not None:
        return cached

    touches = True
    module = sys.modules.get(cls.__module__)
    if module is not None:
        try:
            touches = ".shared" in inspect.getsource(module)
        except (OSError, TypeError):
            touches = True
    _SHARED_CACHE[cls] = touches
    return touches


# --------------------------------------------------------------------------- #
# Short-circuiting
# --------------------------------------------------------------------------- #


def _short_circuit_reason(
    collector: _Collector, run: RunConfig, layer: int, remaining: int
) -> str:
    """Why the run may stop after this layer, or an empty string to continue.

    **Only an adverse verdict may stop a run.** Once substitution is
    established, further probes buy nothing: the answer will not change and the
    remaining budget would be spent confirming it. The reverse is not
    symmetrical, and treating it as though it were is the mistake this function
    exists to avoid.

    Layers 0 and 1 establish *identity* -- what model this is. Layer 2 measures
    *quality* -- whether that model is being served intact. Those are different
    questions, and a quantized deployment of the genuine model answers the first
    one truthfully: same tokenizer, same token accounting, valid thinking-block
    signatures, correct model id. Stopping there on accumulated positive
    evidence would skip precisely the probes that detect a degraded serving
    configuration, and report MATCH for an endpoint whose CJK output is
    corrupted and whose advertised context window is a fiction. A user who asked
    for layer 2 asked that question and must get an answer to it.

    Two further guards apply to the adverse case. The verdict must be one of the
    outer bands, not merely leaning; and at least :data:`SHORT_CIRCUIT_FAMILIES`
    families must have contributed, so that an accusation is never settled by
    several probes all measuring the same property. An explicit ``--probe``
    selection disables stopping entirely: a user who named the experiments wants
    them run.
    """
    if not remaining or run.probes:
        return ""

    report = aggregate(collector.evidence, prior_odds=run.prior_odds)
    if report.verdict not in (Verdict.MISMATCH, Verdict.EVASION):
        return ""

    families = sorted(f for f, total in report.family_totals.items() if abs(total) > 1e-9)
    if len(families) < SHORT_CIRCUIT_FAMILIES:
        return ""

    return (
        f"stopped after layer {layer}: the evidence from {len(families)} families "
        f"({', '.join(families)}) already refutes the claim at "
        f"p={report.probability:.4f}, past the {report.verdict.value} threshold. "
        f"{remaining} further probe(s) were not run, so this is an early exit rather "
        "than a full audit."
    )


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #


def _load_snapshot(warnings: list[str]) -> ReferenceSnapshot | None:
    """Load the bundled snapshot, degrading to ``None`` rather than failing.

    A snapshot that will not parse is a broken install, not a broken endpoint.
    The probes that need reference data skip themselves; the ones that do not --
    metadata, api surface, tokenizer, self-report -- still have something to say,
    and saying it beats refusing to run at all.
    """
    try:
        return load_snapshot()
    except Exception as exc:
        warnings.append(
            f"the reference snapshot could not be loaded ({redact(str(exc))[:200]}); "
            "every probe that compares against published data will skip itself."
        )
        return None


def _check_reference(
    provider: ProviderConfig,
    snapshot: ReferenceSnapshot | None,
    reference: ModelRecord | None,
    warnings: list[str],
) -> None:
    """Record what the snapshot does and does not know about this claim."""
    if snapshot is None:
        return

    age = snapshot_age_days(snapshot)
    if age > STALE_SNAPSHOT_DAYS:
        warnings.append(
            f"the reference snapshot is {age} days old (as of {snapshot.as_of}). "
            "Published scores, pricing and context windows move faster than that; "
            "run `llmverify refresh` and review the diff before trusting a close call."
        )
    elif age < 0:
        warnings.append(
            f"the reference snapshot is dated {snapshot.as_of}, which is in the future. "
            "That means a hand edit or a skewed clock rather than fresh data."
        )

    if reference is None:
        warnings.append(
            f"{provider.target_model!r} is not in the reference snapshot, so there are no "
            "published scores, pricing or token-overhead figures to compare against. "
            "Metadata, API surface, tokenizer and capability evidence is still collected; "
            "benchmark, long-context and token-accounting probes will skip themselves. "
            "Set claimed_model to the canonical id if the provider uses its own naming."
        )


def _budget_for(
    run: RunConfig, provider: ProviderConfig, reference: ModelRecord | None
) -> Budget:
    """Seed the budget with whichever price is higher, official or advertised.

    Cost is estimated rather than reported, so the estimate has to pick a price.
    Taking the larger of the claimed model's first-party price and the price this
    provider advertises keeps the ceiling protective in both directions: a
    reseller advertising a tenth of the official rate does not get ten times the
    samples, and one charging above it does not overshoot the user's real bill.
    :class:`~llmverify.config.BudgetConfig` says the runner errs on the side of
    stopping early, and this is where that happens.
    """
    pricing = reference.pricing if reference is not None else None
    return Budget(
        max_cost_usd=run.budget.max_cost_usd,
        max_samples=run.budget.max_samples,
        max_wall_s=run.budget.max_wall_s,
        price_in_per_mtok=_higher(
            pricing.input_per_mtok if pricing else None, provider.price_in_per_mtok
        ),
        price_out_per_mtok=_higher(
            pricing.output_per_mtok if pricing else None, provider.price_out_per_mtok
        ),
    )


def _higher(left: float | None, right: float | None) -> float | None:
    known = [v for v in (left, right) if v is not None]
    return max(known) if known else None


def _dataset_loader(run: RunConfig, probes: Sequence[Probe]) -> Any | None:
    """One dataset loader per run, when a probe that reads datasets is selected."""
    if not any(probe.name in _DATASET_PROBES for probe in probes):
        return None

    from .benchmarks.datasets import DatasetLoader

    return DatasetLoader(
        run.cache_dir,
        hf_token=os.environ.get(run.hf_token_env),
        token_env=run.hf_token_env,
    )


def _global_timeout(run: RunConfig) -> float | None:
    if run.budget.max_wall_s is None:
        return None
    return run.budget.max_wall_s + GLOBAL_GRACE_S


def _remaining_wall(budget: Budget) -> float | None:
    if budget.max_wall_s is None:
        return None
    return budget.max_wall_s - budget.elapsed_s


async def _close_quietly(*closeables: Any) -> None:
    """Release every adapter and loader, including when the caller cancelled us.

    Each close is shielded so that a cancellation delivered mid-cleanup does not
    abandon the remaining sockets. Swallowing the resulting ``CancelledError``
    here is correct: this runs inside a ``finally``, so the original
    cancellation is still on its way out.
    """
    for closeable in closeables:
        if closeable is None:
            continue
        try:
            await asyncio.shield(asyncio.ensure_future(closeable.aclose()))
        except asyncio.CancelledError:
            continue
        except Exception:
            continue


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #


def _assemble(
    *,
    provider: ProviderConfig,
    run: RunConfig,
    adapter: Adapter,
    snapshot: ReferenceSnapshot | None,
    reference: ModelRecord | None,
    collector: _Collector,
    ctx: ProbeContext,
    warnings: list[str],
    started_at: dt.datetime,
    baseline_used: bool,
) -> RunResult:
    budget = ctx.budget
    evidence = collector.evidence
    report = aggregate(evidence, prior_odds=run.prior_odds)
    if collector.stop_reason:
        report.notes.append(collector.stop_reason)

    all_warnings = list(warnings)
    for message in ctx.shared.get("warnings", []) or []:
        if isinstance(message, str) and message not in all_warnings:
            all_warnings.append(message)
    if collector.stop_reason:
        all_warnings.append(collector.stop_reason)

    result = RunResult(
        provider_name=provider.name,
        requested_model=provider.model,
        claimed_model=provider.target_model,
        api_family=adapter.family.value,
        base_url=provider.base_url or adapter.default_base_url,
        verdict=report,
        started_at=started_at,
        finished_at=dt.datetime.now(dt.timezone.utc),
        reference_as_of=snapshot.as_of if snapshot is not None else None,
        reference_found=reference is not None,
        timings=collector.timings,
        spent_usd=budget.spent_usd,
        input_tokens=budget.input_tokens,
        output_tokens=budget.output_tokens,
        samples=budget.samples,
        seed=run.seed,
        layers=tuple(run.layers),
        baseline_used=baseline_used,
        warnings=all_warnings,
    )

    if provider_unreachable(result):
        result.warnings.insert(
            0,
            "no probe got a usable response out of this endpoint. Check the base URL, "
            "the model id and the API key before reading anything into the verdict.",
        )
    return result


def _failed_result(
    provider: ProviderConfig,
    run: RunConfig,
    snapshot: ReferenceSnapshot | None,
    exc: BaseException,
) -> RunResult:
    """A result standing in for a run that could not start.

    Used only by :func:`verify_many`, so that one misconfigured provider leaves a
    labelled row in the comparison rather than taking the other providers with it.
    """
    now = dt.datetime.now(dt.timezone.utc)
    message = redact(f"{type(exc).__name__}: {exc}")[:300]
    failure = Evidence(
        probe="runner",
        label="run_failed",
        llr=0.0,
        family="misc",
        status=EvidenceStatus.ERROR,
        detail=f"the run could not start: {message}",
        data={"error_type": type(exc).__name__},
    )
    return RunResult(
        provider_name=provider.name,
        requested_model=provider.model,
        claimed_model=provider.target_model,
        api_family=provider.api,
        base_url=provider.base_url,
        verdict=VerdictReport(
            verdict=Verdict.INCONCLUSIVE,
            probability=0.5,
            total_llr=0.0,
            prior_odds=run.prior_odds,
            evidence=[failure],
            family_totals={},
            notes=["the run never started, so nothing was measured."],
        ),
        started_at=now,
        finished_at=now,
        reference_as_of=snapshot.as_of if snapshot is not None else None,
        seed=run.seed,
        layers=tuple(run.layers),
        warnings=[f"the run could not start: {message}"],
    )
