"""Each probe against the persona it was written to catch.

These are the tests the package exists for, so they are written as end-to-end
runs: a real provider config, a real adapter, a real socket, the real runner,
and an assertion about the verdict rather than about an internal. What each one
pins down:

* an HONEST endpoint is never accused;
* a SUBSTITUTED one reaches LIKELY_MISMATCH or MISMATCH;
* an EVADING one reaches EVASION, both when the benchmark probe has already
  bought the two arms and when the evasion probe has to buy its own;
* a QUANTIZED one is caught by the Unicode-integrity echo, which involves no
  reasoning at all and so cannot be excused as the model having a bad day;
* a TRUNCATING one is caught by the long-context ladder, in both the hard-clip
  and rolling-window shapes;
* and the token-accounting probe does not merely say "not what you claimed", it
  names the model whose published overhead the endpoint actually reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import (
    ANTHROPIC_FAMILY_MODEL,
    CLAIMED_MODEL,
    CONTEXT_WINDOW,
    OTHER_MODEL,
    anthropic_config,
    known_item_texts,
    openai_config,
    run_config,
)
from llmverify.evidence import Evidence, EvidenceStatus, Verdict
from llmverify.probes import all_probes, select_probes
from llmverify.reference.schema import ReferenceSnapshot
from llmverify.results import RunResult
from llmverify.runner import provider_unreachable, verify_provider
from mockserver import MockProvider, Persona


async def check(
    server: MockProvider,
    snapshot: ReferenceSnapshot,
    cache: Path,
    *probes: str,
    provider: dict[str, Any] | None = None,
    **run_overrides: Any,
) -> RunResult:
    """Run the named probes against ``server`` and return the whole result."""
    config = openai_config(server, **(provider or {}))
    run = run_config(cache, probes=tuple(probes), **run_overrides)
    return await verify_provider(config, run, snapshot=snapshot)


def findings(result: RunResult, label: str) -> list[Evidence]:
    return [item for item in result.evidence if item.label == label]


def one(result: RunResult, label: str) -> Evidence:
    matching = findings(result, label)
    assert len(matching) == 1, f"expected one {label!r}, got {[e.label for e in result.evidence]}"
    return matching[0]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_every_probe_declares_a_layer_a_family_and_a_description() -> None:
    for name, cls in all_probes().items():
        assert cls.name == name
        assert cls.layer in (0, 1, 2, 3)
        assert cls.family
        assert cls.description.strip()
        assert cls.estimated_requests >= 0


def test_probes_run_cheapest_first(cache_dir: Path) -> None:
    chosen = select_probes(run_config(cache_dir))
    keys = [(probe.layer, probe.order, probe.name) for probe in chosen]
    assert keys == sorted(keys)
    assert chosen[0].layer == 0


def test_naming_probes_selects_exactly_those(cache_dir: Path) -> None:
    chosen = select_probes(run_config(cache_dir, probes=("metadata", "benchmark")))
    assert [probe.name for probe in chosen] == ["metadata", "benchmark"]

    excluded = select_probes(run_config(cache_dir, layers=(0,), exclude_probes=("metadata",)))
    assert "metadata" not in [probe.name for probe in excluded]


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


async def test_an_honest_endpoint_is_not_accused(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST)
    result = await check(
        server, snapshot, cache_dir, "metadata", "token_accounting", "benchmark"
    )
    assert not result.verdict.verdict.is_adverse
    assert result.verdict.verdict is Verdict.MATCH
    assert result.verdict.probability > 0.99
    assert result.reference_found is True


async def test_a_substituted_endpoint_is_a_mismatch(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.SUBSTITUTED, model_id=OTHER_MODEL)
    result = await check(
        server, snapshot, cache_dir, "metadata", "token_accounting", "benchmark"
    )
    assert result.verdict.verdict in (Verdict.LIKELY_MISMATCH, Verdict.MISMATCH)
    assert result.verdict.probability < 0.10


async def test_a_substituted_endpoint_that_still_echoes_the_right_name_is_caught(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """Echoing the claimed id is four characters of work and buys nothing."""
    server = make_provider(Persona.SUBSTITUTED, model_id=CLAIMED_MODEL)
    result = await check(
        server, snapshot, cache_dir, "metadata", "token_accounting", "benchmark"
    )
    assert one(result, "model_echo").llr > 0
    assert result.verdict.verdict in (Verdict.LIKELY_MISMATCH, Verdict.MISMATCH)


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #


async def test_token_accounting_confirms_a_matching_overhead(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST)
    result = await check(server, snapshot, cache_dir, "token_accounting")

    overhead = one(result, "tool_system_prompt")
    assert overhead.status is EvidenceStatus.OK
    assert overhead.llr > 0
    assert overhead.data["system_prompt_auto"] == server.config.tool_overhead_auto
    assert overhead.data["system_prompt_forced"] == server.config.tool_overhead_forced
    assert overhead.data["tool_definition_tokens"] == server.config.tool_definition_tokens
    assert overhead.data["models_matching_both"] == [CLAIMED_MODEL]

    gap = one(result, "forced_choice_delta")
    assert gap.llr > 0


async def test_token_accounting_names_the_model_the_overhead_belongs_to(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """The most useful sentence this tool can produce: not "wrong", but "which"."""
    server = make_provider(Persona.SUBSTITUTED)
    result = await check(server, snapshot, cache_dir, "token_accounting")

    overhead = one(result, "tool_system_prompt")
    assert overhead.llr < 0
    assert overhead.data["models_matching_both"] == [OTHER_MODEL]
    assert OTHER_MODEL in overhead.detail
    assert overhead.data["system_prompt_auto"] == 675
    assert overhead.data["system_prompt_forced"] == 804

    gap = one(result, "forced_choice_delta")
    assert gap.llr < 0
    assert OTHER_MODEL in gap.data["models_matching_gap"]


async def test_token_accounting_reports_an_injected_system_prompt_without_weighing_it(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """A proxy that wraps every request explains a residual matching nothing."""
    server = make_provider(
        Persona.HONEST, envelope_tokens=900, tool_overhead_auto=311, tool_overhead_forced=431
    )
    result = await check(server, snapshot, cache_dir, "token_accounting")

    envelope = one(result, "baseline_envelope")
    assert envelope.llr == 0.0
    assert envelope.data["envelope"]["overhead"] == 900
    assert envelope.data["envelope"]["suggests_injected_prompt"] is True

    overhead = one(result, "tool_system_prompt")
    assert overhead.llr < 0
    # Weighed lightly, because the injected prompt is the innocent explanation.
    assert abs(overhead.llr) < 2.0


async def test_token_accounting_skips_a_model_with_no_published_overhead(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST, model_id=ANTHROPIC_FAMILY_MODEL)
    result = await check(
        server,
        snapshot,
        cache_dir,
        "token_accounting",
        provider={"model": ANTHROPIC_FAMILY_MODEL, "claimed_model": ANTHROPIC_FAMILY_MODEL},
    )
    assert all(item.status is EvidenceStatus.SKIPPED for item in result.evidence)


# --------------------------------------------------------------------------- #
# Multilingual and quantization
# --------------------------------------------------------------------------- #


async def test_a_quantized_endpoint_is_caught_by_the_unicode_echo(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """Echoing a string needs no reasoning, so corruption is about the pipeline."""
    server = make_provider(Persona.QUANTIZED)
    result = await check(server, snapshot, cache_dir, "multilingual")

    integrity = one(result, "unicode_integrity")
    assert integrity.status is EvidenceStatus.OK
    assert integrity.llr < 0
    assert integrity.data["replacement_chars"] > 0
    assert {"han", "kana", "hangul"} <= set(integrity.data["damaged_segments"])
    # The scripts that survive FP4 routing are not reported as damaged.
    assert "arabic" not in integrity.data["damaged_segments"]
    assert "hebrew" not in integrity.data["damaged_segments"]


async def test_an_honest_endpoint_echoes_nine_scripts_intact(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST)
    result = await check(server, snapshot, cache_dir, "multilingual")

    integrity = one(result, "unicode_integrity")
    assert integrity.data["exact"] is True
    assert integrity.llr > 0
    assert one(result, "multilingual_accuracy").llr > 0
    # Which script an answer comes back in is recorded and never weighed.
    assert one(result, "reply_script_fidelity").llr == 0.0


# --------------------------------------------------------------------------- #
# Long context
# --------------------------------------------------------------------------- #


async def test_a_hard_clip_is_caught_by_the_long_context_ladder(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.TRUNCATING, context_limit_tokens=8_000)
    result = await check(server, snapshot, cache_dir, "long_context")

    ceiling = one(result, "effective_context_window")
    assert ceiling.status is EvidenceStatus.OK
    assert ceiling.llr < 0
    assert ceiling.data["claimed_context_window"] == CONTEXT_WINDOW
    assert ceiling.data["verified_tokens"] == 4_000
    assert ceiling.data["failed_at_tokens"] == 16_000

    truncation = one(result, "silent_truncation")
    assert truncation.llr < 0
    assert truncation.data["short_cells"]
    assert result.verdict.verdict.is_adverse or result.verdict.total_llr < 0


async def test_a_rolling_window_shows_up_as_tail_only_retrieval(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """Retrieval that works only near the end of the prompt has one explanation."""
    server = make_provider(
        Persona.TRUNCATING, context_limit_tokens=8_000, truncation_keeps="tail"
    )
    result = await check(server, snapshot, cache_dir, "long_context")

    profile = one(result, "needle_depth_profile")
    assert profile.llr < 0
    assert profile.data["tail_only_sizes"]
    assert one(result, "silent_truncation").llr < 0


async def test_an_honest_endpoint_carries_its_advertised_window(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST)
    result = await check(server, snapshot, cache_dir, "long_context")

    ceiling = one(result, "effective_context_window")
    assert ceiling.llr > 0
    assert ceiling.data["verified_tokens"] == CONTEXT_WINDOW
    assert ceiling.data["failed_at_tokens"] is None
    assert findings(result, "silent_truncation") == []


# --------------------------------------------------------------------------- #
# Evasion
# --------------------------------------------------------------------------- #


async def test_an_evading_endpoint_yields_the_evasion_verdict(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """The most valuable test in the suite.

    The endpoint answers every recognisable item correctly and almost every
    paraphrase wrongly. Its absolute accuracy is therefore whatever the mix of
    variants happens to be, and no single-arm measurement can see the routing.
    """
    server = make_provider(Persona.EVADING, known_items=known_item_texts(600))
    result = await check(server, snapshot, cache_dir, "benchmark", "evasion")

    assert result.verdict.verdict is Verdict.EVASION
    gap = one(result, "variant_accuracy_gap")
    assert gap.llr < 0
    assert gap.data["gap_pp"] > 50.0
    assert gap.data["p_adjusted"] < 0.05
    assert gap.data["source"] == "benchmark probe"
    assert any("Evasion detected" in note for note in result.verdict.notes)


async def test_the_evasion_probe_buys_its_own_paired_split_when_it_must(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """Run alone, it has no benchmark record to read and pays for one itself.

    The split is paired on purpose: the same items in both arms make McNemar
    available, which detects this size of gap on far fewer requests than two
    independent samples would.
    """
    server = make_provider(Persona.EVADING, known_items=known_item_texts(600))
    result = await check(server, snapshot, cache_dir, "evasion")

    assert result.verdict.verdict is Verdict.EVASION
    gap = one(result, "variant_accuracy_gap")
    assert gap.data["source"] == "dedicated canary split"
    assert gap.data["benchmark"] == "mock_arithmetic"
    assert "McNemar" in gap.data["test"]
    assert gap.data["verbatim_rate_pp"] > 90.0
    assert gap.data["paraphrased_rate_pp"] < 30.0


async def test_the_evasion_probe_clears_an_honest_endpoint(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST)
    result = await check(server, snapshot, cache_dir, "evasion")

    assert not result.verdict.verdict.is_adverse
    gap = one(result, "variant_accuracy_gap")
    assert gap.llr >= 0.0
    assert abs(gap.data["gap_pp"]) <= 5.0
    # Corroboration only, and capped at WEAK whatever it finds.
    assert abs(one(result, "marker_sensitivity").llr) <= 1.2


async def test_paraphrasing_can_be_turned_off_and_the_probe_says_so(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.EVADING, known_items=known_item_texts(600))
    result = await check(server, snapshot, cache_dir, "evasion", anti_evasion=False)
    assert [item.status for item in result.evidence] == [EvidenceStatus.SKIPPED]


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #


async def test_the_parameter_matrix_matches_an_honest_openai_endpoint(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST)
    result = await check(server, snapshot, cache_dir, "api_surface")

    matrix = one(result, "parameter_matrix")
    assert matrix.data["signature_consistent"] is True
    assert matrix.data["matrix"]["logprobs"]["support"] == "accepted"
    assert matrix.data["matrix"]["seed"]["support"] == "accepted"
    assert matrix.llr > 0
    assert findings(result, "silently_dropped_parameters") == []


async def test_a_litellm_front_end_shows_as_silently_dropped_parameters(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """A fact about the serving stack, not the weights, so it carries no weight."""
    server = make_provider(Persona.LITELLM)
    result = await check(server, snapshot, cache_dir, "api_surface")

    dropped = one(result, "silently_dropped_parameters")
    assert dropped.llr == 0.0
    assert {"logprobs", "response_format_json_schema"} <= set(dropped.data["ignored"])


async def test_a_claim_to_serve_a_messages_model_that_returns_logprobs(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """Anthropic's protocol has no logprobs, and a proxy cannot synthesise them."""
    server = make_provider(Persona.HONEST, model_id=ANTHROPIC_FAMILY_MODEL)
    result = await check(
        server,
        snapshot,
        cache_dir,
        "api_surface",
        provider={"model": ANTHROPIC_FAMILY_MODEL, "claimed_model": ANTHROPIC_FAMILY_MODEL},
    )
    logprobs = one(result, "anthropic_claim_returns_logprobs")
    assert logprobs.llr < -3.0
    assert one(result, "anthropic_claim_accepts_seed").llr < 0


async def test_a_first_party_messages_endpoint_rejects_what_it_does_not_define(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(
        Persona.HONEST, model_id=ANTHROPIC_FAMILY_MODEL, reject_unknown_params=True
    )
    config = anthropic_config(
        server, model=ANTHROPIC_FAMILY_MODEL, claimed_model=ANTHROPIC_FAMILY_MODEL
    )
    run = run_config(cache_dir, probes=("api_surface",))
    result = await verify_provider(config, run, snapshot=snapshot)

    matrix = one(result, "parameter_matrix")
    assert matrix.data["matrix"]["seed"]["support"] == "rejected"
    assert matrix.data["matrix"]["logprobs"]["support"] == "rejected"
    assert matrix.data["signature_consistent"] is True
    assert findings(result, "anthropic_claim_returns_logprobs") == []


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


async def test_metadata_weighs_a_wrong_model_id_against_the_snapshot(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST, model_id=OTHER_MODEL)
    result = await check(server, snapshot, cache_dir, "metadata")

    echo = one(result, "model_echo")
    assert echo.llr < 0
    assert echo.data["reported_matches_record"] == OTHER_MODEL


async def test_metadata_does_not_weigh_the_envelope_across_protocol_families(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """A compatible proxy rewrites the envelope legitimately; penalising it lies."""
    server = make_provider(Persona.HONEST, model_id=ANTHROPIC_FAMILY_MODEL)
    result = await check(
        server,
        snapshot,
        cache_dir,
        "metadata",
        provider={"model": ANTHROPIC_FAMILY_MODEL, "claimed_model": ANTHROPIC_FAMILY_MODEL},
    )
    envelope = one(result, "response_id_prefix")
    assert envelope.llr == 0.0
    assert "Not weighed" in envelope.detail


async def test_metadata_records_infrastructure_headers_without_weighing_them(
    mock_provider: MockProvider, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    result = await check(mock_provider, snapshot, cache_dir, "metadata")
    headers = one(result, "infrastructure_headers")
    assert headers.llr == 0.0
    assert one(result, "catalogue").data["requested_model_listed"] is True


# --------------------------------------------------------------------------- #
# Runner behaviour
# --------------------------------------------------------------------------- #


async def test_an_endpoint_that_never_answers_is_reported_as_unreachable(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    server = make_provider(Persona.HONEST, fail_status=500)
    result = await check(server, snapshot, cache_dir, "metadata", "self_report")

    assert provider_unreachable(result)
    assert result.verdict.verdict is Verdict.INCONCLUSIVE
    assert any("base URL" in warning for warning in result.warnings)


async def test_one_broken_probe_does_not_abort_the_run(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """A stack that violates its own schema must not take the audit down."""
    server = make_provider(Persona.BROKEN)
    result = await check(server, snapshot, cache_dir, "metadata", "token_accounting")

    assert result.timings
    assert {timing.probe for timing in result.timings} == {"metadata", "token_accounting"}
    # No usage object at all, so the token measurement cannot be taken -- which
    # is an unsupported capability, not a crash.
    accounting = one(result, "tool_system_prompt")
    assert accounting.status in (EvidenceStatus.UNSUPPORTED, EvidenceStatus.ERROR)


async def test_a_named_probe_selection_disables_early_stopping(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """A user who named the experiments wants them run, however clear it gets."""
    server = make_provider(Persona.SUBSTITUTED, model_id=OTHER_MODEL)
    result = await check(
        server, snapshot, cache_dir, "metadata", "token_accounting", "multilingual"
    )
    assert {timing.probe for timing in result.timings} == {
        "metadata",
        "token_accounting",
        "multilingual",
    }
    assert not any("stopped after layer" in note for note in result.verdict.notes)


async def test_a_sample_ceiling_truncates_rather_than_overspending(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    from llmverify.config import BudgetConfig

    server = make_provider(Persona.HONEST)
    result = await check(
        server,
        snapshot,
        cache_dir,
        "benchmark",
        budget=BudgetConfig(max_cost_usd=None, max_samples=6, max_wall_s=120.0),
    )
    assert result.samples <= 6
    statuses = {item.status for item in result.evidence}
    assert EvidenceStatus.OK not in statuses


async def test_layers_zero_and_one_against_an_honest_endpoint(
    make_provider: Any, snapshot: ReferenceSnapshot, cache_dir: Path
) -> None:
    """The whole cheap half of the pipeline, end to end, with nothing erroring."""
    server = make_provider(Persona.HONEST, latency_s=0.01)
    config = openai_config(server)
    run = run_config(cache_dir, layers=(0, 1))
    result = await verify_provider(config, run, snapshot=snapshot)

    assert not result.verdict.verdict.is_adverse
    errored = [item for item in result.evidence if item.status is EvidenceStatus.ERROR]
    assert errored == []
    assert result.samples > 0
    assert result.input_tokens > 0
    assert {timing.probe for timing in result.timings} >= {
        "metadata",
        "api_surface",
        "token_accounting",
        "tokenizer",
        "self_report",
        "determinism",
    }


@pytest.mark.skip(reason="would send requests to a first-party endpoint over the network")
async def test_against_a_real_provider() -> None:
    """Left here as a marker: nothing in this suite may reach the network."""
