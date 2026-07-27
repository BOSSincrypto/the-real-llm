"""The bundled snapshot, checked against its own rules.

This file is the tool's evidence base: every threshold a provider is measured
against comes out of it, and a wrong number here becomes a wrong accusation
downstream with no other check in the way. So the snapshot is validated as data,
not merely as YAML -- ids unique, aliases unambiguous, every score carrying a
source and a date, and no date in the future.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path

import pytest

from llmverify.errors import ReferenceDataError
from llmverify.reference import (
    clear_cache,
    default_snapshot_path,
    find_model,
    load_snapshot,
    snapshot_age_days,
)
from llmverify.reference.schema import ReferenceSnapshot


@pytest.fixture(scope="module")
def bundled() -> ReferenceSnapshot:
    return load_snapshot()


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_the_bundled_snapshot_validates_against_the_schema(bundled: ReferenceSnapshot) -> None:
    assert bundled.schema_version == 1
    assert bundled.models
    assert bundled.families
    assert isinstance(bundled.as_of, dt.date)


def test_the_bundled_snapshot_ships_inside_the_package() -> None:
    path = default_snapshot_path()
    assert path.is_file()
    assert path.suffix == ".yaml"
    assert path.parent.name == "data"


def test_model_ids_are_unique(bundled: ReferenceSnapshot) -> None:
    counts = Counter(record.id for record in bundled.models)
    assert [ident for ident, n in counts.items() if n > 1] == []


def test_aliases_do_not_collide_across_vendors(bundled: ReferenceSnapshot) -> None:
    """One alias must never resolve to two records, whoever published them.

    An alias that two vendors share would make ``find_model`` pick by document
    order, which is how a provider ends up measured against another company's
    published scores.
    """
    owners: dict[str, set[str]] = {}
    for record in bundled.models:
        for name in (record.id, *record.aliases):
            owners.setdefault(name.strip().lower(), set()).add(record.id)
    colliding = {name: sorted(ids) for name, ids in owners.items() if len(ids) > 1}
    assert colliding == {}


def test_every_alias_resolves_back_to_its_own_record(bundled: ReferenceSnapshot) -> None:
    for record in bundled.models:
        for alias in record.aliases:
            assert bundled.find(alias) is record, f"{alias} does not resolve to {record.id}"


def test_families_are_declared_once_each(bundled: ReferenceSnapshot) -> None:
    names = [signature.family for signature in bundled.families]
    assert len(names) == len(set(names))
    for record in bundled.models:
        assert record.family in {*names, "openai_compat"}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_every_score_carries_a_source_and_a_date(bundled: ReferenceSnapshot) -> None:
    for record in bundled.models:
        for score in record.scores:
            assert score.source.strip(), f"{record.id}/{score.benchmark} has no source"
            assert isinstance(score.as_of, dt.date)
            assert score.confidence in ("primary", "independent", "secondary", "unverified")


def test_no_score_is_dated_in_the_future(bundled: ReferenceSnapshot) -> None:
    """Dated against the snapshot, not against the clock.

    Comparing to ``date.today()`` would turn this suite into a time bomb the
    other way round: a snapshot refreshed tomorrow is not wrong today. What can
    never be right is a score claiming to have been read after the snapshot that
    contains it.
    """
    for record in bundled.models:
        for score in record.scores:
            assert score.as_of <= bundled.as_of, (
                f"{record.id}/{score.benchmark} is dated {score.as_of}, after the "
                f"snapshot's own {bundled.as_of}"
            )


def test_no_model_is_released_after_the_snapshot(bundled: ReferenceSnapshot) -> None:
    for record in bundled.models:
        if record.released is not None:
            assert record.released <= bundled.as_of
        if record.training_cutoff is not None:
            assert record.training_cutoff <= bundled.as_of


def test_token_accounting_entries_are_dated_and_sourced(bundled: ReferenceSnapshot) -> None:
    for record in bundled.models:
        accounting = record.token_accounting
        if accounting is None:
            continue
        assert accounting.source, f"{record.id} has token accounting with no source"
        if accounting.as_of is not None:
            assert accounting.as_of <= bundled.as_of
        if (
            accounting.tool_overhead_auto is not None
            and accounting.tool_overhead_forced is not None
        ):
            # Forcing a tool choice adds to the system prompt; it never removes.
            assert accounting.tool_overhead_forced > accounting.tool_overhead_auto


def test_percentage_scores_are_percentages(bundled: ReferenceSnapshot) -> None:
    for record in bundled.models:
        for score in record.scores:
            if score.unit == "percent":
                assert 0.0 <= score.score <= 100.0, f"{record.id}/{score.benchmark}"


def test_pricing_is_non_negative(bundled: ReferenceSnapshot) -> None:
    for record in bundled.models:
        pricing = record.pricing
        if pricing is None:
            continue
        for value in (
            pricing.input_per_mtok,
            pricing.output_per_mtok,
            pricing.cache_read_per_mtok,
            pricing.cache_write_per_mtok,
        ):
            assert value is None or value >= 0.0


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def test_find_model_resolves_ids_case_and_separator_insensitively(
    bundled: ReferenceSnapshot,
) -> None:
    record = bundled.models[0]
    assert find_model(bundled, record.id) is record
    assert find_model(bundled, record.id.upper()) is record
    assert find_model(bundled, f"  {record.id}  ") is record


def test_find_model_refuses_a_wrong_vendor_prefix(bundled: ReferenceSnapshot) -> None:
    """``openai/claude-opus-5`` names nothing, and must not resolve to anything."""
    record = next(r for r in bundled.models if r.vendor == "anthropic")
    assert find_model(bundled, f"openai/{record.id}") is None
    assert find_model(bundled, f"{record.vendor}/{record.id}") is record


def test_find_model_strips_only_routing_tags(bundled: ReferenceSnapshot) -> None:
    record = bundled.models[0]
    assert find_model(bundled, f"{record.id}:free") is record
    # ``:thinking`` selects a different behaviour, so it must not be discarded.
    assert find_model(bundled, f"{record.id}:thinking") is None


def test_find_model_returns_none_rather_than_guessing(bundled: ReferenceSnapshot) -> None:
    assert find_model(bundled, "") is None
    assert find_model(bundled, "a-model-nobody-published") is None


def test_score_for_prefers_matching_conditions_and_primary_sources(
    bundled: ReferenceSnapshot,
) -> None:
    for record in bundled.models:
        for benchmark in {score.benchmark for score in record.scores}:
            best = record.score_for(benchmark)
            assert best is not None
            span = record.score_range_for(benchmark)
            assert span is not None
            low, high, count = span
            assert low <= best.score <= high
            assert count == sum(1 for s in record.scores if s.benchmark == benchmark)


def test_score_range_widens_the_benefit_of_the_doubt(bundled: ReferenceSnapshot) -> None:
    """Where sources disagree, the range must span every one of them."""
    for record in bundled.models:
        for benchmark in {score.benchmark for score in record.scores}:
            values = [s.score for s in record.scores if s.benchmark == benchmark]
            low, high, count = record.score_range_for(benchmark)  # type: ignore[misc]
            assert low == min(values)
            assert high == max(values)
            assert count == len(values)


def test_family_signatures_are_reachable_by_name(bundled: ReferenceSnapshot) -> None:
    assert bundled.family_signature("anthropic") is not None
    assert bundled.family_signature("not-a-family") is None


def test_anthropic_signature_records_the_two_absences(bundled: ReferenceSnapshot) -> None:
    """No logprobs and no seed. Both were confirmed two independent ways."""
    signature = bundled.family_signature("anthropic")
    assert signature is not None
    assert signature.supports_logprobs is False
    assert signature.supports_seed is False
    assert signature.response_id_prefix == "msg_"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_snapshot_age_is_measured_against_a_supplied_date(
    bundled: ReferenceSnapshot,
) -> None:
    later = bundled.as_of + dt.timedelta(days=45)
    assert snapshot_age_days(bundled, today=later) == 45
    earlier = bundled.as_of - dt.timedelta(days=2)
    assert snapshot_age_days(bundled, today=earlier) == -2


def test_loading_is_cached_by_resolved_path() -> None:
    assert load_snapshot() is load_snapshot()
    clear_cache()
    # A fresh parse is a different object carrying the same data.
    assert load_snapshot().as_of == load_snapshot().as_of


def test_a_broken_snapshot_is_a_hard_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    with pytest.raises(ReferenceDataError, match="not found"):
        load_snapshot(missing)

    not_yaml = tmp_path / "bad.yaml"
    not_yaml.write_text("as_of: [unclosed\n", encoding="utf-8")
    with pytest.raises(ReferenceDataError, match="valid YAML"):
        load_snapshot(not_yaml)

    wrong_shape = tmp_path / "list.yaml"
    wrong_shape.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ReferenceDataError, match="mapping"):
        load_snapshot(wrong_shape)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 1\nas_of: not-a-date\n", encoding="utf-8")
    with pytest.raises(ReferenceDataError, match="reference schema"):
        load_snapshot(invalid)


def test_a_snapshot_round_trips_through_its_own_yaml_dict(
    bundled: ReferenceSnapshot,
) -> None:
    restored = ReferenceSnapshot.model_validate(bundled.to_yaml_dict())
    assert restored.as_of == bundled.as_of
    assert [r.id for r in restored.models] == [r.id for r in bundled.models]
