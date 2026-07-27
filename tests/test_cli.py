"""The command line, including the exit code each persona earns.

Every command here runs offline. The listing commands read the bundled snapshot
and the in-process registries; ``check`` talks to the mock provider over a
loopback socket. The one command that must reach the network -- ``refresh`` --
is skipped rather than exercised.

The exit codes are the part worth pinning: a CI job gates on them, so 0, 1, 2, 3
and 4 have to mean what the help text says they mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmverify.cli import (
    EXIT_INCONCLUSIVE,
    EXIT_SUBSTITUTION,
    EXIT_UNREACHABLE,
    EXIT_USAGE,
    EXIT_VERIFIED,
    main,
)
from mockserver import MockProvider, Persona

#: A model in the bundled snapshot whose published tool-use overhead the mock
#: reproduces exactly by default: 286 tokens with tool_choice auto, 406 forced.
BUNDLED_MODEL = "claude-opus-5"

#: And the model whose published overhead the SUBSTITUTED persona reports
#: instead: 675 and 804.
BUNDLED_IMPOSTOR = "claude-opus-4-7"

#: Cheap probes that need no dataset and no network.
OFFLINE_PROBES = ("--probe", "metadata", "--probe", "token_accounting", "--probe", "self_report")


def provider_file(path: Path, server: MockProvider, model: str = BUNDLED_MODEL) -> str:
    path.write_text(
        f"name: mock\napi: openai\nbase_url: {server.openai_url}\n"
        f"model: {model}\nclaimed_model: {model}\nmax_retries: 0\ntimeout_s: 30\n",
        encoding="utf-8",
    )
    return str(path)


# --------------------------------------------------------------------------- #
# Offline commands
# --------------------------------------------------------------------------- #


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "exit codes" in out
    assert "cannot prove which weights ran" in out


def test_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0


def test_no_command_prints_help_and_reports_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == EXIT_USAGE
    assert "usage" in capsys.readouterr().out.lower()


def test_probes_lists_every_probe(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    target = tmp_path / "probes.json"
    assert main(["probes", "--json", str(target)]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert "token_accounting" in out
    assert "evasion" in out

    listing = json.loads(target.read_text(encoding="utf-8"))["probes"]
    assert {row["name"] for row in listing} >= {"metadata", "benchmark", "evasion"}
    assert all(row["layer"] in (0, 1, 2, 3) for row in listing)


def test_benchmarks_lists_the_spread_that_decides_usefulness(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "benchmarks.json"
    assert main(["benchmarks", "--json", str(target)]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert "separates" in out

    listing = json.loads(target.read_text(encoding="utf-8"))["benchmarks"]
    by_name = {row["name"]: row for row in listing}
    assert by_name["gpqa_diamond"]["discriminative"] is False
    assert by_name["gpqa_diamond"]["gated"] is True


def test_models_reads_the_bundled_snapshot(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "models.json"
    assert main(["models", "--json", str(target)]) == EXIT_VERIFIED
    assert BUNDLED_MODEL in capsys.readouterr().out

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["as_of"]
    assert any(row["id"] == BUNDLED_MODEL for row in payload["models"])


def test_models_can_be_filtered_and_says_so_when_nothing_matches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["models", "anthropic"]) == EXIT_VERIFIED
    assert "claude" in capsys.readouterr().out

    assert main(["models", "a-vendor-nobody-has-heard-of"]) == EXIT_VERIFIED
    assert "nothing in the snapshot matches" in capsys.readouterr().out


def test_cache_reports_and_clears(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    assert main(["cache", "--cache-dir", str(cache)]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert "entries   0" in out
    assert "nothing cached yet" in out

    entry = cache / "datasets" / "abc.json"
    entry.parent.mkdir(parents=True)
    entry.write_text('{"_llmverify": {}, "rows": []}', encoding="utf-8")

    assert main(["cache", "--cache-dir", str(cache)]) == EXIT_VERIFIED
    assert "entries   1" in capsys.readouterr().out

    assert main(["cache", "--clear", "--cache-dir", str(cache)]) == EXIT_VERIFIED
    assert "cleared" in capsys.readouterr().out
    assert not entry.exists()


def test_a_bad_layer_selection_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "--model", "m", "--layers", "9"]) == EXIT_USAGE
    assert main(["check", "--model", "m", "--layers", "not-a-number"]) == EXIT_USAGE


def test_check_without_a_model_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check"]) == EXIT_USAGE


def test_an_unknown_probe_name_is_a_usage_error() -> None:
    assert main(["check", "--model", "m", "--probe", "not-a-probe"]) == EXIT_USAGE


@pytest.mark.skip(reason="refresh fetches live leaderboards; no test may touch the network")
def test_refresh() -> None:
    """Left as a marker for the one command that is inherently online."""


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def test_check_returns_zero_for_an_endpoint_that_matches(
    make_provider: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Published overhead of 286 auto / 406 forced, reported exactly."""
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server)

    code = main(["check", path, *OFFLINE_PROBES, "--quiet"])
    assert code == EXIT_VERIFIED
    assert "MATCH" in capsys.readouterr().out


def test_check_returns_one_for_an_endpoint_whose_overhead_is_another_models(
    make_provider: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = make_provider(Persona.SUBSTITUTED, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server)

    # Verbose, because the impostor's name is the finding: the non-verbose
    # table folds the detail column to keep identifiers on one line.
    code = main(["check", path, *OFFLINE_PROBES, "--verbose"])
    assert code == EXIT_SUBSTITUTION
    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert BUNDLED_IMPOSTOR in out


def test_check_returns_two_when_one_probe_settles_nothing(
    make_provider: Any, tmp_path: Path
) -> None:
    """The aggregator refuses to call a run on a single probe, and so does the CLI."""
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server)
    assert main(["check", path, "--probe", "self_report", "--quiet"]) == EXIT_INCONCLUSIVE


def test_check_returns_four_when_nothing_answers(make_provider: Any, tmp_path: Path) -> None:
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL, fail_status=503)
    path = provider_file(tmp_path / "provider.yaml", server)
    assert main(["check", path, "--probe", "metadata", "--probe", "self_report"]) == (
        EXIT_UNREACHABLE
    )


def test_check_writes_a_machine_readable_result(
    make_provider: Any, tmp_path: Path
) -> None:
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server)
    target = tmp_path / "out" / "result.json"

    assert main(["check", path, *OFFLINE_PROBES, "--json", str(target), "--quiet"]) == (
        EXIT_VERIFIED
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["provider"]["claimed_model"] == BUNDLED_MODEL
    assert payload["verdict"]["value"] == "MATCH"
    assert payload["evidence"]
    assert payload["budget"]["samples"] > 0


def test_check_writes_a_self_contained_html_report(
    make_provider: Any, tmp_path: Path
) -> None:
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server)
    target = tmp_path / "report.html"

    assert main(["check", path, *OFFLINE_PROBES, "--html", str(target), "--quiet"]) == (
        EXIT_VERIFIED
    )
    document = target.read_text(encoding="utf-8")
    assert document.lstrip().startswith("<!doctype html")
    assert "MATCH" in document


def test_flags_override_a_provider_file(
    make_provider: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server, model="something-else")

    code = main(
        [
            "check",
            path,
            "--model",
            BUNDLED_MODEL,
            "--claimed-model",
            BUNDLED_MODEL,
            "--probe",
            "metadata",
            "--quiet",
        ]
    )
    assert code in (EXIT_VERIFIED, EXIT_INCONCLUSIVE)
    assert BUNDLED_MODEL in capsys.readouterr().out


def test_a_provider_can_be_described_entirely_with_flags(
    make_provider: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    code = main(
        [
            "check",
            "--api",
            "openai",
            "--base-url",
            server.openai_url,
            "--model",
            BUNDLED_MODEL,
            "--probe",
            "metadata",
            "--quiet",
        ]
    )
    assert code in (EXIT_VERIFIED, EXIT_INCONCLUSIVE)
    # The label falls back to the endpoint host when no name was given.
    assert "127.0.0.1" in capsys.readouterr().out


def test_verbose_shows_every_usable_finding(
    make_provider: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    path = provider_file(tmp_path / "provider.yaml", server)

    main(["check", path, "--probe", "metadata", "--verbose"])
    out = capsys.readouterr().out
    assert "model_echo" in out
    assert "catalogue" in out


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def test_compare_tabulates_two_providers_and_returns_the_worst_code(
    make_provider: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    honest = make_provider(Persona.HONEST, model_id=BUNDLED_MODEL)
    impostor = make_provider(Persona.SUBSTITUTED, model_id=BUNDLED_MODEL)
    first = provider_file(tmp_path / "a.yaml", honest)
    second = provider_file(tmp_path / "b.yaml", impostor)
    target = tmp_path / "compare.json"

    code = main(["compare", first, second, *OFFLINE_PROBES, "--json", str(target)])
    assert code == EXIT_SUBSTITUTION

    out = capsys.readouterr().out
    assert "MATCH" in out and "MISMATCH" in out

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [row["verdict"] for row in payload["providers"]] == ["MATCH", "MISMATCH"]
    assert len(payload["runs"]) == 2
