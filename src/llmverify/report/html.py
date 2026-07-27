"""Self-contained HTML report.

The output is one file with no network dependencies at all: no CDN, no web
fonts, no remote images, no scripts that fetch anything. That is a deliberate
constraint rather than a stylistic one. This document is the artefact somebody
attaches to a support ticket or a procurement dispute, so it has to render the
same in a year, offline, from a mail attachment, and it must not phone anywhere
when a third party opens it.

Charts are hand-written SVG for the same reason, and because there are only
three of them: a diverging bar chart of evidence contributions, an interval
chart per benchmark, and the sequential test's trace. Every one has a table
beside it carrying the same numbers, so nothing is encoded in colour alone.

Everything interpolated into the page goes through :func:`_esc`, which redacts
registered secrets and then escapes markup. Provider error bodies reach this
document verbatim by design -- they are evidence -- and a provider that returns
``<script>`` in an error message must not be able to run it in a reader's
browser.
"""

from __future__ import annotations

import datetime as dt
import html as html_module
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import redact
from ..evidence import FAMILY_CAPS, Evidence, EvidenceStatus, Verdict
from ..results import RunResult, compare_results

__all__ = ["render_comparison_html", "render_html"]

_LN10 = math.log(10)

#: Verdict to CSS colour token. EVASION gets a hue off the good-to-bad axis
#: because it is not a severity: it says the measurements were taken under the
#: provider's control, which is a different kind of finding.
_VERDICT_TOKEN: dict[Verdict, str] = {
    Verdict.MATCH: "good",
    Verdict.LIKELY_MATCH: "good",
    Verdict.INCONCLUSIVE: "warn",
    Verdict.LIKELY_MISMATCH: "serious",
    Verdict.MISMATCH: "critical",
    Verdict.DEGRADED: "serious",
    Verdict.EVASION: "evasion",
}

_VERDICT_GLOSS: dict[Verdict, str] = {
    Verdict.MATCH: "The observations are what the claimed model should produce.",
    Verdict.LIKELY_MATCH: (
        "Consistent with the claimed model, on less evidence than a clean match."
    ),
    Verdict.INCONCLUSIVE: (
        "Not enough usable evidence to decide either way. This is a statement about the "
        "run, not about the provider."
    ),
    Verdict.LIKELY_MISMATCH: (
        "The endpoint behaves unlike the claimed model, on evidence short of the "
        "strongest band."
    ),
    Verdict.MISMATCH: "The endpoint behaves unlike the claimed model.",
    Verdict.DEGRADED: (
        "This is the claimed model, but the deployment is not delivering it: a "
        "capability it should have measurably is not there. Identity is not in "
        "question here, so this sits off the match/mismatch axis -- the answer to "
        "\"are they serving what I paid for\" is no, for a different reason than "
        "substitution."
    ),
    Verdict.EVASION: (
        "The endpoint's behaviour depends on whether an input is recognisable as a "
        "benchmark item. Every other measurement in this run was therefore taken under "
        "conditions the provider chose rather than conditions you chose, and none of them "
        "can be read as a match or a mismatch."
    ),
}

_LAYER_TITLES: dict[int, str] = {
    0: "metadata, free",
    1: "cheap behavioural fingerprints",
    2: "capability probes",
    3: "statistical",
}

_LIMITS_STATEMENT = (
    "This report measures behaviour over the requests this run actually made, at the "
    "settings it used, at the time it ran. Agreement with the claimed model's published "
    "behaviour is evidence that the endpoint served that model for those requests. It is "
    "not proof: an endpoint that routes a fraction of its traffic to the genuine model, or "
    "that serves genuine weights at a lower numeric precision, can produce the best result "
    "on this page. No purely software-side method can settle which weights ran; only "
    "hardware attestation of the serving stack can. Read an adverse verdict the same way "
    "-- as a measurement of this sample, with the evidence and its weights shown above so "
    "that it can be checked."
)

_CSS = """
:root {
  color-scheme: light dark;
  --plane: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --pos: #2a78d6;
  --neg: #e34948;
  --neutral: #f0efec;
  --good: #0ca30c;
  --warn: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
  --evasion: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --plane: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --pos: #3987e5;
    --neg: #e66767;
    --neutral: #383835;
    --evasion: #9085e9;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2rem 1rem 4rem;
  background: var(--plane);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.55;
}
main { max-width: 68rem; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 0.25rem; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 0.75rem; letter-spacing: -0.005em; }
h3 { font-size: 0.95rem; margin: 1.5rem 0 0.5rem; }
p { margin: 0.6rem 0; }
a { color: inherit; }
.sub { color: var(--ink-2); font-size: 0.9rem; margin: 0 0 1.5rem; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.6rem;
  padding: 1.1rem 1.25rem;
  margin: 0 0 1rem;
}
.meta { display: grid; grid-template-columns: minmax(8rem, auto) 1fr; gap: 0.15rem 1.25rem; }
.meta dt { color: var(--muted); font-size: 0.85rem; }
.meta dd { margin: 0; font-size: 0.9rem; word-break: break-word; }
.verdict { border-left: 0.4rem solid var(--tone); }
.verdict .tag {
  display: inline-block;
  font-size: 0.95rem;
  font-weight: 650;
  letter-spacing: 0.06em;
  color: var(--tone);
  border: 1px solid var(--tone);
  border-radius: 0.35rem;
  padding: 0.1rem 0.55rem;
}
.hero { font-size: 2.6rem; font-weight: 600; line-height: 1.1; margin: 0.75rem 0 0; }
.hero small { display: block; font-size: 0.8rem; font-weight: 400; color: var(--muted); }
.stats {
  display: flex;
  flex-wrap: wrap;
  gap: 1.75rem;
  margin-top: 1rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--border);
}
.stats div { font-size: 1.05rem; }
.stats span { display: block; font-size: 0.75rem; color: var(--muted); letter-spacing: 0.02em; }
.note { color: var(--ink-2); font-size: 0.88rem; }
.caption { color: var(--muted); font-size: 0.82rem; margin: 0.5rem 0 0; }
.callout {
  border: 1px solid var(--border);
  border-left: 0.25rem solid var(--tone, var(--serious));
  border-radius: 0.4rem;
  padding: 0.6rem 0.85rem;
  margin: 0.75rem 0;
  font-size: 0.88rem;
  background: var(--surface);
}
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
th, td { text-align: left; padding: 0.42rem 0.6rem; border-bottom: 1px solid var(--grid); }
th { color: var(--muted); font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.detail { color: var(--ink-2); min-width: 20rem; }
tbody tr:last-child td { border-bottom: none; }
.chip {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  border: 1px solid currentColor;
  border-radius: 0.3rem;
  padding: 0 0.35rem;
  white-space: nowrap;
}
.ok { color: var(--ink-2); }
.skipped { color: var(--muted); }
.unsupported { color: var(--ink-2); }
.error { color: var(--critical); }
.truncated { color: var(--serious); }
.legend { display: flex; flex-wrap: wrap; gap: 1rem; margin: 0.5rem 0 0; font-size: 0.8rem; }
.legend i { display: inline-block; width: 0.85rem; height: 0.55rem; border-radius: 0.15rem; }
figure { margin: 0; }
svg { display: block; height: auto; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink-2); }
.tick { fill: var(--muted); font-size: 10px; }
.rowlab { fill: var(--ink-2); font-size: 11px; }
.grp { fill: var(--muted); font-size: 10px; letter-spacing: 0.06em; }
.val { fill: var(--ink); font-size: 10px; font-variant-numeric: tabular-nums; }
.axisline { stroke: var(--axis); stroke-width: 1; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.refline { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; }
.pos { fill: var(--pos); }
.neg { fill: var(--neg); }
.serieline { fill: none; stroke: var(--pos); stroke-width: 2; }
.band { fill: var(--pos); opacity: 0.16; }
details { border-top: 1px solid var(--grid); padding: 0.5rem 0; }
summary { cursor: pointer; font-size: 0.88rem; }
summary::marker { color: var(--muted); }
.kv { display: grid; grid-template-columns: minmax(9rem, auto) 1fr; gap: 0.15rem 1rem; }
.kv dt { color: var(--muted); font-size: 0.8rem; word-break: break-word; }
.kv dd { margin: 0; font-size: 0.84rem; word-break: break-word; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82em; }
footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--ink-2);
  font-size: 0.84rem;
}
"""


# --------------------------------------------------------------------------- #
# Escaping and formatting
# --------------------------------------------------------------------------- #


def _esc(value: object) -> str:
    """Redact secrets, then escape for HTML. The only way text enters the page."""
    return html_module.escape(redact("" if value is None else str(value)), quote=True)


def _num(value: Any) -> float | None:
    """Coerce a value from an evidence data dict to a float, or give up."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _ban(nats: float) -> str:
    return f"{nats / _LN10:+.2f}"


def _probability(probability: float) -> str:
    if probability >= 0.9999:
        return ">99.99%"
    if probability <= 0.0001:
        return "<0.01%"
    if probability >= 0.99 or probability <= 0.01:
        return f"{probability:.2%}"
    return f"{probability:.1%}"


def _value_text(value: Any) -> str:
    """Render one data-dict value as a readable string."""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


# --------------------------------------------------------------------------- #
# SVG primitives
# --------------------------------------------------------------------------- #


def _svg(
    width: float, height: float, body: str, *, label: str, min_width: str = "32rem"
) -> str:
    """Wrap SVG body in a responsive, horizontally scrollable frame.

    No ``xmlns``: inline SVG in an HTML document is already in the SVG
    namespace, and leaving it out keeps the file free of any URL that is not a
    citation a reader can see.
    """
    return (
        f'<div class="scroll"><svg viewBox="0 0 {width:g} {height:g}" width="100%" '
        f'style="min-width:{min_width}" role="img" aria-label="{_esc(label)}">{body}</svg></div>'
    )


def _bar_path(x_zero: float, x_end: float, y: float, height: float, radius: float = 4.0) -> str:
    """A bar anchored at the zero axis with its data end rounded."""
    r = max(0.0, min(radius, abs(x_end - x_zero), height / 2.0))
    top, bottom = y, y + height
    if x_end >= x_zero:
        return (
            f"M{x_zero:.1f} {top:.1f}H{x_end - r:.1f}"
            f"a{r:.1f} {r:.1f} 0 0 1 {r:.1f} {r:.1f}"
            f"V{bottom - r:.1f}a{r:.1f} {r:.1f} 0 0 1 {-r:.1f} {r:.1f}"
            f"H{x_zero:.1f}Z"
        )
    return (
        f"M{x_zero:.1f} {top:.1f}H{x_end + r:.1f}"
        f"a{r:.1f} {r:.1f} 0 0 0 {-r:.1f} {r:.1f}"
        f"V{bottom - r:.1f}a{r:.1f} {r:.1f} 0 0 0 {r:.1f} {r:.1f}"
        f"H{x_zero:.1f}Z"
    )


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    cls: str = "tick",
    anchor: str = "start",
    fill: str | None = None,
    halo: bool = False,
) -> str:
    """One text node. ``halo`` outlines it in the surface colour.

    The halo is how a label stays legible when it has to sit on top of a filled
    mark: painting the stroke first leaves the glyphs themselves untouched, so
    the text keeps its ink colour instead of taking the fill's contrast problem.
    """
    extra = f' fill="{fill}"' if fill else ""
    if halo:
        extra += ' stroke="var(--surface)" stroke-width="3" style="paint-order:stroke"'
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>'
        f"{_esc(text)}</text>"
    )


def _line(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def _end_label(text: str, x_end: float, y: float, *, outward: float, limit: float) -> str:
    """Label a bar's data end, moving inside the bar when it would not fit outside.

    Bars that reach the axis limit are exactly the ones worth labelling, and a
    label pushed past the plot would land on the row labels of the category
    axis, so those flip inward and take the surface colour instead.
    """
    width = len(text) * 5.6
    outside = x_end + outward * 6.0
    if (outward > 0 and outside + width <= limit) or (outward < 0 and outside - width >= limit):
        return _svg_text(
            outside, y, text, cls="val", anchor="start" if outward > 0 else "end"
        )
    return _svg_text(
        x_end - outward * 6.0,
        y,
        text,
        cls="val",
        anchor="end" if outward > 0 else "start",
        halo=True,
    )


def _elide(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Evidence waterfall
# --------------------------------------------------------------------------- #


def _effective_family_cap(raw_nats: float, damped_nats: float, family: str) -> tuple[float, bool]:
    """Recover the damping cap this run actually used.

    Damping is ``cap * tanh(raw / cap)``, which is invertible in ``cap`` for any
    observed pair where damping visibly bit. Solving for it means the reference
    line drawn on the chart is the run's real ceiling even when the caller
    overrode the defaults; when nothing was damped there is nothing to solve, and
    the package default is drawn and labelled as such.
    """
    default = FAMILY_CAPS.get(family, FAMILY_CAPS["misc"])
    raw, damped = abs(raw_nats), abs(damped_nats)
    if damped <= 0.0 or raw - damped < 1e-6:
        return default, False

    low, high = damped + 1e-9, max(default, raw) * 8.0 + 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if mid * math.tanh(raw / mid) < damped:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0, True


def _waterfall(result: RunResult) -> str:
    """Diverging bars, grouped by family, with each family's cap as a reference line."""
    usable = [e for e in result.usable_evidence if abs(e.bans) > 5e-3]
    if not usable:
        return '<p class="note">No probe produced a non-zero contribution.</p>'

    raw_by_family: dict[str, float] = {}
    for item in result.usable_evidence:
        raw_by_family[item.family] = raw_by_family.get(item.family, 0.0) + item.llr

    groups: dict[str, list[Evidence]] = {}
    for item in usable:
        groups.setdefault(item.family, []).append(item)

    totals = result.verdict.family_totals
    order = sorted(groups, key=lambda f: -abs(totals.get(f, 0.0)))
    caps: dict[str, tuple[float, bool]] = {}
    for family in order:
        caps[family] = _effective_family_cap(
            raw_by_family.get(family, 0.0), totals.get(family, 0.0), family
        )

    width, label_w, right_pad = 900.0, 250.0, 86.0
    row_h, bar_h, group_gap, top_pad = 22.0, 11.0, 16.0, 14.0
    plot_left, plot_right = label_w, width - right_pad
    x_zero = (plot_left + plot_right) / 2.0
    half = (plot_right - plot_left) / 2.0

    largest = max(abs(item.bans) for item in usable)
    # Caps are drawn to scale, but a cap far outside the observed evidence would
    # squash every bar into the axis, so the domain stops at 2.5x the data and
    # any cap past it is called out in words instead of drawn.
    cap_bans = [cap / _LN10 for cap, _ in caps.values()]
    domain = max(largest, min(max(cap_bans, default=0.0), largest * 2.5), 0.5) * 1.06

    def x_of(bans: float) -> float:
        return x_zero + max(-1.0, min(1.0, bans / domain)) * half

    parts: list[str] = []
    y = top_pad
    ranked = sorted(usable, key=lambda e: -abs(e.bans))
    labelled = {id(item) for item in ranked[:5]}
    beyond: list[str] = []

    for family in order:
        cap = caps[family][0]
        group_top = y
        parts.append(_svg_text(6, y + 10, family.replace("_", " ").upper(), cls="grp"))
        y += 14.0

        for item in sorted(groups[family], key=lambda e: -abs(e.bans)):
            row_mid = y + row_h / 2.0
            label = f"{item.probe} / {item.label}"
            parts.append(
                _svg_text(label_w - 10, row_mid + 4, _elide(label, 38), cls="rowlab", anchor="end")
            )
            x_end = x_of(item.bans)
            tone = "pos" if item.supports else "neg"
            parts.append(
                f'<path d="{_bar_path(x_zero, x_end, row_mid - bar_h / 2.0, bar_h)}" '
                f'class="{tone}"><title>{_esc(label)}: {_esc(f"{item.bans:+.3f}")} ban'
                f"</title></path>"
            )
            if id(item) in labelled:
                parts.append(
                    _end_label(
                        f"{item.bans:+.2f}", x_end, row_mid + 4,
                        outward=1.0 if item.supports else -1.0,
                        # A supporting label may use the right margin; a refuting
                        # one may not cross into the row labels.
                        limit=width - 6.0 if item.supports else plot_left,
                    )
                )
            y += row_h

        band_bottom = y - 2.0
        if cap / _LN10 <= domain:
            for sign in (1.0, -1.0):
                x_cap = x_of(sign * cap / _LN10)
                parts.append(_line(x_cap, group_top + 12, x_cap, band_bottom, "refline"))
            # On the family's own heading line, where no bar or row label can be.
            parts.append(
                _svg_text(x_of(cap / _LN10), group_top + 10, f"cap {cap / _LN10:.1f}",
                          cls="tick", anchor="middle")
            )
        else:
            beyond.append(f"{family} ({cap / _LN10:.1f} ban)")
        y += group_gap

    height, axis_y = y + 50.0, y + 8.0
    parts.append(_line(plot_left, axis_y, plot_right, axis_y, "axisline"))
    for fraction in (-1.0, -0.5, 0.0, 0.5, 1.0):
        value = fraction * domain
        x = x_of(value)
        parts.append(_line(x, top_pad - 6, x, axis_y, "gridline" if fraction else "axisline"))
        parts.append(_svg_text(x, axis_y + 14, f"{value:+.1f}" if fraction else "0", cls="tick",
                               anchor="middle"))
    parts.append(_svg_text(plot_left, axis_y + 30, "refutes the claim", cls="tick"))
    parts.append(_svg_text(plot_right, axis_y + 30, "supports the claim", cls="tick", anchor="end"))

    legend = (
        '<div class="legend">'
        '<span><i style="background:var(--pos)"></i> supports the claim</span>'
        '<span><i style="background:var(--neg)"></i> refutes the claim</span>'
        '<span><i style="background:var(--muted);height:0;border-top:2px dashed var(--muted)">'
        "</i> family damping cap</span></div>"
    )
    caption = (
        "One bar per probe that produced usable evidence, in bans (a ban is a 10:1 "
        "likelihood ratio). Bars are grouped by evidence family; the dashed lines are the "
        "ceiling that family's combined evidence is damped toward, so a family whose bars "
        "reach its line contributed less to the total than their sum suggests. Values for "
        "every probe, damped and undamped, are in the tables below."
    )
    if beyond:
        caption += (
            " Cap lines outside the plotted range are not drawn: "
            + ", ".join(sorted(beyond))
            + "."
        )
    if any(not derived for _, derived in caps.values()):
        caption += (
            " Where damping did not bite, the line shown is the package default cap for "
            "that family rather than a value recovered from this run."
        )
    chart = _svg(
        width,
        height,
        "".join(parts),
        label="Evidence contribution per probe, in bans, grouped by evidence family.",
    )
    return chart + legend + f'<p class="caption">{_esc(caption)}</p>'


def _family_table(result: RunResult) -> str:
    """The waterfall's table twin: raw sum, damped contribution, cap, share."""
    raw: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in result.usable_evidence:
        raw[item.family] = raw.get(item.family, 0.0) + item.llr
        counts[item.family] = counts.get(item.family, 0) + 1

    totals = result.verdict.family_totals
    if not totals:
        return ""
    magnitude = sum(abs(v) for v in totals.values()) or 1.0

    rows = []
    any_default = False
    for family, damped in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
        cap, derived = _effective_family_cap(raw.get(family, 0.0), damped, family)
        any_default = any_default or not derived
        rows.append(
            "<tr>"
            f"<td>{_esc(family)}</td>"
            f'<td class="num">{counts.get(family, 0)}</td>'
            f'<td class="num">{_esc(_ban(raw.get(family, 0.0)))}</td>'
            f'<td class="num">{_esc(_ban(damped))}</td>'
            f'<td class="num">{_esc(f"{cap / _LN10:.2f}")}'
            f'{"" if derived else " *"}</td>'
            f'<td class="num">{abs(damped) / magnitude:.0%}</td>'
            "</tr>"
        )

    top_family, top_total = max(totals.items(), key=lambda kv: abs(kv[1]))
    concentration = ""
    if len(totals) > 1 and abs(top_total) / magnitude >= 0.6:
        concentration = (
            f'<p class="note">{abs(top_total) / magnitude:.0%} of the evidence weight comes '
            f"from the {_esc(top_family)} family alone. A verdict resting on one family is a "
            "narrower result than the same number of bans drawn from several.</p>"
        )
    footnote = (
        '<p class="caption">* package default cap; this run never reached it, so the value '
        "could not be recovered from the run itself.</p>"
        if any_default
        else ""
    )
    return (
        '<div class="scroll"><table><thead><tr><th>family</th><th class="num">probes</th>'
        '<th class="num">raw sum (ban)</th><th class="num">after damping (ban)</th>'
        '<th class="num">cap (ban)</th><th class="num">share</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
        + footnote
        + concentration
    )


# --------------------------------------------------------------------------- #
# Benchmark charts
# --------------------------------------------------------------------------- #


def _benchmark_evidence(result: RunResult) -> list[Evidence]:
    return [
        item
        for item in result.evidence
        if isinstance(item.data.get("benchmark"), str)
        and _num(item.data.get("observed_accuracy_pp")) is not None
    ]


def _interval_chart(item: Evidence) -> str:
    """Measured accuracy with its Wilson interval against the published score."""
    data = item.data
    observed = _num(data.get("observed_accuracy_pp")) or 0.0
    interval = data.get("wilson_95_pp") or []
    low = _num(interval[0]) if len(interval) == 2 else None
    high = _num(interval[1]) if len(interval) == 2 else None
    published = _num(data.get("reference_score_pp"))
    # The line worth drawing is the accuracy the test was actually run against,
    # which is the published score less whatever tolerance the incomparable
    # conditions bought. The distance between the two lines is that tolerance.
    null_accuracy = _num(data.get("null_accuracy"))
    used = null_accuracy * 100.0 if null_accuracy is not None else _num(
        data.get("reference_score_used_pp")
    )

    width, height = 900.0, 150.0
    left, right = 60.0, 40.0
    axis_y, row_y = 96.0, 52.0

    def x_of(pp: float) -> float:
        return left + max(0.0, min(100.0, pp)) / 100.0 * (width - left - right)

    parts: list[str] = []
    for tick in range(0, 101, 10):
        x = x_of(tick)
        parts.append(_line(x, 24, x, axis_y, "gridline"))
        if tick % 20 == 0:
            parts.append(_svg_text(x, axis_y + 15, str(tick), cls="tick", anchor="middle"))
    parts.append(_line(left, axis_y, x_of(100), axis_y, "axisline"))
    parts.append(_svg_text(x_of(100), axis_y + 30, "accuracy (%)", cls="tick", anchor="end"))

    if low is not None and high is not None:
        x_low, x_high = x_of(low), x_of(high)
        parts.append(
            f'<rect x="{x_low:.1f}" y="{row_y - 3:.1f}" width="{max(2.0, x_high - x_low):.1f}" '
            f'height="6" rx="3" class="pos" opacity="0.28"><title>Wilson 95% interval '
            f'{_esc(f"{low:.1f}")}-{_esc(f"{high:.1f}")}%</title></rect>'
        )
        for x in (x_low, x_high):
            parts.append(_line(x, row_y - 8, x, row_y + 8, "axisline"))

    x_obs = x_of(observed)
    parts.append(
        f'<circle cx="{x_obs:.1f}" cy="{row_y:.1f}" r="5.5" class="pos" stroke="var(--surface)" '
        f'stroke-width="2"><title>measured {_esc(f"{observed:.1f}")}%</title></circle>'
    )
    parts.append(
        _svg_text(x_obs, row_y - 14, f"{observed:.1f}%", cls="val", anchor="middle")
    )

    if published is not None:
        x_pub = x_of(published)
        parts.append(_line(x_pub, 24, x_pub, axis_y, "refline"))
        parts.append(
            f'<path d="M{x_pub:.1f} {row_y - 7:.1f}l6 7-6 7-6-7Z" fill="var(--neg)" '
            f'stroke="var(--surface)" stroke-width="2"><title>published '
            f'{_esc(f"{published:.1f}")}%</title></path>'
        )
        parts.append(
            _svg_text(x_pub, 20, f"published {published:.1f}%", cls="tick", anchor="middle")
        )
    if used is not None and (published is None or abs(used - published) > 0.05):
        x_used = x_of(used)
        parts.append(_line(x_used, 24, x_used, axis_y, "refline"))
        parts.append(
            _svg_text(x_used, axis_y - 6, f"null used {used:.1f}%", cls="tick", anchor="middle")
        )

    caveat = _effort_caveat(data)
    if caveat:
        parts.append(
            f'<text x="{left:.1f}" y="{height - 8:.1f}" class="tick" '
            f'fill="var(--serious)">{_esc(caveat)}</text>'
        )

    legend = (
        '<div class="legend">'
        '<span><i style="background:var(--pos);border-radius:50%;width:0.6rem;height:0.6rem">'
        "</i> measured accuracy</span>"
        '<span><i style="background:var(--pos);opacity:0.28"></i> Wilson 95% interval</span>'
        '<span><i style="background:var(--neg)"></i> published reference score</span>'
        '<span><i style="background:var(--muted);height:0;border-top:2px dashed var(--muted)">'
        "</i> null the test used</span></div>"
    )
    label = (
        f"Measured accuracy {observed:.1f}% with its 95% interval, against the published "
        f"reference score."
    )
    return _svg(width, height, "".join(parts), label=label, min_width="30rem") + legend


def _effort_caveat(data: dict[str, Any]) -> str:
    """The one sentence that matters more than the number, when it applies."""
    reference_effort = data.get("reference_effort")
    provider_effort = data.get("provider_effort")
    if reference_effort is None:
        return (
            "Caveat: the published score does not state the reasoning effort it was measured "
            "at, so this comparison is approximate."
        )
    if provider_effort is None:
        return (
            f"Caveat: the published score was measured at effort {reference_effort!r}; this run "
            "pinned no effort, so the endpoint's default is unknown."
        )
    if str(provider_effort) != str(reference_effort):
        return (
            f"Caveat: the published score was measured at effort {reference_effort!r} and this "
            f"run pinned {provider_effort!r}. Scores are not comparable across efforts."
        )
    return ""


def _benchmark_conditions(data: dict[str, Any]) -> str:
    """Source, URL, as-of date and evaluation conditions, as a definition list."""
    rows: list[tuple[str, str]] = []
    source = data.get("reference_source")
    if source:
        rows.append(("source", str(source)))
    url = data.get("reference_source_url") or data.get("source_url")
    if url:
        # Printed as text, never as a link: this document loads nothing remote,
        # and a citation a reader can copy is worth more than one they can click.
        rows.append(("source URL", str(url)))
    else:
        rows.append(("source URL", "not recorded in the reference snapshot"))
    for key, label in (
        ("reference_key", "reference score key"),
        ("reference_as_of", "as of"),
        ("reference_confidence", "confidence"),
        ("reference_effort", "reference effort"),
        ("provider_effort", "effort pinned for this run"),
        ("reference_optimized", "benchmark-optimised settings"),
        ("tolerance_allowance_pp", "tolerance added (pp)"),
        ("null_accuracy", "null accuracy"),
        ("alternative_accuracy", "alternative accuracy"),
        ("n_graded", "items graded"),
        ("stopped_for", "stopped because"),
        ("discriminative", "discriminative benchmark"),
    ):
        if key in data and data[key] is not None:
            rows.append((label, _value_text(data[key])))

    reasons = data.get("tolerance_reasons")
    if isinstance(reasons, list) and reasons:
        rows.append(("why tolerance was added", "; ".join(str(r) for r in reasons)))

    body = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    return f'<dl class="kv">{body}</dl>'


# --------------------------------------------------------------------------- #
# SPRT trace
# --------------------------------------------------------------------------- #


_Path = list[tuple[float, float]]


def _sprt_series(data: dict[str, Any]) -> tuple[_Path, _Path, str]:
    """Return the statistic's path, an optional envelope partner, and a caption.

    When the probe recorded a per-item trace, the first path is that trace and
    the second is empty. When it did not, the two are the envelope of every path
    consistent with the recorded ``k`` correct out of ``n``: all wrong answers
    first is the highest the statistic could have gone, all correct first the
    lowest. Drawing a single straight line between the endpoints instead would
    invent an ordering the run never recorded.
    """
    trace = data.get("sprt_trace") or data.get("llr_trace")
    if isinstance(trace, list) and trace:
        points: _Path = []
        for index, entry in enumerate(trace, start=1):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                n, llr = _num(entry[0]), _num(entry[1])
            else:
                n, llr = float(index), _num(entry)
            if n is not None and llr is not None:
                points.append((n, llr))
        if points:
            return points, [], "The per-item path of the statistic, as recorded."

    n = _num(data.get("n_graded"))
    successes = _num(data.get("successes"))
    p0 = _num(data.get("null_accuracy"))
    p1 = _num(data.get("alternative_accuracy"))
    final = _num(data.get("sprt_llr_nats")) or 0.0
    if n is None or successes is None or n <= 0:
        return [], [], ""
    if p0 is None or p1 is None or not 0.0 < p1 < p0 < 1.0:
        return (
            [(0.0, 0.0), (n, final)],
            [],
            "Only the endpoint of the statistic was recorded; the line between it and the "
            "origin is a guide, not the path the run took.",
        )

    step_correct = math.log(p1 / p0)
    step_wrong = math.log((1.0 - p1) / (1.0 - p0))
    wrong = n - successes
    return (
        [(0.0, 0.0), (wrong, wrong * step_wrong), (n, final)],
        [(0.0, 0.0), (successes, successes * step_correct), (n, final)],
        "The per-item order was not recorded, so the band covers every path consistent with "
        f"{successes:.0f} correct out of {n:.0f}; both edges end where the test stopped.",
    )


def _sprt_chart(item: Evidence) -> str:
    data = item.data
    upper_bound = _num(data.get("sprt_upper_bound"))
    lower_bound = _num(data.get("sprt_lower_bound"))
    if upper_bound is None or lower_bound is None:
        return ""
    upper, lower, caption = _sprt_series(data)
    if not upper:
        return ""

    width, height = 900.0, 250.0
    left, right, top, bottom = 64.0, 40.0, 34.0, 40.0
    n_max = max(point[0] for point in upper + lower) or 1.0
    values = [point[1] for point in upper + lower] + [upper_bound, lower_bound]
    pad = (max(values) - min(values)) * 0.12 or 0.5
    y_hi, y_lo = max(values) + pad, min(values) - pad

    def x_of(n: float) -> float:
        return left + (n / n_max) * (width - left - right)

    def y_of(llr: float) -> float:
        span = (y_hi - y_lo) or 1.0
        return top + (y_hi - llr) / span * (height - top - bottom)

    parts: list[str] = []
    parts.append(_line(left, y_of(y_hi), left, y_of(y_lo), "axisline"))
    parts.append(_line(left, y_of(y_lo), x_of(n_max), y_of(y_lo), "axisline"))
    parts.append(_line(left, y_of(0.0), x_of(n_max), y_of(0.0), "gridline"))
    parts.append(_svg_text(left - 8, y_of(0.0) + 4, "0", cls="tick", anchor="end"))

    for bound, label in (
        (upper_bound, "boundary: accept H1, the endpoint is degraded"),
        (lower_bound, "boundary: accept H0, the reference accuracy"),
    ):
        y = y_of(bound)
        parts.append(_line(left, y, x_of(n_max), y, "refline"))
        parts.append(_svg_text(left + 6, y - 5, f"{label}  {bound:+.2f}", cls="tick"))

    def points_of(path: _Path) -> str:
        return " ".join(f"{x_of(n):.1f},{y_of(v):.1f}" for n, v in path)

    if lower:
        band = points_of(upper) + " " + points_of(list(reversed(lower)))
        parts.append(f'<polygon points="{band}" class="band"/>')
        parts.append(f'<polyline points="{points_of(lower)}" class="serieline" opacity="0.5"/>')
    parts.append(
        f'<polyline points="{points_of(upper)}" class="serieline" '
        f'opacity="{0.5 if lower else 1.0}"/>'
    )

    stop_n, stop_llr = upper[-1]
    parts.append(
        f'<circle cx="{x_of(stop_n):.1f}" cy="{y_of(stop_llr):.1f}" r="5" class="pos" '
        f'stroke="var(--surface)" stroke-width="2"><title>stopped at n={_esc(f"{stop_n:.0f}")}, '
        f'LLR {_esc(f"{stop_llr:+.2f}")}</title></circle>'
    )
    parts.append(
        _svg_text(x_of(stop_n) - 8, y_of(stop_llr) - 10,
                  f"stopped at n={stop_n:.0f}, LLR {stop_llr:+.2f}", cls="val", anchor="end")
    )

    for fraction in (0.0, 0.5, 1.0):
        n = n_max * fraction
        parts.append(_svg_text(x_of(n), height - 18, f"{n:.0f}", cls="tick", anchor="middle"))
    parts.append(_svg_text(left, height - 4, "items graded", cls="tick"))
    parts.append(
        _svg_text(6, 14, "log-likelihood ratio (nats), positive favours degradation", cls="tick")
    )

    decision = data.get("sprt_decision", "unknown")
    stopped_for = data.get("stopped_for", "")
    note = (
        f"{caption} The test stopped with decision {decision!r}"
        + (f" ({stopped_for})." if stopped_for else ".")
        + " Positive movement favours the degraded hypothesis, which is the opposite sign "
        "from this probe's evidence contribution."
    )
    return _svg(
        width,
        height,
        "".join(parts),
        label="Sequential test: log-likelihood ratio against items graded, with its decision "
        "boundaries.",
        min_width="30rem",
    ) + f'<p class="caption">{_esc(note)}</p>'


# --------------------------------------------------------------------------- #
# Tables and detail sections
# --------------------------------------------------------------------------- #


def _evidence_table(result: RunResult) -> str:
    layer_of = {timing.probe: timing.layer for timing in result.timings}
    rows = []
    ordered = sorted(
        result.evidence,
        key=lambda e: (layer_of.get(e.probe, 99), -abs(e.bans), e.probe, e.label),
    )
    for item in ordered:
        layer = layer_of.get(item.probe)
        usable = item.status is EvidenceStatus.OK
        rows.append(
            "<tr>"
            f'<td class="num">{"" if layer is None else layer}</td>'
            f"<td>{_esc(item.probe)}</td>"
            f"<td>{_esc(item.label)}</td>"
            f"<td>{_esc(item.family)}</td>"
            f'<td><span class="chip {item.status.value}">{_esc(item.status.value)}</span></td>'
            f'<td class="num">{_esc(f"{item.bans:+.2f}") if usable else "&mdash;"}</td>'
            f'<td class="num">{_esc(f"{item.cap / _LN10:.2f}")}</td>'
            f'<td class="detail">{_esc(item.detail)}</td>'
            "</tr>"
        )
    layers = "".join(
        f"<li>layer {layer}: {_esc(title)}</li>" for layer, title in sorted(_LAYER_TITLES.items())
    )
    return (
        '<div class="scroll"><table><thead><tr>'
        '<th class="num">layer</th><th>probe</th><th>check</th><th>family</th><th>status</th>'
        '<th class="num">ban</th><th class="num">cap</th><th>detail</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        f'<ul class="caption">{layers}</ul>'
    )


def _probe_details(result: RunResult) -> str:
    blocks = []
    for item in result.evidence:
        if not item.data and not item.detail:
            continue
        rows = "".join(
            f"<dt>{_esc(key)}</dt><dd>{_esc(_value_text(value))}</dd>"
            for key, value in item.data.items()
        )
        detail = f'<p class="note">{_esc(item.detail)}</p>' if item.detail else ""
        spend = ""
        if item.cost_usd or item.tokens or item.duration_s:
            spend = (
                f'<p class="caption">cost ${item.cost_usd:.6f} &middot; {item.tokens} tokens '
                f"&middot; {item.duration_s:.2f}s</p>"
            )
        blocks.append(
            "<details><summary>"
            f"{_esc(item.probe)} / {_esc(item.label)} "
            f'<span class="chip {item.status.value}">{_esc(item.status.value)}</span> '
            f'<span class="note">{_esc(f"{item.bans:+.2f}")} ban</span>'
            f"</summary>{detail}"
            + (f'<dl class="kv">{rows}</dl>' if rows else "")
            + spend
            + "</details>"
        )
    return "".join(blocks)


def _timings_table(result: RunResult) -> str:
    if not result.timings:
        return ""
    rows = "".join(
        "<tr>"
        f'<td class="num">{timing.layer}</td>'
        f"<td>{_esc(timing.probe)}</td>"
        f"<td>{_esc(timing.status)}</td>"
        f'<td class="num">{timing.requests}</td>'
        f'<td class="num">{timing.duration_s:.2f}</td>'
        f'<td class="detail">{_esc(timing.error or "")}</td>'
        "</tr>"
        for timing in result.timings
    )
    return (
        '<div class="scroll"><table><thead><tr><th class="num">layer</th><th>probe</th>'
        '<th>status</th><th class="num">requests</th><th class="num">seconds</th>'
        "<th>error</th></tr></thead><tbody>" + rows + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )


def _endpoint_host(base_url: str | None) -> str:
    """Host and port of an endpoint, never its credentials.

    ``urlsplit().hostname`` drops any ``user:password@`` prefix, which is the
    point: a key pasted into a base URL must not reach a report.
    """
    if not base_url:
        return "(adapter default)"
    parts = urlsplit(base_url)
    if not parts.hostname:
        return "(unparsable base_url)"
    return f"{parts.hostname}:{parts.port}" if parts.port else parts.hostname


def _verdict_card(result: RunResult) -> str:
    report = result.verdict
    verdict = report.verdict
    token = _VERDICT_TOKEN[verdict]
    return (
        f'<section class="card verdict" style="--tone:var(--{token})">'
        f'<span class="tag">{_esc(verdict.value)}</span>'
        f'<p class="hero">{_esc(_probability(report.probability))}'
        "<small>posterior probability that this endpoint serves the claimed model, from a "
        f"prior of {_esc(f'{report.prior_odds:g}')}:1 odds</small></p>"
        '<div class="stats">'
        f"<div>{_esc(f'{report.total_bans:+.2f}')} ban<span>weight of evidence "
        "(negative refutes)</span></div>"
        f"<div>${result.spent_usd:.4f}<span>estimated cost</span></div>"
        f"<div>{result.duration_s:.1f}s<span>duration</span></div>"
        f"<div>{result.samples}<span>samples</span></div>"
        f"<div>{len(result.usable_evidence)} / {len(result.evidence)}"
        "<span>probes with usable evidence</span></div>"
        "</div>"
        f'<p class="note">{_esc(_VERDICT_GLOSS[verdict])}</p>'
        "</section>"
    )


def _header_card(result: RunResult) -> str:
    if result.reference_as_of is None:
        reference = "no snapshot loaded"
    else:
        found = "claimed model found" if result.reference_found else "claimed model NOT in snapshot"
        reference = f"{result.reference_as_of.isoformat()} ({found})"
    rows = [
        ("provider", result.provider_name),
        ("endpoint", _endpoint_host(result.base_url)),
        ("requested model", result.requested_model),
        ("claimed model", result.claimed_model),
        ("api family", result.api_family),
        ("reference snapshot", reference),
        ("layers", ", ".join(str(n) for n in result.layers) or "none"),
        ("seed", result.seed),
        ("baseline A/B", "yes" if result.baseline_used else "no"),
        ("started", result.started_at.isoformat(timespec="seconds")),
    ]
    body = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    return f'<section class="card"><dl class="meta">{body}</dl></section>'


def _messages(result: RunResult) -> str:
    blocks = [
        f'<div class="callout" style="--tone:var(--ink-2)">{_esc(note)}</div>'
        for note in result.verdict.notes
    ]
    blocks += [
        f'<div class="callout" style="--tone:var(--warn)">{_esc(warning)}</div>'
        for warning in result.warnings
    ]
    if not blocks:
        return ""
    return "<h2>Notes and warnings</h2>" + "".join(blocks)


def _footer(
    *, tool_version: str, reference_as_of: dt.date | None, seed: int | None, generated: dt.datetime
) -> str:
    facts = [
        f"llmverify {tool_version}",
        f"reference snapshot {reference_as_of.isoformat() if reference_as_of else 'none'}",
    ]
    if seed is not None:
        facts.append(f"seed {seed}")
    facts.append(f"generated {generated.isoformat(timespec='seconds')}")
    return (
        "<footer><p>"
        + _esc(" · ".join(facts))
        + "</p><p><strong>What this establishes.</strong> "
        + _esc(_LIMITS_STATEMENT)
        + "</p></footer>"
    )


def render_html(result: RunResult, *, path: Path) -> Path:
    """Write the full report for one run and return the path written."""
    sections: list[str] = [
        f"<h1>{_esc(result.provider_name)} &mdash; {_esc(result.claimed_model)}</h1>",
        f'<p class="sub">{_esc(_endpoint_host(result.base_url))} &middot; verified '
        f"{_esc(result.finished_at.isoformat(timespec='seconds'))}</p>",
        _verdict_card(result),
        _header_card(result),
        _messages(result),
        "<h2>Where the evidence came from</h2>",
        _waterfall(result),
        _family_table(result),
    ]

    benchmarks = _benchmark_evidence(result)
    if benchmarks:
        sections.append("<h2>Benchmarks</h2>")
        for item in benchmarks:
            name = str(item.data.get("benchmark", item.label))
            sections.append(f"<h3>{_esc(name)}</h3>")
            caveat = _effort_caveat(item.data)
            if caveat:
                sections.append(f'<div class="callout">{_esc(caveat)}</div>')
            sections.append("<figure>" + _interval_chart(item) + "</figure>")
            sections.append(_benchmark_conditions(item.data))
            trace = _sprt_chart(item)
            if trace:
                sections.append("<figure>" + trace + "</figure>")

    sections.extend(
        [
            "<h2>All evidence</h2>",
            _evidence_table(result),
            "<h2>Per-probe detail</h2>",
            _probe_details(result),
            "<h2>Probe timings</h2>",
            _timings_table(result),
            _footer(
                tool_version=result.tool_version,
                reference_as_of=result.reference_as_of,
                seed=result.seed,
                generated=result.finished_at,
            ),
        ]
    )

    title = f"llmverify: {result.provider_name} / {result.claimed_model}"
    return _write(path, _page(title, "".join(sections)))


def render_comparison_html(results: list[RunResult], *, path: Path) -> Path:
    """Write a side-by-side report for several providers and return the path."""
    if not results:
        body = "<h1>Provider comparison</h1><p class=\"note\">No runs to compare.</p>"
        return _write(path, _page("llmverify: comparison", body))

    summary = compare_results(results)
    pending: dict[str, list[RunResult]] = {}
    for run in results:
        pending.setdefault(run.provider_name, []).append(run)
    ranked: list[RunResult] = []
    for row in summary["providers"]:
        bucket = pending.get(str(row["name"]))
        if bucket:
            ranked.append(bucket.pop(0))

    rows = []
    for row in summary["providers"]:
        verdict = Verdict(str(row["verdict"]))
        token = _VERDICT_TOKEN[verdict]
        rows.append(
            "<tr>"
            f"<td>{_esc(row['name'])}</td>"
            f"<td>{_esc(row['claimed_model'])}</td>"
            f'<td><span class="chip" style="color:var(--{token})">{_esc(verdict.value)}</span></td>'
            f'<td class="num">{_esc(_probability(float(row["probability_genuine"])))}</td>'
            f'<td class="num">{float(row["total_bans"]):+.2f}</td>'
            f'<td class="num">${float(row["estimated_cost_usd"]):.4f}</td>'
            f'<td class="num">{float(row["duration_s"]):.1f}s</td>'
            "</tr>"
        )

    warnings = "".join(
        f'<div class="callout" style="--tone:var(--warn)">{_esc(w)}</div>'
        for w in summary["warnings"]
    )

    body = (
        "<h1>Provider comparison</h1>"
        f'<p class="sub">{_esc(", ".join(summary["claimed_models"]))} &middot; '
        f"{len(results)} endpoints</p>"
        + warnings
        + '<div class="scroll"><table><thead><tr><th>provider</th><th>claimed model</th>'
        '<th>verdict</th><th class="num">P(claimed)</th><th class="num">ban</th>'
        '<th class="num">cost</th><th class="num">time</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
        + "<h2>Weight of evidence</h2>"
        + _comparison_chart(ranked)
        + _footer(
            tool_version=results[0].tool_version,
            reference_as_of=results[0].reference_as_of,
            seed=None,
            generated=max(r.finished_at for r in results),
        )
    )
    return _write(path, _page("llmverify: comparison", body))


def _comparison_chart(results: list[RunResult]) -> str:
    """One diverging bar per provider, on a shared scale."""
    if not results:
        return ""
    width, label_w, right_pad = 900.0, 220.0, 70.0
    row_h, bar_h, top_pad = 30.0, 13.0, 14.0
    plot_left, plot_right = label_w, width - right_pad
    x_zero = (plot_left + plot_right) / 2.0
    half = (plot_right - plot_left) / 2.0
    domain = max((abs(r.verdict.total_bans) for r in results), default=0.5) * 1.15 or 0.5

    parts: list[str] = []
    y = top_pad
    for result in results:
        bans = result.verdict.total_bans
        mid = y + row_h / 2.0
        parts.append(
            _svg_text(label_w - 10, mid + 4, _elide(result.provider_name, 30), cls="rowlab",
                      anchor="end")
        )
        x_end = x_zero + max(-1.0, min(1.0, bans / domain)) * half
        tone = "pos" if bans >= 0 else "neg"
        parts.append(
            f'<path d="{_bar_path(x_zero, x_end, mid - bar_h / 2.0, bar_h)}" class="{tone}">'
            f"<title>{_esc(result.provider_name)}: {_esc(f'{bans:+.2f}')} ban</title></path>"
        )
        anchor = "start" if bans >= 0 else "end"
        parts.append(
            _svg_text(x_end + (6 if bans >= 0 else -6), mid + 4,
                      f"{bans:+.2f}", cls="val", anchor=anchor)
        )
        y += row_h

    axis_y = y + 6.0
    parts.append(_line(x_zero, top_pad - 4, x_zero, axis_y, "axisline"))
    parts.append(_svg_text(plot_left, axis_y + 14, "refutes the claim", cls="tick"))
    parts.append(_svg_text(plot_right, axis_y + 14, "supports the claim", cls="tick", anchor="end"))
    caption = (
        "Total weight of evidence per provider, in bans, after per-family damping. Providers "
        "checked against different claimed models are not directly comparable; any such "
        "mismatch is listed above the table."
    )
    chart = _svg(
        width,
        axis_y + 26.0,
        "".join(parts),
        label="Total weight of evidence per provider, in bans.",
    )
    return chart + f'<p class="caption">{_esc(caption)}</p>'


def _write(path: Path, document: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
