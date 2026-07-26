"""Three machine-gradable images, and the difference between accepting and seeing.

A text-only model wearing a multimodal name is one of the easiest substitutions
to make and one of the easiest to catch, because there is no partial credit: a
model that cannot see answers at chance, and chance on these three tasks is one
in six, one in four and one in nine.

**The images are generated here, in pure Python.** :func:`png` writes a minimal
RGB PNG with :mod:`zlib` and :mod:`struct` and nothing else -- no Pillow, no
image assets in the repository. That matters beyond dependency hygiene: an image
generated fresh from the run's seed cannot have been seen during training, and
cannot be recognised by a provider that special-cases known probe images.

The three tasks are chosen to be unambiguous to a human and unguessable without
sight: a solid colour drawn from a fixed palette, a count of well-separated
squares, and the position of the odd cell in a three-by-three grid. Each has a
single correct answer that grades by string or integer comparison, so no judge
model is involved.

**"Accepts images" and "sees images" are different claims, and this probe
reports them separately.** A proxy can accept an image part, drop it, and pass
the text through to a text-only model; the request succeeds, the usage object
looks plausible, and the answers are guesses. An endpoint that *rejects* image
content while claiming a multimodal model is strong evidence against the claim,
and so is one that accepts the image and then answers at chance -- the second
being the case a naive check misses entirely.

**The likelihood model is stated, not hidden.** Each task contributes
``ln(P(observed | sighted) / P(observed | blind))``, using an assumed per-task
accuracy for a genuinely sighted frontier model and the exact chance rate for a
blind one. The sighted rates are assumptions -- 0.95 for colour, 0.85 for
counting and for grid position, reflecting that counting and spatial indexing
are where sighted models genuinely do slip -- and they are deliberately
conservative, since underestimating a real model's accuracy shrinks the evidence
against an endpoint rather than inflating it.
"""

from __future__ import annotations

import random
import re
import struct
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import STRONG, Evidence, EvidenceStatus, llr_from_probability
from ..types import ChatRequest, ImagePart, Message, Role, TextPart
from . import Probe, ProbeContext, register_probe

__all__ = ["PALETTE", "VisionProbe", "png"]

#: Named colours, well separated in RGB so that a sighted model naming any of
#: them names the right one. Each entry is the RGB triple and the words that
#: count as naming it.
PALETTE: tuple[tuple[str, tuple[int, int, int], tuple[str, ...]], ...] = (
    ("red", (215, 35, 35), ("red", "crimson", "scarlet")),
    ("green", (30, 155, 60), ("green", "emerald")),
    ("blue", (35, 70, 200), ("blue", "azure", "cobalt")),
    ("yellow", (240, 210, 45), ("yellow", "gold")),
    ("purple", (130, 50, 175), ("purple", "violet")),
    ("orange", (240, 135, 30), ("orange", "amber")),
)

#: Assumed per-task accuracy for an endpoint that genuinely sees the image.
#: Assumptions, not measurements; see the module docstring.
SIGHTED_ACCURACY: dict[str, float] = {"colour": 0.95, "count": 0.85, "grid": 0.85}

#: Exact chance rate for an endpoint that cannot see the image and must guess.
BLIND_ACCURACY: dict[str, float] = {"colour": 1 / 6, "count": 1 / 4, "grid": 1 / 9}

_ANSWER_RE = re.compile(r"ANSWER\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

#: Substrings that mark a refusal as being about the image rather than about
#: anything else in the request.
_IMAGE_REFUSAL_MARKERS: tuple[str, ...] = (
    "image",
    "vision",
    "multimodal",
    "modality",
    "image_url",
    "media_type",
    "does not support images",
    "unsupported content",
)

#: Phrases a text-only model uses when it was handed a prompt about an image it
#: never received. Recorded because it explains a chance-level score.
_BLIND_DISCLAIMERS: tuple[str, ...] = (
    "cannot see",
    "can't see",
    "unable to see",
    "no image",
    "i do not have access to",
    "i don't have access to",
    "not able to view",
    "cannot view",
    "as a text-based",
)


@dataclass(slots=True)
class _Task:
    """One image, one question, one graded answer."""

    key: str
    kind: str
    image: bytes
    prompt: str
    expected: str
    correct: bool | None = None
    answer: str | None = None
    error: str | None = None
    disclaimed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expected": self.expected,
            "answer": self.answer,
            "correct": self.correct,
            "image_bytes": len(self.image),
            "blind_disclaimer": self.disclaimed,
            "error": self.error,
        }


@register_probe
class VisionProbe(Probe):
    """Colour, count and grid-position tasks on freshly generated PNGs."""

    name: ClassVar[str] = "vision"
    layer: ClassVar[int] = 2
    family: ClassVar[str] = "vision"
    order: ClassVar[int] = 90
    estimated_requests: ClassVar[int] = 3
    description: ClassVar[str] = (
        "Three procedurally generated PNGs -- solid colour, square count, odd cell in "
        "a 3x3 grid -- graded exactly, separating image rejection from image blindness."
    )

    max_tokens: ClassVar[int] = 256

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only when an image claim exists to be tested.

        Both halves are required: the claimed model must be recorded as taking
        image input, and the protocol must be able to carry one. Sending an
        image to an endpoint whose claimed model is text-only would measure
        nothing and could only produce a false finding.
        """
        if not ctx.adapter.capabilities.vision or ctx.reference is None:
            return False
        modalities = {m.strip().lower() for m in ctx.reference.modalities}
        return bool(modalities & {"image", "vision", "images"})

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        rng = ctx.rng("vision")
        tasks = _build_tasks(rng)

        cost = 0.0
        tokens = 0
        rejection: str | None = None
        truncated = False

        for task in tasks:
            try:
                ctx.budget.check()
            except BudgetExhausted:
                truncated = True
                break

            response, error = await ctx.adapter.try_chat(
                ChatRequest(
                    messages=(
                        Message(
                            Role.USER,
                            (ImagePart(task.image, "image/png"), TextPart(task.prompt)),
                        ),
                    ),
                    max_tokens=self.max_tokens,
                )
            )
            if response is None:
                ctx.budget.charge(None, None)
                task.error = redact(str(error))[:220]
                if rejection is None and _is_image_refusal(error):
                    rejection = task.error
                continue

            cost += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            tokens += response.usage.total_tokens or 0
            text = response.text or ""
            task.answer = redact(text)[:200]
            task.disclaimed = any(marker in text.lower() for marker in _BLIND_DISCLAIMERS)
            task.correct = _grade(task, text)

        elapsed = time.perf_counter() - started
        data: dict[str, Any] = {
            "tasks": {task.key: task.as_dict() for task in tasks},
            "sighted_accuracy_assumed": SIGHTED_ACCURACY,
            "blind_accuracy_chance": {k: round(v, 4) for k, v in BLIND_ACCURACY.items()},
        }
        charged = {"cost_usd": cost, "tokens": tokens, "duration_s": elapsed}

        if rejection is not None:
            return [self._rejected(ctx, rejection, data, charged)]
        return [self._sight(ctx, tasks, data, truncated, charged)]

    # ------------------------------------------------------------ interpretation

    def _rejected(
        self,
        ctx: ProbeContext,
        rejection: str,
        data: dict[str, Any],
        charged: dict[str, Any],
    ) -> Evidence:
        return self._ev(
            "image_input_accepted",
            -STRONG,
            detail=(
                f"the endpoint refused image content: {rejection}. "
                f"{ctx.provider.target_model!r} is recorded as taking image input, so an "
                "endpoint that cannot accept an image is not serving it. This is about the "
                "endpoint as a whole -- a text-only model behind a multimodal name, or a "
                "proxy that never implemented the image path."
            ),
            data={**data, "rejection": rejection},
            **charged,
        )

    def _sight(
        self,
        ctx: ProbeContext,
        tasks: list[_Task],
        data: dict[str, Any],
        truncated: bool,
        charged: dict[str, Any],
    ) -> Evidence:
        """Weigh the graded answers against sighted and blind likelihoods."""
        graded = [task for task in tasks if task.correct is not None]
        if not graded:
            return self._ev(
                "image_comprehension",
                0.0,
                status=EvidenceStatus.TRUNCATED if truncated else EvidenceStatus.ERROR,
                detail=(
                    "no image task produced a gradable answer, so nothing can be said about "
                    "whether the endpoint sees images."
                ),
                data=data,
                **charged,
            )

        llr = 0.0
        for task in graded:
            sighted = SIGHTED_ACCURACY[task.kind]
            blind = BLIND_ACCURACY[task.kind]
            if task.correct:
                llr += llr_from_probability(sighted, blind)
            else:
                llr += llr_from_probability(1.0 - sighted, 1.0 - blind)

        correct = sum(1 for task in graded if task.correct)
        disclaimed = [task.key for task in graded if task.disclaimed]
        data = {
            **data,
            "graded": len(graded),
            "correct": correct,
            "blind_disclaimers": disclaimed,
            "llr_before_cap": round(llr, 3),
        }
        status = EvidenceStatus.TRUNCATED if truncated and len(graded) < 2 else EvidenceStatus.OK

        summary = (
            f"the endpoint accepted every image and answered {correct} of {len(graded)} "
            "tasks correctly"
        )
        if correct == 0:
            detail = (
                f"{summary}, which is what a model that never saw them would score. "
                "Accepting an image and seeing an image are different claims: a proxy can "
                "take the image part, drop it, and forward the text to a text-only model, "
                "and every part of the response except the answers looks normal."
            )
            if disclaimed:
                detail += (
                    " The endpoint said in as many words that it could not see an image in "
                    f"{len(disclaimed)} of the replies."
                )
        elif correct == len(graded):
            detail = (
                f"{summary}: the colour, the square count and the odd cell's row and column "
                "were all right on images generated for this run, so the endpoint is "
                "genuinely reading pixels."
            )
        else:
            detail = (
                f"{summary}. Partial sight is the ambiguous case -- counting and spatial "
                "indexing are where sighted models genuinely slip -- so the weight follows "
                "the assumed per-task accuracies rather than a verdict."
            )

        return self._ev(
            "image_comprehension",
            llr,
            status=status,
            detail=detail,
            data=data,
            **charged,
        )

    # ----------------------------------------------------------------- helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        status: EvidenceStatus = EvidenceStatus.OK,
        detail: str = "",
        data: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        tokens: int = 0,
        duration_s: float = 0.0,
    ) -> Evidence:
        return Evidence(
            probe=self.name,
            label=label,
            llr=llr,
            cap=STRONG,
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# PNG writing
# --------------------------------------------------------------------------- #


def png(width: int, height: int, pixel: Callable[[int, int], tuple[int, int, int]]) -> bytes:
    """Encode an RGB PNG from a function of ``(x, y)``.

    Eight-bit truecolour, no interlacing, and filter type 0 on every scanline.
    Filtering exists to help compression, and these images are flat colour
    blocks that zlib already handles well, so the simplest correct encoder is
    also a perfectly good one.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            red, green, blue = pixel(x, y)
            raw.extend((red & 0xFF, green & 0xFF, blue & 0xFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            _chunk(b"IEND", b""),
        )
    )


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """One length-tag-payload-CRC PNG chunk."""
    return b"".join(
        (
            struct.pack(">I", len(payload)),
            tag,
            payload,
            struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF),
        )
    )


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #

_WHITE = (255, 255, 255)
_INK = (25, 45, 105)
_GRID_BASE = (200, 205, 215)
_GRID_ODD = (205, 45, 45)


def _build_tasks(rng: random.Random) -> list[_Task]:
    """Generate the three images and their questions for this run's seed."""
    return [_colour_task(rng), _count_task(rng), _grid_task(rng)]


def _colour_task(rng: random.Random) -> _Task:
    name, rgb, _ = PALETTE[rng.randrange(len(PALETTE))]
    image = png(256, 256, lambda x, y: rgb)
    prompt = (
        "This image is one solid colour. Which colour is it? Choose from red, green, "
        "blue, yellow, purple and orange, and reply with that one word formatted "
        'exactly as "ANSWER: <colour>".'
    )
    return _Task("colour", "colour", image, prompt, name)


def _count_task(rng: random.Random) -> _Task:
    """A white field with ``k`` well-separated squares, ``k`` between three and six."""
    size = 384
    cell = 96
    inset = 18
    count = rng.randrange(3, 7)
    occupied = frozenset(rng.sample(range(16), count))

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        index = (y // cell) * 4 + (x // cell)
        if index not in occupied:
            return _WHITE
        within_x = x % cell
        within_y = y % cell
        if inset <= within_x < cell - inset and inset <= within_y < cell - inset:
            return _INK
        return _WHITE

    prompt = (
        "Count the filled squares in this image. Reply with the count alone as a "
        'digit, formatted exactly as "ANSWER: <number>".'
    )
    return _Task("count", "count", png(size, size, pixel), prompt, str(count))


def _grid_task(rng: random.Random) -> _Task:
    """A three-by-three grid with exactly one cell in a different colour."""
    size = 384
    cell = size // 3
    gutter = 8
    row = rng.randrange(3)
    column = rng.randrange(3)

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        if x % cell < gutter or y % cell < gutter:
            return _WHITE
        if (y // cell, x // cell) == (row, column):
            return _GRID_ODD
        return _GRID_BASE

    prompt = (
        "This image is a 3 by 3 grid of coloured cells. Exactly one cell is a "
        "different colour from the other eight. Give its row and column, both counted "
        'from 1 starting at the top left, formatted exactly as "ANSWER: <row>,<column>".'
    )
    return _Task("grid", "grid", png(size, size, pixel), prompt, f"{row + 1},{column + 1}")


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def _grade(task: _Task, text: str) -> bool:
    """Exact grading per task kind, tolerant of formatting but not of guessing."""
    answer = _extract(text) or text
    if task.kind == "colour":
        return _grade_colour(answer, task.expected)
    if task.kind == "count":
        number = _first_number(answer)
        return number is not None and str(number) == task.expected
    return _grade_grid(answer, task.expected)


def _extract(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip().strip("*`\"'.") if matches else None


def _grade_colour(answer: str, expected: str) -> bool:
    """The expected colour must be named, and no other palette colour may be.

    The second half matters: a model hedging with "red or orange" has not
    identified the colour, and counting that as correct would hand a blind
    endpoint a way to score above chance by listing options.
    """
    lowered = answer.lower()
    named = {
        name
        for name, _, words in PALETTE
        if any(re.search(rf"\b{word}\b", lowered) for word in words)
    }
    return named == {expected}


def _first_number(answer: str) -> int | None:
    match = re.search(r"\d+", answer)
    if match:
        return int(match.group())
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", answer.lower()):
            return value
    return None


def _grade_grid(answer: str, expected: str) -> bool:
    numbers = [int(n) for n in re.findall(r"\d+", answer)]
    if len(numbers) < 2:
        return False
    return f"{numbers[0]},{numbers[1]}" == expected


def _is_image_refusal(error: Exception | None) -> bool:
    """Whether a failed request was refused specifically over the image part."""
    status = getattr(error, "status", None)
    if status is None or not 400 <= int(status) < 500:
        return False
    haystack = str(error).lower()
    body = getattr(error, "body", None)
    if isinstance(body, str):
        haystack += " " + body.lower()
    return any(marker in haystack for marker in _IMAGE_REFUSAL_MARKERS)
