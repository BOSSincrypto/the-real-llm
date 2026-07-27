# Probes

A probe is one self-contained experiment against an endpoint. It turns
observations into `Evidence`: a signed log-likelihood ratio, a family, a cap, a
one-line human explanation and a machine-readable payload. Nothing else in the
package decides anything — the aggregator adds the numbers up, the reporters
show them.

List them at any time:

```console
$ llmverify probes
$ llmverify probes --json probes.json
```

## How to read the tables below

**Layer** is roughly what it costs. Layer 0 is free metadata. Layer 1 is cheap
fingerprints — fractions of a cent. Layer 2 buys seconds of generation. Layer 3
costs real money. Probes run cheapest first, so a blatant substitution is
settled before anything expensive starts.

**Requests** is the probe's declared `estimated_requests`, used for budget
planning and ordering. The real number varies: several probes stop early, and
several shrink to fit the remaining budget.

**Family** groups probes that measure overlapping things. Within a family the
LLRs are summed and then damped toward the family cap with
`cap * tanh(total / cap)`, so five tokenizer measurements saturate instead of
compounding into five independent opinions.

**Cap** is that family ceiling, expressed in *bans* — base-10 log-likelihood
ratios, one ban being 10:1. The named magnitudes are:

| name | nats | bans | likelihood ratio |
|---|---|---|---|
| `WEAK` | 1.10 | 0.48 | 3:1 |
| `MODERATE` | 2.30 | 1.00 | 10:1 |
| `STRONG` | 4.61 | 2.00 | 100:1 |
| `DECISIVE` | 9.21 | 4.00 | 10,000:1 |

A verdict needs the posterior to cross 0.99 for `MATCH` or 0.01 for `MISMATCH`,
which from even prior odds is ±2 bans. So a `WEAK` family cannot reach a verdict
even when everything in it agrees; that is the intended property, not a
limitation to be worked around.

## Summary

| probe | layer | requests | family | family cap |
|---|---|---|---|---|
| `metadata` | 0 | 1 | `metadata` | MODERATE (1.00 ban) |
| `thinking_signature` | 1 | 3 | `cryptographic` | DECISIVE (4.00) |
| `token_accounting` | 1 | 6 | `token_accounting` | DECISIVE (4.00) |
| `api_surface` | 1 | 8 | `api_surface` | STRONG (2.00) |
| `self_report` | 1 | 3 | `self_report` | WEAK (0.48) |
| `determinism` | 1 | 5 | `determinism` (+ `misc`) | WEAK (0.48) / MODERATE |
| `tokenizer` | 1 | 11 | `tokenizer` | STRONG (2.00) |
| `knowledge_cutoff` | 1 | 7 | `knowledge` | MODERATE (1.00) |
| `long_context` | 2 | 9 | `long_context` | STRONG (2.00) |
| `tool_calling` | 2 | 5 | `tool_use` | MODERATE (1.00) |
| `structured_output` | 2 | 4 | `capability` | STRONG (2.00) |
| `vision` | 2 | 3 | `vision` | STRONG (2.00) |
| `multilingual` | 2 | 8 | `multilingual` | MODERATE (1.00) |
| `performance` | 2 | 5 | `performance` | WEAK (0.48) |
| `logprobs` | 3 | 24 | `distribution` | STRONG (2.00) |
| `benchmark` | 3 | 60 | `benchmark` | DECISIVE (4.00) |
| `evasion` | 3 | 70 | `evasion` | DECISIVE (4.00) |

Select or drop probes by name:

```console
$ llmverify check provider.yaml --probe token_accounting --probe api_surface
$ llmverify check provider.yaml --exclude-probe long_context
```

Naming probes explicitly disables the runner's early stop, since you asked for
those specific probes.

---

# Layer 0

## `metadata`

**Measures.** One 16-token completion plus one catalogue read. From them: the
echoed `model` field, the response-id prefix, the shape of the usage object, the
finish-reason vocabulary, `system_fingerprint`, infrastructure headers, the
catalogue's shape, and — when the endpoint is OpenRouter — every upstream
provider's self-declared quantization. Its response is left in
`ctx.shared["warmup_response"]` so no later probe pays for the same information
twice.

**Cost.** One request. Free in any practical sense.

**Cap.** `metadata`, MODERATE. Individual items are mostly WEAK or zero; the one
item that reaches STRONG is an echoed model id that the snapshot resolves to a
*different* model.

**Reads.** The response envelope — id prefix, usage keys, finish reasons,
`system_fingerprint` — is produced by whatever software answered the HTTP
request, not by the weights. It is weighed only when the claimed model's own
first-party API is the protocol being spoken. A reseller fronting Claude behind
an OpenAI-compatible route rewrites the envelope legitimately, and is not
penalised for it.

**False positives.** Any legitimate proxy or gateway: it rewrites ids, drops
`system_fingerprint`, adds its own headers and serves its own catalogue. The
probe's design already discounts this to near zero, which is why the family cap
is MODERATE and why an adverse `metadata` finding never convicts alone.

**False negatives.** Everything here is trivially forgeable. An echoed model id
is a string the provider chose; a response-id prefix is four characters. A
reseller that takes ten minutes to make its envelope consistent passes this
probe completely. It runs first because carelessness is common, not because it is
hard to defeat.

---

# Layer 1

## `thinking_signature`

**Measures.** Anthropic returns extended-thinking blocks with a `signature`
field, documented as existing so the API can verify a thinking block was
generated by Claude when it is passed back; a tampered signature is rejected with
a 400. The probe elicits a thinking block, replays it verbatim, then replays it
again with the middle of the base64 signature corrupted to the same length. A
genuine endpoint must accept the first and reject the second.

**Cost.** Up to 3 requests. Stops as soon as one step settles the question.

**Cap.** `cryptographic`, DECISIVE. Individual items run MODERATE to STRONG.

**Why it is the strongest identity test available for the Claude family.** A
reseller that is not routing through genuine Anthropic inference cannot produce a
block that survives replay, and cannot tell a good signature from a corrupted
one. Signatures are portable across Anthropic, Bedrock and Vertex, so a
first-party gateway on any of the three still passes.

**The three negative outcomes are not equivalent.**

- *No thinking blocks at all* is weakest — a capability gap, not proof of a
  different model. MODERATE.
- *Thinking blocks with no signature* is stronger: the feature exists and the
  field that makes it verifiable is absent. STRONG rather than conclusive,
  because some legitimate proxies strip or re-sign blocks.
- *A corrupted signature accepted* is the single most conclusive negative result
  this tool can produce. It is only ever emitted after the verbatim replay
  succeeded, so the endpoint demonstrably tolerates the replay structure.

**False positives.** A proxy that strips or re-signs thinking blocks with its own
key looks exactly like one that never had a real signature. And an endpoint that
ignores replayed thinking blocks entirely — neither validating nor using them —
accepts both the intact and the corrupted replay for a reason that has nothing to
do with forgery. That alternative is named in the evidence detail every time the
finding fires.

**False negatives.** Any endpoint that genuinely routes to Anthropic passes, even
if it substitutes a different model for every request that does not enable
thinking. The probe verifies the inference path, not the model on it.

**Deliberately scored at zero.** A verbatim replay rejected with a 400 has two
explanations — the endpoint refused a genuine block, or this tool built the
replay wrongly — and from outside they are indistinguishable. That case stops the
probe and contributes nothing at all.

## `token_accounting`

**Measures.** Anthropic publishes the exact token cost of the system prompt the
API injects when a request carries tools, and that integer differs by model
generation: 286 for one, 354 for another, 675 for a third, 496 for a fourth. The
probe recovers it by double differencing — a no-tool baseline removes the request
envelope, and a second tool identical but for its name removes the tool
definition — leaving the system prompt alone.

**Cost.** 6 requests, on the free token-counting path where the adapter has one.

**Cap.** `token_accounting`, DECISIVE. Individual items are MODERATE (the
system-prompt residual) or WEAK (the forced-choice delta).

**Why the family cap is DECISIVE.** Almost every other probe measures a
behaviour, and behaviours drift. This one measures an integer that a model
generation either produces or does not. Tolerance is ±2 tokens, against a table
whose closest pair differs by four — so the slack costs no discriminating power.
A residual matching a *different* model in the snapshot is the most useful
sentence this tool can produce: not "not what you claimed", but which model it
actually is.

**False positives.** A proxy that injects its own system prompt inflates every
measurement equally. That cancels in the differencing, and is measured and
reported anyway because it is the most common innocent explanation for a residual
matching nothing. A proxy that re-serialises the tool schema with different
whitespace moves the number by a token or two, which the tolerance absorbs.

**False negatives.** The residual is computed from numbers the provider reports
about its own usage. A provider willing to fabricate `usage.input_tokens`
consistently across four different requests defeats it — at which point every
token-based measurement in this package is defeated too. The probe also does not
run at all when the claimed model has no published tool-use overhead, which is
every non-Anthropic model in the snapshot.

## `api_surface`

**Measures.** Which of eight request parameters the endpoint accepts, rejects, or
accepts and silently drops: `logprobs`, `seed`, `prompt_logprobs`,
`response_format` with a JSON schema, `logit_bias`, `min_p`, `top_k`,
`presence_penalty`. Where the effect can be observed the probe checks for it, so
a 2xx with no effect is reported as `IGNORED` rather than as support.

**Cost.** 8 requests, all short.

**Cap.** `api_surface`, STRONG. The decisive item — a Claude claim returning real
logprobs — is capped at STRONG on its own.

**Three distinctions carry the weight.**

- *Accepted versus ignored.* Several ignored parameters at once is the signature
  of a LiteLLM front end running `drop_params=True`. That is a fact about the
  serving stack, not about the weights, and is reported at zero LLR.
- *Logprobs against a Claude claim.* Anthropic's protocol has no logprobs and no
  seed. A translating proxy can accept a `seed` field and throw it away, so
  acceptance there is only WEAK. It cannot manufacture per-token logprobs without
  running a model that produces them — and unusually for this package, that
  evidence is *not* weakened by the endpoint speaking an OpenAI-compatible
  protocol, because the observation is about what computed the tokens rather than
  how they were packaged.
- *`prompt_logprobs`* — logprobs over the input tokens — is a vLLM extension.
  SGLang spells its equivalent `return_logprob` / `top_logprobs_num` and has no
  `prompt_logprobs` at all. Support says vLLM, which is neutral for open weights
  and hard to reconcile with a claim to serve closed ones.

**False positives.** A gateway that adds its own parameter validation in front of
a genuine upstream will report rejections the upstream would have accepted. The
parameter matrix itself is therefore reported as describing *the protocol being
spoken*, not the claimed model's own API.

**False negatives.** A reseller that implements a faithful rejection table
defeats everything except the logprobs finding, and can defeat that too by simply
not returning logprobs.

## `self_report`

**Measures.** Asks the endpoint to identify itself under three different
framings, and extracts a vendor name with a conservative regex that honours
negation ("I am not GPT, I am Claude" reads as Claude).

**Cost.** 3 requests.

**Cap.** `self_report`, WEAK — and every individual item is WEAK too. Three
probes agreeing here move the posterior about as much as one response-id prefix.

**Why it is weighted at almost nothing.** A system prompt overrides it
completely: anything upstream of the model can make it assert any identity, and
it will do so confidently. There is nothing to detect, because the model is not
lying — it is answering the question it was given. Distillation corrupts it in
the other direction: a model trained on another lab's outputs inherits that lab's
self-descriptions and will sincerely identify as a competitor's.

**Asymmetry.** A *correct* self-report is nearly worthless. A *confidently wrong*
one, naming a different vendor outright, is worth slightly more — not because it
proves substitution, but because a provider claiming model X has had every chance
to stop its endpoint saying otherwise. Disagreement between phrasings is worth
noting on its own.

**False positives.** One injected line of system prompt. Distilled models.
**False negatives.** One injected line of system prompt.

## `determinism`

**Measures.** Five sequential samples of one prompt at temperature 0 with a
pinned seed. Reports repeat agreement, mean normalised edit distance and output
length variance.

**Cost.** 5 requests, sent sequentially — concurrent identical requests tend to
land in one batch, which is exactly the condition under which outputs agree for
reasons unrelated to the model.

**Cap.** `determinism`, WEAK. The cached-response signature is filed under `misc`
(MODERATE) instead, deliberately, so it is not damped along with a drift
measurement that is measuring something else entirely.

**Drift is expected and is not evidence.** A forward pass at fixed batch
composition is deterministic; production inference is not, because dynamic
batching makes your result depend on how many other requests were batched
alongside it. Floating-point addition is not associative and GPU reduction
kernels pick their tiling from the runtime batch shape. Every hosted provider
disclaims bitwise determinism, and Anthropic states that even at temperature 0
results will not be fully deterministic. High variability therefore contributes
an LLR of essentially zero — it is reported because a human wants the number.

Perfect bitwise stability is very weak *positive* evidence at most, and only
indirectly: batch-invariant kernels are opt-in and cost 25–55% of throughput,
which is not an expense a reseller cutting corners on weights would choose.

**What is actually worth weight** is *degenerate* sameness: every sample
byte-identical and suspiciously short, or repeat latency collapsing toward zero.
That is a cache or a lookup table answering instead of a model.

**False positives.** A genuinely terse model answering a short prompt identically
five times trips the cached-response signature. The evidence detail says so, and
distinguishes the latency-collapse case from the identical-text case.

**False negatives.** A cache with jitter added to its latency, or one that varies
whitespace, passes.

## `tokenizer`

**Measures.** Token count of `CANONICAL_TEXT`, a fixed Latin / Cyrillic / CJK /
Arabic / emoji / code / digit string, plus per-script ratios per segment. Two
paths: a real token-counting endpoint where the adapter has one (Anthropic,
Gemini), and usage differencing otherwise, with a calibration step that cancels
the request envelope exactly.

**Cost.** 11 requests. Free on the token-counting path.

**Cap.** `tokenizer`, STRONG; individual items STRONG.

**Where the weight actually comes from.** Claude 4.7 and later, including Fable
and Mythos, produce roughly 30% more tokens for identical text than 4.6 and
earlier. `GENERATION_RATIO` is 1.30 with an 8% band. An endpoint claiming a
post-4.7 model whose canonical count sits near the claimed value divided by 1.30
is running a pre-4.7 tokenizer, and no prompt engineering changes that. The check
only fires *within one vendor* — across vendors a 30% difference means nothing at
all.

The per-script profile is reported at zero LLR. No reference profile exists to
compare it against, so it is an honest measurement a human can read, not a guess
dressed as evidence. Same for the canonical count when the snapshot records no
`canonical_text_tokens` for the claimed model.

**False positives.** A proxy whose injected system prompt varies per request
breaks the differencing assumption. A stack that normalises Unicode before
tokenizing shifts the count.

**False negatives.** Two models sharing a tokenizer are indistinguishable here,
which is most substitutions within one vendor generation.

## `knowledge_cutoff`

**Measures.** Binary search over eleven dated rungs, each a fact with an exact
machine-checkable answer that could not be known before its date. Reports the
*interval* between the latest rung answered correctly and the earliest one
answered wrongly, never a point estimate.

**Cost.** About 7 requests — four for the search, plus outright confirmation of
the two rungs bracketing the transition, because a single flaky answer at the
transition is the one error that would move the estimate.

**Cap.** `knowledge`, MODERATE; individual items MODERATE.

**Design details that matter.** Every rung comes from a verified source; facts
whose first-public date could not be pinned were left out rather than dated by
guesswork, so the ladder is sparse and has a gap between November 2025 and March
2026 that the estimate simply cannot resolve. Rungs naming the *claimed model
itself* are dropped before the search starts: vendors post-train models to know
their own names and release dates, sometimes for events after the training
cutoff, so such a rung would be answered correctly by the genuine article and
read as knowledge from beyond its own boundary — a false accusation against
exactly the endpoint we are trying to clear. Grading is exact match against a
small accepted-answer set, never a judge model. Non-monotone results are reported
rather than smoothed away.

**False positives.** A retrieval-augmented proxy answers post-cutoff questions
legitimately and looks like a forgery here. Models are unreliable narrators about
events near their boundary, recalling some and not others essentially at random.

**False negatives.** A model that declines to answer is indistinguishable, from
outside, from one that never knew. Two models with the same cutoff are
indistinguishable, and cutoffs cluster.

---

# Layer 2

## `long_context`

**Measures.** Needle-in-a-haystack retrieval at 10%, 50% and 90% depth over a
ladder of prompt sizes. The haystack is numbered lines of random words: trivial
to generate at megabyte scale, carrying no meaning a model could reconstruct from
priors, and not compressible into a summary — an endpoint that silently
summarises a long prompt before feeding it to a smaller model cannot preserve a
random line through that step. The needle is a random ten-character key.

**Cost.** 9 requests nominally, but this probe can cost more than every other
probe combined: one 1M-token rung at flagship input pricing is several dollars.
It refuses to spend more than `BUDGET_SHARE` (0.5) of the run's cost ceiling,
degrades to a single middle-depth probe when that is all it can afford, refuses
to send more than `UNPRICED_TOKEN_CEILING` (512,000) tokens when no pricing is
known, and reports `TRUNCATED` rather than overspending.

**Cap.** `long_context`, STRONG; individual items STRONG.

**Why three depths.** A stack that silently keeps only the last N tokens passes
at 90% and fails at 10% — a signature no prompt engineering imitates. The middle
is where lossy long-context attention degrades first, so a genuinely
long-context-capable model running at a reduced KV budget fails there first.

**Three failure modes, told apart.** An explicit context-length error (honest,
and gives a hard ceiling); acceptance with silent truncation (visible as
`usage.input_tokens` far below what was sent); acceptance with no retrieval.
Each implicates a different part of the stack.

**No statistics are used, deliberately.** When a window is genuinely below its
claim, retrieval collapses from near-certain to near-impossible over one rung. A
few samples settle it; sequential testing would spend budget sharpening a
decision that is not close.

**False positives.** A first-party endpoint under a context-management or
compaction feature legitimately does not see the whole prompt.

**False negatives.** An endpoint that serves the full window but a worse model
inside it passes completely — retrieval is easy.

## `tool_calling`

**Measures.** Five short experiments: tool-call emission on an unambiguous
prompt, parallel calls, `tool_choice` compliance, argument well-formedness, and
adherence to an enum/nested/closed schema.

**Cost.** 5 requests.

**Cap.** `tool_use`, MODERATE; individual items WEAK.

**The stack matters more than the weights here, and the weighting says so.** The
field evidence is unambiguous: identical open weights served by different
providers produced wildly different tool reliability, and the cause was serving
software — old builds, wrong defaults, home-grown parsers for the model's call
syntax — far more often than quantization or a different checkpoint. Malformed
`arguments_raw` in particular is a parser bug, so it is reported with almost no
weight and with that explanation in the evidence detail.

The reference snapshot records no per-model tool expectations, so most of this
probe is a profile rather than a verdict. Two findings are model-independent
enough to weigh: no tool call at all for a single unambiguous prompt, and a call
emitted after `tool_choice: "none"`.

**False positives.** Any serving stack with a home-grown tool-call parser.
**False negatives.** A competent stack in front of a weak model passes.

## `structured_output`

**Measures.** A JSON Schema response format against a prompt that argues with the
schema — an incident whose severity is not in the enum, whose downtime exceeds
the numeric bound, and which volunteers a fact with nowhere to live under
`additionalProperties: false`. Grades into *enforced* (constrained decoding: the
output cannot violate the schema), *best effort* (the model was asked nicely and
usually complies), or *ignored* (2xx and prose — the LiteLLM `drop_params`
shape).

**Cost.** `SAMPLES` (3) + 1 requests.

**Cap.** `capability`, STRONG; individual items STRONG.

**Why the prompt pulls against the schema.** Without that tension the two
outcomes look identical: a cooperative model given an easy schema produces valid
output either way.

**Two schemas.** A 4xx on the full schema is retried once with a core schema
using only `type`, `required`, `enum`, `properties`, `items` and
`additionalProperties`. Without that retry an endpoint that enforces schemas
properly but implements a narrower keyword set would be recorded as rejecting
structured output altogether — a false finding produced entirely by this tool's
choice of test schema.

Outright *rejection* of the parameter is deliberately not scored here; that
belongs to `api_surface`, and double-counting it would let one observation
contribute twice.

**False positives.** A provider that has simply not implemented constrained
decoding for a model that supports it upstream.
**False negatives.** Constrained decoding sits outside the weights entirely — an
endpoint can enforce a schema perfectly over any model at all.

## `vision`

**Measures.** Three procedurally generated PNGs, written in pure Python with
`zlib` and `struct`: a solid colour from a fixed palette, a count of
well-separated squares, and the position of the odd cell in a 3×3 grid. Chance is
1/6, 1/4 and 1/9.

**Cost.** 3 requests.

**Cap.** `vision`, STRONG; individual items STRONG.

**The images are generated fresh from the run's seed**, so they cannot have been
seen during training and cannot be recognised by a provider that special-cases
known probe images. No image assets ship in this repository and no Pillow
dependency is needed.

**"Accepts images" and "sees images" are different claims, and this probe reports
them separately.** A proxy can accept an image part, drop it, and pass the text
through to a text-only model: the request succeeds, the usage object looks
plausible, and the answers are guesses. An endpoint that *rejects* image content
while claiming a multimodal model is strong evidence against the claim, and so is
one that accepts it and answers at chance — the case a naive check misses
entirely.

**The likelihood model is stated, not hidden.** Each task contributes
`ln(P(observed | sighted) / P(observed | blind))` using assumed sighted accuracy
of 0.95 for colour and 0.85 for counting and grid position, against the exact
chance rate for a blind model. Those rates are assumptions, deliberately
conservative: underestimating a real model's accuracy shrinks the evidence
against an endpoint rather than inflating it.

**False positives.** Three items is a small sample. A sighted model having a bad
run on counting is not impossible, which is what the conservative sighted rates
are for.
**False negatives.** Any multimodal model passes, including a much weaker one.

## `multilingual`

**Measures.** Two separate things, reported separately.

*Unicode integrity* is the important half: a fixed string — Han, kana, hangul,
Arabic and Hebrew RTL runs, decomposed combining marks, an Indic
consonant-vowel cluster, a Thai tone mark, a ZWJ emoji, a regional-indicator
pair — echoed back and compared character for character. There is no reasoning in
this task at all: any model that receives the string intact can return it intact.
Replacement characters, dropped combining marks or mangled CJK are a statement
about the pipeline the bytes travelled through, and CJK corruption under
low-precision routing is a failure observed in the field, not a hypothetical.

*Task accuracy* is the weaker half: one sentence translated into Spanish,
Russian, Chinese and Arabic, graded for required content words, plus a factual
question with a number or proper-noun answer asked in three of those. The
Spanish arm is a control — an endpoint that handles Spanish and fails the other
three has a script problem, one that fails all four has a capability problem.

**Cost.** 8 requests.

**Cap.** `multilingual`, MODERATE; individual items MODERATE.

**Scored at zero on purpose.** Normalisation-only differences: a reply that
differs from the original only by Unicode normalisation form has lost nothing,
and calling that corruption would accuse an honest endpoint of something a linter
did. Also which script the answer comes back in — answering a Chinese question in
English is a post-training preference, not evidence about weights.

**False positives.** Any transport, proxy or logging layer in the path that
re-encodes text. The finding is about the pipeline, and the pipeline may not be
the model's.
**False negatives.** A substituted model that handles Unicode correctly, which
most do.

## `performance`

**Measures.** Time-to-first-token and output tokens per second over at least
`SAMPLES` (5) streamed generations of a fixed prompt, sent sequentially. TTFT is
taken at the transport layer at the first frame carrying content, so provider
pings and role frames do not flatter it. Median and interquartile range, not
mean — one slow sample from a cold route otherwise dominates a five-sample
average. Cost per correct answer is computed when benchmark results are present,
as an economics figure for a human, at zero LLR.

**Cost.** 5 streamed requests.

**Cap.** `performance`, WEAK; individual items WEAK. It cannot move a verdict on
its own, by construction. Three such probes could not reach a verdict between
them.

**Why it is nearly weightless.** Every innocent explanation for these numbers is
at least as plausible as the guilty one. Throughput far above a model's
first-party figure is what better hardware looks like, what an under-subscribed
cluster looks like, what a newer serving stack looks like, and what a shorter
output looks like. A price far below the official one is what a loss leader looks
like, what committed-capacity pricing looks like, and what a different cost base
looks like. First-party throughput also moves without announcement as vendors
change their own infrastructure.

Individual measurements are reported at zero LLR. Only the *joint* observation —
throughput far above the claimed model's typical rate *together with* a price far
below its official one — carries any weight at all.

**False positives.** Better hardware, quiet hours, a shorter answer.
**False negatives.** A reseller pricing at parity and rate-limiting to match.

---

# Layer 3

## `logprobs`

**Measures.** With a baseline: three tests comparing what two endpoints report
for the same prompts. Primary is a two-sample Kolmogorov–Smirnov test on pooled
chosen-token logprobs. Secondary are a chi-square on which token the candidate
chose, categorised by that token's rank in the baseline's top-k; and a
rank-uniformity check randomised within its own atom. Positions are compared only
up to and including the first token where the two completions diverge — after
that the two models are continuing different prefixes, so position *i* is not the
same random variable on both sides.

Without a baseline: only that the endpoint's own numbers form a distribution —
entries sorted descending, chosen token present among the alternatives, values at
or below zero, an entropy profile that is not obviously manufactured. OpenAI's
`-9999.0` sentinel for untracked tokens is excluded from every statistic rather
than averaged in as a probability of 1e-4343.

**Cost.** 24 requests (2 × the prompt set), 24 max tokens each, `top_logprobs` 5,
temperature 1.0.

**Cap.** `distribution`, STRONG. Every individual item is capped at MODERATE and
then further multiplied by `CONFIGURATION_DISCOUNT` (0.5).

**`UNSUPPORTED` is the ordinary outcome.** Anthropic's protocol has no logprobs
and no seed, Gemini's are unconfirmed, and reasoning models across vendors
decline them in practice. That is reported as an ordinary outcome, not a failure.

**Temperature is 1.0 on purpose.** All three tests treat the emitted token as a
draw from the reported distribution; a greedy decode would make it the argmax by
construction and both rank tests would reject the null against every endpoint,
honest ones included. Since an endpoint may ignore the temperature it was sent,
the assumption is *checked*: if either side emits its most likely token
materially more often than its own probabilities predict, the rank tests are
skipped and said to be skipped, and only KS is reported.

**What a significant result is worth.** arXiv:2504.04715 is blunt about the
ceiling: methods using log probabilities "are defeated by inherent inference
nondeterminism in production". Batch composition, kernel selection, quantization
and speculative decoding all move a logprob without touching a weight, and
providers change all four without notice. A significant KS result says the two
endpoints are running *different serving configurations* — which may mean
different weights, different precision, or a different GPU generation on a
Tuesday.

**False positives.** Different hardware, a different batch regime, a
quantization change on either side, speculative decoding on one side only.
**False negatives.** No logprobs, no test. Which is most endpoints.

The complementary inference — a Claude claim returning logprobs at all — belongs
to `api_surface` and is deliberately not repeated here. Counting one observation
in two families would defeat the damping.

## `benchmark`

**Measures.** Accuracy against the claimed model's published score, tested
sequentially with an SPRT. Items are drawn without replacement from a seeded
shuffle and fed in one at a time, so a blatant substitution settles in a few
dozen questions and a subtle one keeps sampling.

**Cost.** 60 requests nominally, up to `MAX_ITEMS_PER_BENCHMARK` (150), at 2048
max tokens each. This is where the money goes.

**Cap.** `benchmark`, DECISIVE; individual items DECISIVE or STRONG.

**Benchmark choice decides everything, and the famous ones are useless.** GPQA
Diamond cannot separate 2026 frontier models: published scores cluster between 87
and 93, and separating Claude Opus 5 (91.8) from GPT-5.6 Sol (91.3) at 95%
confidence and 80% power needs about 48,500 items per arm against the 198 that
exist. SimpleQA Verified spans 67.7 points and separates the same pair in about
47 items; ARC-AGI-2 spans 87.6 and does it in about 7. Benchmarks are ranked by
information per request, and when the only reference score available sits on a
saturated benchmark the user is told so in as many words. A non-significant
result there means the benchmark is too coarse — never that the endpoint is
genuine.

**Conditions, not just numbers.** Reasoning effort alone moves scores by tens of
points: DeepSeek reports V4-Pro at 90.1 on GPQA Diamond in Think-Max and 72.9 in
Non-Think. When the provider pins an effort the reference score was not measured
at, the comparison is invalid as stated. Rather than refuse, the probe widens its
tolerance by `EFFORT_MISMATCH_ALLOWANCE_PP` (15.0 pp) and says why. Scores
published under benchmark-optimised settings get a further
`OPTIMIZED_ALLOWANCE_PP` (3.0 pp). Both allowances are judgement calls, not
measurements, and are labelled as such in the evidence data.

Where several sources publish a score for the same model under the same
conditions they disagree — Opus 5 on ARC-AGI-2 is 90.4 by ARC Prize and 88.3 by
Epoch AI — so the null is set at the *lowest* published value in the matching
range. Disagreement between sources widens the benefit of the doubt instead of
silently becoming this tool's bias.

**Two kinds of wrong.** An item the grader could not extract an answer from is
not an item answered incorrectly. Extraction failures are excluded from the SPRT
and reported with their own rate, because a provider mangling the output format
is failing differently from one reasoning badly — and because the reference score
was measured by a harness whose extractor worked. Exclusion is not free: if an
endpoint mangles output only on hard items, excluding those inflates the observed
accuracy. That is why the rate is evidence in its own right.

**Truncation is not a decision.** A test that hits the sample, cost or wall-clock
ceiling before crossing a boundary reports `TRUNCATED`, never whichever boundary
happened to be nearer. Forcing a decision destroys the error guarantees the whole
design exists to provide.

**False positives.** A different harness, a different effort, a different
system prompt, a stricter grader than the lab's. This package grades SimpleQA by
string matching rather than with a judge model, which systematically *under*-
scores every endpoint by an unmeasured amount — the direction that produces
missed detections rather than false accusations, which is the safer error.
**False negatives.** A provider that routes benchmark-shaped inputs to the real
model, which is what `evasion` exists to catch. A substitution too small to
detect at the sample size you paid for.

## `evasion`

**Measures.** The same underlying questions put twice: once verbatim, as
published and therefore string-matchable against any corpus, and once
paraphrased — same question, same gold answer, no longer equal or nearly equal to
anything in a lookup table. An endpoint whose behaviour does not depend on
recognisability scores the same on both arms within sampling noise.

Independently, a *structural* check: the same question sent with and without a
marker a naive detector would key on — a canary-style header and the distinctive
`ANSWER:` instruction format published harnesses use — comparing response length
and latency between the arms.

**Cost.** Up to 70 requests, but usually free: when `benchmark` has already run,
its per-item results with the variant recorded are already in
`ctx.shared["benchmark_results"]` and this probe simply reads them. Only when
that record is missing or too small does it buy its own canary split, defaulting
to `CANARY_ITEMS` (30) items per arm and shrinking to fit the remaining budget.

**Cap.** `evasion`, DECISIVE. The accuracy items are DECISIVE, the structural
items WEAK.

**Why it is not on the match/mismatch axis.** A strong finding here promotes the
run to the `EVASION` verdict rather than folding into the posterior. It does not
say the endpoint is worse than claimed; it says the measurements were taken under
conditions the provider chose, so every other number in the run is suspect.

**Paired beats unpaired.** When the same items appear in both arms — which the
canary split arranges deliberately — McNemar's test conditions on the items where
the arms disagreed, which is where all the information lives. `two_proportion_z`
is the fallback for the interleaved case.

**The null is not zero.** Paraphrasing perturbs difficulty a little, and the
neutral framing sentences the paraphraser adds can shift accuracy on their own; a
few points either way is the expected honest result. A gap of tens of points with
a small p-value is something else.

**Multiplicity.** Up to three tests, corrected with Holm–Bonferroni. Running
several comparisons at a nominal 5% and reporting whichever came out smallest is
exactly how a tool like this manufactures accusations against honest providers.
Holm is valid under any dependence structure, which matters because the accuracy
and structural tests are not independent.

**False positives.** A paraphrase that is genuinely harder than the original. A
provider whose cache serves verbatim items faster than paraphrased ones, which
moves latency without any routing decision. Length and latency differ between
prompts for entirely innocent reasons — which is why the structural check is
capped at WEAK.

**False negatives.** The structural marker is synthetic. No real dataset's canary
string ships here, so a detector keyed to a specific corpus will not fire — the
structural check is a lower bound on detectability, not a measurement of it. A
provider that routes on *semantics* rather than on string match survives
paraphrasing entirely. Turn the whole thing off with `--no-anti-evasion` only
when you are deliberately reproducing a published number.
