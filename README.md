# llmverify

[English](README.md) · [Русский](README.ru.md)

![licence Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Check whether a custom LLM provider actually serves the model it claims.**

You are paying frontier prices through a third-party endpoint, and you have no
way to know what answered your request. The model id in the response is a string
the provider chose. The gap this opens is not theoretical and it is not small: in
August 2025 the *identical* `gpt-oss-120b` weights scored 93.3% at six providers,
86.7% at Groq, 80.0% at Azure and **36.7% at CompactifAI** on the same AIME25
benchmark — a 56.6-point spread under one model name, caused mostly by old vLLM
builds silently defaulting `reasoning_effort` to medium. Quantization labels are
self-reported and unaudited: in one transparency audit 13 of 42 OpenRouter
endpoints — **31%** — declared their quantization as `unknown`, and seven
providers disclosed nothing at all. A NeurIPS 2025 result had to be invalidated
because its authors never pinned which provider served their model, and a
follow-up audit found 31 of 32 influential AI-safety codebases pinning nothing
either. Nobody in any of those cases was necessarily lying. That is the point:
you cannot tell.

llmverify probes an endpoint from the outside, gathers evidence across four
layers, weighs each piece as a log-likelihood ratio with an explicit cap, and
prints a verdict that shows its work. It never asks you to trust it — every
number in the report carries the observation it came from and the alternative
explanations that were considered.

It cannot prove which weights ran. **No software-only method can.** That is a
published result, not a missing feature, and [it has its own
page](docs/limitations.md).

## Install

Python 3.10 or newer. Four runtime dependencies — `httpx`, `pydantic`, `pyyaml`,
`rich` — and nothing else. No numpy, no scipy, no pandas, no `datasets`, no
`tiktoken`: every statistical test is pure Python against the standard library,
and benchmark items are read over the HuggingFace datasets-server REST API a page
at a time. A tool that audits other people's supply chains has no business adding
a long transitive tail to its own.

```console
$ git clone https://github.com/BOSSincrypto/the-real-llm.git
$ cd the-real-llm
$ pip install -e .
```

## 30-second quickstart

Three commands that cost nothing and need no key:

```console
$ llmverify probes          # every probe, its layer, its cost, what it measures
$ llmverify benchmarks      # every benchmark, and whether it can separate models
$ llmverify models opus     # what the reference snapshot knows about a model
```

Then point it at an endpoint. The key never appears on the command line — you
name the environment variable that holds it:

```console
$ export ACME_API_KEY=...
$ llmverify check \
    --api openai \
    --base-url https://api.acme-inference.example/v1 \
    --model opus5-turbo \
    --claimed-model claude-opus-5 \
    --api-key-env ACME_API_KEY \
    --layers 0,1
```

Layers 0 and 1 cost fractions of a cent and catch most careless substitutions. To
run everything with a spend ceiling, using a config file:

```console
$ llmverify check examples/openai-compatible.yaml --max-cost 0.25 --verbose
$ llmverify compare providers/*.yaml --layers 0,1,2 --json results.json
```

Exit codes are meant for CI:

| code | meaning |
|---|---|
| 0 | verified — `MATCH` or `LIKELY_MATCH` |
| 1 | substitution — `LIKELY_MISMATCH`, `MISMATCH` or `EVASION` |
| 2 | inconclusive — not enough evidence either way |
| 3 | usage error — bad configuration or command line |
| 4 | unreachable — no probe got a usable response |

Full command list: `check`, `compare`, `refresh`, `models`, `probes`,
`benchmarks`, `cache`. Every one takes `--help`.

## How it works

Seventeen probes across four layers. Probes run cheapest first, so a blatant fake
is settled before anything expensive starts; the run stops early only when the
evidence has crossed a threshold on at least three *independent* evidence
families, and every probe that never ran is recorded as skipped with its reason —
an early stop must never read like a clean bill of health.

| layer | cost | probes | what it can establish |
|---|---|---|---|
| **0** metadata | free — 1 request | `metadata` | Whether the response envelope, catalogue and declared quantization are consistent with the claim. Everything here is forgeable in minutes, so it is weighed at almost nothing — it runs first because carelessness is common |
| **1** fingerprints | fractions of a cent — ~43 requests | `thinking_signature`, `token_accounting`, `api_surface`, `self_report`, `determinism`, `tokenizer`, `knowledge_cutoff` | That the tokenizer generation, tool-use token overhead, parameter-support matrix or knowledge boundary is inconsistent with the claim. For Claude, whether a thinking-block signature survives replay. This is where the tool earns its keep |
| **2** capability | seconds of generation — ~34 requests | `long_context`, `tool_calling`, `structured_output`, `vision`, `multilingual`, `performance` | That the served context window, vision, tool calling, Unicode handling or structured output falls short of the claim. Not the converse: a stronger model passes too |
| **3** statistical | real money — 100+ requests | `logprobs`, `benchmark`, `evasion` | That accuracy is below the published number by a stated effect at a stated error rate; that behaviour differs between recognisable and paraphrased inputs; that two endpoints' logprob distributions differ |

Each probe emits `Evidence` carrying a signed log-likelihood ratio — positive
supports the provider's claim, negative refutes it — so combining evidence is
addition, and the posterior follows from Bayes' rule in odds form. Two departures
from naive addition keep it honest. **Per-probe caps**: no single observation may
convict. **Per-family damping**: probes measuring overlapping things are summed
and then squashed toward a family ceiling with `cap · tanh(total / cap)`, so five
tokenizer measurements are one opinion rather than five independent experiments.
Neither is statistically exact — exactness would need a joint model of probe
correlations that nobody has — and both are conservative in the direction that
matters: they make it *harder* to accuse a provider.

Full details, including every probe's known false-positive and false-negative
modes: [`docs/probes.md`](docs/probes.md).

## Three findings that make this sharper than a benchmark runner

### 1. Token accounting identifies the model generation exactly, for free

Anthropic publishes the exact token cost of the system prompt its API injects
when a request carries tools, and that integer differs by model generation:

| model | `tool_choice` auto/none | `tool_choice` any/tool |
|---|---|---|
| Claude Opus 5 | **286** | **406** |
| Claude Sonnet 5 | **354** | **474** |
| Claude Opus 4.8 | 290 | 410 |
| Claude Opus 4.7 | 675 | 804 |
| Opus 4.6 / Sonnet 4.6 / Opus 4.5 / Haiku 4.5 | 496–497 | 588–589 |

This is not a statistic and not a behaviour. It is an integer the endpoint
reports about itself, and it either matches the claimed model or it matches a
different one. The `token_accounting` probe recovers it by double differencing —
a no-tool baseline removes the request envelope, and a second tool identical but
for its name removes the tool definition — so a proxy's own injected system
prompt cancels out. Tolerance is ±2 tokens against a table whose closest pair
differs by four, so the slack costs no discriminating power at all. A residual
matching a *different* model in the snapshot is the most useful sentence this
tool can produce: not "not what you claimed", but which model it actually is.

The `tokenizer` probe rests on the same kind of fact. Claude 4.7 and later
produce roughly **30% more tokens for identical text** than 4.6 and earlier, so
an endpoint claiming Opus 5 whose count on a fixed probe string sits near the
claimed value ÷ 1.30 is running a pre-4.7 tokenizer.

### 2. Thinking-block signatures cannot be forged

Anthropic returns extended-thinking blocks with a `signature` field, documented
as existing so the API can verify that a thinking block was generated by Claude
when it is passed back. A tampered signature is rejected with a 400.

Whatever the signature is over, one property follows from the rejection behaviour
alone: **an endpoint not routing through genuine Anthropic inference cannot
produce a block that survives replay, and cannot tell a good signature from a
corrupted one.** The probe elicits a thinking block, replays it verbatim, then
replays it again with eight characters in the middle of the base64 signature
corrupted — same length, everything else byte-identical. A genuine endpoint must
accept the first and reject the second. This is the only cryptographic test in
the package and the strongest identity test that exists for the Claude family;
signatures are portable across Anthropic, Bedrock and Vertex, so a first-party
gateway on any of the three still passes.

One honest alternative explanation survives, and is named in every report that
carries the finding: an endpoint that ignores replayed thinking blocks entirely —
neither validating nor using them — accepts both requests for a reason that has
nothing to do with forgery.

### 3. GPQA Diamond cannot separate 2026 frontier models. SimpleQA Verified can.

This is the finding that most changes how a run should be designed, and it is
purely arithmetic.

| benchmark | models | min | max | spread | sd |
|---|---|---|---|---|---|
| ARC-AGI-2 | 21 | 4.9 | 92.5 | 87.6 | **30.6** |
| SimpleQA Verified | 36 | 9.6 | 77.3 | 67.7 | **16.7** |
| ARC-AGI v1 | 21 | 44.7 | 98.0 | 53.3 | 16.0 |
| OTIS Mock AIME | 35 | 80.0 | 100.0 | 20.0 | 5.3 |
| SWE-bench Verified | 17 | 57.9 | 83.5 | 25.6 | 5.2 |
| **GPQA Diamond** | 35 | 71.3 | 92.8 | 21.5 | **5.0** |

Now the sample sizes. Two-proportion z-test, 95% confidence, 80% power, per arm:

| separating | on | items needed per arm |
|---|---|---|
| Opus 5 (91.8) vs GPT-5.6 Sol (91.3) | GPQA Diamond | **48,519** — against 198 in existence |
| Opus 5 (91.8) vs Sonnet 5 (87.4) | GPQA Diamond | 754 |
| GPT-5.6 Sol (71.6) vs GPT-5.6 Terra (43.1) | SimpleQA Verified | **47** |
| GPT-5.6 Sol (92.5) vs GLM-5.2 (22.8) | ARC-AGI-2 | **7** |

Separating two current flagships on GPQA Diamond is not expensive — it is
*impossible*, by a factor of 245. Every 2026 flagship lands between 87 and 93,
and the benchmark has 198 items. A naive verifier that runs the most prestigious
benchmark it knows will spend real money to produce a non-significant result and
then read that result as a clean bill of health.

So llmverify ranks benchmarks by information per request, declares
`discriminative = False` on the saturated ones, weights them down, and says so in
`llmverify benchmarks`. Saturated benchmarks stay available — a substituted model
is often nowhere near frontier — but a non-significant result on one means the
benchmark is too coarse, never that the endpoint is genuine.

The corollary, and the reason the `evasion` probe exists: accuracy alone is the
easiest thing in this package to defeat. A provider that routes anything looking
like a benchmark item to the real model, and serves everything else from
something cheaper, beats every accuracy measurement at negligible cost. So items
are also sent paraphrased, and the gap between the two arms is measured directly.

## Configuration

A provider config is YAML. Nothing in it is secret — it names the *environment
variable* holding your key, never the key — so it is safe to commit, which is the
point: an audit you cannot re-run from a checked-in file is an anecdote.

```yaml
# Label used in reports and in the `compare` table.
name: acme-reseller

# Adapter to speak: openai, anthropic, gemini, or one from a plugin. `openai`
# covers vLLM, SGLang, LiteLLM, OpenRouter, Ollama, LM Studio and essentially
# every reseller, because they all implement that protocol.
api: openai

# Endpoint root, no trailing path. The adapter appends /chat/completions and
# /models itself. Omit to use the adapter's first-party URL.
base_url: https://api.acme-inference.example/v1

# What to send in the `model` field: the provider's own naming.
model: opus5-turbo

# The canonical model this endpoint claims to serve. This is what the reference
# snapshot is looked up by and what the verdict is about. Set it whenever
# `model` is not already a canonical id -- otherwise half the probes have
# nothing to compare against. Check it with: llmverify models opus
claimed_model: claude-opus-5

# The NAME of an environment variable holding the API key. Never the key.
api_key_env: ACME_API_KEY

# How the key is presented. Defaults to the adapter's norm -- bearer for openai,
# x-api-key for anthropic, query for gemini. One of: bearer, x-api-key, query,
# none.
auth_scheme: bearer

# Merged into every request.
headers:
  x-acme-tenant: research
query_params:
  api-version: "2026-05-01"

# Transport. Defaults shown. Lower max_concurrency for an endpoint that
# rate-limits hard -- 429s otherwise read as capability failures.
timeout_s: 180.0
connect_timeout_s: 20.0
max_concurrency: 4
max_retries: 3

# Only ever disable TLS verification against something you control, on a network
# you control. A verifier that skips certificate checks can be
# machine-in-the-middled by exactly the party it is auditing.
verify_tls: true
# proxy: http://127.0.0.1:8080

# Pin reasoning effort. Published scores are effort-conditional -- DeepSeek
# reports V4-Pro at 90.1 on GPQA Diamond in Think-Max and 72.9 in Non-Think --
# so leaving this unset makes benchmark comparison approximate, and the run says
# so.
reasoning_effort: high
# thinking_budget: 4096

# Prices the provider advertises, per million tokens. Used by the performance
# probe to compare advertised economics against first-party pricing, and to turn
# token usage into the estimated spend that --max-cost is enforced against.
price_in_per_mtok: 1.20
price_out_per_mtok: 6.00
```

`${ENV_VAR}` references are expanded anywhere in the file, and a reference to an
unset variable is a hard error rather than an empty string.

Five ready-to-edit configs live in [`examples/`](examples/), each annotated field
by field: a generic reseller, OpenRouter (with a note on pinning a provider and
reading declared quantization), Anthropic first-party as an A/B baseline, a local
vLLM server, and a file carrying a `run:` block that demonstrates A/B mode with a
tightened budget.

Run settings — layers, probe selection, seed, budget ceilings, α, β,
`min_effect_pp`, prior odds and an inline baseline — live under a `run:` key that
`llmverify.config.load_run_config` reads. The CLI builds its run config from flags
instead; `examples/with-baseline.yaml` spells out both forms side by side.

## Sample output

A real run against a deliberately dishonest local endpoint — an OpenAI-compatible
server claiming `claude-opus-5` while returning logprobs, which Anthropic's
protocol cannot express:

```console
$ llmverify check --api openai --base-url http://127.0.0.1:8931/v1 \
      --model claude-opus-5 --layers 0,1
  done 127.0.0.1:8931/metadata          0.1s  7 finding(s), +0.72 bans
  fail 127.0.0.1:8931/token_accounting  0.3s  2 finding(s), -1.00 bans
  done 127.0.0.1:8931/self_report       0.1s  1 finding(s), +0.00 bans
  done 127.0.0.1:8931/determinism       0.2s  3 finding(s), -0.48 bans
  done 127.0.0.1:8931/api_surface       0.4s  3 finding(s), -2.24 bans
  done 127.0.0.1:8931/tokenizer         0.4s  2 finding(s), +0.00 bans

MISMATCH 127.0.0.1:8931 claiming claude-opus-5 -- p(genuine)=0.0038, -2.42 bans, $0.0124, 1s
openai at http://127.0.0.1:8931/v1, model claude-opus-5, seed 20260726,
reference as of 2026-07-26, model found
evidence

 probe              finding                            bans    what it means
 ─────────────────────────────────────────────────────────────────────────────────
 api_surface        anthropic_claim_returns_logprobs   -2.00   the endpoint claims
                                                               'claude-opus-5' and returned
                                                               real per-token logprobs.
                                                               Anthropic's protocol has no
                                                               logprobs at all [...] and a
                                                               translating proxy cannot
                                                               synthesise them without
                                                               running a model that
                                                               produces them.
 token_accounting   forced_choice_delta                -1.00   forcing tool use added 0
                                                               tokens where 'claude-opus-5'
                                                               should add 120. The tool
                                                               definition cancels in this
                                                               difference, so it cannot be
                                                               blamed on serialisation.
 api_surface        anthropic_claim_accepts_seed       -0.48   [...] Only weak: a proxy that
                                                               accepts the field and drops
                                                               it is indistinguishable from
                                                               one that honours it.
 determinism        cached_response_signature          -0.48   every sample was byte-
                                                               identical and only 2
                                                               characters long [...] Latency
                                                               did not collapse, so this may
                                                               be a very terse model rather
                                                               than a canned answer.
 metadata           model_echo                         +0.48   the endpoint echoed
                                                               'claude-opus-5' exactly. This
                                                               is the cheapest field in the
                                                               response to forge, so it is
                                                               capped at weak.
 metadata           catalogue                          +0.24   'claude-opus-5' appears in a
                                                               catalogue of 1 models;
                                                               serving stack looks like
                                                               vllm.

family totals (bans): api_surface -1.61, token_accounting -0.98, metadata +0.61,
misc -0.44, self_report +0.00, determinism +0.00, tokenizer +0.00
18 ok, 3 skipped, 1 error; 32 requests, 2214+54 tokens
$ echo $?
1
```

Read it from the bottom up. The family totals show the finding rests on two
independent families, not one probe shouting. Supporting evidence is shown
alongside refuting evidence — a reader who sees only half of an adverse verdict
has been misled — and every explanation names its own weakness. `-2.42 bans` is a
base-10 log-likelihood ratio: about 260:1 against the claim. `--verbose` shows
every piece of evidence rather than the headline ones; `--quiet` prints the
verdict line alone.

`compare` runs the same checks across several endpoints, one at a time so they do
not contend for one uplink and flatter or slander each other's measured speed:

```console
$ llmverify compare acme.yaml gateway.yaml --layers 0,1
comparison

 provider        claimed model   verdict        p(genuine)   bans    cost
 ──────────────────────────────────────────────────────────────────────────
 corp-gateway    claude-opus-5   INCONCLUSIVE   0.3607       -0.25   $0.0223
 acme-reseller   claude-opus-5   MISMATCH       0.0038       -2.42   $0.0124
```

### The HTML report

`llmverify.report.render_html(result, path=Path("report.html"))` writes a single
self-contained document — around 45 KB — with **no network dependencies at all**:
no CDN, no web fonts, no remote images, nothing that fetches. That is deliberate
rather than stylistic. This is the artefact somebody attaches to a support ticket
or a procurement dispute, so it has to render identically in a year, offline, from
a mail attachment, and it must not phone anywhere when a third party opens it.

It contains a verdict card with a plain-language gloss, a run header, a diverging
bar chart of every evidence contribution, a per-family table showing which caps
bound the total, per-benchmark interval charts with the sequential test's trace,
the full evidence table, per-probe detail with the raw measurements, and probe
timings. Charts are hand-written SVG and every one has a table beside it carrying
the same numbers, so nothing is encoded in colour alone. Everything interpolated
into the page is redacted for secrets and then escaped — provider error bodies
reach the document verbatim by design, because they are evidence, and a provider
that returns `<script>` in an error message must not be able to run it in a
reader's browser.

`render_comparison_html(results, path=...)` does the same for several runs.

> **Known gap.** The CLI's `--html` flag looks for a `write_html` hook on
> `llmverify.report`, which the bundled reporting package does not currently
> export. Passing `--html` therefore prints a warning and writes nothing. Use
> `--json` from the command line, which carries the same data in a stable,
> secret-free schema, or call `render_html` from Python as above.

Everything printed anywhere — error bodies, base URLs, probe details — passes
through the redactor first. Provider errors routinely echo request headers back,
and a report that leaks the key it authenticated with is worse than no report.

## The reference snapshot

`src/llmverify/reference/data/reference.yaml` is what a verdict is measured
against: **47 model records across 13 vendors, 3 API family signatures and 473
benchmark scores**, dated 2026-07-26. It is versioned in git rather than fetched
at run time, so that rerunning last month's command against last month's commit
produces last month's answer, and so that when this tool accuses someone a human
can read the commit that moved the threshold.

Every number carries its source, the date it was read, and a confidence level:

| level | means | count |
|---|---|---|
| `primary` | the lab's own publication | 104 |
| `independent` | Epoch AI, ARC Prize, Artificial Analysis | 341 |
| `secondary` | press or third-party write-ups, including figures triangulated from convergent summaries where a lab published only images | 28 |
| `unverified` | flagged as unconfirmed at the source | 0 |

Scores also carry the *conditions* that make them comparable — effort, tools,
shot count, harness, item count, whether benchmark-optimised settings were used —
and those fields are filled **only when the publisher stated them**. Null means
unknown, never "default". Where several sources disagree about the same model at
the same settings — Opus 5 on ARC-AGI-2 is 90.4 by ARC Prize and 88.3 by Epoch
AI — the null is set at the *lowest* value in the matching range, so disagreement
widens the benefit of the doubt instead of quietly becoming this tool's own bias.

`llmverify refresh` fetches live data, diffs it against the file, and by default
**stops there**:

```console
$ llmverify refresh
reference refresh 2026-07-26T23:59:27+00:00
  mode: dry run
  source epoch: ok, 232 scores across 33 known models
  source openrouter: ok, 343 models listed, 45 matched to records

no changes: the snapshot matches every source consulted

unmatched upstream models (13)
  ! GLM-5 (model_version glm-5)
  ! Kimi K2.6 (model_version kimi-k2.6)
  ! Qwen 3.6 Max (Preview) (model_version qwen3.6-max-preview)
  ...
  add a row to EPOCH_MODEL_IDS after confirming which model each one is
```

`--write` applies the merge; `--source` narrows it to `epoch` or `openrouter`.
A source that fails is recorded and the others still run. Nothing is ever
deleted — a snapshot model missing from the live catalogue is listed as "no longer
listed upstream, not deleted", because a catalogue hiccup must not erase a
reference threshold. Unmatched upstream models are reported for a human to map,
never guessed at: fuzzy name matching is how `gpt-5.6-luna` ends up compared
against `gpt-5.6-sol`'s numbers.

Schema, every source with its URL and licence, confidence levels, and how to add
a model by hand: [`docs/reference-data.md`](docs/reference-data.md).

## Extending it

Three plugin points, all discovered through entry points, all skipping any plugin
that fails to import — a broken plugin must not break the tool.

**An adapter** teaches llmverify a wire protocol. Subclass `Adapter`, provide
`name`, `family`, `default_base_url`, `build_payload`, `parse_response` and
`parse_stream`, and publish it:

```toml
[project.entry-points."llmverify.adapters"]
nimbus = "llmverify_nimbus.adapter:NimbusAdapter"
```

Two rules are not optional: omit unset fields rather than nulling them — the
`api_surface` probe reads exactly that difference to tell "rejected" from
"silently dropped" — and keep everything the endpoint volunteered in
`ChatResponse.raw`. A complete worked example is in
[`docs/adapters.md`](docs/adapters.md).

**A probe** adds an experiment. Subclass `Probe`, declare `layer`, `family`,
`order` and `estimated_requests`, and implement
`async def run(ctx) -> list[Evidence]`:

```toml
[project.entry-points."llmverify.probes"]
my_probe = "my_package.probes:MyProbe"
```

A probe must never raise for an expected negative — "this endpoint has no
logprobs" is `ctx.unsupported(...)`, not a crash — and must use `ctx.rng(salt)`
rather than global randomness, so that skipping one probe cannot change every
later probe's sampling.

**A benchmark** adds items. Subclass `Benchmark`, declare `reference_key`, the
dataset spec, `licence`, `gated`, `discriminative` and `score_spread`, and
implement `load`:

```toml
[project.entry-points."llmverify.benchmarks"]
my_benchmark = "my_package.benchmarks:MyBenchmark"
```

Grading must be deterministic and must never call another model. An LLM judge
would add a second trust assumption to a tool whose entire purpose is to check a
trust assumption.

Development setup:

```console
$ pip install -e '.[dev]'
$ pytest
$ ruff check src tests
```

## Limitations

**No software-only method can prove which weights ran.** This is the finding of
arXiv:2504.04715, which states it directly: *"software-only methods are
fundamentally unreliable: statistical tests on text outputs are query-intensive
and fail against subtle substitutions, while methods using log probabilities are
defeated by inherent inference nondeterminism in production."* Its conclusion is
that only trusted execution environments with hardware attestation are robust.
Everything in this package is a software-only method and is inside that finding.

**A provider that routes a fraction of traffic to the genuine model defeats every
statistical test here.** Serve the real weights to 10% of requests, or only to
requests that look like tests, and the mixture is indistinguishable from honesty
at any sample size you would pay for. The provider sees your queries before
deciding how to answer them. That asymmetry is structural, not a tuning problem.

**Serving infrastructure legitimately changes behaviour at fixed weights, and
Anthropic says so explicitly:** *"Model weights are fixed for a given ID, but the
serving infrastructure around the model can change over time... infrastructure
updates produce minor differences in observable behavior even when the model ID
and weights have not changed."* That is a vendor telling you, in advance, that a
behavioural-drift detector will produce false positives against its genuine
first-party endpoint. Independently, arXiv:2605.19537 measured backend choice
alone moving benchmark scores by up to **16.6 percentage points** — larger than
most gaps between adjacent frontier models, and larger than this tool's default
detection threshold of 8 points.

**Benchmark reference numbers are effort-conditional, and comparing across efforts
is invalid.** DeepSeek reports V4-Pro at 90.1 on GPQA Diamond in Think-Max and
72.9 in Non-Think: a 17.2-point swing from one setting, same weights. Disclosure
is regressing — OpenAI's GPT-5.6 announcement states no effort level for any of
its numbers.

**Drift at temperature 0 is expected and is not evidence.** A forward pass at
fixed batch composition is deterministic; production inference is not, because
dynamic batching makes your result depend on how many other requests were batched
alongside it. Every hosted provider disclaims bitwise determinism. llmverify
scores this at essentially zero and caps the whole `determinism` family at a 3:1
likelihood ratio.

Other things it will miss: a competent forger who makes their envelope
consistent; a provider that routes on semantics rather than on string match; a
substitution between two models within a few points of each other on every
benchmark you can afford; a *stronger* model served in place of the claimed one; a
substitution that only manifests on workloads this tool does not exercise.

**llmverify is designed to make substitution expensive to hide and to show its
work, not to deliver a proof.** Read an adverse verdict as "this endpoint does not
behave like the model it claims", never as proof of fraud; read a favourable one
as "nothing here contradicts the claim, at the sample size paid for".

The long version — with citations, a per-layer table of what can and cannot be
established, and every false-positive mode spelled out — is in
[`docs/limitations.md`](docs/limitations.md). Read it before you act on a verdict.

## Documentation

| | |
|---|---|
| [`docs/limitations.md`](docs/limitations.md) | what this cannot establish, with citations |
| [`docs/probes.md`](docs/probes.md) | every probe: cost, family, cap, failure modes |
| [`docs/adapters.md`](docs/adapters.md) | writing and registering an adapter, probe or benchmark |
| [`docs/reference-data.md`](docs/reference-data.md) | snapshot schema, sources, confidence levels |
| [`examples/`](examples/) | five annotated provider configs |

## Licence and attribution

llmverify is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

Benchmark scores in the reference snapshot sourced from **Epoch AI's Capabilities
Index** (`epoch.ai/data/eci_benchmarks.csv`) are used under **CC-BY**, which
permits redistribution with credit; that credit is in [`NOTICE`](NOTICE).
Artificial Analysis index values reach the snapshot embedded in OpenRouter's
public API and remain Artificial Analysis's data under its own terms. Model
identifiers, context lengths, pricing and self-declared quantization come from
OpenRouter's documented public API.

Evaluation datasets are **not** redistributed here. They are fetched on demand
over the HuggingFace datasets-server and each remains under its own licence — MIT
(SimpleQA, MMLU-Pro), CC-BY-4.0 (GPQA, gated), CC-BY-NC-SA-4.0 (AIME 2025/2026,
non-commercial), Apache-2.0 (IFEval). Using a benchmark through llmverify does not
change the terms you accepted from its publisher. [`NOTICE`](NOTICE) names each
one with what it covers.
