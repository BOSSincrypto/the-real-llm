# Limitations

Read this before you act on a verdict.

llmverify measures what an endpoint *does*. It cannot prove what an endpoint
*is*. The gap between those two sentences is not an implementation gap that a
future release closes — it is a result in the literature, and this page is where
the tool states it plainly instead of burying it.

## 1. The ceiling: no software-only method can prove which weights ran

The most direct statement of the problem is "Are You Getting What You Pay For?
Auditing Model Substitution in LLM APIs" (arXiv:2504.04715), which concludes,
verbatim:

> software-only methods are fundamentally unreliable: statistical tests on text
> outputs are query-intensive and fail against subtle substitutions, while
> methods using log probabilities are defeated by inherent inference
> nondeterminism in production.

Its conclusion is that only trusted execution environments with hardware
attestation are robust against a determined provider. Everything in this package
is a software-only method. Every one of them is therefore inside that finding.

The concrete attack that defeats all of them is cheap and requires no
sophistication: **route a fraction of traffic to the genuine model.** A provider
that serves the real weights to 10% of requests, or only to requests that look
like tests, produces a distribution that no test here can distinguish from an
honest one at any sample size you would pay for. Statistical tests measure a
mixture; the mixture is under the provider's control; and the provider observes
your queries before deciding how to answer them. That asymmetry is structural.

Related refinements do not remove the ceiling, they move it:

- **TOPLOC** (arXiv:2501.16007, ICML 2025) commits to activations with a locality-
  sensitive hash, about 258 bytes per 32 tokens, robust to GPU nondeterminism.
- **DiFR** (arXiv:2511.20621) reports detecting 4-bit quantization at AUC ≫ 0.999
  within 300 tokens using shared-seed post-Gumbel logit comparison.

Both need cooperation from the serving side — the provider must emit the
commitment, or you must be able to run the reference yourself. Neither is
something you can do from the outside against an uncooperative endpoint, which is
the case llmverify exists for.

## 2. What each layer can and cannot establish

| layer | can establish | cannot establish |
|---|---|---|
| **0** — metadata | that the endpoint's envelope is or is not consistent with the claimed model's first-party API; what an OpenRouter route declares for its own quantization | anything about the weights. Every field here is a string the provider chose, forgeable in minutes |
| **1** — cheap fingerprints | that the tokenizer generation, the tool-use token overhead, the parameter-support matrix or the knowledge boundary is inconsistent with the claim; for Claude, that a thinking-block signature does or does not survive replay | that a *consistent* fingerprint means the claimed model ran. A provider routing through genuine inference passes the signature test while substituting on every request that does not enable thinking |
| **2** — capability | that the served context window, vision, tool-calling, Unicode handling or structured-output behaviour falls short of the claim | that meeting the claim means the claimed model. A stronger model, a weaker model in the same family, and the real model all pass |
| **3** — statistical | that accuracy on a discriminative benchmark is below the published number by a stated effect size, at a stated error rate; that behaviour differs between recognisable and paraphrased inputs; that two endpoints' logprob distributions differ | *which* model produced the difference, or that no difference means no substitution. Also nothing at all about a provider mixing traffic |

The aggregate verdict inherits every one of those limits. It is a posterior over
"does this endpoint behave like the claimed model", not over "did the claimed
model run".

## 3. False positives: honest endpoints that this tool can accuse

These are the failure modes that matter most, because a false accusation is the
worst thing a tool like this can do. Each is real, each has been designed around,
and none is eliminated.

### 3.1 A legitimate proxy injecting a system prompt

Enterprise gateways, safety layers, routers and internal LLM platforms routinely
prepend a system prompt, rewrite identifiers, strip fields and re-serialise
payloads. Every one of those is visible to this tool and none of them is
substitution.

What it breaks: self-identification becomes whatever the prompt says, at no cost
to the provider's honesty. Token counts inflate by the prompt's length. The
response envelope — id prefix, `system_fingerprint`, usage-object shape, catalogue
entries — becomes the proxy's rather than the model's. Thinking-block signatures
may be stripped or re-signed with the proxy's own key. Tokenizer differencing
assumes a fixed envelope, and a proxy whose injected prompt varies per request
breaks that assumption outright.

What llmverify does about it: `self_report` is capped at WEAK in both directions.
Envelope findings are weighed only when the claimed model's own first-party API
is the protocol being spoken; a reseller fronting Claude behind an
OpenAI-compatible route is not penalised for rewriting the envelope, because doing
so is legitimate. Token accounting double-differences, so a fixed injected prompt
cancels — and the measured envelope size is reported anyway, because it is the
most common innocent explanation for a residual matching nothing.

What remains: a proxy with a *variable* injected prompt defeats the differencing.
A proxy that strips signatures is indistinguishable from an endpoint that never
had one.

### 3.2 A provider on different hardware

"The Silent Hyperparameter" (arXiv:2605.19537) measured backend choice alone
moving benchmark scores by up to **16.6 percentage points** at fixed weights.
That is larger than most gaps between adjacent frontier models and larger than
this package's default `min_effect_pp` of 8.0.

The field evidence agrees. In August 2025 identical `gpt-oss-120b` weights scored
93.3% at six providers, 86.7% at Groq, 80.0% at Azure and 36.7% at CompactifAI on
AIME25 — a 56.6-point spread under one model name, largely old vLLM builds
defaulting `reasoning_effort` to medium. In November 2024 aider measured the same
nominal model at 72.2% on one provider and 0.5% on the worst.

Those numbers cut both ways. They are the reason this tool exists, and they are
the reason a low score is not proof of different weights. Serving software,
kernel versions, GPU generation, batch regime, speculative decoding and
quantization all move behaviour without touching a checkpoint.

What llmverify does: throughput and latency findings are capped at WEAK and
individual measurements are reported at zero LLR; only the joint
fast-*and*-cheap observation carries anything. Tool-calling findings are
discounted because the field evidence says serving software, not weights, drives
tool reliability. Logprob findings are capped at MODERATE and further multiplied
by a 0.5 configuration discount, and the evidence detail says a significant result
means "different serving configuration", not "different weights".

What remains: an honest provider on unusual hardware accumulates small negative
evidence across several families, and enough of it reaches a verdict.

### 3.3 A model served at a different reasoning effort

**Comparing scores across efforts is invalid.** DeepSeek reports V4-Pro at 90.1
on GPQA Diamond in Think-Max mode and 72.9 in Non-Think — a 17.2-point swing from
one setting, with the same weights. Anthropic's ladder is max / xhigh / high /
medium / low with an API default of high; OpenAI's adds `none` with a default of
medium and its eval footnote states evals were run at xhigh except where
specified; Gemini's is high / medium / low / minimal and is mandatory, with no
default at all.

Worse, disclosure is regressing. OpenAI's GPT-5.6 announcement states no effort
level for any of its numbers.

What llmverify does: `BenchmarkScore.effort` is filled only when the publisher
stated it, and null means unknown rather than "default". When a provider pins an
effort the reference score was not measured at, the benchmark probe widens its
tolerance by `EFFORT_MISMATCH_ALLOWANCE_PP` (15.0 pp) and says why in the report;
optimised-settings scores get a further 3.0 pp. Where several sources disagree
about the same model at the same settings, the null is set at the *lowest*
published value in the matching range, so disagreement widens the benefit of the
doubt instead of becoming this tool's bias.

What remains: both allowances are judgement calls, not measurements. They are
labelled as such in the evidence data. An unpinned run against a reference score
whose effort is also unknown is comparing two things that may not be comparable,
and the report says so rather than pretending otherwise.

### 3.4 First-party infrastructure changes

Anthropic states this outright in its own documentation:

> Model weights are fixed for a given ID, but the serving infrastructure around
> the model can change over time... infrastructure updates produce minor
> differences in observable behavior even when the model ID and weights have not
> changed.

That is a vendor telling you, in advance, that a behavioural-drift detector will
produce false positives against its genuine first-party endpoint. It is not a
hypothetical: it is the documented expected behaviour of the most trustworthy
endpoint you can point this tool at.

The practical consequence: two runs against `api.anthropic.com`, a month apart,
can disagree without anything being wrong. An A/B against a first-party baseline
inherits the same property — the baseline is not a fixed reference, it is another
endpoint that also drifts.

### 3.5 Nondeterminism at temperature 0

The most common way an amateur verifier accuses an honest provider is to send the
same prompt twice at temperature 0, get different answers, and call it evidence.

Thinking Machines Lab's "Defeating Nondeterminism in LLM Inference" (September
2025) explains why that reasoning is wrong. A forward pass at *fixed batch
composition* is deterministic. Nondeterminism enters because dynamic batching
makes your result depend on how many other requests were batched alongside it:
floating-point addition is not associative, and GPU reduction kernels select their
tiling and Split-K strategy from the runtime batch shape, so
`torch.mm(a[:1], b) != torch.mm(a, b)[:1]`. The fix is batch-invariant kernels,
which are opt-in and cost throughput — vLLM's `VLLM_BATCH_INVARIANT=1` is beta,
needs NVIDIA SM8.0 or later, identical hardware and the same vLLM build; SGLang's
`--enable-deterministic-inference` costs 25–55% of throughput.

Every hosted provider disclaims bitwise determinism. Anthropic states that even
at a temperature of 0.0 the results will not be fully deterministic. OpenAI's
`seed` parameter, which exists only on Chat Completions and not on the Responses
API, is documented as not guaranteeing determinism.

What llmverify does: **drift under identical seed and temperature 0 contributes an
LLR of essentially zero.** The `determinism` family caps at WEAK. The number is
reported because a human reading the report wants it, not because it argues either
way. What does carry weight is *degenerate* sameness — every sample
byte-identical and suspiciously short, or repeat latency collapsing toward zero —
which is a cache answering instead of a model.

## 4. False negatives: substitutions this tool will miss

- **Traffic mixing.** Covered above. This is the one that matters.
- **Semantic routing.** The evasion probe paraphrases items so they no longer
  string-match a published corpus. A provider that routes on *semantics* rather
  than on string match survives paraphrasing entirely. The structural check uses a
  synthetic marker, not any real dataset's canary string, so a detector keyed to a
  specific corpus will not fire — it is a lower bound on detectability, not a
  measurement of it.
- **A competent forger.** Every layer-0 field is a string. A reseller that spends
  an afternoon making its envelope, catalogue and error vocabulary consistent
  passes the whole free layer. Token accounting is computed from numbers the
  provider reports about its own usage: a provider willing to fabricate
  `usage.input_tokens` consistently across four differenced requests defeats it —
  at which point every token-based measurement here is defeated too.
- **A near-neighbour substitution.** Separating Claude Opus 5 (91.8) from GPT-5.6
  Sol (91.3) on GPQA Diamond at 95% confidence and 80% power needs roughly 48,500
  items per arm. GPQA Diamond has 198. That is an impossibility, not an expense.
  Models within a few points of each other on every benchmark you can afford are
  not separable by accuracy at all.
- **A stronger model.** Nothing here penalises an endpoint for being better than
  claimed. If a reseller quietly upgrades you, the verdict is `MATCH`.
- **Substitution outside the probes' shape.** Everything is measured on short
  prompts, small item counts and specific capabilities. A substitution that only
  manifests on 200k-token agentic workloads is not in this tool's reach.

## 5. What the statistical tests actually buy

The prior art is worth reading in full; the short version is that the tests here
are real but their power is bounded.

**Model Equality Testing** (Gao, Liang & Guestrin, arXiv:2410.20247, ICLR 2025)
frames the problem as two-sample distribution testing over completions. Its best
statistic is maximum mean discrepancy with a Hamming string kernel, reaching a
median of **77.4% power at about 10 samples per prompt** — a useful, honest
number, and notably not 99%. The paper found **11 of 31** commercial Llama
endpoints, including ones on Bedrock and Azure AI Studio, serving distributions
that differed from the reference. Code: `github.com/i-gao/model-equality-testing`.

Two things follow. First, 10 samples per prompt across a prompt set is a real
cost, and this package's budget ceilings will often stop short of it. Second, the
test needs a *reference distribution* — which in practice means either a
first-party baseline endpoint you are paying for as well, or GPUs of your own
running the same weights.

**Rank-based uniformity testing** (arXiv:2506.06975) applies a Cramér–von Mises
test to log-rank percentiles and is more query-efficient than MMD. It caught a
tokenization mismatch on HuggingFace Inference for a Mistral model — a real find,
and also an illustration of the recurring problem: the thing detected was a
serving-stack bug, not substituted weights.

llmverify's `logprobs` probe implements the same family of ideas — a two-sample KS
test on pooled chosen-token logprobs, a chi-square on chosen-token rank, and a
randomised rank-uniformity check. It is capped at MODERATE per item with a further
0.5 discount, and `UNSUPPORTED` is its ordinary outcome: Anthropic's protocol has
no logprobs at all, Gemini's could not be reconfirmed, and reasoning models across
vendors decline them in practice.

Other prior art, none of it implemented here, all of it worth knowing:
**LLMmap** (arXiv:2407.15847, USENIX Security 2025) reports >95% accuracy over 42
model versions in 8 queries; **TRAP** (arXiv:2402.12991) uses adversarial-suffix
shibboleths; **Model Provenance Testing** (arXiv:2502.00706) reports 90–95%
precision over 600+ models; **"One Token Is Enough"** (arXiv:2607.10252) reports
under 11% equal error rate over 165 OpenRouter models with about 100 queries;
**"Accuracy Is Not All You Need"** (arXiv:2407.09141, NeurIPS 2024) shows
compressed models matching aggregate accuracy while individual answers flip, and
argues for KL-divergence and answer-flip metrics over raw accuracy.

## 6. Multiplicity, and the temptation this tool resists

llmverify runs seventeen probes producing dozens of evidence items. Running many
comparisons at a nominal 5% and reporting whichever came out smallest is exactly
how a tool like this manufactures accusations against honest providers.

Three mechanisms push back. Per-probe caps mean no single item can convict.
Per-family damping — `cap * tanh(total / cap)` — means correlated probes saturate
instead of compounding, so five tokenizer measurements are one opinion, not five.
And a run with fewer than three contributing probes is forced to `INCONCLUSIVE`
regardless of the arithmetic, while an early stop additionally requires three
distinct *families*.

None of that is statistically exact. Exactness would need a joint model of probe
correlations that nobody has. They are conservative in the direction that matters:
they make it harder, not easier, to accuse a provider.

The evasion probe additionally corrects across its own tests with
Holm–Bonferroni, which is valid under any dependence structure — necessary,
because its accuracy and structural tests are not independent.

## 7. Reference data limits

- **Benchmark numbers are effort-conditional**, and comparing across efforts is
  invalid. See §3.3.
- **Sources disagree about the same model at the same settings.** Opus 5 on
  ARC-AGI-2 is 90.4 by ARC Prize and 88.3 by Epoch AI.
- **Some numbers are `secondary`.** Anthropic renders comparison tables as
  images; several figures in the snapshot were triangulated from convergent
  third-party summaries and are labelled as such.
- **The grader here is stricter than the labs'.** Published SimpleQA figures come
  from judge-based grading, which accepts any semantically correct phrasing. This
  package matches strings instead, because a judge model would put a second
  unverified model inside a tool whose job is to verify one. That systematically
  *under*-scores every endpoint by an amount this repository has not measured. The
  direction is the safe one — it produces missed detections rather than false
  accusations — but it is a real bias.
- **The snapshot goes stale.** `llmverify models` warns past 30 days.
  `llmverify refresh` diffs against live sources but writes nothing without
  `--write`, and most fields — token accounting, effort ladders, primary lab
  scores — cannot be refreshed automatically at all.

## 8. Operational limits

- **Costs are estimated, not measured.** Providers do not report spend, so
  `--max-cost` is enforced against an estimate computed from reference pricing.
  Where pricing is unknown the estimate is zero and only `--max-samples` and
  `--max-wall` bind.
- **An early stop is a partial audit.** Every probe that never ran is recorded as
  skipped with its reason and the verdict carries a note. An early stop must never
  read like a clean bill of health.
- **`INCONCLUSIVE` is a statement about the run, not about the provider.** It
  usually means the budget ran out, the endpoint refused the capabilities the
  probes needed, or the claimed model is not in the snapshot.
- **A truncated sequential test has not decided anything.** It is reported as
  `TRUNCATED`, never as whichever boundary happened to be nearer. Forcing a
  decision would destroy the error guarantees the design exists to provide.
- **Benchmarks that need a sandbox are excluded, not approximated.** SWE-bench
  and Terminal-Bench are among the most discriminative measurements available and
  llmverify cannot run either.
- **The dataset cache makes runs reproducible and also makes them stale.**
  `llmverify cache --clear` when that matters.

## 9. How to read a verdict

| verdict | read it as |
|---|---|
| `MATCH` / `LIKELY_MATCH` | nothing observed contradicts the claim, at the sample size you paid for |
| `INCONCLUSIVE` | the run did not gather enough usable evidence. About the run, not the provider |
| `LIKELY_MISMATCH` / `MISMATCH` | this endpoint does not behave like the model it claims. **Not** proof of fraud |
| `EVASION` | behaviour depends on whether an input is recognisable as a benchmark item. Every other measurement in the run was taken under conditions the provider chose |

An adverse verdict is the start of a conversation with your provider, supported
by a report that shows every observation, every likelihood ratio and every
alternative explanation the tool knows of. It is not a finding of fact.

llmverify is designed to make substitution expensive to hide and to show its
work. It is not designed to deliver a proof, and it does not have one to give.

## Sources cited

| ref | title |
|---|---|
| arXiv:2504.04715 | Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs |
| arXiv:2410.20247 | Model Equality Testing: Which Model Is This API Serving? (ICLR 2025) |
| arXiv:2506.06975 | A rank-based uniformity test for LLM API auditing |
| arXiv:2605.19537 | The Silent Hyperparameter: backend choice in LLM evaluation |
| arXiv:2501.16007 | TOPLOC: locality-sensitive hashing for verifiable inference (ICML 2025) |
| arXiv:2511.20621 | DiFR: token- and activation-level inference verification |
| arXiv:2407.15847 | LLMmap: fingerprinting for LLM-integrated applications (USENIX Security 2025) |
| arXiv:2402.12991 | TRAP: targeted random adversarial prompt honeypots |
| arXiv:2502.00706 | Model Provenance Testing for LLMs |
| arXiv:2607.10252 | One Token Is Enough |
| arXiv:2407.09141 | Accuracy Is Not All You Need (NeurIPS 2024) |
| arXiv:2508.19843 | SoK: fingerprinting and auditing of LLM services |
| — | Defeating Nondeterminism in LLM Inference, Thinking Machines Lab, September 2025 |
