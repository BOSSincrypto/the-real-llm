# The reference snapshot

The snapshot answers one question: *what should the real model look like?* It is
a single YAML file versioned in git rather than a service fetched at run time, so
that a verdict is reproducible and reviewable. When this tool accuses a provider
of substitution, a human can read the commit that changed the threshold.

```
src/llmverify/reference/data/reference.yaml
```

As shipped it carries 47 model records across 13 vendors, 3 API family
signatures and 473 benchmark scores, dated 2026-07-26.

Browse it:

```console
$ llmverify models              # everything
$ llmverify models opus         # substring match on id, alias, vendor or name
$ llmverify models --json models.json
$ llmverify models --snapshot ./my-snapshot.yaml
```

`llmverify models` prints a warning when the snapshot is more than 30 days old.
Labs ship models faster than that, and a threshold measured against a model that
has since been superseded is the quiet way to get a verdict wrong.

## Schema

Defined and validated in `llmverify.reference.schema`. Unknown keys are rejected,
not ignored: a typo in a field name must fail loudly rather than silently drop a
threshold.

### `ReferenceSnapshot` — the file

| field | type | meaning |
|---|---|---|
| `schema_version` | int | currently `1` |
| `as_of` | date | when the numbers were read |
| `generated_by` | str | `manual`, or `refresh:epoch+openrouter` after a write |
| `sources` | map | name → URL, for provenance |
| `families` | list | `FamilySignature` per API family |
| `models` | list | `ModelRecord` per model |

### `FamilySignature` — what a first-party endpoint of a protocol looks like

| field | type | meaning |
|---|---|---|
| `family` | str | `anthropic`, `openai`, `gemini` |
| `response_id_prefix` | str? | e.g. `msg_` for Anthropic |
| `supports_logprobs` | bool? | whether the protocol expresses them at all |
| `supports_seed` | bool? | same |
| `has_system_fingerprint` | bool? | same |
| `usage_keys` | list | keys expected in the usage object |
| `finish_reasons` | list | values `finish_reason` / `stop_reason` may take |
| `notes` | str? | free text |

This catches the crudest substitutions. An endpoint claiming a Claude model while
accepting `seed` and returning `logprobs` is not talking to Anthropic, whatever
its `model` field says.

### `ModelRecord` — one claimed model

| field | type | meaning |
|---|---|---|
| `id` | str | canonical API model identifier |
| `vendor` | str | `anthropic`, `openai`, `google`, `z-ai`, … |
| `family` | str | API family: `openai`, `anthropic`, `gemini`, `openai_compat` |
| `display_name` | str? | human name |
| `aliases` | list | other identifiers that resolve here, including OpenRouter slugs |
| `released` | date? | |
| `training_cutoff` | date? | the cutoff the vendor publishes |
| `knowledge_cutoff` | date? | a cutoff **measured by probing**, usually earlier. Null throughout the shipped snapshot |
| `context_window` | int? | |
| `max_output_tokens` | int? | |
| `reasoning` | bool? | |
| `open_weights` | bool | |
| `modalities` | list | defaults to `["text"]` |
| `default_effort` | str? | API default on the effort ladder |
| `effort_ladder` | list | e.g. `[max, xhigh, high, medium, low]` |
| `pricing` | `Pricing`? | |
| `token_accounting` | `TokenAccounting`? | |
| `scores` | list | `BenchmarkScore` entries |
| `typical_output_tps` | float? | median output tokens/second on first-party infrastructure. Only ever weak evidence — hardware differs legitimately |
| `notes` | str? | |

The `training_cutoff` / `knowledge_cutoff` split is deliberate. The first is what
a vendor states; the second is what a probe could measure, which is usually
earlier because models are unreliable narrators near their boundary. Nothing in
the shipped snapshot fills the second, and the `knowledge_cutoff` probe compares
against the published figure with that gap in mind.

### `Pricing`

`input_per_mtok`, `output_per_mtok`, `cache_read_per_mtok`,
`cache_write_per_mtok`, all USD per million tokens, plus a `source` URL.

### `TokenAccounting` — deterministic fingerprints

| field | meaning |
|---|---|
| `tool_overhead_auto` | system-prompt tokens with `tool_choice` auto/none |
| `tool_overhead_forced` | …and with `tool_choice` any/tool |
| `bash_tool_overhead` | tokens added by a server-side bash tool |
| `canonical_text_tokens` | token count for `llmverify.probes.tokenizer.CANONICAL_TEXT` |
| `source`, `as_of` | provenance |

Anthropic publishes the first three per model. Sending one request with a single
tool and reading back `usage.input_tokens` therefore identifies the model
generation with no statistics, no logprobs and essentially no cost — the
strongest cheap signal available for the Claude family. Claude Opus 5 is 286 and
406; Opus 4.7 is 675 and 804.

`canonical_text_tokens` distinguishes tokenizer generations within one vendor.
`CANONICAL_TEXT` must never change: changing it would silently invalidate every
value recorded under this key.

### `BenchmarkScore` — one published number, with its conditions

| field | type | meaning |
|---|---|---|
| `benchmark` | str | key, e.g. `gpqa_diamond`, `arc_agi_2`, `simpleqa_verified` |
| `score` | float | percent, or the benchmark's native unit |
| `unit` | enum | `percent` (default), `elo`, `score` |
| `effort` | str? | reasoning effort the score was measured at |
| `tools` | bool? | whether tools were available |
| `shots` | int? | |
| `optimized` | bool? | whether benchmark-optimised settings were used |
| `harness` | str? | e.g. `Terminus-2`, `mini-SWE-agent on GKE` |
| `n_items` | int? | |
| `source` | str | required |
| `source_url` | str? | |
| `as_of` | date | required |
| `confidence` | enum | see below |
| `notes` | str? | |

**A score without its conditions is close to meaningless.** Labs report the same
benchmark at different reasoning efforts, harnesses and tool settings, and the
spread between conditions routinely exceeds the gap between models: DeepSeek
reports V4-Pro at 90.1 on GPQA Diamond in Think-Max and 72.9 in Non-Think, a
17.2-point swing from one setting. `effort`, `tools` and `optimized` are
therefore filled **only when the publisher stated them**. Null means unknown,
never "default".

Benchmark keys distinguish variants that are not comparable to each other:
`swe_bench_verified` / `swe_bench_pro` / `swe_bench_pro_public`, and
`terminal_bench` / `terminal_bench_1_0` / `terminal_bench_2_0` /
`terminal_bench_2_1` / `terminal_bench_hard`.

Two lookup methods matter. `ModelRecord.score_for(benchmark, effort=, tools=)`
returns the best matching single score, preferring exact condition matches and
ranking `primary` above `independent` above `secondary` above `unverified`.
`ModelRecord.score_range_for(...)` returns `(lowest, highest, count)` across
matching entries — and the benchmark probe uses *that*, taking the conservative
end, because independent evaluators disagree about the same model at the same
settings. Claude Opus 5 on ARC-AGI-2 at max effort is 90.4 by ARC Prize and 88.3
by Epoch AI, both independent, both current, 2.1 points apart. Picking one and
calling it *the* reference score would decide by accident of sort order whether
an honest endpoint starts two points in the hole.

## Confidence levels

| level | means |
|---|---|
| `primary` | the lab's own publication |
| `independent` | Epoch AI, ARC Prize, Artificial Analysis — a third party that ran the eval |
| `secondary` | press or third-party write-ups, including figures triangulated from convergent summaries when the lab published only images |
| `unverified` | flagged as unconfirmed at the source |

In the shipped snapshot: 341 `independent`, 104 `primary`, 28 `secondary`, 0
`unverified`.

`secondary` is not a euphemism for "guessed". Anthropic renders its comparison
tables as images, so several Claude Sonnet 5 figures were triangulated from three
convergent secondary sources and labelled accordingly. Where a number could not
be pinned at all it was left out rather than estimated: Kimi K3's official
announcement contains zero benchmark numbers, so no circulating figure for it
appears here.

`primary` ranking above `independent` has a sharp edge worth knowing. A lab
number with no stated effort will win `score_for` over Epoch's conditioned run
unless the caller passes `effort=`. Callers should pass it.

## Sources

Every URL in the snapshot's `sources:` block, with its licensing position.
Attribution lives in [`NOTICE`](../NOTICE).

| source | URL | licence / terms | what it supplies |
|---|---|---|---|
| Epoch AI Capabilities Index | `https://epoch.ai/data/eci_benchmarks.csv` | **CC-BY** — redistribution permitted with credit | most `independent` benchmark scores |
| OpenRouter models | `https://openrouter.ai/api/v1/models` | public API, no key | model ids, context lengths, pricing |
| OpenRouter endpoints | `https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints` | public API, no key | per-provider self-declared quantization |
| Artificial Analysis | `https://artificialanalysis.ai/api/v2/data/llms/models` | Artificial Analysis's terms | `aa_intelligence_index`, `aa_coding_index`, `aa_agentic_index` |
| ARC Prize | `https://arcprize.org/leaderboard` | — | ARC-AGI-1/2/3 scores |
| Terminal-Bench | `https://www.tbench.ai/leaderboard/terminal-bench/2.0` | — | Terminal-Bench scores |
| Aider polyglot | `https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml` | Apache-2.0 | Aider polyglot scores |
| Anthropic docs | `https://platform.claude.com/docs/` | — | ids, ladders, token-accounting table |
| Anthropic, Claude Opus 5 | `https://www.anthropic.com/news/claude-opus-5` | — | `primary` Opus 5 scores |
| OpenAI docs | `https://developers.openai.com/api/docs/` | — | ids, context, pricing |
| OpenAI, GPT-5.6 | `https://openai.com/index/gpt-5-6/` | — | `primary` GPT-5.6 scores |
| xAI docs | `https://docs.x.ai/` | — | Grok ids and pricing |
| Stanford HELM | `https://storage.googleapis.com/crfm-helm-public/` | — | cross-checks |
| HF datasets-server | `https://datasets-server.huggingface.co/` | per dataset | benchmark items at run time |
| LMArena | HF `lmarena-ai/leaderboard-dataset` | CC-BY-4.0 | Elo cross-checks |

The Artificial Analysis API returns 401 without a key, so those index values are
not fetched directly; they reach the snapshot embedded in OpenRouter's model
records and are attributed to Artificial Analysis, with the source string
"Artificial Analysis, relayed by OpenRouter". They are composites, not single
benchmarks, and their conditions are not published per model, so treat them as
weak evidence.

Two sources are deliberately *not* used. The HuggingFace Open LLM Leaderboard was
retired on 2025-03-13 and frozen since 2025-03-20. And no model's own
self-description is ever a source, for the reasons in
[`docs/probes.md`](probes.md#self_report).

## `llmverify refresh`

```console
$ llmverify refresh                          # dry run: fetch, diff, print
$ llmverify refresh --source epoch           # one source only
$ llmverify refresh --write                  # apply the merge
$ llmverify refresh --snapshot ./mine.yaml   # refresh a different file
```

Known sources are `epoch` and `openrouter`; the default is both.

**It is a dry run by default and that is not a convenience.** The number in this
file is the number a provider gets accused of failing to reach, so a human should
read the change in a pull request before it takes effect. A real run today:

```
reference refresh 2026-07-26T23:59:27+00:00
  mode: dry run
  source epoch: ok, 232 scores across 33 known models
  source openrouter: ok, 343 models listed, 45 matched to records

no changes: the snapshot matches every source consulted

unmatched upstream models (13)
  ! GLM-5 (model_version glm-5)
  ! Gemma 4 31B IT (model_version gemma-4-31b-it)
  ! Grok 4.20 (model_version grok-4.20-0309-reasoning)
  ! Kimi K2.6 (model_version kimi-k2.6)
  ! Qwen 3.6 Max (Preview) (model_version qwen3.6-max-preview)
  ...
  add a row to EPOCH_MODEL_IDS after confirming which model each one is
```

Five behaviours are worth knowing.

- **A partial refresh is a result.** A source that fails is recorded and the
  others still run; the report says which half succeeded. Exit code 4 only when
  *no* source was reached.
- **Nothing is ever deleted.** A snapshot model missing from the live OpenRouter
  catalogue is listed under "no longer listed upstream, not deleted". A catalogue
  hiccup must not erase a reference threshold.
- **Unmatched upstream models are reported, not guessed at.** Epoch publishes
  models this snapshot has no record for. Adding a mapping means editing
  `EPOCH_MODEL_IDS` in `llmverify/reference/refresh.py` after confirming which
  model each one is — never by fuzzy name matching, which is how `gpt-5.6-luna`
  ends up compared against `gpt-5.6-sol`'s numbers.
- **Only benchmarks in `EPOCH_BENCHMARKS` are imported.** A benchmark whose
  variant cannot be named is a benchmark whose scores cannot be compared, so it
  is ignored rather than guessed at.
- **Writing loses the file's comments.** `--write` re-serialises from the schema.
  The leading comment block is preserved; per-record prose is not. Keep
  provenance in `notes` fields, which survive.

## Adding a model by hand

Most records cannot be refreshed automatically — token accounting, effort
ladders, `primary` lab scores and anything a lab published as an image all have
to be typed in. Append to `models:` in `reference.yaml`:

```yaml
- id: acme-titan-2
  vendor: acme
  # API family: openai | anthropic | gemini | openai_compat
  family: openai_compat
  display_name: Acme Titan 2
  aliases:
    - acme/acme-titan-2          # OpenRouter-style slug
    - acme-titan-2-20260701      # dated snapshot id
  released: '2026-07-01'
  # The cutoff the vendor publishes. Leave knowledge_cutoff null unless you have
  # actually measured one by probing.
  training_cutoff: '2026-03-01'
  reasoning: true
  default_effort: medium
  effort_ladder: [high, medium, low]
  context_window: 262144
  max_output_tokens: 65536
  modalities: [text, image]
  open_weights: false
  pricing:
    input_per_mtok: 2.0
    output_per_mtok: 8.0
    source: https://acme.example/pricing
  scores:
    - benchmark: simpleqa_verified
      score: 61.4
      # Fill effort/tools/optimized ONLY if the publisher stated them.
      effort: high
      tools: false
      n_items: 1000
      source: Acme, Titan 2 announcement
      source_url: https://acme.example/blog/titan-2
      as_of: '2026-07-01'
      confidence: primary
    - benchmark: gpqa_diamond
      score: 89.7
      source: Epoch AI Capabilities Index
      source_url: https://epoch.ai/data/eci_benchmarks.csv
      as_of: '2026-07-26'
      confidence: independent
      notes: Saturated benchmark; never sufficient on its own.
  notes: >-
    Effort not stated for the GPQA figure, so no effort key is recorded.
```

Then check it:

```console
$ llmverify models acme-titan-2
$ python -c "from llmverify.reference import load_snapshot; load_snapshot()"
```

A malformed snapshot raises `ReferenceDataError` rather than warning, because
running with a half-parsed reference would silently drop the thresholds a verdict
depends on.

Four rules for hand edits.

1. **Never invent a number.** Omitting a score costs you one comparison. A wrong
   score produces a confident false accusation. If a lab published a chart image
   and you are reading a third-party transcription, mark it `secondary` and say
   so in `notes`.
2. **Record conditions or leave them null.** Never fill `effort` with what you
   assume the default was.
3. **Aliases are exact, not fuzzy.** Identifier resolution normalises casing,
   separators, a trailing date stamp, an OpenRouter `vendor/` prefix and routing
   tags such as `:free` — but never guesses. An identifier that does not resolve
   exactly resolves to `None` and the run reports the model as unknown, which is
   the correct outcome. Resolution also refuses to answer when a normalised form
   matches more than one record.
4. **Bump `as_of` on the snapshot** when you change numbers, so the staleness
   warning stays truthful.

## Using your own snapshot

Nothing requires the bundled file. `--snapshot PATH` works on `models` and
`refresh`, and from Python:

```python
from pathlib import Path

from llmverify.reference import load_snapshot

snapshot = load_snapshot(Path("./my-snapshot.yaml"))
```

This is the supported way to disagree with a threshold. Copy the file, change the
number, keep the diff, and the verdict a run produces is auditable against your
own reference rather than against this repository's.
