# Workload benchmark

`ds4-bench --workload-file` measures realistic request sequences while keeping
prefill and decode Expert-cache counters separate. The legacy
`speed-bench/promessi_sposi.txt` benchmark remains useful for fixed hardware
comparisons; this mode is intended for SSD-streaming and cache-policy work.

## Public datasets

The preparation script downloads and normalizes:

- **MT-Bench**: multi-domain, two-turn chat tasks. Second-turn cases include
  the published reference first turn for reproducible context. Apache-2.0.
- **ELYZA-tasks-100**: 100 diverse Japanese instruction tasks. CC-BY-SA-4.0.
- **OpenAssistant OASST1**: human-authored multilingual SFT conversations. A
  deterministic subset of reviewed roots and follow-ups is used. Apache-2.0.
- **LongBench v1**: long-document QA, summarization, meeting, and code tasks.
  A length-spread sample from six task families is used. MIT.

Downloaded and derived third-party data lives under `workload-bench/data/` and
is gitignored. Source URLs, licenses, selection settings, and SHA-256 hashes are
written to `data/manifest.json`. OASST1 is public user content and may contain
sensitive or objectionable text; use it only as local test input and never
execute model-generated commands.

## Prepare

No Python packages are required.

```sh
python3 workload-bench/prepare_workloads.py
```

This produces the shuffled full `data/workload.txt`, a source-interleaved
`data/workload-balanced.txt`, and a four-case `data/workload-quick.txt`.
Use the balanced file with `--workload-limit`: its first 20 records contain five
cases from each source.

## Run

On a 48 GiB Mac, start with `workload-quick.txt`, a one-case limit, and a small
generation count. Increase one dimension at a time after checking Activity
Monitor. The repository's plain `make test` is not a workload smoke test: it
defaults to the full model-quality suite and begins with a 30k+ token
long-context case. Do not run it while tuning cache memory or thermals.

Warm-service mode preserves the Expert cache across independent KV sessions:

```sh
./ds4-bench \
  --metal \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 24GB \
  --ctx-alloc 32768 \
  --gen-tokens 128 \
  --workload-file workload-bench/data/workload-balanced.txt \
  --workload-warmup 4 \
  --workload-limit 20 \
  --csv workload-warm.csv
```

Cold-request mode releases resident Expert slabs and resets counters before
each case. It is intentionally much slower, so normally use a small limit:

```sh
./ds4-bench \
  --metal \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ssd-streaming-cache-experts 24GB \
  --ctx-alloc 32768 \
  --gen-tokens 32 \
  --workload-file workload-bench/data/workload-quick.txt \
  --workload-cold \
  --csv workload-cold.csv
```

The CSV reports request TTFT, prefill throughput, decode-step and inter-token
throughput, decode p50/p95 latency, and phase-specific cache hits, misses,
evictions, SSD bytes, and `pread` time. `ttft_ms` ends when the first output
token is sampled; `first_decode_ms` is the first token re-evaluation and is not
TTFT.

Aggregate a run by source (or use `--group-by category`, `cache_mode`, or
`all`):

```sh
python3 workload-bench/summarize.py workload-warm.csv --group-by source
```

For optimization diagnostics, an optional intrusive mode also records selected
layer calls, all-resident/mixed/all-missing layers, resident and missing Expert
references, bind time, SSD load time remaining after resident submission, and
the subsequent wait for the resident GPU stage:

```sh
./ds4-bench \
  --metal \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 24GB \
  --ctx-alloc 32768 \
  --gen-tokens 32 \
  --workload-file workload-bench/data/workload-quick.txt \
  --workload-warmup 1 \
  --workload-limit 1 \
  --workload-detailed-expert-timing \
  --csv workload-detail.csv
```

Detailed timing adds clocks and synchronization diagnostics inside the Metal
streaming path. Use it to locate I/O and overlap opportunities, but compare TPS
between implementations with the flag disabled.

## Expert route traces and cache replay

The intrusive Expert route trace is deliberately separate from the workload
CSV. Version 1 is UTF-8 RFC 4180 CSV with this exact header (the final column is
optional):

```csv
trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size,expert_bytes
1,prefill,0,0,3,17;42;9,101,2048,7077888
1,decode,0,0,3,17;7;9,101,2048,7077888
```

- `phase` is `prefill` or `decode`. `request` and `token` are local ordinal
  counters, not text or tokenizer IDs.
- `selected_ids` contains unique non-negative Expert IDs in routing order,
  separated by semicolons. `hit_mask` has one `0` or `1` in the same order.
- `cache_size` is the global resident-entry count observed immediately before
  that row's lookups and admissions. This makes it useful for runtime replay
  agreement without requiring a post-load synchronization point.
- `expert_bytes`, when present, is the positive number of SSD bytes read for
  one missing Expert in that row. If it is omitted or empty, pass the slab's
  per-Expert byte size to the simulator with `--expert-bytes`.

The schema permits no prompt text, generated text, token IDs, logits, user
metadata, or unknown columns. The reader rejects unknown columns rather than
risk treating an accidentally enriched trace as privacy-safe. Start tracing
before warmup if the simulator must reproduce runtime hit masks: a non-empty
initial cache cannot be reconstructed from its size alone. When the trace
contains benchmark warmups, `--warmup-requests N` replays their cache effects
but excludes the first `N` request ordinals from reported metrics.

Version 1 writes routed-prefill batches as consecutive per-token rows. By
default, the simulator reconstructs each maximal contiguous
`(request, layer)` sequence whose token ordinal increases by one. It applies
all route-frequency updates first, loads unique Experts in first-occurrence
order while pinning the full unique set, and counts a later occurrence in the
same batch as a local reuse hit. `--no-prefill-batches` retains row-at-a-time
replay only as a diagnostic.

Replay all policies at one or more exact slot budgets printed by the runtime:

```sh
python3 workload-bench/simulate_expert_cache.py expert-route.csv \
  --capacity 1915 \
  --capacity 2522 \
  --capacity 3128 \
  --warmup-requests 2 \
  --expert-bytes 7077888
```

Those are the current model's measured 16/20/24 GiB dynamic-cache budgets;
use the runtime's printed slot counts when the model or reserved prefill budget
changes.

The simulator includes the current 16-decode-token decayed-hotness/LRU policy,
including its metadata-only hotness reset at each new prefill request while
resident entries and LRU ages survive. It also includes LRU, lifetime LFU,
segmented LRU, exact-frequency TinyLFU admission in front of SLRU, and Belady's
offline lower bound. A selected set is one atomic lookup: all hit bits are
measured against pre-event residency, and online replacement does not evict
another Expert selected by the same event. Belady may load but bypass a
one-shot candidate, which is intentional for a lower bound.

`--policy current-layer-quota` is an optional imbalance diagnostic. It divides
the capacity as evenly as possible across all routed layers (the lowest layer
IDs receive the remainder) and applies the current hotness/LRU victim score
inside each strict partition. It is intentionally not part of the default
policy set because unused quota cannot be borrowed. On the balanced v1 trace at
1,915 slots, the strict quota reached a 75.0440% decode hit rate and 455.72 MiB
missing per token, versus 75.7773% and 421.84 MiB for the global current policy.
The 8.0% increase in missing bytes makes layer imbalance an unlikely explanation
for the gap to Belady on this workload.

`--phase-isolated` models the production SLRU boundary: a prefill probation hit
refreshes probation but does not promote, a prefill hit does not change
protected recency, and prefill admissions cannot evict protected decode
entries. If every probation victim is pinned by the current selected set, the
simulator counts the loaded Expert as a transient `bypass`. Version 1 does not
record transient-slot occupancy or stalls, so those costs remain a runtime
measurement rather than a replay result.

Routed prefill can use a separate batch reserve, and some long-prefill paths do
not emit per-token route rows. The reusable-cache simulator does not model that
reserve or invent missing routes. Consequently, use decode hit-rate agreement
as the current-policy reproduction gate; `cache_size_agreement` and prefill
agreement are diagnostics that expose incomplete or batch-deferred state.

The current balanced v1 trace remains just outside the Phase A decode gate:
after two warmups, inferred batch replay gives 75.7773% versus the runtime's
75.1527%, a +0.6246 percentage-point difference. Overall agreement is closer
(-0.1013 points), but that does not satisfy the 0.5-point decode criterion.
The smallest robust v2 fix is to add explicit `batch_id`/`batch_size` metadata
and emit every long-prefill route. If a path cannot emit those rows, add a
privacy-safe decode-start snapshot of `(layer, expert, hotness, last_used_rank)`
so replay can restore reusable state without any prompt or token content.

Each policy/capacity has `all`, `prefill`, and `decode` rows. Metrics include
hit rate, observed-trace hit rate and delta, hit-mask and pre-event cache-size
agreement, missing bytes/token, and evictions/token. Missing-byte metrics are
reported as `n/a` unless every miss has a byte size from the trace or
`--expert-bytes`. Useful variants are:

```sh
# Require current-policy decode to reproduce runtime within 0.5 points.
python3 workload-bench/simulate_expert_cache.py expert-route.csv \
  --capacity 2048 \
  --expert-bytes 7077888 \
  --policy current \
  --require-current-agreement-pp 0.5

# Retain at least two entries per active layer; model cold-request traces.
python3 workload-bench/simulate_expert_cache.py expert-route.csv \
  --capacity 2048 \
  --expert-bytes 7077888 \
  --layer-reserve 2 \
  --reset-per-request \
  --format csv

# Sweep production phase-isolated SLRU protected targets in one replay.
python3 workload-bench/simulate_expert_cache.py expert-route.csv \
  --capacity 2048 \
  --policy slru \
  --phase-isolated \
  --protected-ratio 0.70 \
  --protected-ratio 0.75 \
  --protected-ratio 0.80

# Check whether a strict equal per-layer partition helps at the 16 GiB budget.
python3 workload-bench/simulate_expert_cache.py expert-route.csv \
  --capacity 1915 \
  --warmup-requests 2 \
  --policy current-layer-quota
```

Run the dependency-free focused tests with:

```sh
python3 -m unittest discover -s tests -p 'test_simulate_expert_cache.py' -v
```

For the lowest-risk smoke test, use a 16 GiB cache and only one generated token
(prefill and TTFT, but no decode step):

```sh
./ds4-bench \
  --metal \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 16GB \
  --ctx-alloc 32768 \
  --gen-tokens 1 \
  --workload-file workload-bench/data/workload-quick.txt \
  --workload-limit 1 \
  --csv workload-smoke.csv
```
