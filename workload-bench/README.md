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
references, bind time, SSD load time, and the synchronization wait on missing
Experts:

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
