#!/usr/bin/env python3
"""Summarize ds4-bench workload CSV without third-party dependencies."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = ("all", "source", "category", "cache_mode")


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def rate(hits: float, misses: float) -> float:
    total = hits + misses
    return hits / total if total else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(group: str, rows: list[dict[str, str]]) -> dict[str, object]:
    prompt_tokens = sum(number(row, "prompt_tokens") for row in rows)
    output_tokens = sum(number(row, "output_tokens") for row in rows)
    decode_steps = sum(number(row, "decode_steps") for row in rows)
    prefill_seconds = sum(
        ratio(number(row, "prompt_tokens"), number(row, "prefill_tps"))
        for row in rows
        if number(row, "prefill_tps") > 0
    )
    decode_seconds = sum(
        ratio(number(row, "decode_steps"), number(row, "decode_step_tps"))
        for row in rows
        if number(row, "decode_step_tps") > 0
    )
    prefill_hits = sum(number(row, "prefill_cache_hits") for row in rows)
    prefill_misses = sum(number(row, "prefill_cache_misses") for row in rows)
    decode_hits = sum(number(row, "decode_cache_hits") for row in rows)
    decode_misses = sum(number(row, "decode_cache_misses") for row in rows)
    ttft = [number(row, "ttft_ms") for row in rows]
    decoded_rows = [row for row in rows if number(row, "decode_steps") > 0]
    case_decode_p50 = [number(row, "decode_p50_ms") for row in decoded_rows]
    case_decode_p95 = [number(row, "decode_p95_ms") for row in decoded_rows]
    decode_missing = sum(number(row, "decode_missing_experts") for row in rows)
    decode_load_ms = sum(number(row, "decode_missing_load_ms") for row in rows)
    decode_wait_ms = sum(number(row, "decode_resident_wait_ms") for row in rows)
    detailed = any(number(row, "detailed_timing") != 0 for row in rows)

    return {
        "group": group,
        "cases": len(rows),
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "prefill_tps": ratio(prompt_tokens, prefill_seconds),
        "decode_tps": ratio(decode_steps, decode_seconds),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "case_decode_p50_median_ms":
            statistics.median(case_decode_p50) if case_decode_p50 else 0.0,
        "case_decode_p95_p95_ms": percentile(case_decode_p95, 0.95),
        "prefill_cache_hit_rate": rate(prefill_hits, prefill_misses),
        "decode_cache_hit_rate": rate(decode_hits, decode_misses),
        "prefill_pread_mib": sum(number(row, "prefill_pread_mib") for row in rows),
        "decode_pread_mib": sum(number(row, "decode_pread_mib") for row in rows),
        "decode_missing_experts_per_step":
            ratio(decode_missing, decode_steps) if detailed else "",
        "decode_missing_load_ms_per_step":
            ratio(decode_load_ms, decode_steps) if detailed else "",
        "decode_resident_wait_ms_per_step":
            ratio(decode_wait_ms, decode_steps) if detailed else "",
    }


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(records: list[dict[str, object]]) -> None:
    fields = list(records[0])
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join("---" for _ in fields) + " |")
    for record in records:
        print("| " + " | ".join(format_value(record[field]) for field in fields) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV produced by ds4-bench --workload-file")
    parser.add_argument(
        "--group-by", choices=GROUP_FIELDS, default="source",
        help="aggregation key (default: source)",
    )
    parser.add_argument("--format", choices=("csv", "markdown"), default="markdown")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error(f"no workload records in {args.csv}")
    required = {"prompt_tokens", "prefill_tps", "decode_steps", "decode_step_tps"}
    missing = required.difference(rows[0])
    if missing:
        parser.error(f"not a workload CSV; missing columns: {', '.join(sorted(missing))}")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = "all" if args.group_by == "all" else row.get(args.group_by, "")
        groups[key].append(row)
    records = [summarize(key, groups[key]) for key in sorted(groups)]

    if args.format == "markdown":
        write_markdown(records)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
