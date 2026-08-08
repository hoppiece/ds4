#!/usr/bin/env python3
"""Causally evaluate simple next-token Expert prefetch predictors.

Predictions are issued after a decode source row has updated the current cache
policy.  Entries resident at that issue point are not submitted for I/O.  The
raw and submitted sets are retained until the next decode row for the same
request/layer, where the trace hit mask supplies authoritative demand misses.

This is an optimistic feasibility ceiling: speculative slots, bandwidth,
deadlines, cancellation, and demand interference are intentionally deferred.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import simulate_expert_cache as cache_sim


PREDICTOR_DEPTHS = {
    "previous-token": ("top1", "top2", "full"),
    "recent-frequency": ("top1", "top2"),
    "transition": ("top1", "top2"),
    "static-hotlist": ("top1", "top2"),
}


class AnalysisError(ValueError):
    """The requested prefetch analysis cannot be evaluated."""


@dataclass(frozen=True)
class PendingPrediction:
    source_token: int
    raw: frozenset[int]
    submitted: frozenset[int]
    suppressed: frozenset[int]
    expert_bytes: int


@dataclass
class _Metrics:
    eligible_events: int = 0
    raw_predictions: int = 0
    raw_correct_predictions: int = 0
    eligible_selected_refs: int = 0
    resident_suppressed: int = 0
    submitted_io: int = 0
    useful_submitted_io: int = 0
    wasted_submitted_io: int = 0
    pin_only_misses: int = 0
    eligible_demand_misses: int = 0
    submitted_bytes: int = 0
    useful_bytes: int = 0
    wasted_bytes: int = 0
    pin_only_bytes: int = 0
    eligible_demand_bytes: int = 0


@dataclass(frozen=True)
class PrefetchReport:
    predictor: str
    depth: str
    capacity: int
    warmup_requests: int
    recent_window: int
    measured_decode_events: int
    eligible_events: int
    raw_predictions: int
    raw_correct_predictions: int
    eligible_selected_refs: int
    raw_prediction_precision: float | None
    raw_route_coverage: float | None
    resident_suppressed: int
    submitted_io: int
    useful_submitted_io: int
    wasted_submitted_io: int
    pin_only_misses: int
    eligible_demand_misses: int
    global_demand_misses: int
    submitted_io_precision: float | None
    eligible_demand_miss_coverage: float | None
    global_demand_miss_coverage: float | None
    submitted_bytes: int
    useful_bytes: int
    wasted_bytes: int
    pin_only_bytes: int
    eligible_demand_bytes: int
    global_demand_bytes: int
    wasted_over_demand_bytes: float | None
    simulated_hit_bit_agreement: float
    simulated_hit_rate: float
    observed_hit_rate: float
    hit_rate_delta_pp: float
    precision_gate_pass: bool
    coverage_gate_pass: bool
    waste_gate_pass: bool
    passes_gates: bool


class _PreviousToken:
    name = "previous-token"

    def observe_and_rank(self, event: cache_sim.TraceEvent) -> tuple[int, ...]:
        return event.selected_ids


class _RecentFrequency:
    name = "recent-frequency"

    def __init__(self, window: int) -> None:
        self.history: dict[tuple[int, int], deque[tuple[int, ...]]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def observe_and_rank(self, event: cache_sim.TraceEvent) -> tuple[int, ...]:
        history = self.history[(event.request, event.layer)]
        history.append(event.selected_ids)
        counts: Counter[int] = Counter()
        last_seen: dict[int, tuple[int, int]] = {}
        for age, selected in enumerate(history):
            for route_rank, expert in enumerate(selected):
                counts[expert] += 1
                last_seen[expert] = (age, -route_rank)
        return tuple(
            sorted(
                counts,
                key=lambda expert: (
                    -counts[expert],
                    -last_seen[expert][0],
                    -last_seen[expert][1],
                    expert,
                ),
            )
        )


class _OnlineTransition:
    name = "transition"

    def __init__(self) -> None:
        self.previous: dict[tuple[int, int], tuple[int, ...]] = {}
        self.occurrences: Counter[tuple[int, int]] = Counter()
        self.transitions: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    def observe_and_rank(self, event: cache_sim.TraceEvent) -> tuple[int, ...]:
        request_layer = (event.request, event.layer)
        previous = self.previous.get(request_layer)
        if previous is not None:
            targets = set(event.selected_ids)
            for source in previous:
                key = (event.layer, source)
                self.occurrences[key] += 1
                self.transitions[key].update(targets)
        self.previous[request_layer] = event.selected_ids

        scores: dict[int, float] = defaultdict(float)
        for source in event.selected_ids:
            key = (event.layer, source)
            denominator = self.occurrences[key]
            if denominator == 0:
                continue
            for target, count in self.transitions[key].items():
                scores[target] += count / denominator
        return tuple(sorted(scores, key=lambda expert: (-scores[expert], expert)))


class _StaticHotlist:
    name = "static-hotlist"

    def __init__(self, hotlists: dict[int, tuple[int, ...]]) -> None:
        self.hotlists = hotlists

    def observe_and_rank(self, event: cache_sim.TraceEvent) -> tuple[int, ...]:
        return self.hotlists.get(event.layer, ())


def read_trace_strict(path: str | Path) -> list[cache_sim.TraceEvent]:
    """Read a v1 trace while requiring canonical CRLF physical records."""

    trace_path = Path(path)
    raw = trace_path.read_bytes()
    if not raw:
        raise cache_sim.TraceFormatError(f"{trace_path}: empty trace")
    if not raw.endswith(b"\r\n"):
        raise cache_sim.TraceFormatError(
            f"{trace_path}: trace records must end with CRLF"
        )
    remainder = raw.replace(b"\r\n", b"")
    if b"\r" in remainder or b"\n" in remainder:
        raise cache_sim.TraceFormatError(
            f"{trace_path}: bare CR or LF is not valid trace v1 CSV"
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise cache_sim.TraceFormatError(
            f"{trace_path}: trace must be UTF-8"
        ) from error
    return cache_sim.read_trace(io.StringIO(text, newline=""))


def _request_partition(
    events: Sequence[cache_sim.TraceEvent], warmup_requests: int
) -> tuple[frozenset[int], frozenset[int]]:
    request_order = tuple(dict.fromkeys(event.request for event in events))
    if warmup_requests < 0 or warmup_requests >= len(request_order):
        raise AnalysisError("warmup_requests must leave at least one measured request")
    return (
        frozenset(request_order[:warmup_requests]),
        frozenset(request_order[warmup_requests:]),
    )


def _event_bytes(event: cache_sim.TraceEvent, fallback: int | None) -> int:
    value = event.expert_bytes if event.expert_bytes is not None else fallback
    if value is None or value <= 0:
        raise AnalysisError(
            "every decode event needs expert_bytes or --expert-bytes for byte metrics"
        )
    return value


def _static_hotlists(
    events: Sequence[cache_sim.TraceEvent], warmup_ids: frozenset[int]
) -> dict[int, tuple[int, ...]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for event in events:
        if event.phase == "decode" and event.request in warmup_ids:
            counts[event.layer].update(event.selected_ids)
    return {
        layer: tuple(
            sorted(layer_counts, key=lambda expert: (-layer_counts[expert], expert))
        )
        for layer, layer_counts in counts.items()
    }


def _depth_candidates(
    predictor: str, depth: str, ranking: Sequence[int]
) -> frozenset[int]:
    if depth == "top1":
        limit = 1
    elif depth == "top2":
        limit = 2
    elif predictor == "previous-token" and depth == "full":
        limit = len(ranking)
    else:
        raise AnalysisError(f"invalid predictor depth: {predictor}/{depth}")
    return frozenset(ranking[:limit])


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze(
    events: Sequence[cache_sim.TraceEvent],
    *,
    capacity: int,
    warmup_requests: int = 0,
    expert_bytes: int | None = None,
    recent_window: int = 4,
    hotness_decay_tokens: int = 16,
    layer_reserve: int = 0,
    prefill_batches: bool = True,
    min_precision: float = 0.70,
    min_coverage: float = 0.20,
    max_waste_ratio: float = 0.15,
) -> list[PrefetchReport]:
    """Replay current cache state and evaluate all bounded predictors."""

    if not events:
        raise AnalysisError("trace has no events")
    if capacity <= 0:
        raise AnalysisError("capacity must be positive")
    if recent_window <= 0:
        raise AnalysisError("recent_window must be positive")
    if hotness_decay_tokens <= 0:
        raise AnalysisError("hotness_decay_tokens must be positive")
    if layer_reserve < 0:
        raise AnalysisError("layer_reserve must be non-negative")
    thresholds = (min_precision, min_coverage, max_waste_ratio)
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
        raise AnalysisError("gate thresholds must be finite and non-negative")
    if min_precision > 1.0 or min_coverage > 1.0:
        raise AnalysisError("precision and coverage gates cannot exceed 1")

    warmup_ids, measured_ids = _request_partition(events, warmup_requests)
    decode_events = [event for event in events if event.phase == "decode"]
    if not decode_events:
        raise AnalysisError("trace has no decode events")
    if capacity < max(len(event.selected_ids) for event in events):
        raise AnalysisError("capacity is smaller than a selected Expert set")
    if expert_bytes is not None and expert_bytes <= 0:
        raise AnalysisError("expert_bytes must be positive")
    for event in decode_events:
        _event_bytes(event, expert_bytes)

    predictors = (
        _PreviousToken(),
        _RecentFrequency(recent_window),
        _OnlineTransition(),
        _StaticHotlist(_static_hotlists(events, warmup_ids)),
    )
    metrics = {
        (name, depth): _Metrics()
        for name, depths in PREDICTOR_DEPTHS.items()
        for depth in depths
    }
    pending: dict[tuple[str, str, int, int], PendingPrediction] = {}
    policy = cache_sim.CurrentHotnessPolicy(
        capacity, layer_reserve, hotness_decay_tokens
    )

    measured_decode_events = 0
    global_demand_misses = 0
    global_demand_bytes = 0
    agreement_refs = 0
    agreement_bits = 0
    simulated_hits = 0
    observed_hits = 0
    previous_request: int | None = None
    event_index = 0
    while event_index < len(events):
        event = events[event_index]
        if previous_request is None or event.request != previous_request:
            policy.begin_request()
        previous_request = event.request

        batch_end = (
            cache_sim._prefill_batch_end(events, event_index)
            if prefill_batches
            else event_index + 1
        )
        batch = events[event_index:batch_end]
        outcomes = (
            policy.access_prefill_batch(batch, event_index)
            if prefill_batches and event.phase == "prefill"
            else [policy.access(event, event_index)]
        )
        for source_event, outcome in zip(batch, outcomes):
            if source_event.phase != "decode":
                continue
            size = _event_bytes(source_event, expert_bytes)
            measured = source_event.request in measured_ids
            target_misses = frozenset(
                expert
                for expert, hit in zip(source_event.selected_ids, source_event.hit_mask)
                if not hit
            )
            if measured:
                measured_decode_events += 1
                global_demand_misses += len(target_misses)
                global_demand_bytes += len(target_misses) * size
                agreement_refs += len(source_event.hit_mask)
                simulated_hits += sum(outcome.hits)
                observed_hits += sum(source_event.hit_mask)
                agreement_bits += sum(
                    simulated == observed
                    for simulated, observed in zip(outcome.hits, source_event.hit_mask)
                )

            for predictor in predictors:
                for depth in PREDICTOR_DEPTHS[predictor.name]:
                    key = (
                        predictor.name,
                        depth,
                        source_event.request,
                        source_event.layer,
                    )
                    issued = pending.get(key)
                    if (
                        measured
                        and issued is not None
                        and source_event.token == issued.source_token + 1
                    ):
                        counter = metrics[(predictor.name, depth)]
                        useful = issued.submitted & target_misses
                        pin_only = issued.suppressed & target_misses
                        target_selected = frozenset(source_event.selected_ids)
                        counter.eligible_events += 1
                        counter.raw_predictions += len(issued.raw)
                        counter.raw_correct_predictions += len(
                            issued.raw & target_selected
                        )
                        counter.eligible_selected_refs += len(target_selected)
                        counter.resident_suppressed += len(issued.suppressed)
                        counter.submitted_io += len(issued.submitted)
                        counter.useful_submitted_io += len(useful)
                        counter.wasted_submitted_io += len(issued.submitted - useful)
                        counter.pin_only_misses += len(pin_only)
                        counter.eligible_demand_misses += len(target_misses)
                        counter.submitted_bytes += (
                            len(issued.submitted) * issued.expert_bytes
                        )
                        counter.useful_bytes += len(useful) * issued.expert_bytes
                        counter.wasted_bytes += (
                            len(issued.submitted - useful) * issued.expert_bytes
                        )
                        counter.pin_only_bytes += len(pin_only) * issued.expert_bytes
                        counter.eligible_demand_bytes += len(target_misses) * size

                ranking = predictor.observe_and_rank(source_event)
                resident = {
                    expert
                    for layer, expert in policy.resident
                    if layer == source_event.layer
                }
                for depth in PREDICTOR_DEPTHS[predictor.name]:
                    raw = _depth_candidates(predictor.name, depth, ranking)
                    suppressed = raw & resident
                    pending[
                        (
                            predictor.name,
                            depth,
                            source_event.request,
                            source_event.layer,
                        )
                    ] = PendingPrediction(
                        source_token=source_event.token,
                        raw=raw,
                        submitted=raw - resident,
                        suppressed=suppressed,
                        expert_bytes=size,
                    )
        event_index = batch_end

    agreement = agreement_bits / agreement_refs if agreement_refs else 1.0
    simulated_hit_rate = simulated_hits / agreement_refs if agreement_refs else 0.0
    observed_hit_rate = observed_hits / agreement_refs if agreement_refs else 0.0
    reports: list[PrefetchReport] = []
    for (predictor, depth), counter in metrics.items():
        precision = _ratio(counter.useful_submitted_io, counter.submitted_io)
        raw_precision = _ratio(counter.raw_correct_predictions, counter.raw_predictions)
        raw_coverage = _ratio(
            counter.raw_correct_predictions, counter.eligible_selected_refs
        )
        eligible_coverage = _ratio(
            counter.useful_submitted_io, counter.eligible_demand_misses
        )
        global_coverage = _ratio(counter.useful_submitted_io, global_demand_misses)
        waste_ratio = _ratio(counter.wasted_bytes, global_demand_bytes)
        precision_pass = precision is not None and precision >= min_precision
        coverage_pass = global_coverage is not None and global_coverage >= min_coverage
        waste_pass = waste_ratio is not None and waste_ratio <= max_waste_ratio
        reports.append(
            PrefetchReport(
                predictor=predictor,
                depth=depth,
                capacity=capacity,
                warmup_requests=warmup_requests,
                recent_window=recent_window,
                measured_decode_events=measured_decode_events,
                eligible_events=counter.eligible_events,
                raw_predictions=counter.raw_predictions,
                raw_correct_predictions=counter.raw_correct_predictions,
                eligible_selected_refs=counter.eligible_selected_refs,
                raw_prediction_precision=raw_precision,
                raw_route_coverage=raw_coverage,
                resident_suppressed=counter.resident_suppressed,
                submitted_io=counter.submitted_io,
                useful_submitted_io=counter.useful_submitted_io,
                wasted_submitted_io=counter.wasted_submitted_io,
                pin_only_misses=counter.pin_only_misses,
                eligible_demand_misses=counter.eligible_demand_misses,
                global_demand_misses=global_demand_misses,
                submitted_io_precision=precision,
                eligible_demand_miss_coverage=eligible_coverage,
                global_demand_miss_coverage=global_coverage,
                submitted_bytes=counter.submitted_bytes,
                useful_bytes=counter.useful_bytes,
                wasted_bytes=counter.wasted_bytes,
                pin_only_bytes=counter.pin_only_bytes,
                eligible_demand_bytes=counter.eligible_demand_bytes,
                global_demand_bytes=global_demand_bytes,
                wasted_over_demand_bytes=waste_ratio,
                simulated_hit_bit_agreement=agreement,
                simulated_hit_rate=simulated_hit_rate,
                observed_hit_rate=observed_hit_rate,
                hit_rate_delta_pp=(simulated_hit_rate - observed_hit_rate) * 100.0,
                precision_gate_pass=precision_pass,
                coverage_gate_pass=coverage_pass,
                waste_gate_pass=waste_pass,
                passes_gates=precision_pass and coverage_pass and waste_pass,
            )
        )
    return reports


REPORT_FIELDS = tuple(PrefetchReport.__dataclass_fields__)


def write_reports(reports: Sequence[PrefetchReport], output_format: str) -> None:
    records = [asdict(report) for report in reports]
    if output_format == "json":
        json.dump(records, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=REPORT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity", type=_positive_int, required=True)
    parser.add_argument("--warmup-requests", type=_non_negative_int, default=0)
    parser.add_argument("--expert-bytes", type=_positive_int)
    parser.add_argument("--recent-window", type=_positive_int, default=4)
    parser.add_argument("--hotness-decay-tokens", type=_positive_int, default=16)
    parser.add_argument("--layer-reserve", type=_non_negative_int, default=0)
    parser.add_argument("--no-prefill-batches", action="store_true")
    parser.add_argument("--min-precision", type=float, default=0.70)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--max-waste-ratio", type=float, default=0.15)
    parser.add_argument("--require-gates", action="store_true")
    parser.add_argument("--require-current-agreement-pp", type=float)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args(argv)

    try:
        if args.require_current_agreement_pp is not None and (
            not math.isfinite(args.require_current_agreement_pp)
            or args.require_current_agreement_pp < 0.0
        ):
            raise AnalysisError("agreement threshold must be finite and non-negative")
        events = read_trace_strict(args.trace)
        reports = analyze(
            events,
            capacity=args.capacity,
            warmup_requests=args.warmup_requests,
            expert_bytes=args.expert_bytes,
            recent_window=args.recent_window,
            hotness_decay_tokens=args.hotness_decay_tokens,
            layer_reserve=args.layer_reserve,
            prefill_batches=not args.no_prefill_batches,
            min_precision=args.min_precision,
            min_coverage=args.min_coverage,
            max_waste_ratio=args.max_waste_ratio,
        )
    except (
        OSError,
        AnalysisError,
        cache_sim.TraceFormatError,
        cache_sim.SimulationError,
    ) as error:
        print(f"analyze-expert-prefetch: error: {error}", file=sys.stderr)
        return 2

    write_reports(reports, args.format)
    if args.require_current_agreement_pp is not None:
        threshold = args.require_current_agreement_pp
        if reports and abs(reports[0].hit_rate_delta_pp) > threshold:
            print(
                "analyze-expert-prefetch: current-policy hit rate is outside "
                f"agreement threshold ({reports[0].hit_rate_delta_pp:+.3f} pp)",
                file=sys.stderr,
            )
            return 1
    if args.require_gates and not any(report.passes_gates for report in reports):
        print(
            "analyze-expert-prefetch: no predictor passed precision/coverage/waste gates",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
