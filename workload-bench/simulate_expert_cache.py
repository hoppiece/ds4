#!/usr/bin/env python3
"""Replay privacy-safe Expert route traces against cache policies.

Trace version 1 is an RFC 4180 CSV with these columns::

    trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size

``selected_ids`` is a semicolon-separated list of unique non-negative Expert
IDs. ``hit_mask`` has one ``0`` or ``1`` per selected ID. ``cache_size`` is the
runtime resident-entry count immediately before the event. An optional positive
``expert_bytes`` column records the SSD bytes needed to load one selected
Expert; ``--expert-bytes`` supplies a fallback when it is absent.

No prompt, generated text, token IDs, logits, or user metadata belongs in this
format. Unknown columns are rejected deliberately so an accidentally enriched
trace does not silently become a shareable artifact.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Sequence


TRACE_VERSION = "1"
PHASES = ("prefill", "decode")
REQUIRED_TRACE_FIELDS = (
    "trace_version",
    "phase",
    "request",
    "token",
    "layer",
    "selected_ids",
    "hit_mask",
    "cache_size",
)
OPTIONAL_TRACE_FIELDS = ("expert_bytes",)
POLICY_NAMES = (
    "current",
    "current-layer-quota",
    "lru",
    "lfu",
    "slru",
    "tinylfu-slru",
    "belady",
)
DEFAULT_POLICY_NAMES = tuple(
    name for name in POLICY_NAMES if name != "current-layer-quota"
)
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

CacheKey = tuple[int, int]  # (layer, expert)


class TraceFormatError(ValueError):
    """The input is not a valid, privacy-safe version 1 route trace."""


class SimulationError(ValueError):
    """The requested policy configuration cannot replay the trace."""


@dataclass(frozen=True)
class TraceEvent:
    phase: str
    request: int
    token: int
    layer: int
    selected_ids: tuple[int, ...]
    hit_mask: tuple[bool, ...]
    cache_size: int
    expert_bytes: int | None = None

    @property
    def keys(self) -> tuple[CacheKey, ...]:
        return tuple((self.layer, expert) for expert in self.selected_ids)

    @property
    def token_key(self) -> tuple[str, int, int]:
        return (self.phase, self.request, self.token)


@dataclass(frozen=True)
class AccessOutcome:
    hits: tuple[bool, ...]
    resident_before: int
    evictions: int = 0
    admissions: int = 0
    bypasses: int = 0


def _batch_layout(
    events: Sequence[TraceEvent],
    initially_resident: set[CacheKey],
) -> tuple[tuple[CacheKey, ...], tuple[bool, ...], tuple[tuple[bool, ...], ...]]:
    """Return unique accesses and synthetic per-row hits for one prefill batch."""

    seen: set[CacheKey] = set()
    unique: list[CacheKey] = []
    row_hits: list[tuple[bool, ...]] = []
    for event in events:
        hits: list[bool] = []
        for key in event.keys:
            hits.append(key in initially_resident or key in seen)
            if key not in seen:
                seen.add(key)
                unique.append(key)
        row_hits.append(tuple(hits))
    return (
        tuple(unique),
        tuple(key in initially_resident for key in unique),
        tuple(row_hits),
    )


def _batch_outcomes(
    aggregate: AccessOutcome,
    row_hits: Sequence[tuple[bool, ...]],
) -> list[AccessOutcome]:
    """Attach batch mutations once while preserving every source trace row."""

    outcomes = [
        AccessOutcome(hits=hits, resident_before=aggregate.resident_before)
        for hits in row_hits
    ]
    outcomes[-1] = AccessOutcome(
        hits=outcomes[-1].hits,
        resident_before=aggregate.resident_before,
        evictions=aggregate.evictions,
        admissions=aggregate.admissions,
        bypasses=aggregate.bypasses,
    )
    return outcomes


def _prefill_batch_end(events: Sequence[TraceEvent], start: int) -> int:
    """Find a contiguous request/layer prefill batch with token steps of one."""

    first = events[start]
    if first.phase != "prefill":
        return start + 1
    end = start + 1
    previous_token = first.token
    while end < len(events):
        event = events[end]
        if (
            event.phase != "prefill"
            or event.request != first.request
            or event.layer != first.layer
            or event.token != previous_token + 1
        ):
            break
        previous_token = event.token
        end += 1
    return end


@dataclass
class _Accumulator:
    events: int = 0
    references: int = 0
    hits: int = 0
    trace_hits: int = 0
    matching_hit_bits: int = 0
    matching_cache_sizes: int = 0
    evictions: int = 0
    admissions: int = 0
    bypasses: int = 0
    missing_bytes: int = 0
    missing_bytes_known: bool = True
    token_keys: set[tuple[str, int, int]] = field(default_factory=set)

    def record(
        self,
        event: TraceEvent,
        outcome: AccessOutcome,
        fallback_expert_bytes: int | None,
    ) -> None:
        self.events += 1
        self.token_keys.add(event.token_key)
        self.references += len(outcome.hits)
        self.hits += sum(outcome.hits)
        self.trace_hits += sum(event.hit_mask)
        self.matching_hit_bits += sum(
            simulated == observed
            for simulated, observed in zip(outcome.hits, event.hit_mask)
        )
        self.matching_cache_sizes += outcome.resident_before == event.cache_size
        self.evictions += outcome.evictions
        self.admissions += outcome.admissions
        self.bypasses += outcome.bypasses

        misses = len(outcome.hits) - sum(outcome.hits)
        if misses:
            expert_bytes = (
                event.expert_bytes
                if event.expert_bytes is not None
                else fallback_expert_bytes
            )
            if expert_bytes is None:
                self.missing_bytes_known = False
            else:
                self.missing_bytes += misses * expert_bytes


@dataclass(frozen=True)
class ReportRow:
    policy: str
    capacity: int
    warmup_requests: int
    layer_reserve: int
    protected_ratio: float
    phase_isolated: bool
    strict_layer_quota: bool
    prefill_batched: bool
    phase: str
    events: int
    tokens: int
    references: int
    hits: int
    misses: int
    hit_rate: float
    trace_hit_rate: float
    hit_rate_delta_pp: float
    hit_mask_agreement: float
    cache_size_agreement: float
    missing_bytes: int | None
    missing_bytes_per_token: float | None
    evictions: int
    evictions_per_token: float
    admissions: int
    bypasses: int


def _trace_error(line: int, message: str) -> TraceFormatError:
    location = "header" if line == 1 else f"line {line}"
    return TraceFormatError(f"{location}: {message}")


def _parse_uint(
    raw: str | None,
    field_name: str,
    line: int,
    *,
    maximum: int = UINT64_MAX,
    positive: bool = False,
) -> int:
    if raw is None or not raw or not raw.isascii() or not raw.isdigit():
        qualifier = "positive" if positive else "non-negative"
        raise _trace_error(line, f"{field_name} must be a {qualifier} integer")
    value = int(raw)
    if positive and value == 0:
        raise _trace_error(line, f"{field_name} must be a positive integer")
    if value > maximum:
        raise _trace_error(line, f"{field_name} exceeds {maximum}")
    return value


def _read_trace_handle(handle: IO[str]) -> list[TraceEvent]:
    reader = csv.DictReader(handle)
    header = reader.fieldnames
    if header is None:
        raise _trace_error(1, "empty trace")
    if len(header) != len(set(header)):
        duplicates = sorted(name for name in set(header) if header.count(name) > 1)
        raise _trace_error(1, f"duplicate columns: {', '.join(duplicates)}")

    required = set(REQUIRED_TRACE_FIELDS)
    allowed = required | set(OPTIONAL_TRACE_FIELDS)
    missing = required.difference(header)
    unknown = set(header).difference(allowed)
    if missing:
        raise _trace_error(1, f"missing columns: {', '.join(sorted(missing))}")
    if unknown:
        raise _trace_error(
            1,
            "unknown columns are not privacy-safe: " + ", ".join(sorted(unknown)),
        )

    events: list[TraceEvent] = []
    for row in reader:
        line = reader.line_num
        if None in row:
            raise _trace_error(line, "row has more values than the header")
        if any(value is None for value in row.values()):
            raise _trace_error(line, "row has fewer values than the header")
        if row["trace_version"] != TRACE_VERSION:
            raise _trace_error(
                line,
                f"unsupported trace_version {row['trace_version']!r}; expected {TRACE_VERSION}",
            )
        phase = row["phase"]
        if phase not in PHASES:
            raise _trace_error(line, "phase must be 'prefill' or 'decode'")

        request = _parse_uint(row["request"], "request", line, maximum=UINT32_MAX)
        token = _parse_uint(row["token"], "token", line, maximum=UINT32_MAX)
        layer = _parse_uint(row["layer"], "layer", line, maximum=UINT32_MAX)
        cache_size = _parse_uint(
            row["cache_size"], "cache_size", line, maximum=UINT32_MAX
        )

        raw_ids = row["selected_ids"]
        if not raw_ids:
            raise _trace_error(line, "selected_ids must not be empty")
        selected_ids = tuple(
            _parse_uint(part, "selected_ids", line, maximum=UINT32_MAX)
            for part in raw_ids.split(";")
        )
        if len(selected_ids) != len(set(selected_ids)):
            raise _trace_error(line, "selected_ids must be unique within an event")

        raw_mask = row["hit_mask"]
        if len(raw_mask) != len(selected_ids) or any(
            bit not in "01" for bit in raw_mask
        ):
            raise _trace_error(
                line,
                "hit_mask must contain one '0' or '1' per selected Expert",
            )
        hit_mask = tuple(bit == "1" for bit in raw_mask)

        expert_bytes: int | None = None
        if "expert_bytes" in row and row["expert_bytes"]:
            expert_bytes = _parse_uint(
                row["expert_bytes"], "expert_bytes", line, positive=True
            )
        events.append(
            TraceEvent(
                phase=phase,
                request=request,
                token=token,
                layer=layer,
                selected_ids=selected_ids,
                hit_mask=hit_mask,
                cache_size=cache_size,
                expert_bytes=expert_bytes,
            )
        )

    if not events:
        raise _trace_error(1, "trace has a header but no events")
    return events


def read_trace(source: str | Path | IO[str]) -> list[TraceEvent]:
    """Read and validate a version 1 route trace.

    Paths are decoded with ``utf-8-sig`` so a spreadsheet-added UTF-8 BOM does
    not change the first header name. File-like objects are read as supplied.
    """

    if hasattr(source, "read"):
        return _read_trace_handle(source)  # type: ignore[arg-type]
    path = Path(source)
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return _read_trace_handle(handle)
    except UnicodeDecodeError as error:
        raise TraceFormatError(f"{path}: trace must be UTF-8") from error


class CachePolicy(ABC):
    name: str

    def __init__(self, capacity: int, layer_reserve: int = 0) -> None:
        self.capacity = capacity
        self.layer_reserve = layer_reserve

    @property
    @abstractmethod
    def resident(self) -> set[CacheKey]:
        """Return the currently reusable cache keys."""

    @abstractmethod
    def access(self, event: TraceEvent, event_index: int) -> AccessOutcome:
        """Replay one atomic selected-Expert event."""

    @abstractmethod
    def reset(self) -> None:
        """Drop reusable entries and policy history at a cold request boundary."""

    def begin_request(self) -> None:
        """Apply metadata-only policy work at a warm request boundary."""

    def access_prefill_batch(
        self,
        events: Sequence[TraceEvent],
        first_event_index: int,
    ) -> list[AccessOutcome]:
        """Replay a reconstructed batch; policies may override atomicity."""

        return [
            self.access(event, first_event_index + offset)
            for offset, event in enumerate(events)
        ]


class _ScoredPolicy(CachePolicy):
    """Common admission path for LRU and route-frequency policies."""

    def __init__(
        self,
        capacity: int,
        layer_reserve: int = 0,
        layer_quotas: dict[int, int] | None = None,
    ) -> None:
        super().__init__(capacity, layer_reserve)
        self.layer_quotas = layer_quotas
        self.entries: dict[CacheKey, int] = {}
        self.layer_counts: dict[int, int] = defaultdict(int)
        self.clock = 0
        self.victim_heap: list[tuple[int, int, CacheKey]] = []

    @property
    def resident(self) -> set[CacheKey]:
        return set(self.entries)

    def reset(self) -> None:
        self.entries.clear()
        self.layer_counts.clear()
        self.clock = 0
        self.victim_heap.clear()
        self._reset_history()

    def _reset_history(self) -> None:
        pass

    def _before_event(self, event: TraceEvent) -> None:
        pass

    @abstractmethod
    def _victim_score(self, key: CacheKey) -> tuple[int, int, CacheKey]:
        """Return a score where the smallest entry is evicted."""

    def _push_score(self, key: CacheKey) -> None:
        heapq.heappush(self.victim_heap, self._victim_score(key))

    def _rebuild_heap(self) -> None:
        self.victim_heap = [self._victim_score(key) for key in self.entries]
        heapq.heapify(self.victim_heap)

    def _choose_victim(
        self,
        selected: frozenset[CacheKey],
        candidate: CacheKey,
    ) -> CacheKey | None:
        layer_counts = (
            self.layer_counts
            if self.layer_reserve or self.layer_quotas is not None
            else None
        )
        quota_layer: int | None = None
        if self.layer_quotas is not None and layer_counts is not None:
            if layer_counts[candidate[0]] >= self.layer_quotas[candidate[0]]:
                quota_layer = candidate[0]
        deferred: list[tuple[int, int, CacheKey]] = []
        victim: CacheKey | None = None
        while self.victim_heap:
            score = heapq.heappop(self.victim_heap)
            key = score[-1]
            if key not in self.entries or score != self._victim_score(key):
                continue
            if key in selected or (quota_layer is not None and key[0] != quota_layer):
                deferred.append(score)
                continue
            if (
                quota_layer is None
                and layer_counts is not None
                and self.layer_reserve
                and key[0] != candidate[0]
                and layer_counts[key[0]] <= self.layer_reserve
            ):
                deferred.append(score)
                continue
            victim = key
            break
        for score in deferred:
            heapq.heappush(self.victim_heap, score)
        return victim

    def _apply_accesses(
        self,
        keys: Sequence[CacheKey],
        hits: Sequence[bool],
        resident_before: int,
        event_index: int,
    ) -> AccessOutcome:
        selected = frozenset(keys)
        evictions = 0
        admissions = 0
        bypasses = 0
        for key, hit in zip(keys, hits):
            if not hit:
                continue
            self.clock += 1
            self.entries[key] = self.clock
            self._push_score(key)
        for key, hit in zip(keys, hits):
            if hit:
                continue
            self.clock += 1
            quota_full = (
                self.layer_quotas is not None
                and self.layer_counts[key[0]] >= self.layer_quotas[key[0]]
            )
            if len(self.entries) >= self.capacity or quota_full:
                victim = self._choose_victim(selected, key)
                if victim is None:
                    if self.layer_quotas is not None:
                        bypasses += 1
                        continue
                    raise SimulationError(
                        f"{self.name}: no victim at event {event_index}; "
                        "capacity/reserve cannot hold the selected set"
                    )
                del self.entries[victim]
                self.layer_counts[victim[0]] -= 1
                evictions += 1
            self.entries[key] = self.clock
            self.layer_counts[key[0]] += 1
            self._push_score(key)
            admissions += 1
        if len(self.victim_heap) > self.capacity * 4:
            self._rebuild_heap()
        return AccessOutcome(
            tuple(hits),
            resident_before,
            evictions,
            admissions,
            bypasses,
        )

    def access(self, event: TraceEvent, event_index: int) -> AccessOutcome:
        keys = event.keys
        resident_before = len(self.entries)
        hits = tuple(key in self.entries for key in keys)
        self._before_event(event)
        return self._apply_accesses(keys, hits, resident_before, event_index)

    def access_prefill_batch(
        self,
        events: Sequence[TraceEvent],
        first_event_index: int,
    ) -> list[AccessOutcome]:
        resident_before = len(self.entries)
        unique, unique_hits, row_hits = _batch_layout(events, self.resident)
        # Runtime prepare_selected_batch records all token frequencies before
        # looking up or replacing any unique Expert.
        for event in events:
            self._before_event(event)
        aggregate = self._apply_accesses(
            unique,
            unique_hits,
            resident_before,
            first_event_index,
        )
        return _batch_outcomes(aggregate, row_hits)


class LRUPolicy(_ScoredPolicy):
    name = "lru"

    def _victim_score(self, key: CacheKey) -> tuple[int, int, CacheKey]:
        return (0, self.entries[key], key)


class LFUPolicy(_ScoredPolicy):
    name = "lfu"

    def __init__(self, capacity: int, layer_reserve: int = 0) -> None:
        super().__init__(capacity, layer_reserve)
        self.frequency: dict[CacheKey, int] = defaultdict(int)

    def _reset_history(self) -> None:
        self.frequency.clear()

    def _before_event(self, event: TraceEvent) -> None:
        for key in event.keys:
            self.frequency[key] += 1

    def _victim_score(self, key: CacheKey) -> tuple[int, int, CacheKey]:
        return (self.frequency[key], self.entries[key], key)


class CurrentHotnessPolicy(_ScoredPolicy):
    """Model the runtime's decayed route-hotness, with LRU tie-breaking."""

    name = "current"

    def __init__(
        self,
        capacity: int,
        layer_reserve: int = 0,
        decay_tokens: int = 16,
        layer_quotas: dict[int, int] | None = None,
    ) -> None:
        super().__init__(capacity, layer_reserve, layer_quotas)
        self.decay_tokens = decay_tokens
        self.hotness: dict[CacheKey, int] = defaultdict(int)
        self.decode_token_count = 0
        self.decay_anchor = 0
        self.seen_decode_tokens: set[tuple[int, int]] = set()

    def _reset_history(self) -> None:
        self.hotness.clear()
        self.decode_token_count = 0
        self.decay_anchor = 0
        self.seen_decode_tokens.clear()

    def begin_request(self) -> None:
        # The runtime clears route scores at prefill start while retaining the
        # reusable entries and their LRU clock. Rebuild the lazy victim heap so
        # every resident immediately has the new zero hotness score.
        self._reset_history()
        self._rebuild_heap()

    def _before_event(self, event: TraceEvent) -> None:
        decode_key = (event.request, event.token)
        if event.phase == "decode" and decode_key not in self.seen_decode_tokens:
            self.seen_decode_tokens.add(decode_key)
            self.decode_token_count += 1
            if self.decay_anchor == 0:
                self.decay_anchor = self.decode_token_count
            while self.decode_token_count - self.decay_anchor >= self.decay_tokens:
                self.hotness = defaultdict(
                    int,
                    {
                        key: value >> 1
                        for key, value in self.hotness.items()
                        if value >> 1
                    },
                )
                self.decay_anchor += self.decay_tokens
                self._rebuild_heap()
        for key in event.keys:
            self.hotness[key] = min(UINT32_MAX, self.hotness[key] + 1)

    def _victim_score(self, key: CacheKey) -> tuple[int, int, CacheKey]:
        return (self.hotness[key], self.entries[key], key)


class CurrentLayerQuotaPolicy(CurrentHotnessPolicy):
    """Current score with a deterministic strict per-layer partition."""

    name = "current-layer-quota"

    def __init__(
        self,
        capacity: int,
        layers: Sequence[int],
        decay_tokens: int = 16,
    ) -> None:
        ordered_layers = tuple(sorted(set(layers)))
        per_layer, remainder = divmod(capacity, len(ordered_layers))
        if per_layer == 0:
            raise SimulationError(
                "current-layer-quota requires at least one slot per active layer"
            )
        quotas = {
            layer: per_layer + (index < remainder)
            for index, layer in enumerate(ordered_layers)
        }
        super().__init__(
            capacity,
            decay_tokens=decay_tokens,
            layer_quotas=quotas,
        )


class SLRUPolicy(CachePolicy):
    name = "slru"

    def __init__(
        self,
        capacity: int,
        layer_reserve: int = 0,
        protected_ratio: float = 0.8,
        tiny_lfu: bool = False,
        sample_multiplier: int = 10,
        phase_isolated: bool = False,
    ) -> None:
        super().__init__(capacity, layer_reserve)
        self.protected_ratio = protected_ratio
        self.protected_capacity = (
            0
            if capacity == 1
            else min(capacity - 1, max(1, int(capacity * protected_ratio)))
        )
        self.tiny_lfu = tiny_lfu
        self.phase_isolated = phase_isolated
        self.sample_size = max(1, capacity * sample_multiplier)
        self.probation: OrderedDict[CacheKey, None] = OrderedDict()
        self.protected: OrderedDict[CacheKey, None] = OrderedDict()
        self.frequency: dict[CacheKey, int] = defaultdict(int)
        self.frequency_samples = 0
        if tiny_lfu:
            self.name = "tinylfu-slru"

    @property
    def resident(self) -> set[CacheKey]:
        return set(self.probation) | set(self.protected)

    def reset(self) -> None:
        self.probation.clear()
        self.protected.clear()
        self.frequency.clear()
        self.frequency_samples = 0

    def _record_frequency(self, keys: Sequence[CacheKey]) -> None:
        if not self.tiny_lfu:
            return
        for key in keys:
            if self.frequency_samples >= self.sample_size:
                self.frequency = defaultdict(
                    int,
                    {
                        old_key: count >> 1
                        for old_key, count in self.frequency.items()
                        if count >> 1
                    },
                )
                self.frequency_samples = 0
            self.frequency[key] += 1
            self.frequency_samples += 1

    def _promote(self, key: CacheKey) -> None:
        del self.probation[key]
        self.protected[key] = None
        if len(self.protected) > self.protected_capacity:
            demoted, _ = self.protected.popitem(last=False)
            self.probation[demoted] = None

    def _victim(
        self,
        selected: frozenset[CacheKey],
        candidate: CacheKey,
        allow_protected: bool = True,
    ) -> CacheKey | None:
        layer_counts = (
            Counter(layer for layer, _ in self.resident) if self.layer_reserve else None
        )
        segments = (
            (self.probation, self.protected) if allow_protected else (self.probation,)
        )
        for segment in segments:
            for key in segment:
                if key in selected:
                    continue
                if (
                    layer_counts is not None
                    and key[0] != candidate[0]
                    and layer_counts[key[0]] <= self.layer_reserve
                ):
                    continue
                return key
        return None

    def _remove(self, key: CacheKey) -> None:
        if key in self.probation:
            del self.probation[key]
        else:
            del self.protected[key]

    def _apply_accesses(
        self,
        event: TraceEvent,
        keys: Sequence[CacheKey],
        hits: Sequence[bool],
        resident_before: int,
        event_index: int,
        *,
        record_frequency: bool,
    ) -> AccessOutcome:
        selected = frozenset(keys)
        if record_frequency:
            self._record_frequency(keys)

        # Promotions happen before admissions, matching a batched lookup whose
        # complete hit mask is known before any missing Expert is installed.
        for key, hit in zip(keys, hits):
            if not hit:
                continue
            if key in self.protected:
                if not self.phase_isolated or event.phase == "decode":
                    self.protected.move_to_end(key)
            elif self.phase_isolated and event.phase == "prefill":
                self.probation.move_to_end(key)
            else:
                self._promote(key)

        evictions = 0
        admissions = 0
        bypasses = 0
        for key, hit in zip(keys, hits):
            if hit:
                continue
            victim: CacheKey | None = None
            if len(self.probation) + len(self.protected) >= self.capacity:
                protect_decode = self.phase_isolated and event.phase == "prefill"
                victim = self._victim(
                    selected,
                    key,
                    allow_protected=not protect_decode,
                )
                if victim is None:
                    if protect_decode:
                        # The runtime uses a transient slot here. The offline
                        # reusable-cache model records the load as a bypass;
                        # transient capacity and stalls are intentionally out
                        # of scope for this trace version.
                        bypasses += 1
                        continue
                    raise SimulationError(
                        f"{self.name}: no victim at event {event_index}; "
                        "capacity/reserve cannot hold the selected set"
                    )
                reserve_fill = False
                if self.layer_reserve:
                    layer_count = sum(layer == key[0] for layer, _ in self.resident)
                    reserve_fill = layer_count < self.layer_reserve
                if (
                    self.tiny_lfu
                    and not reserve_fill
                    and self.frequency[key] <= self.frequency[victim]
                ):
                    bypasses += 1
                    continue
                self._remove(victim)
                evictions += 1
            self.probation[key] = None
            admissions += 1

        return AccessOutcome(
            tuple(hits),
            resident_before,
            evictions=evictions,
            admissions=admissions,
            bypasses=bypasses,
        )

    def access(self, event: TraceEvent, event_index: int) -> AccessOutcome:
        keys = event.keys
        resident_before = len(self.probation) + len(self.protected)
        hits = tuple(key in self.probation or key in self.protected for key in keys)
        return self._apply_accesses(
            event,
            keys,
            hits,
            resident_before,
            event_index,
            record_frequency=True,
        )

    def access_prefill_batch(
        self,
        events: Sequence[TraceEvent],
        first_event_index: int,
    ) -> list[AccessOutcome]:
        resident_before = len(self.probation) + len(self.protected)
        unique, unique_hits, row_hits = _batch_layout(events, self.resident)
        self._record_frequency(tuple(key for event in events for key in event.keys))
        aggregate = self._apply_accesses(
            events[0],
            unique,
            unique_hits,
            resident_before,
            first_event_index,
            record_frequency=False,
        )
        return _batch_outcomes(aggregate, row_hits)


class BeladyPolicy(CachePolicy):
    """Offline lower bound with admission bypass and perfect next-use data."""

    name = "belady"

    def __init__(
        self,
        capacity: int,
        events: Sequence[TraceEvent],
        layer_reserve: int = 0,
        reset_per_request: bool = False,
    ) -> None:
        super().__init__(capacity, layer_reserve)
        self.events = events
        self.reset_per_request = reset_per_request
        self.entries: set[CacheKey] = set()
        self.future: dict[CacheKey, deque[int]] = defaultdict(deque)
        self.victim_heap: list[tuple[float, int, int, CacheKey]] = []
        for event_index, event in enumerate(events):
            for key in event.keys:
                self.future[key].append(event_index)

    @property
    def resident(self) -> set[CacheKey]:
        return set(self.entries)

    def reset(self) -> None:
        self.entries.clear()
        self.victim_heap.clear()

    def _next_use(self, key: CacheKey, request: int) -> float:
        if not self.future[key]:
            return math.inf
        event_index = self.future[key][0]
        if self.reset_per_request and self.events[event_index].request != request:
            return math.inf
        return float(event_index)

    def _heap_score(
        self, key: CacheKey, request: int
    ) -> tuple[float, int, int, CacheKey]:
        # heapq is a min-heap. Negating next use and the deterministic key gives
        # the same victim as max((next_use, key)) without a resident-set scan.
        return (-self._next_use(key, request), -key[0], -key[1], key)

    def _push_score(self, key: CacheKey, request: int) -> None:
        heapq.heappush(self.victim_heap, self._heap_score(key, request))

    def _rebuild_heap(self, request: int) -> None:
        self.victim_heap = [self._heap_score(key, request) for key in self.entries]
        heapq.heapify(self.victim_heap)

    def _unconstrained_victim(self, request: int) -> CacheKey | None:
        while self.victim_heap:
            score = heapq.heappop(self.victim_heap)
            key = score[-1]
            if key in self.entries and score == self._heap_score(key, request):
                return key
        return None

    def _apply_accesses(
        self,
        event: TraceEvent,
        keys: Sequence[CacheKey],
        hits: Sequence[bool],
        old_entries: set[CacheKey],
        resident_before: int,
    ) -> AccessOutcome:
        missing = {key for key, hit in zip(keys, hits) if not hit}
        for key in keys:
            self.entries.add(key)
            self._push_score(key, event.request)
        bypasses = 0
        evictions = 0
        while len(self.entries) > self.capacity:
            if self.layer_reserve:
                layer_counts = Counter(layer for layer, _ in self.entries)
                candidates = [
                    key
                    for key in self.entries
                    if layer_counts[key[0]] > self.layer_reserve
                ]
                victim = (
                    max(
                        candidates,
                        key=lambda key: (self._next_use(key, event.request), key),
                    )
                    if candidates
                    else None
                )
            else:
                victim = self._unconstrained_victim(event.request)
            if victim is None:
                raise SimulationError("belady: layer reserve leaves no evictable entry")
            self.entries.remove(victim)
            if victim in old_entries:
                evictions += 1
            elif victim in missing:
                bypasses += 1

        admissions = sum(key in self.entries for key in missing)
        if len(self.victim_heap) > self.capacity * 4:
            self._rebuild_heap(event.request)
        return AccessOutcome(
            tuple(hits),
            resident_before,
            evictions=evictions,
            admissions=admissions,
            bypasses=bypasses,
        )

    def _consume_future(self, event: TraceEvent, event_index: int) -> None:
        for key in event.keys:
            if not self.future[key] or self.future[key][0] != event_index:
                raise SimulationError(
                    "Belady next-use index is inconsistent with trace order"
                )
            self.future[key].popleft()

    def access(self, event: TraceEvent, event_index: int) -> AccessOutcome:
        old_entries = set(self.entries)
        hits = tuple(key in old_entries for key in event.keys)
        self._consume_future(event, event_index)
        return self._apply_accesses(
            event,
            event.keys,
            hits,
            old_entries,
            len(old_entries),
        )

    def access_prefill_batch(
        self,
        events: Sequence[TraceEvent],
        first_event_index: int,
    ) -> list[AccessOutcome]:
        old_entries = set(self.entries)
        unique, unique_hits, row_hits = _batch_layout(events, old_entries)
        for offset, event in enumerate(events):
            self._consume_future(event, first_event_index + offset)
        aggregate = self._apply_accesses(
            events[0],
            unique,
            unique_hits,
            old_entries,
            len(old_entries),
        )
        return _batch_outcomes(aggregate, row_hits)


def _make_policy(
    name: str,
    capacity: int,
    events: Sequence[TraceEvent],
    *,
    layer_reserve: int,
    protected_ratio: float,
    hotness_decay_tokens: int,
    tinylfu_sample_multiplier: int,
    reset_per_request: bool,
    phase_isolated: bool,
) -> CachePolicy:
    if name == "current":
        return CurrentHotnessPolicy(capacity, layer_reserve, hotness_decay_tokens)
    if name == "current-layer-quota":
        if layer_reserve:
            raise SimulationError(
                "current-layer-quota cannot be combined with layer_reserve"
            )
        return CurrentLayerQuotaPolicy(
            capacity,
            [event.layer for event in events],
            hotness_decay_tokens,
        )
    if name == "lru":
        return LRUPolicy(capacity, layer_reserve)
    if name == "lfu":
        return LFUPolicy(capacity, layer_reserve)
    if name == "slru":
        return SLRUPolicy(
            capacity,
            layer_reserve,
            protected_ratio,
            phase_isolated=phase_isolated,
        )
    if name == "tinylfu-slru":
        return SLRUPolicy(
            capacity,
            layer_reserve,
            protected_ratio,
            tiny_lfu=True,
            sample_multiplier=tinylfu_sample_multiplier,
            phase_isolated=phase_isolated,
        )
    if name == "belady":
        return BeladyPolicy(
            capacity,
            events,
            layer_reserve,
            reset_per_request=reset_per_request,
        )
    raise SimulationError(f"unknown policy {name!r}")


def _report_row(
    policy: str,
    capacity: int,
    warmup_requests: int,
    layer_reserve: int,
    protected_ratio: float,
    phase_isolated: bool,
    strict_layer_quota: bool,
    prefill_batched: bool,
    phase: str,
    counters: _Accumulator,
) -> ReportRow:
    references = counters.references
    tokens = len(counters.token_keys)
    hit_rate = counters.hits / references if references else 0.0
    trace_hit_rate = counters.trace_hits / references if references else 0.0
    missing_bytes = counters.missing_bytes if counters.missing_bytes_known else None
    missing_bytes_per_token = (
        missing_bytes / tokens if missing_bytes is not None and tokens else None
    )
    return ReportRow(
        policy=policy,
        capacity=capacity,
        warmup_requests=warmup_requests,
        layer_reserve=layer_reserve,
        protected_ratio=protected_ratio,
        phase_isolated=phase_isolated,
        strict_layer_quota=strict_layer_quota,
        prefill_batched=prefill_batched,
        phase=phase,
        events=counters.events,
        tokens=tokens,
        references=references,
        hits=counters.hits,
        misses=references - counters.hits,
        hit_rate=hit_rate,
        trace_hit_rate=trace_hit_rate,
        hit_rate_delta_pp=(hit_rate - trace_hit_rate) * 100.0,
        hit_mask_agreement=(
            counters.matching_hit_bits / references if references else 1.0
        ),
        cache_size_agreement=(
            counters.matching_cache_sizes / counters.events if counters.events else 1.0
        ),
        missing_bytes=missing_bytes,
        missing_bytes_per_token=missing_bytes_per_token,
        evictions=counters.evictions,
        evictions_per_token=counters.evictions / tokens if tokens else 0.0,
        admissions=counters.admissions,
        bypasses=counters.bypasses,
    )


def simulate(
    events: Sequence[TraceEvent],
    capacities: Sequence[int],
    policies: Sequence[str] = DEFAULT_POLICY_NAMES,
    *,
    expert_bytes: int | None = None,
    layer_reserve: int = 0,
    protected_ratio: float | Sequence[float] = 0.8,
    hotness_decay_tokens: int = 16,
    tinylfu_sample_multiplier: int = 10,
    reset_per_request: bool = False,
    phase_isolated: bool = False,
    warmup_requests: int = 0,
    prefill_batches: bool = True,
) -> list[ReportRow]:
    """Replay ``events`` and return overall and phase-specific metrics."""

    if not events:
        raise SimulationError("trace has no events")
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise SimulationError("capacities must contain positive slot counts")
    if len(set(capacities)) != len(capacities):
        raise SimulationError("capacities must not contain duplicates")
    if not policies:
        raise SimulationError("at least one policy is required")
    unknown_policies = set(policies).difference(POLICY_NAMES)
    if unknown_policies:
        raise SimulationError(
            "unknown policies: " + ", ".join(sorted(unknown_policies))
        )
    if len(set(policies)) != len(policies):
        raise SimulationError("policies must not contain duplicates")
    if expert_bytes is not None and expert_bytes <= 0:
        raise SimulationError("expert_bytes must be positive")
    if warmup_requests < 0:
        raise SimulationError("warmup_requests must be non-negative")
    if layer_reserve < 0:
        raise SimulationError("layer_reserve must be non-negative")
    if isinstance(protected_ratio, (float, int)):
        protected_ratios = (float(protected_ratio),)
    else:
        protected_ratios = tuple(protected_ratio)
    if not protected_ratios or any(
        not math.isfinite(ratio) or not 0.0 < ratio < 1.0 for ratio in protected_ratios
    ):
        raise SimulationError("protected_ratio values must be between 0 and 1")
    if len(set(protected_ratios)) != len(protected_ratios):
        raise SimulationError("protected_ratio values must not contain duplicates")
    if hotness_decay_tokens <= 0:
        raise SimulationError("hotness_decay_tokens must be positive")
    if tinylfu_sample_multiplier <= 0:
        raise SimulationError("tinylfu_sample_multiplier must be positive")

    max_selected = max(len(event.selected_ids) for event in events)
    if prefill_batches:
        index = 0
        while index < len(events):
            end = _prefill_batch_end(events, index)
            if events[index].phase == "prefill":
                unique = {key for event in events[index:end] for key in event.keys}
                max_selected = max(max_selected, len(unique))
            index = end
    request_order = tuple(dict.fromkeys(event.request for event in events))
    if warmup_requests >= len(request_order):
        raise SimulationError(
            "warmup_requests must leave at least one measured request"
        )
    warmup_request_ids = frozenset(request_order[:warmup_requests])
    layers = {event.layer for event in events}
    if reset_per_request:
        layers_by_request: dict[int, set[int]] = defaultdict(set)
        for event in events:
            layers_by_request[event.request].add(event.layer)
        reserve_layer_count = max(map(len, layers_by_request.values()))
    else:
        reserve_layer_count = len(layers)
    for capacity in capacities:
        if capacity < max_selected:
            raise SimulationError(
                f"capacity {capacity} is smaller than maximum selected set "
                f"({max_selected})"
            )
        if layer_reserve and capacity < layer_reserve * reserve_layer_count:
            raise SimulationError(
                f"capacity {capacity} cannot reserve {layer_reserve} slots for "
                f"each of {reserve_layer_count} simultaneously retained layers"
            )

    reports: list[ReportRow] = []
    for capacity in capacities:
        for policy_name in policies:
            ratios = (
                protected_ratios
                if policy_name in ("slru", "tinylfu-slru")
                else protected_ratios[:1]
            )
            for ratio in ratios:
                policy = _make_policy(
                    policy_name,
                    capacity,
                    events,
                    layer_reserve=layer_reserve,
                    protected_ratio=ratio,
                    hotness_decay_tokens=hotness_decay_tokens,
                    tinylfu_sample_multiplier=tinylfu_sample_multiplier,
                    reset_per_request=reset_per_request,
                    phase_isolated=phase_isolated,
                )
                counters = {
                    "all": _Accumulator(),
                    "prefill": _Accumulator(),
                    "decode": _Accumulator(),
                }
                previous_request: int | None = None
                event_index = 0
                while event_index < len(events):
                    event = events[event_index]
                    if previous_request is None or event.request != previous_request:
                        if reset_per_request and previous_request is not None:
                            policy.reset()
                        else:
                            policy.begin_request()
                    previous_request = event.request
                    batch_end = (
                        _prefill_batch_end(events, event_index)
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
                        if source_event.request in warmup_request_ids:
                            continue
                        counters["all"].record(source_event, outcome, expert_bytes)
                        counters[source_event.phase].record(
                            source_event, outcome, expert_bytes
                        )
                    event_index = batch_end

                for phase in ("all", "prefill", "decode"):
                    reports.append(
                        _report_row(
                            policy.name,
                            capacity,
                            warmup_requests,
                            layer_reserve,
                            ratio,
                            phase_isolated,
                            policy.name == "current-layer-quota",
                            prefill_batches,
                            phase,
                            counters[phase],
                        )
                    )
    return reports


REPORT_FIELDS = tuple(ReportRow.__dataclass_fields__)


def _display(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_reports(reports: Sequence[ReportRow], output_format: str) -> None:
    records = [asdict(report) for report in reports]
    if output_format == "json":
        json.dump(records, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
        return
    print("| " + " | ".join(REPORT_FIELDS) + " |")
    print("| " + " | ".join("---" for _ in REPORT_FIELDS) + " |")
    for record in records:
        print(
            "| " + " | ".join(_display(record[field]) for field in REPORT_FIELDS) + " |"
        )


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
    parser.add_argument("trace", type=Path, help="version 1 Expert route trace CSV")
    parser.add_argument(
        "--capacity",
        type=_positive_int,
        action="append",
        required=True,
        metavar="SLOTS",
        help="reusable Expert slot budget; repeat to sweep budgets",
    )
    parser.add_argument(
        "--policy",
        choices=POLICY_NAMES,
        action="append",
        help="policy to replay; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--expert-bytes",
        type=_positive_int,
        help="fallback bytes read per missing Expert",
    )
    parser.add_argument(
        "--layer-reserve",
        type=_non_negative_int,
        default=0,
        metavar="SLOTS",
        help="minimum resident slots retained per layer (default: 0)",
    )
    parser.add_argument(
        "--protected-ratio",
        type=float,
        action="append",
        help="SLRU protected target; repeat to sweep ratios (default: 0.8)",
    )
    parser.add_argument(
        "--hotness-decay-tokens",
        type=_positive_int,
        default=16,
        help="current-policy decode-token decay cadence (default: 16)",
    )
    parser.add_argument(
        "--tinylfu-sample-multiplier",
        type=_positive_int,
        default=10,
        help="TinyLFU halving window in capacity multiples (default: 10)",
    )
    parser.add_argument(
        "--reset-per-request",
        action="store_true",
        help="clear cache and policy history when request changes",
    )
    parser.add_argument(
        "--warmup-requests",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="replay but exclude the first N request ordinals from metrics",
    )
    parser.add_argument(
        "--phase-isolated",
        action="store_true",
        help="keep prefill hits/admissions from changing protected decode entries",
    )
    parser.add_argument(
        "--no-prefill-batches",
        action="store_true",
        help="disable inferred atomic request/layer prefill replay",
    )
    parser.add_argument(
        "--require-current-agreement-pp",
        type=float,
        metavar="PP",
        help="fail when current-policy decode hit rate differs by more than PP",
    )
    parser.add_argument(
        "--format", choices=("markdown", "csv", "json"), default="markdown"
    )
    args = parser.parse_args(argv)

    try:
        events = read_trace(args.trace)
        policies = args.policy or list(DEFAULT_POLICY_NAMES)
        reports = simulate(
            events,
            args.capacity,
            policies,
            expert_bytes=args.expert_bytes,
            layer_reserve=args.layer_reserve,
            protected_ratio=args.protected_ratio or 0.8,
            hotness_decay_tokens=args.hotness_decay_tokens,
            tinylfu_sample_multiplier=args.tinylfu_sample_multiplier,
            reset_per_request=args.reset_per_request,
            phase_isolated=args.phase_isolated,
            warmup_requests=args.warmup_requests,
            prefill_batches=not args.no_prefill_batches,
        )
    except (OSError, TraceFormatError, SimulationError) as error:
        parser.error(str(error))

    if args.require_current_agreement_pp is not None:
        if (
            not math.isfinite(args.require_current_agreement_pp)
            or args.require_current_agreement_pp < 0
        ):
            parser.error("--require-current-agreement-pp must be non-negative")
        current = [
            report
            for report in reports
            if report.policy == "current" and report.phase == "decode"
        ]
        if current and all(report.references == 0 for report in current):
            current = [
                report
                for report in reports
                if report.policy == "current" and report.phase == "all"
            ]
        if not current:
            parser.error("--require-current-agreement-pp requires --policy current")
        failures = [
            report
            for report in current
            if abs(report.hit_rate_delta_pp) > args.require_current_agreement_pp
        ]
        if failures:
            detail = ", ".join(
                f"capacity {report.capacity}: {report.hit_rate_delta_pp:+.3f} pp"
                for report in failures
            )
            parser.error(
                "current-policy hit rate is outside the agreement threshold: " + detail
            )

    write_reports(reports, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
