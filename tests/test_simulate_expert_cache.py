#!/usr/bin/env python3
"""Deterministic tests for the dependency-free Expert cache simulator."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "workload-bench" / "simulate_expert_cache.py"
SPEC = importlib.util.spec_from_file_location("simulate_expert_cache", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
simulator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = simulator
SPEC.loader.exec_module(simulator)


def event(
    token: int,
    *experts: int,
    layer: int = 0,
    request: int = 0,
    phase: str = "decode",
    hit_mask: tuple[bool, ...] | None = None,
    cache_size: int = 0,
    expert_bytes: int | None = 10,
):
    if hit_mask is None:
        hit_mask = tuple(False for _ in experts)
    return simulator.TraceEvent(
        phase=phase,
        request=request,
        token=token,
        layer=layer,
        selected_ids=tuple(experts),
        hit_mask=hit_mask,
        cache_size=cache_size,
        expert_bytes=expert_bytes,
    )


def overall(rows, policy: str):
    return next(row for row in rows if row.policy == policy and row.phase == "all")


class TraceParsingTests(unittest.TestCase):
    def test_reads_versioned_trace_and_optional_bytes(self):
        trace = io.StringIO(
            "trace_version,phase,request,token,layer,selected_ids,hit_mask,"
            "cache_size,expert_bytes\n"
            "1,prefill,0,2,3,7;12;4,101,99,4096\n"
        )
        events = simulator.read_trace(trace)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].keys, ((3, 7), (3, 12), (3, 4)))
        self.assertEqual(events[0].hit_mask, (True, False, True))
        self.assertEqual(events[0].cache_size, 99)
        self.assertEqual(events[0].expert_bytes, 4096)

    def test_rejects_unknown_columns_to_keep_trace_privacy_safe(self):
        trace = io.StringIO(
            "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size,prompt\n"
            "1,decode,0,0,0,1,0,0,secret\n"
        )
        with self.assertRaisesRegex(
            simulator.TraceFormatError, "unknown columns are not privacy-safe: prompt"
        ):
            simulator.read_trace(trace)

    def test_rejects_bad_version_duplicate_ids_and_mask_length(self):
        cases = (
            (
                "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size\n"
                "2,decode,0,0,0,1,0,0\n",
                "unsupported trace_version",
            ),
            (
                "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size\n"
                "1,decode,0,0,0,1;1,00,0\n",
                "selected_ids must be unique",
            ),
            (
                "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size\n"
                "1,decode,0,0,0,1;2,0,0\n",
                "hit_mask must contain",
            ),
        )
        for raw, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(simulator.TraceFormatError, message):
                    simulator.read_trace(io.StringIO(raw))

    def test_rejects_non_ascii_or_signed_integer_fields(self):
        for bad_token in ("-1", "+1", " 1", "١"):
            trace = io.StringIO(
                "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size\n"
                f"1,decode,0,{bad_token},0,1,0,0\n"
            )
            with self.subTest(token=bad_token):
                with self.assertRaisesRegex(simulator.TraceFormatError, "token must"):
                    simulator.read_trace(trace)


class PolicyTests(unittest.TestCase):
    def test_current_hotness_retains_frequent_entry_that_lru_evicts(self):
        events = [
            event(0, 0, cache_size=0),
            event(1, 0, hit_mask=(True,), cache_size=1),
            event(2, 1, cache_size=1),
            event(3, 2, cache_size=2),
            event(4, 0, hit_mask=(True,), cache_size=2),
        ]
        rows = simulator.simulate(events, [2], ["current", "lru"])
        current = overall(rows, "current")
        lru = overall(rows, "lru")
        self.assertEqual((current.hits, current.evictions), (2, 1))
        self.assertEqual((lru.hits, lru.evictions), (1, 2))
        self.assertEqual(current.hit_mask_agreement, 1.0)
        self.assertEqual(current.cache_size_agreement, 1.0)

    def test_current_hotness_halves_after_sixteen_decode_token_intervals(self):
        policy = simulator.CurrentHotnessPolicy(3, decay_tokens=16)
        policy.access(event(0, 0), 0)
        for token in range(1, 16):
            policy.access(event(token, 1), token)
        policy.access(event(16, 2), 16)
        self.assertEqual(policy.hotness.get((0, 0), 0), 0)
        self.assertEqual(policy.hotness[(0, 1)], 7)
        self.assertEqual(policy.hotness[(0, 2)], 1)

    def test_current_request_boundary_resets_hotness_but_keeps_residency(self):
        policy = simulator.CurrentHotnessPolicy(2)
        policy.access(event(0, 0), 0)
        policy.access(event(1, 0), 1)
        resident_before = policy.resident
        clock_before = policy.clock
        policy.begin_request()
        self.assertEqual(policy.resident, resident_before)
        self.assertEqual(policy.clock, clock_before)
        self.assertTrue(all(value == 0 for value in policy.hotness.values()))

    def test_prefill_batch_counts_frequency_before_unique_atomic_loads(self):
        policy = simulator.CurrentHotnessPolicy(4)
        policy.access(event(0, 0, phase="decode"), 0)
        policy.begin_request()
        batch = [
            event(0, 0, 1, phase="prefill", layer=0),
            event(1, 1, 2, phase="prefill", layer=0),
        ]
        outcomes = policy.access_prefill_batch(batch, 1)
        self.assertEqual(outcomes[0].hits, (True, False))
        self.assertEqual(outcomes[1].hits, (True, False))
        self.assertEqual(sum(outcome.admissions for outcome in outcomes), 2)
        self.assertEqual(policy.hotness[(0, 0)], 1)
        self.assertEqual(policy.hotness[(0, 1)], 2)
        self.assertEqual(policy.hotness[(0, 2)], 1)

    def test_simulate_infers_contiguous_prefill_batch_and_snapshot(self):
        events = [
            event(0, 0, 1, phase="prefill", cache_size=0),
            event(1, 1, 2, phase="prefill", cache_size=0),
        ]
        batched = overall(simulator.simulate(events, [3], ["current"]), "current")
        rowwise = overall(
            simulator.simulate(
                events,
                [3],
                ["current"],
                prefill_batches=False,
            ),
            "current",
        )
        self.assertEqual((batched.hits, batched.cache_size_agreement), (1, 1.0))
        self.assertEqual(rowwise.cache_size_agreement, 0.5)
        self.assertTrue(batched.prefill_batched)
        self.assertFalse(rowwise.prefill_batched)

    def test_strict_layer_quota_prevents_hot_layer_from_evicting_cold_layer(self):
        events = [
            event(0, 0, layer=0),
            event(1, 0, layer=1),
            event(2, 0, layer=0),
            event(3, 1, layer=0),
            event(4, 0, layer=1),
        ]
        rows = simulator.simulate(
            events,
            [2],
            ["current", "current-layer-quota"],
        )
        current = overall(rows, "current")
        quota = overall(rows, "current-layer-quota")
        self.assertEqual(current.hits, 1)
        self.assertEqual(quota.hits, 2)
        self.assertTrue(quota.strict_layer_quota)

    def test_belady_bypasses_one_shot_candidate_and_is_lower_bound(self):
        events = [
            event(0, 0),
            event(1, 1),
            event(2, 0),
            event(3, 2),
            event(4, 0),
            event(5, 1),
        ]
        rows = simulator.simulate(events, [2], ["lru", "belady"])
        lru = overall(rows, "lru")
        belady = overall(rows, "belady")
        self.assertEqual(lru.hits, 2)
        self.assertEqual(belady.hits, 3)
        self.assertEqual((belady.evictions, belady.bypasses), (0, 1))

    def test_tinylfu_requires_candidate_frequency_to_beat_victim(self):
        events = [event(0, 0), event(1, 1), event(2, 2), event(3, 2)]
        rows = simulator.simulate(events, [2], ["tinylfu-slru"])
        tiny = overall(rows, "tinylfu-slru")
        self.assertEqual(tiny.hits, 0)
        self.assertEqual(tiny.admissions, 3)
        self.assertEqual(tiny.bypasses, 1)
        self.assertEqual(tiny.evictions, 1)

    def test_slru_promotes_probation_hits(self):
        events = [
            event(0, 0),
            event(1, 1),
            event(2, 0),
            event(3, 2),
            event(4, 3),
            event(5, 0),
        ]
        rows = simulator.simulate(events, [3], ["slru"], protected_ratio=2.0 / 3.0)
        slru = overall(rows, "slru")
        self.assertEqual(slru.hits, 2)
        self.assertEqual(slru.evictions, 1)

    def test_phase_isolated_slru_keeps_prefill_hits_out_of_protected(self):
        events = [
            event(0, 0, phase="decode"),
            event(1, 0, phase="decode"),
            event(0, 1, phase="prefill"),
            event(1, 1, phase="prefill"),
            event(2, 2, phase="prefill"),
            event(2, 0, phase="decode"),
        ]
        normal = overall(
            simulator.simulate(events, [2], ["slru"], protected_ratio=0.5),
            "slru",
        )
        isolated = overall(
            simulator.simulate(
                events,
                [2],
                ["slru"],
                protected_ratio=0.5,
                phase_isolated=True,
            ),
            "slru",
        )
        self.assertEqual(normal.hits, 2)
        self.assertEqual(isolated.hits, 3)
        self.assertTrue(isolated.phase_isolated)

    def test_phase_isolated_slru_bypasses_when_only_protected_is_evictable(self):
        events = [
            event(0, 0, phase="decode"),
            event(1, 0, phase="decode"),
            event(0, 1, phase="prefill"),
            event(1, 1, 2, phase="prefill"),
        ]
        isolated = overall(
            simulator.simulate(
                events,
                [2],
                ["slru"],
                protected_ratio=0.5,
                phase_isolated=True,
            ),
            "slru",
        )
        self.assertEqual(isolated.bypasses, 1)
        self.assertEqual(isolated.evictions, 0)

    def test_segment_ratio_sweep_only_repeats_segmented_policies(self):
        rows = simulator.simulate(
            [event(0, 0)],
            [2],
            ["current", "slru"],
            protected_ratio=[0.5, 0.75],
        )
        current = [row for row in rows if row.policy == "current"]
        slru = [row for row in rows if row.policy == "slru"]
        self.assertEqual(len(current), 3)
        self.assertEqual(len(slru), 6)
        self.assertEqual({row.protected_ratio for row in slru}, {0.5, 0.75})

    def test_layer_reserve_preserves_a_layer_minimum(self):
        events = [
            event(0, 0, layer=0),
            event(1, 0, layer=1),
            event(2, 1, layer=0),
            event(3, 0, layer=0),
            event(4, 1, layer=0),
            event(5, 2, layer=0),
            event(6, 0, layer=1),
        ]
        no_reserve = overall(
            simulator.simulate(events, [3], ["lru"], layer_reserve=0), "lru"
        )
        reserve = overall(
            simulator.simulate(events, [3], ["lru"], layer_reserve=1), "lru"
        )
        self.assertEqual(no_reserve.hits, 2)
        self.assertEqual(reserve.hits, 3)
        self.assertEqual(reserve.evictions, 1)

    def test_reset_per_request_models_cold_cache(self):
        events = [
            event(0, 7, request=0, cache_size=0),
            event(0, 7, request=1, cache_size=0),
        ]
        warm = overall(
            simulator.simulate(events, [1], ["lru"], reset_per_request=False),
            "lru",
        )
        cold = overall(
            simulator.simulate(events, [1], ["lru"], reset_per_request=True),
            "lru",
        )
        self.assertEqual(warm.hits, 1)
        self.assertEqual(cold.hits, 0)
        self.assertEqual(cold.cache_size_agreement, 1.0)

    def test_warmup_requests_prime_cache_but_are_excluded_from_metrics(self):
        events = [
            event(0, 7, request=0),
            event(0, 7, request=1),
        ]
        measured = overall(
            simulator.simulate(
                events,
                [1],
                ["lru"],
                warmup_requests=1,
            ),
            "lru",
        )
        self.assertEqual((measured.references, measured.hits), (1, 1))
        self.assertEqual(measured.warmup_requests, 1)

    def test_batch_events_use_pre_event_hits_and_require_enough_slots(self):
        events = [event(0, 0, 1), event(1, 2, 3)]
        row = overall(simulator.simulate(events, [2], ["lru"]), "lru")
        self.assertEqual(row.hits, 0)
        self.assertEqual(row.evictions, 2)
        with self.assertRaisesRegex(simulator.SimulationError, "maximum selected set"):
            simulator.simulate(events, [1], ["lru"])


class MetricsAndCliTests(unittest.TestCase):
    def test_phase_metrics_count_unique_tokens_not_layer_rows(self):
        events = [
            event(0, 0, phase="prefill", layer=0, expert_bytes=100),
            event(0, 0, phase="prefill", layer=1, expert_bytes=100),
            event(0, 0, phase="decode", layer=0, expert_bytes=100),
        ]
        rows = simulator.simulate(events, [2], ["lru"])
        prefill = next(row for row in rows if row.phase == "prefill")
        decode = next(row for row in rows if row.phase == "decode")
        self.assertEqual(prefill.tokens, 1)
        self.assertEqual(prefill.missing_bytes, 200)
        self.assertEqual(prefill.missing_bytes_per_token, 200.0)
        self.assertEqual(decode.hits, 1)
        self.assertEqual(decode.missing_bytes_per_token, 0.0)

    def test_missing_byte_metric_is_explicitly_unavailable_without_size(self):
        rows = simulator.simulate([event(0, 0, expert_bytes=None)], [1], ["lru"])
        self.assertIsNone(overall(rows, "lru").missing_bytes_per_token)
        fallback = simulator.simulate(
            [event(0, 0, expert_bytes=None)],
            [1],
            ["lru"],
            expert_bytes=64,
        )
        self.assertEqual(overall(fallback, "lru").missing_bytes_per_token, 64.0)

    def test_json_cli_runs_all_policies(self):
        raw = (
            "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size\n"
            "1,decode,0,0,0,3,0,0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(raw, encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = simulator.main(
                    [
                        str(path),
                        "--capacity",
                        "1",
                        "--expert-bytes",
                        "32",
                        "--format",
                        "json",
                    ]
                )
        self.assertEqual(rc, 0)
        records = json.loads(stdout.getvalue())
        self.assertEqual(len(records), len(simulator.DEFAULT_POLICY_NAMES) * 3)
        self.assertEqual(
            {record["policy"] for record in records},
            set(simulator.DEFAULT_POLICY_NAMES),
        )


if __name__ == "__main__":
    unittest.main()
