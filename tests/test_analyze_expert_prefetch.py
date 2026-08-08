#!/usr/bin/env python3
"""Deterministic tests for offline Expert-prefetch feasibility analysis."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_BENCH = ROOT / "workload-bench"
sys.path.insert(0, str(WORKLOAD_BENCH))

import analyze_expert_prefetch as analyzer  # noqa: E402
import simulate_expert_cache as simulator  # noqa: E402


def event(
    token: int,
    *experts: int,
    layer: int = 0,
    request: int = 0,
    phase: str = "decode",
    hit_mask: tuple[bool, ...] | None = None,
    cache_size: int = 0,
    expert_bytes: int = 10,
) -> simulator.TraceEvent:
    return simulator.TraceEvent(
        phase=phase,
        request=request,
        token=token,
        layer=layer,
        selected_ids=tuple(experts),
        hit_mask=(tuple(False for _ in experts) if hit_mask is None else hit_mask),
        cache_size=cache_size,
        expert_bytes=expert_bytes,
    )


def report(
    reports: list[analyzer.PrefetchReport], predictor: str, depth: str
) -> analyzer.PrefetchReport:
    return next(
        row for row in reports if row.predictor == predictor and row.depth == depth
    )


def static_fixture_events() -> list[simulator.TraceEvent]:
    return [
        event(0, 3, request=0),
        event(0, 1, 2, request=1),
        event(1, 3, 4, request=1),
    ]


def trace_bytes(events: list[simulator.TraceEvent]) -> bytes:
    rows = [
        "trace_version,phase,request,token,layer,selected_ids,hit_mask,"
        "cache_size,expert_bytes"
    ]
    for item in events:
        rows.append(
            ",".join(
                (
                    "1",
                    item.phase,
                    str(item.request),
                    str(item.token),
                    str(item.layer),
                    ";".join(map(str, item.selected_ids)),
                    "".join("1" if hit else "0" for hit in item.hit_mask),
                    str(item.cache_size),
                    str(item.expert_bytes),
                )
            )
        )
    return ("\r\n".join(rows) + "\r\n").encode()


class StrictTraceTests(unittest.TestCase):
    def test_requires_crlf_and_reuses_v1_privacy_schema(self) -> None:
        raw = trace_bytes([event(0, 1)])
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.csv"
            good.write_bytes(raw)
            self.assertEqual(len(analyzer.read_trace_strict(good)), 1)

            bare_lf = Path(directory) / "lf.csv"
            bare_lf.write_bytes(raw.replace(b"\r\n", b"\n"))
            with self.assertRaisesRegex(simulator.TraceFormatError, "CRLF"):
                analyzer.read_trace_strict(bare_lf)

            unsafe = Path(directory) / "unsafe.csv"
            unsafe.write_bytes(
                raw.replace(
                    b"cache_size,expert_bytes\r\n",
                    b"cache_size,expert_bytes,prompt\r\n",
                ).replace(b",0,10\r\n", b",0,10,secret\r\n")
            )
            with self.assertRaisesRegex(simulator.TraceFormatError, "unknown columns"):
                analyzer.read_trace_strict(unsafe)


class PredictorTests(unittest.TestCase):
    def test_issue_time_resident_suppression_is_not_target_time_submission(
        self,
    ) -> None:
        events = [
            event(0, 1, 2, layer=0),
            event(0, 9, 10, layer=1),
            event(1, 1, 3, layer=0),
        ]
        rows = analyzer.analyze(events, capacity=2)
        previous = report(rows, "previous-token", "top1")

        self.assertEqual(previous.raw_predictions, 1)
        self.assertEqual(previous.resident_suppressed, 1)
        self.assertEqual(previous.submitted_io, 0)
        self.assertEqual(previous.pin_only_misses, 1)
        self.assertEqual(previous.eligible_demand_misses, 2)
        self.assertEqual(previous.global_demand_misses, 6)

    def test_static_hotlist_is_frozen_from_warmup_and_warmup_is_unmeasured(
        self,
    ) -> None:
        rows = analyzer.analyze(static_fixture_events(), capacity=2, warmup_requests=1)
        hotlist = report(rows, "static-hotlist", "top1")

        self.assertEqual(hotlist.measured_decode_events, 2)
        self.assertEqual(hotlist.eligible_events, 1)
        self.assertEqual(hotlist.submitted_io, 1)
        self.assertEqual(hotlist.useful_submitted_io, 1)
        self.assertEqual(hotlist.submitted_io_precision, 1.0)
        self.assertEqual(hotlist.eligible_demand_miss_coverage, 0.5)
        self.assertEqual(hotlist.global_demand_miss_coverage, 0.25)
        self.assertEqual(hotlist.wasted_over_demand_bytes, 0.0)
        self.assertTrue(hotlist.passes_gates)

    def test_nonconsecutive_decode_rows_are_not_scored_as_next_token(self) -> None:
        rows = analyzer.analyze([event(0, 1), event(2, 1)], capacity=1)
        previous = report(rows, "previous-token", "top1")
        self.assertEqual(previous.eligible_events, 0)
        self.assertEqual(previous.raw_predictions, 0)
        self.assertEqual(previous.global_demand_misses, 2)

    def test_recent_history_is_request_local_and_deterministic(self) -> None:
        predictor = analyzer._RecentFrequency(2)
        self.assertEqual(predictor.observe_and_rank(event(0, 2, 1)), (2, 1))
        self.assertEqual(predictor.observe_and_rank(event(1, 1, 3)), (1, 3, 2))
        self.assertEqual(predictor.observe_and_rank(event(0, 7, request=1)), (7,))

    def test_transition_is_online_and_never_crosses_request_boundary(self) -> None:
        predictor = analyzer._OnlineTransition()
        self.assertEqual(predictor.observe_and_rank(event(0, 1)), ())
        self.assertEqual(predictor.observe_and_rank(event(1, 2)), ())
        self.assertEqual(predictor.observe_and_rank(event(2, 1)), (2,))
        self.assertEqual(predictor.observe_and_rank(event(0, 1, request=1)), (2,))

    def test_transition_scoring_is_causal_end_to_end(self) -> None:
        rows = analyzer.analyze(
            [event(0, 1), event(1, 2), event(2, 1), event(3, 2)],
            capacity=1,
        )
        transition = report(rows, "transition", "top1")

        self.assertEqual(transition.eligible_events, 3)
        self.assertEqual(transition.submitted_io, 1)
        self.assertEqual(transition.useful_submitted_io, 1)
        self.assertEqual(transition.eligible_demand_misses, 3)
        self.assertEqual(transition.global_demand_misses, 4)
        self.assertEqual(transition.submitted_io_precision, 1.0)
        self.assertAlmostEqual(transition.eligible_demand_miss_coverage, 1 / 3)
        self.assertEqual(transition.global_demand_miss_coverage, 0.25)

    def test_raw_route_metrics_are_distinct_from_submitted_io_metrics(self) -> None:
        rows = analyzer.analyze([event(0, 1, 2), event(1, 1, 3)], capacity=2)
        previous = report(rows, "previous-token", "top2")
        self.assertEqual(previous.raw_correct_predictions, 1)
        self.assertEqual(previous.raw_prediction_precision, 0.5)
        self.assertEqual(previous.raw_route_coverage, 0.5)
        self.assertEqual(previous.submitted_io, 0)
        self.assertIsNone(previous.submitted_io_precision)

    def test_observed_hit_mask_is_authoritative_when_current_policy_disagrees(
        self,
    ) -> None:
        rows = analyzer.analyze(
            [
                event(0, 2, request=0),
                event(0, 1, request=1),
                event(1, 2, request=1, hit_mask=(True,)),
            ],
            capacity=1,
            warmup_requests=1,
        )
        hotlist = report(rows, "static-hotlist", "top1")

        self.assertEqual(hotlist.submitted_io, 1)
        self.assertEqual(hotlist.useful_submitted_io, 0)
        self.assertEqual(hotlist.wasted_submitted_io, 1)
        self.assertEqual(hotlist.global_demand_misses, 1)
        self.assertEqual(hotlist.simulated_hit_rate, 0.0)
        self.assertEqual(hotlist.observed_hit_rate, 0.5)
        self.assertEqual(hotlist.hit_rate_delta_pp, -50.0)

    def test_prefill_batch_populates_current_policy_before_decode(self) -> None:
        rows = analyzer.analyze(
            [
                event(0, 1, phase="prefill"),
                event(1, 1, phase="prefill"),
                event(2, 1, hit_mask=(True,)),
            ],
            capacity=1,
        )
        previous = report(rows, "previous-token", "top1")

        self.assertEqual(previous.measured_decode_events, 1)
        self.assertEqual(previous.simulated_hit_bit_agreement, 1.0)
        self.assertEqual(previous.simulated_hit_rate, 1.0)
        self.assertEqual(previous.observed_hit_rate, 1.0)


class CliTests(unittest.TestCase):
    def run_cli(self, raw: bytes, *extra: str) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_bytes(raw)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = analyzer.main(
                    [
                        str(path),
                        "--capacity",
                        "2",
                        "--warmup-requests",
                        "1",
                        *extra,
                    ]
                )
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_json_output_and_gate_exit_convention(self) -> None:
        raw = trace_bytes(static_fixture_events())
        rc, stdout, _ = self.run_cli(raw, "--require-gates")
        self.assertEqual(rc, 0)
        records = json.loads(stdout)
        self.assertTrue(any(record["passes_gates"] for record in records))

        rc, stdout, stderr = self.run_cli(
            raw,
            "--require-gates",
            "--min-coverage",
            "0.9",
        )
        self.assertEqual(rc, 1)
        self.assertTrue(json.loads(stdout))
        self.assertIn("no predictor passed", stderr)

    def test_trace_error_exits_two_and_csv_header_is_stable(self) -> None:
        raw = trace_bytes(static_fixture_events())
        rc, _, stderr = self.run_cli(raw.replace(b"\r\n", b"\n"))
        self.assertEqual(rc, 2)
        self.assertIn("CRLF", stderr)

        rc, stdout, _ = self.run_cli(raw, "--format", "csv")
        self.assertEqual(rc, 0)
        self.assertEqual(
            stdout.splitlines()[0].split(","), list(analyzer.REPORT_FIELDS)
        )


if __name__ == "__main__":
    unittest.main()
