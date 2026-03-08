#!/usr/bin/env python3
"""
Integration Test Suite - Step 5
Simulates a flaky test failure caused by a race condition / non-deterministic
assertion in an asynchronous integration test.

In real CI/CD this surfaces when:
  - Tests share global state or temp files without proper isolation
  - Async callbacks resolve in a different order on CI than locally
  - A test asserts on timing (sleep-based synchronisation) that is slower on CI
  - External service mocks return responses in a different order under load
"""

import sys
import traceback


# ---------------------------------------------------------------------------
# Simulated test helpers
# ---------------------------------------------------------------------------

def fake_sleep(seconds: float) -> None:
    """Replacement for time.sleep that prints to the log."""
    print(f"[INFO]   waiting {seconds}s for async callback...")


def run_test_model_prediction_order() -> None:
    """
    Test: predictions returned by the async batch API must arrive in the
    same order as the input requests.

    On CI the async worker pool is smaller (2 threads vs 8 locally), so
    requests queued later can complete first, causing the ordering assertion
    to fail non-deterministically.
    """
    print("[INFO] test_model_prediction_order ... ", end="", flush=True)

    input_ids = [f"req-{i:03d}" for i in range(10)]
    fake_sleep(0.1)

    # On CI the 2-worker pool processes pairs; req-001 and req-002 are
    # dispatched together but req-002 (simpler input) finishes before req-001.
    # The response list arrives out of order.
    response_ids = [
        "req-000", "req-002", "req-001",   # <-- req-002 overtook req-001
        "req-003", "req-004", "req-005",
        "req-006", "req-007", "req-009", "req-008",  # <-- req-009 overtook req-008
    ]

    if response_ids != input_ids:
        mismatches = [
            (i, inp, got)
            for i, (inp, got) in enumerate(zip(input_ids, response_ids))
            if inp != got
        ]
        details = "\n".join(
            f"    position {i}: expected {inp!r}, got {got!r}"
            for i, inp, got in mismatches
        )
        raise AssertionError(
            "FAIL\n\n"
            "  test_model_prediction_order — prediction order mismatch\n\n"
            "  The async batch endpoint did not preserve input ordering.\n"
            f"  {len(mismatches)} position(s) out of order:\n"
            f"{details}\n\n"
            "  Root cause: worker pool size on CI (2) differs from local (8).\n"
            "  The test assumes deterministic ordering but the batch handler\n"
            "  does not guarantee it — results are emitted as soon as each\n"
            "  worker finishes, not in submission order.\n\n"
            "  This test has failed on CI 3 out of the last 10 runs (flaky).\n"
            "  Fix: sort results by request ID before asserting, or add an\n"
            "  order-preserving queue in the batch handler."
        )

    print("PASS")


def run_test_feature_cache_ttl() -> None:
    """
    Test: cached feature vectors must expire after the configured TTL.

    Passes locally because the developer's machine has low latency to the
    local Redis instance.  On CI the shared Redis container is under load,
    introducing ~200 ms extra latency, causing the TTL assertion to fire
    200 ms early.
    """
    print("[INFO] test_feature_cache_ttl    ... ", end="", flush=True)
    fake_sleep(0.05)
    print("PASS")


def run_test_db_rollback_on_error() -> None:
    """
    Test: database transaction is rolled back if the downstream HTTP call fails.
    Passes reliably.
    """
    print("[INFO] test_db_rollback_on_error ... ", end="", flush=True)
    fake_sleep(0.02)
    print("PASS")


def run_integration_tests() -> None:
    tests = [
        run_test_db_rollback_on_error,
        run_test_feature_cache_ttl,
        run_test_model_prediction_order,   # this one will fail
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as exc:
            print(f"FAIL", flush=True)
            raise

    print(f"\n[INFO] Results: {passed} passed, {failed} failed.")


if __name__ == "__main__":
    print("[INFO] ============================================================")
    print("[INFO]  Step 5: Integration Test Suite")
    print("[INFO] ============================================================")
    print("[INFO] Script  : integration_tests.py")
    print("[INFO] Purpose : Run end-to-end integration tests against the ML serving stack")
    print("[INFO] Runner  : 2 async workers (CI), 8 async workers (local)")
    print()

    try:
        run_integration_tests()
        print("[INFO] Step 5 PASSED.")
    except AssertionError as e:
        print()
        print("[ERROR] *** INTEGRATION TEST FAILURE (FLAKY TEST) ***", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "[ERROR] Step 5 FAILED: one or more integration tests produced unexpected results.",
            file=sys.stderr,
        )
        print("[ERROR] Traceback (most recent call last):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
