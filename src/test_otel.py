#!/usr/bin/env python3
"""
Test OpenTelemetry Integration - Task 6

Tests the optional OTEL integration for Credit Intelligence.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.otel_config import (
    is_otel_available,
    is_otel_enabled,
    setup_otel,
    get_tracer,
    otel_span,
    record_llm_call,
    record_tool_call,
    shutdown_otel,
    NoOpSpan,
)


def test_otel_availability():
    """Test if OpenTelemetry SDK is available."""
    print("\n" + "=" * 60)
    print("TEST: OTEL SDK Availability")
    print("=" * 60)

    available = is_otel_available()
    print(f"  OpenTelemetry SDK installed: {available}")

    if not available:
        print("  Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp")
        print("  Result: SKIP (SDK not installed)")
        return None

    print("  Result: PASS")
    return True


def test_otel_disabled_by_default():
    """Test that OTEL is disabled by default."""
    print("\n" + "=" * 60)
    print("TEST: OTEL Disabled by Default")
    print("=" * 60)

    # Clear the env var to test default
    original = os.environ.pop("OTEL_TRACES_ENABLED", None)

    try:
        enabled = is_otel_enabled()
        print(f"  OTEL_TRACES_ENABLED not set -> enabled={enabled}")

        if enabled:
            print("  Result: FAIL (should be disabled by default)")
            return False

        print("  Result: PASS (correctly disabled by default)")
        return True

    finally:
        if original:
            os.environ["OTEL_TRACES_ENABLED"] = original


def test_otel_noop_when_disabled():
    """Test that OTEL operations are no-op when disabled."""
    print("\n" + "=" * 60)
    print("TEST: OTEL No-Op When Disabled")
    print("=" * 60)

    # Ensure OTEL is disabled
    os.environ["OTEL_TRACES_ENABLED"] = "false"

    try:
        # Get tracer should return None
        tracer = get_tracer()
        print(f"  Tracer when disabled: {tracer}")

        # otel_span should return NoOpSpan
        with otel_span("test_span") as span:
            span_type = type(span).__name__
            print(f"  Span type when disabled: {span_type}")

            # NoOpSpan operations should not raise
            span.set_attribute("key", "value")
            span.add_event("event")

        # record functions should not raise
        record_llm_call("test-model", "prompt", "response", tokens=100)
        record_tool_call("test-tool", {"input": "data"})

        print("  All no-op operations completed without error")
        print("  Result: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")
        return False


def test_otel_setup():
    """Test OTEL setup when enabled."""
    print("\n" + "=" * 60)
    print("TEST: OTEL Setup")
    print("=" * 60)

    if not is_otel_available():
        print("  OTEL SDK not available - skipping")
        print("  Result: SKIP")
        return None

    # Enable OTEL
    os.environ["OTEL_TRACES_ENABLED"] = "true"

    try:
        # Setup with console exporter for testing
        success = setup_otel(
            service_name="credit-intelligence-test",
            use_console_exporter=False,  # Don't spam console
        )

        print(f"  Setup successful: {success}")

        if not success:
            print("  Result: FAIL (setup returned False)")
            return False

        # Get tracer
        tracer = get_tracer()
        print(f"  Tracer obtained: {tracer is not None}")

        if tracer is None:
            print("  Result: FAIL (tracer is None after setup)")
            return False

        # Test creating a span
        with otel_span("test_span", {"test.attribute": "value"}) as span:
            print(f"  Span created: {type(span).__name__}")
            span.set_attribute("additional.attr", 123)

        print("  Result: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")
        return False

    finally:
        shutdown_otel()
        os.environ["OTEL_TRACES_ENABLED"] = "false"


def test_otel_llm_recording():
    """Test recording LLM calls."""
    print("\n" + "=" * 60)
    print("TEST: OTEL LLM Call Recording")
    print("=" * 60)

    if not is_otel_available():
        print("  OTEL SDK not available - skipping")
        print("  Result: SKIP")
        return None

    # Enable OTEL
    os.environ["OTEL_TRACES_ENABLED"] = "true"

    try:
        setup_otel(service_name="credit-intelligence-test")

        # Record an LLM call
        record_llm_call(
            model="llama-3.3-70b-versatile",
            prompt="Test prompt for credit assessment",
            response="This is a test response",
            tokens=150,
            latency_ms=250.5,
            success=True,
        )

        print("  LLM call recorded successfully")
        print("  Result: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")
        return False

    finally:
        shutdown_otel()
        os.environ["OTEL_TRACES_ENABLED"] = "false"


def test_otel_tool_recording():
    """Test recording tool calls."""
    print("\n" + "=" * 60)
    print("TEST: OTEL Tool Call Recording")
    print("=" * 60)

    if not is_otel_available():
        print("  OTEL SDK not available - skipping")
        print("  Result: SKIP")
        return None

    # Enable OTEL
    os.environ["OTEL_TRACES_ENABLED"] = "true"

    try:
        setup_otel(service_name="credit-intelligence-test")

        # Record a tool call
        record_tool_call(
            tool_name="fetch_sec_data",
            input_data={"company": "Apple Inc", "form_type": "10-K"},
            output_data={"filings_count": 5},
            latency_ms=1500.0,
            success=True,
        )

        # Record a failed tool call
        record_tool_call(
            tool_name="fetch_market_data",
            input_data={"ticker": "INVALID"},
            latency_ms=500.0,
            success=False,
            error="Ticker not found",
        )

        print("  Tool calls recorded successfully")
        print("  Result: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")
        return False

    finally:
        shutdown_otel()
        os.environ["OTEL_TRACES_ENABLED"] = "false"


def main():
    """Run all OTEL integration tests."""
    print("\n" + "=" * 60)
    print("OPENTELEMETRY INTEGRATION TESTS - Task 6")
    print("=" * 60)

    results = {
        "SDK Availability": test_otel_availability(),
        "Disabled by Default": test_otel_disabled_by_default(),
        "No-Op When Disabled": test_otel_noop_when_disabled(),
        "Setup": test_otel_setup(),
        "LLM Recording": test_otel_llm_recording(),
        "Tool Recording": test_otel_tool_recording(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for test_name, result in results.items():
        status = "PASS" if result is True else ("FAIL" if result is False else "SKIP")
        print(f"  {test_name}: {status}")

    print(f"\n  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed > 0:
        print("\n  Some tests FAILED!")
        return 1

    print("\n  All tests passed (or skipped due to missing SDK)!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
