#!/usr/bin/env python3
"""
Test Langfuse Integration - Task 22

Tests the Langfuse tracing/logging integration for Credit Intelligence.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.langfuse_config import (
    is_langfuse_available,
    get_langfuse_handler,
    get_default_langfuse_handler,
    create_trace,
    log_score,
    flush_langfuse,
    LangfuseTracer,
)


def test_langfuse_availability():
    """Test if Langfuse is available and configured."""
    print("\n" + "=" * 60)
    print("TEST: Langfuse Availability")
    print("=" * 60)

    # Check if SDK is installed
    try:
        import langfuse
        sdk_installed = True
        version = getattr(langfuse, "__version__", "unknown")
        print(f"  Langfuse SDK installed: True (v{version})")
    except ImportError:
        sdk_installed = False
        print("  Langfuse SDK installed: False")
        print("  Install with: pip install langfuse")
        return False

    # Check if configured
    available = is_langfuse_available()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    print(f"  LANGFUSE_PUBLIC_KEY set: {bool(public_key)}")
    print(f"  LANGFUSE_SECRET_KEY set: {bool(secret_key)}")
    print(f"  Langfuse configured: {available}")

    if not available:
        print("\n  To enable Langfuse tracing:")
        print("  1. Create account at https://cloud.langfuse.com")
        print("  2. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        print("  Result: PASS (SDK installed, not configured)")
        return True  # SDK installed is enough for the test to pass

    print("  Result: PASS (Langfuse fully configured)")
    return True


def test_langfuse_handler():
    """Test creating a Langfuse callback handler."""
    print("\n" + "=" * 60)
    print("TEST: Langfuse Callback Handler")
    print("=" * 60)

    handler = get_langfuse_handler(
        session_id="test_session_123",
        user_id="test_user",
        metadata={"test": True},
        tags=["test", "credit-intelligence"],
    )

    if handler:
        print(f"  Handler type: {type(handler).__name__}")
        print("  Result: PASS")
        return True
    else:
        print("  Handler: None (Langfuse not configured)")
        print("  Result: SKIP")
        return None


def test_langfuse_trace():
    """Test creating a Langfuse trace."""
    print("\n" + "=" * 60)
    print("TEST: Langfuse Trace Creation")
    print("=" * 60)

    trace = create_trace(
        name="test_credit_assessment",
        session_id="test_run_456",
        metadata={"company": "Test Corp"},
        tags=["test"],
    )

    if trace:
        print(f"  Trace ID: {trace.id}")
        print("  Result: PASS")

        # Test scoring
        success = log_score(
            trace_id=trace.id,
            name="test_score",
            value=0.95,
            comment="Test score from integration test",
        )
        print(f"  Score logged: {success}")

        flush_langfuse()
        return True
    else:
        print("  Trace: None (Langfuse not configured)")
        print("  Result: SKIP")
        return None


def test_langfuse_tracer_context():
    """Test the LangfuseTracer context manager."""
    print("\n" + "=" * 60)
    print("TEST: LangfuseTracer Context Manager")
    print("=" * 60)

    if not is_langfuse_available():
        print("  Langfuse not available - skipping")
        print("  Result: SKIP")
        return None

    try:
        with LangfuseTracer(
            name="test_workflow",
            session_id="test_ctx_789",
            metadata={"test": True},
            tags=["test", "context-manager"],
        ) as tracer:
            # Simulate workflow steps
            span = tracer.span("step_1_data_collection")
            # ... simulate work ...
            tracer.end_span(output={"data": "collected"})

            span = tracer.span("step_2_analysis")
            # ... simulate work ...
            tracer.end_span(output={"analysis": "complete"})

            # Set final output
            tracer.set_output({
                "result": "success",
                "score": 0.92,
            })

            # Log a score
            tracer.score("accuracy", 0.92, "Test accuracy score")

            # Get handler for LangChain integration
            handler = tracer.get_handler()
            print(f"  Handler from tracer: {type(handler).__name__ if handler else 'None'}")

        print("  Context manager completed successfully")
        print("  Result: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")
        return False


def test_langfuse_with_langchain():
    """Test Langfuse handler with a simple LangChain call."""
    print("\n" + "=" * 60)
    print("TEST: Langfuse with LangChain")
    print("=" * 60)

    if not is_langfuse_available():
        print("  Langfuse not available - skipping")
        print("  Result: SKIP")
        return None

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("  GROQ_API_KEY not set - skipping LangChain test")
            print("  Result: SKIP")
            return None

        # Get Langfuse handler
        handler = get_langfuse_handler(
            session_id="langchain_test",
            metadata={"test": "langchain_integration"},
        )

        if not handler:
            print("  Could not create handler")
            print("  Result: FAIL")
            return False

        # Create LLM with callback
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=50,
        )

        # Invoke with Langfuse tracing
        response = llm.invoke(
            [HumanMessage(content="Say 'Langfuse test successful' in exactly 3 words")],
            config={"callbacks": [handler]},
        )

        print(f"  LLM Response: {response.content[:50]}...")
        print("  Trace sent to Langfuse")

        flush_langfuse()
        print("  Result: PASS")
        return True

    except ImportError as e:
        print(f"  Import error: {e}")
        print("  Result: SKIP (missing dependencies)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")
        return False


def main():
    """Run all Langfuse integration tests."""
    print("\n" + "=" * 60)
    print("LANGFUSE INTEGRATION TESTS - Task 22")
    print("=" * 60)

    results = {
        "Availability": test_langfuse_availability(),
        "Callback Handler": test_langfuse_handler(),
        "Trace Creation": test_langfuse_trace(),
        "Context Manager": test_langfuse_tracer_context(),
        "LangChain Integration": test_langfuse_with_langchain(),
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

    print("\n  All tests passed (or skipped due to missing config)!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
