#!/usr/bin/env python3
"""Test the tool-based workflow with sample companies."""

import os
import sys
import json
import logging as stdlib_logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configure logging
stdlib_logging.basicConfig(
    level=stdlib_logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = stdlib_logging.getLogger(__name__)


def test_tool_executor():
    """Test the tool executor."""
    print("\n" + "=" * 60)
    print("TEST: Tool Executor")
    print("=" * 60)

    from tools import get_tool_executor

    executor = get_tool_executor()

    # Get tool specs
    specs = executor.get_tool_specs()
    print(f"\nRegistered tools: {[s['name'] for s in specs]}")

    # Test SEC tool
    print("\nTesting SEC tool with 'Apple Inc'...")
    result = executor.execute_tool("fetch_sec_data", company_identifier="Apple Inc")
    print(f"  Success: {result.success}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
    if result.data:
        print(f"  Found: {result.data.get('found', False)}")

    # Test Web Search tool
    print("\nTesting Web Search tool with 'Acme Corp'...")
    result = executor.execute_tool("web_search", company_name="Acme Corp")
    print(f"  Success: {result.success}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")

    return True


def test_tool_supervisor():
    """Test the tool supervisor."""
    print("\n" + "=" * 60)
    print("TEST: Tool Supervisor")
    print("=" * 60)

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Skipping supervisor test.")
        return False

    from agents import ToolSupervisor

    supervisor = ToolSupervisor(model="primary")

    print("\nTesting tool selection for 'Apple Inc'...")
    selection = supervisor.select_tools("Apple Inc")

    if "error" in selection.get("selection", {}):
        print(f"  Error: {selection['selection']['error']}")
        return False

    tools = [t.get("name") for t in selection.get("selection", {}).get("tools_to_use", [])]
    print(f"  Selected tools: {tools}")
    print(f"  LLM execution time: {selection.get('llm_metrics', {}).get('execution_time_ms', 0):.2f}ms")
    print(f"  Tokens used: {selection.get('llm_metrics', {}).get('total_tokens', 0)}")

    return True


def test_full_assessment():
    """Test full assessment workflow."""
    print("\n" + "=" * 60)
    print("TEST: Full Assessment Workflow")
    print("=" * 60)

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Skipping full assessment test.")
        return False

    from agents import ToolSupervisor

    supervisor = ToolSupervisor(model="fast")  # Use fast model for testing

    print("\nRunning full assessment for 'Microsoft Corporation'...")
    result = supervisor.run_full_assessment("Microsoft Corporation")

    print(f"\n  Run ID: {result.get('run_id')}")
    print(f"  Total execution time: {result.get('total_execution_time_ms', 0):.2f}ms")

    # Tool selection
    selection = result.get("tool_selection", {}).get("selection", {})
    tools = [t.get("name") for t in selection.get("tools_to_use", [])]
    print(f"  Tools selected: {tools}")

    # Assessment
    assessment = result.get("assessment", {}).get("assessment", {})
    print(f"\n  Risk Level: {assessment.get('risk_level', 'N/A')}")
    print(f"  Credit Score: {assessment.get('credit_score', 'N/A')}")
    print(f"  Confidence: {assessment.get('confidence', 'N/A')}")

    return True


def test_evaluation():
    """Test the evaluation framework."""
    print("\n" + "=" * 60)
    print("TEST: Evaluation Framework")
    print("=" * 60)

    from evaluation import ToolSelectionEvaluator, WorkflowEvaluator

    # Test tool selection evaluator
    evaluator = ToolSelectionEvaluator()

    # Test with Apple (public US company)
    result = evaluator.evaluate(
        company_name="Apple Inc",
        selected_tools=["fetch_sec_data", "fetch_market_data"],
        selection_reasoning={
            "company_analysis": {"is_likely_public": True},
            "tools_to_use": [
                {"name": "fetch_sec_data", "reason": "Public company"},
                {"name": "fetch_market_data", "reason": "Stock data"},
            ],
        },
    )

    print(f"\nTool Selection Evaluation for 'Apple Inc':")
    print(f"  Expected tools: {result.expected_tools}")
    print(f"  Selected tools: {result.selected_tools}")
    print(f"  Precision: {result.precision:.2f}")
    print(f"  Recall: {result.recall:.2f}")
    print(f"  F1 Score: {result.f1_score:.2f}")
    print(f"  Is Correct: {result.is_correct}")

    # Test with private company
    result2 = evaluator.evaluate(
        company_name="Smith's Local Shop LLC",
        selected_tools=["web_search", "fetch_legal_data"],
    )

    print(f"\nTool Selection Evaluation for 'Smith's Local Shop LLC':")
    print(f"  Expected tools: {result2.expected_tools}")
    print(f"  Selected tools: {result2.selected_tools}")
    print(f"  F1 Score: {result2.f1_score:.2f}")

    # Get summary
    summary = evaluator.get_summary()
    print(f"\nOverall Summary:")
    print(f"  Total evaluations: {summary['total']}")
    print(f"  Average F1: {summary['avg_f1']:.2f}")

    return True


def test_logging():
    """Test the logging infrastructure."""
    print("\n" + "=" * 60)
    print("TEST: Logging Infrastructure")
    print("=" * 60)

    from run_logging import MetricsCollector, get_run_logger

    # Test metrics collector
    import uuid
    run_id = str(uuid.uuid4())
    collector = MetricsCollector(run_id, "Test Company")

    collector.start_step("tool_selection")
    import time
    time.sleep(0.1)  # Simulate work
    collector.end_step(
        success=True,
        tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    metrics = collector.to_dict()
    print(f"\nMetrics Collector Test:")
    print(f"  Run ID: {metrics['run_id']}")
    print(f"  Execution time: {metrics['total_execution_time_ms']:.2f}ms")
    print(f"  Total tokens: {metrics['total_tokens']}")
    print(f"  Steps recorded: {len(metrics['steps'])}")

    # Test cost estimation
    cost = collector.estimate_cost()
    print(f"\nCost Estimate:")
    print(f"  Input cost: ${cost['input_cost_usd']:.6f}")
    print(f"  Output cost: ${cost['output_cost_usd']:.6f}")
    print(f"  Total cost: ${cost['total_cost_usd']:.6f}")

    # Test run logger
    run_logger = get_run_logger()
    print(f"\nRun Logger:")
    print(f"  Connected to MongoDB: {run_logger.is_connected()}")

    if run_logger.is_connected():
        stats = run_logger.get_stats()
        print(f"  Total runs: {stats.get('total_runs', 0)}")
        print(f"  Completed runs: {stats.get('completed_runs', 0)}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CREDIT INTELLIGENCE - TOOL-BASED WORKFLOW TESTS")
    print("=" * 60)

    tests = [
        ("Tool Executor", test_tool_executor),
        ("Tool Supervisor", test_tool_supervisor),
        ("Full Assessment", test_full_assessment),
        ("Evaluation Framework", test_evaluation),
        ("Logging Infrastructure", test_logging),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "PASS" if success else "FAIL/SKIP"
        print(f"  {name}: {status}")

    passed = sum(1 for _, s in results if s)
    print(f"\n  Total: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
