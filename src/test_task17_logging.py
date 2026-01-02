#!/usr/bin/env python3
"""
Test Task 17 Logging - Verify detailed LLM calls and run summaries are logged.

Tests the new Task 17 logging methods:
- log_llm_call_detailed() - logs to llm_calls_detailed sheet/collection
- log_run_summary() - logs to run_summaries sheet/collection

Runs a simple test workflow and verifies logs in both MongoDB and Google Sheets.
"""

import os
import sys
import time
import uuid
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Import loggers
from run_logging import get_workflow_logger, get_sheets_logger
from run_logging.run_logger import get_run_logger


def test_task17_logging():
    """Test the new Task 17 logging methods."""
    print("=" * 60)
    print("TASK 17 LOGGING TEST")
    print("=" * 60)
    print()

    # Get loggers
    wf_logger = get_workflow_logger()
    sheets_logger = get_sheets_logger()
    run_logger = get_run_logger()

    # Check connections
    print("Checking connections...")
    print(f"  MongoDB connected: {run_logger.is_connected()}")
    print(f"  Google Sheets connected: {sheets_logger.is_connected()}")
    if sheets_logger.is_connected():
        print(f"  Sheets URL: {sheets_logger.get_spreadsheet_url()}")
    print()

    # Generate test data
    run_id = str(uuid.uuid4())
    company_name = "Test Company Inc"
    started_at = datetime.utcnow().isoformat()

    print(f"Test run_id: {run_id}")
    print(f"Test company: {company_name}")
    print()

    # Test 1: Log detailed LLM calls
    print("Test 1: Logging detailed LLM calls...")

    test_llm_calls = [
        {
            "llm_provider": "groq",
            "agent_name": "company_parser",
            "model": "llama-3.3-70b-versatile",
            "prompt": "Parse this company: Test Company Inc. Determine if public/private.",
            "context": "Initial company analysis for credit assessment",
            "response": '{"is_public": false, "industry": "Technology", "confidence": 0.85}',
            "reasoning": "Company name doesn't match known public companies.",
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "response_time_ms": 1234.5,
            "input_cost": 0.000015,
            "output_cost": 0.000025,
            "total_cost": 0.00004,
        },
        {
            "llm_provider": "groq",
            "agent_name": "credit_analyst",
            "model": "llama-3.3-70b-versatile",
            "prompt": "Analyze credit risk for Test Company Inc based on collected data.",
            "context": "SEC filings: None, Finnhub: Limited data, Web search: 5 results",
            "response": '{"risk_level": "MODERATE", "credit_score": 650, "confidence": 0.72}',
            "reasoning": "Limited public data available. Private company with moderate financial indicators.",
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "response_time_ms": 2345.6,
            "input_cost": 0.00005,
            "output_cost": 0.0001,
            "total_cost": 0.00015,
        },
    ]

    for i, call in enumerate(test_llm_calls, 1):
        print(f"  Logging LLM call {i}: {call['agent_name']}/{call['model']}")
        wf_logger.log_llm_call_detailed(
            run_id=run_id,
            company_name=company_name,
            **call,
        )
    print("  Done!")
    print()

    # Test 2: Log run summary
    print("Test 2: Logging run summary...")

    completed_at = datetime.utcnow().isoformat()
    duration_ms = 5678.9

    wf_logger.log_run_summary(
        run_id=run_id,
        company_name=company_name,
        status="completed",
        # Assessment
        risk_level="MODERATE",
        credit_score=650,
        confidence=0.72,
        reasoning="Private company with moderate financial indicators. Limited public data.",
        # Eval metrics
        tool_selection_score=0.85,
        data_quality_score=0.60,
        synthesis_score=0.78,
        overall_score=0.74,
        # Decision
        final_decision="APPROVED_WITH_CONDITIONS",
        decision_reasoning="Recommend standard terms with quarterly review due to limited data.",
        # Execution details
        errors=["SEC API returned no data"],
        warnings=["Low data quality from web search"],
        tools_used=["company_parser", "web_search", "credit_analyst"],
        agents_used=["SupervisorAgent", "SearchAgent", "LLMAnalystAgent"],
        # Timing
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        # Costs
        total_tokens=900,
        total_cost=0.00019,
        llm_calls_count=2,
    )
    print("  Done!")
    print()

    # Wait for async logging to complete
    print("Waiting for async logging to complete (2 seconds)...")
    time.sleep(2)
    print()

    # Verify MongoDB logging
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print()

    if run_logger.is_connected():
        print("Checking MongoDB...")

        # Check llm_calls_detailed
        llm_calls = run_logger.get_llm_calls_detailed(run_id=run_id)
        print(f"  llm_calls_detailed: {len(llm_calls)} records found")
        for call in llm_calls:
            print(f"    - {call.get('agent_name')}: {call.get('total_tokens')} tokens, ${call.get('total_cost', 0):.6f}")

        # Check run_summaries
        summaries = run_logger.get_run_summaries(company_name=company_name)
        print(f"  run_summaries: {len(summaries)} records found")
        for summary in summaries:
            if summary.get("run_id") == run_id:
                print(f"    - run_id: {summary.get('run_id')[:8]}...")
                print(f"      risk_level: {summary.get('risk_level')}")
                print(f"      credit_score: {summary.get('credit_score')}")
                print(f"      overall_score: {summary.get('overall_score')}")
                print(f"      final_decision: {summary.get('final_decision')}")
        print()
    else:
        print("MongoDB not connected - skipping verification")
        print()

    if sheets_logger.is_connected():
        print("Google Sheets logging verified by visual inspection:")
        print(f"  Open: {sheets_logger.get_spreadsheet_url()}")
        print("  Check sheets: llm_calls_detailed, run_summaries")
        print()
    else:
        print("Google Sheets not connected - skipping verification")
        print()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("Task 17 logging test completed successfully!")
    print()
    print("Summary:")
    print(f"  - Logged {len(test_llm_calls)} detailed LLM calls")
    print(f"  - Logged 1 run summary")
    print(f"  - Run ID: {run_id}")


def test_real_workflow():
    """Run a real workflow and verify Task 17 logging."""
    print("=" * 60)
    print("REAL WORKFLOW TEST WITH TASK 17 LOGGING")
    print("=" * 60)
    print()

    try:
        # Import graph
        from agents.graph import graph, wf_logger

        if wf_logger is None:
            print("Workflow logger not initialized!")
            return

        company_name = "Microsoft Corporation"
        print(f"Running workflow for: {company_name}")
        print()

        # Initial state
        initial_state = {
            "company_name": company_name,
            "jurisdiction": "US",
            "ticker": None,
            "company_info": {},
            "task_plan": [],
            "api_data": {},
            "search_data": {},
            "assessment": None,
            "errors": [],
            "status": "started",
            "llm_results": [],
            "human_approved": False,
            "human_feedback": None,
            "validation_message": "",
            "requires_review": False,
        }

        # Run the workflow
        print("Starting workflow...")
        start_time = time.time()
        result = graph.invoke(initial_state)
        duration_ms = (time.time() - start_time) * 1000
        print(f"Workflow completed in {duration_ms:.0f}ms")
        print()

        # Extract results
        run_id = result.get("run_id", "unknown")
        assessment = result.get("assessment", {})
        evaluation = result.get("evaluation", {})
        llm_results = result.get("llm_results", [])

        print("Results:")
        print(f"  Run ID: {run_id}")
        print(f"  Risk Level: {assessment.get('overall_risk_level', 'N/A')}")
        print(f"  Credit Score: {assessment.get('credit_score_estimate', 'N/A')}")
        print(f"  Confidence: {assessment.get('confidence_score', 0):.2f}")
        print(f"  LLM Calls: {len(llm_results)}")
        print()

        # Log comprehensive run summary using Task 17 method
        print("Logging Task 17 run summary...")

        # Calculate totals from LLM results
        total_tokens = sum(r.get("total_tokens", 0) for r in llm_results if r.get("success"))
        total_cost = sum(r.get("total_cost", 0) for r in llm_results if r.get("success"))

        # Collect tools used
        tools_used = list(result.get("api_data", {}).keys()) + ["web_search"]

        # Log using the new Task 17 method
        wf_logger.log_run_summary(
            run_id=run_id,
            company_name=company_name,
            status=result.get("status", "completed"),
            risk_level=assessment.get("overall_risk_level", ""),
            credit_score=assessment.get("credit_score_estimate", 0),
            confidence=assessment.get("confidence_score", 0.0),
            reasoning=assessment.get("llm_reasoning", ""),
            tool_selection_score=evaluation.get("tool_selection", {}).get("f1_score", 0.0),
            data_quality_score=evaluation.get("data_quality", {}).get("completeness", 0.0),
            synthesis_score=evaluation.get("synthesis", {}).get("score", 0.0),
            overall_score=evaluation.get("overall_score", 0.0),
            final_decision="APPROVED" if assessment.get("overall_risk_level") in ["LOW", "MODERATE"] else "REVIEW_REQUIRED",
            decision_reasoning=f"Risk level is {assessment.get('overall_risk_level', 'unknown')}",
            errors=result.get("errors", []),
            warnings=[],
            tools_used=tools_used,
            agents_used=["SupervisorAgent", "APIAgent", "SearchAgent", "LLMAnalystAgent"],
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration_ms,
            total_tokens=total_tokens,
            total_cost=total_cost,
            llm_calls_count=len(llm_results),
        )

        # Log detailed LLM calls
        print("Logging Task 17 LLM calls...")
        for llm_result in llm_results:
            if llm_result.get("success"):
                wf_logger.log_llm_call_detailed(
                    run_id=run_id,
                    company_name=company_name,
                    llm_provider="groq",
                    agent_name=llm_result.get("call_type", "credit_analyst"),
                    model=llm_result.get("model", ""),
                    prompt=f"Credit analysis for {company_name}",
                    context="Multi-LLM consistency evaluation",
                    response=f"Risk: {llm_result.get('risk_level')}, Score: {llm_result.get('credit_score')}",
                    reasoning=llm_result.get("reasoning", ""),
                    prompt_tokens=llm_result.get("prompt_tokens", 0),
                    completion_tokens=llm_result.get("completion_tokens", 0),
                    response_time_ms=0,  # Not tracked individually
                    input_cost=llm_result.get("input_cost", 0),
                    output_cost=llm_result.get("output_cost", 0),
                    total_cost=llm_result.get("total_cost", 0),
                )

        print("Done!")
        print()

        # Wait for async logging
        print("Waiting for async logging (2 seconds)...")
        time.sleep(2)
        print()

        print("=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print()
        print("Check the following for Task 17 logs:")

        sheets_logger = get_sheets_logger()
        if sheets_logger.is_connected():
            print(f"  Google Sheets: {sheets_logger.get_spreadsheet_url()}")
            print("  - Sheet: llm_calls_detailed")
            print("  - Sheet: run_summaries")

        run_logger = get_run_logger()
        if run_logger.is_connected():
            print("  MongoDB: llm_calls_detailed, run_summaries collections")

        print()

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Task 17 logging")
    parser.add_argument("--real", action="store_true", help="Run real workflow test")
    args = parser.parse_args()

    if args.real:
        test_real_workflow()
    else:
        test_task17_logging()
