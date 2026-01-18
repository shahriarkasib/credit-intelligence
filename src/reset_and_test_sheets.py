#!/usr/bin/env python3
"""
Reset all Google Sheets and run test analyses to verify data logging.

This script:
1. Clears all data from all sheets
2. Reinitializes headers
3. Runs 3 company analyses
4. Lists all columns and verifies data
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import gspread
from google.oauth2.service_account import Credentials


def get_sheets_client():
    """Connect to Google Sheets."""
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")

    if not creds_path or not spreadsheet_id:
        print("ERROR: GOOGLE_CREDENTIALS_PATH or GOOGLE_SPREADSHEET_ID not set")
        return None, None

    # Resolve path
    project_root = Path(__file__).parent.parent
    full_path = project_root / creds_path

    if not full_path.exists():
        print(f"ERROR: Credentials file not found: {full_path}")
        return None, None

    creds = Credentials.from_service_account_file(str(full_path), scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(spreadsheet_id)

    return client, spreadsheet


def clear_all_sheets(spreadsheet):
    """Clear all data from all sheets."""
    print("\n" + "="*60)
    print("STEP 1: CLEARING ALL SHEETS")
    print("="*60)

    worksheets = spreadsheet.worksheets()

    for ws in worksheets:
        try:
            # Get all values to count rows
            all_values = ws.get_all_values()
            row_count = len(all_values)

            if row_count > 1:  # Has data beyond header
                # Clear everything except header
                ws.delete_rows(2, row_count)
                print(f"  ✓ Cleared {ws.title}: deleted {row_count - 1} data rows")
            elif row_count == 1:
                print(f"  ○ {ws.title}: already empty (header only)")
            else:
                print(f"  ○ {ws.title}: completely empty")

        except Exception as e:
            print(f"  ✗ Error clearing {ws.title}: {e}")

    print("\n  All sheets cleared!")


def reinitialize_sheets(spreadsheet):
    """Reinitialize all sheet headers."""
    print("\n" + "="*60)
    print("STEP 2: REINITIALIZING SHEET HEADERS")
    print("="*60)

    # Import sheet configs from sheets_logger
    from run_logging.sheets_logger import SheetsLogger

    # Get sheet configs by creating a temp instance
    sheet_configs = {
        "runs": [
            "run_id", "company_name", "node", "agent_name", "model", "temperature",
            "status", "started_at", "completed_at",
            "risk_level", "credit_score", "confidence", "total_time_ms",
            "total_steps", "total_llm_calls", "tools_used", "evaluation_score",
            "timestamp", "generated_by"
        ],
        "tool_calls": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "tool_name", "tool_input", "tool_output",
            "execution_time_ms", "status", "error",
            "timestamp", "generated_by"
        ],
        "assessments": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "model", "temperature", "prompt",
            "risk_level", "credit_score", "confidence", "reasoning", "recommendations",
            "duration_ms", "status",
            "timestamp", "generated_by"
        ],
        "evaluations": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
            "tool_selection_score", "tool_reasoning",
            "data_quality_score", "data_reasoning",
            "synthesis_score", "synthesis_reasoning", "overall_score",
            "duration_ms", "status",
            "timestamp", "generated_by"
        ],
        "tool_selections": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
            "selected_tools", "expected_tools", "correct_tools", "missing_tools", "extra_tools",
            "precision", "recall", "f1_score", "reasoning",
            "duration_ms", "status",
            "timestamp", "generated_by"
        ],
        "step_logs": [
            "run_id", "company_name", "node", "node_type", "agent_name",
            "step_name", "step_number", "model", "temperature",
            "input_summary", "output_summary",
            "execution_time_ms", "status", "error",
            "timestamp", "generated_by"
        ],
        "llm_calls": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "call_type", "model", "temperature", "context",
            "prompt_summary", "response_summary",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_cost", "output_cost", "total_cost",
            "execution_time_ms", "status",
            "timestamp", "generated_by"
        ],
        "consistency_scores": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "model_name", "evaluation_type", "num_runs",
            "risk_level_consistency", "score_consistency", "score_std",
            "overall_consistency", "risk_levels", "credit_scores",
            "duration_ms", "status",
            "timestamp", "generated_by"
        ],
        "data_sources": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "source_name", "records_found", "data_summary",
            "execution_time_ms", "status", "error",
            "timestamp", "generated_by"
        ],
        "langsmith_traces": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "step_name", "run_type", "model", "temperature",
            "input_preview", "output_preview",
            "latency_ms", "status", "error",
            "timestamp", "generated_by"
        ],
        "langgraph_events": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "event_type", "event_name", "model", "temperature", "tokens",
            "input_preview", "output_preview",
            "duration_ms", "status", "error",
            "timestamp", "generated_by"
        ],
        "llm_calls_detailed": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "llm_provider", "model", "temperature",
            "prompt", "context", "response", "reasoning",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_cost", "output_cost", "total_cost",
            "response_time_ms", "status", "error",
            "timestamp", "generated_by"
        ],
        "run_summaries": [
            "run_id", "company_name", "node", "node_type", "model", "temperature",
            "status", "risk_level", "credit_score", "confidence", "reasoning",
            "tool_selection_score", "data_quality_score", "synthesis_score", "overall_score",
            "final_decision", "decision_reasoning",
            "errors", "warnings", "tools_used", "agents_used", "total_steps",
            "started_at", "completed_at", "duration_ms",
            "total_tokens", "total_cost", "llm_calls_count",
            "timestamp", "generated_by"
        ],
        "agent_metrics": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
            "intent_correctness", "plan_quality", "tool_choice_correctness",
            "tool_completeness", "trajectory_match", "final_answer_quality",
            "step_count", "tool_calls", "latency_ms",
            "overall_score",
            "intent_details", "plan_details", "tool_details",
            "trajectory_details", "answer_details",
            "status", "timestamp", "generated_by"
        ],
        "unified_metrics": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
            "faithfulness", "hallucination", "answer_relevancy",
            "factual_accuracy", "final_answer_quality", "accuracy_score",
            "same_model_consistency", "cross_model_consistency",
            "risk_level_agreement", "semantic_similarity", "consistency_score",
            "intent_correctness", "plan_quality", "tool_choice_correctness",
            "tool_completeness", "trajectory_match", "agent_final_answer",
            "agent_efficiency_score",
            "overall_quality_score", "libraries_used", "evaluation_time_ms",
            "status", "timestamp", "generated_by"
        ],
        "llm_judge_results": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "model_used", "temperature",
            "accuracy_score", "completeness_score", "consistency_score",
            "actionability_score", "data_utilization_score", "overall_score",
            "accuracy_reasoning", "completeness_reasoning", "consistency_reasoning",
            "actionability_reasoning", "data_utilization_reasoning", "overall_reasoning",
            "benchmark_alignment", "benchmark_comparison",
            "suggestions",
            "tokens_used", "evaluation_cost", "duration_ms", "status",
            "timestamp", "generated_by"
        ],
        "model_consistency": [
            "eval_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "model_name", "num_runs",
            "risk_level_consistency", "credit_score_mean", "credit_score_std",
            "confidence_variance", "reasoning_similarity",
            "risk_factors_overlap", "recommendations_overlap",
            "overall_consistency", "is_consistent", "consistency_grade",
            "llm_judge_analysis", "llm_judge_concerns",
            "run_details",
            "duration_ms", "status", "timestamp", "generated_by"
        ],
        "cross_model_eval": [
            "eval_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "models_compared", "num_models",
            "risk_level_agreement", "credit_score_mean", "credit_score_std",
            "credit_score_range", "confidence_agreement",
            "best_model", "best_model_reasoning",
            "cross_model_agreement",
            "llm_judge_analysis", "model_recommendations",
            "model_results", "pairwise_comparisons",
            "duration_ms", "status", "timestamp", "generated_by"
        ],
        "deepeval_metrics": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "model_used",
            "answer_relevancy", "faithfulness", "hallucination",
            "contextual_relevancy", "bias", "toxicity",
            "overall_score",
            "answer_relevancy_reason", "faithfulness_reason", "hallucination_reason",
            "contextual_relevancy_reason", "bias_reason",
            "input_query", "context_summary", "assessment_summary",
            "evaluation_model", "evaluation_time_ms", "status",
            "timestamp", "generated_by"
        ],
        "openevals_metrics": [
            "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
            "model_used",
            "intent_correctness", "plan_quality",
            "tool_choice_correctness", "tool_completeness",
            "trajectory_match", "final_answer_quality",
            "step_count", "tool_calls", "latency_ms",
            "overall_score",
            "intent_details", "plan_details", "tool_details",
            "trajectory_details", "answer_details",
            "evaluation_time_ms", "status",
            "timestamp", "generated_by"
        ]
    }

    existing_sheets = {ws.title: ws for ws in spreadsheet.worksheets()}

    for sheet_name, headers in sheet_configs.items():
        try:
            if sheet_name in existing_sheets:
                ws = existing_sheets[sheet_name]
                # Clear and update header
                ws.clear()
                ws.append_row(headers)
                print(f"  ✓ Reset {sheet_name}: {len(headers)} columns")
            else:
                # Create new sheet
                ws = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=len(headers))
                ws.append_row(headers)
                print(f"  + Created {sheet_name}: {len(headers)} columns")

        except Exception as e:
            print(f"  ✗ Error with {sheet_name}: {e}")

    print(f"\n  Total sheets: {len(sheet_configs)}")


def run_company_analyses():
    """Run 3 company analyses."""
    print("\n" + "="*60)
    print("STEP 3: RUNNING 2 COMPANY ANALYSES (PUBLIC + PRIVATE)")
    print("="*60)

    from agents.graph import run_sync_with_logging

    companies = ["Apple Inc", "Private Tech Solutions LLC"]  # Public and private company
    results = []

    for company in companies:
        print(f"\n  Analyzing: {company}")
        print("  " + "-"*40)

        try:
            start = time.time()
            result = run_sync_with_logging(company)
            elapsed = time.time() - start

            assessment = result.get("assessment", {}) or {}
            risk = assessment.get("overall_risk_level", "N/A")
            score = assessment.get("credit_score_estimate", 0)
            conf = assessment.get("confidence_score", 0)

            print(f"  ✓ Completed in {elapsed:.1f}s")
            print(f"    Risk: {risk}, Score: {score}, Confidence: {conf:.2f}")

            results.append({
                "company": company,
                "status": "success",
                "risk_level": risk,
                "credit_score": score,
                "confidence": conf,
                "time_seconds": elapsed,
            })

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                "company": company,
                "status": "failed",
                "error": str(e),
            })

        # Small delay between companies
        time.sleep(2)

    return results


def verify_sheet_data(spreadsheet):
    """Verify all sheets have data and list columns."""
    print("\n" + "="*60)
    print("STEP 4: VERIFYING SHEET DATA")
    print("="*60)

    worksheets = spreadsheet.worksheets()

    summary = []

    for ws in worksheets:
        try:
            all_values = ws.get_all_values()

            if not all_values:
                summary.append({
                    "sheet": ws.title,
                    "status": "EMPTY",
                    "columns": 0,
                    "data_rows": 0,
                })
                continue

            headers = all_values[0]
            data_rows = len(all_values) - 1

            # Check which columns have data
            cols_with_data = []
            cols_empty = []

            if data_rows > 0:
                for i, header in enumerate(headers):
                    has_data = any(
                        len(row) > i and row[i] and row[i].strip()
                        for row in all_values[1:]
                    )
                    if has_data:
                        cols_with_data.append(header)
                    else:
                        cols_empty.append(header)

            summary.append({
                "sheet": ws.title,
                "status": "HAS_DATA" if data_rows > 0 else "NO_DATA",
                "columns": len(headers),
                "data_rows": data_rows,
                "cols_with_data": len(cols_with_data),
                "cols_empty": len(cols_empty),
                "empty_columns": cols_empty[:5] if cols_empty else [],  # First 5
            })

        except Exception as e:
            summary.append({
                "sheet": ws.title,
                "status": "ERROR",
                "error": str(e),
            })

    # Print summary
    print("\n  SHEET SUMMARY:")
    print("  " + "-"*70)
    print(f"  {'Sheet':<25} {'Status':<12} {'Cols':<6} {'Rows':<6} {'Data%':<8}")
    print("  " + "-"*70)

    for item in summary:
        if item["status"] == "ERROR":
            print(f"  {item['sheet']:<25} {'ERROR':<12}")
            continue

        cols = item.get("columns", 0)
        rows = item.get("data_rows", 0)
        cols_data = item.get("cols_with_data", 0)

        pct = f"{cols_data/cols*100:.0f}%" if cols > 0 else "N/A"
        status = item["status"]

        print(f"  {item['sheet']:<25} {status:<12} {cols:<6} {rows:<6} {pct:<8}")

        # Show empty columns if any
        if item.get("empty_columns"):
            print(f"    └─ Empty columns: {', '.join(item['empty_columns'][:3])}...")

    print("  " + "-"*70)

    return summary


def list_all_columns(spreadsheet):
    """List all columns for each sheet."""
    print("\n" + "="*60)
    print("STEP 5: ALL COLUMNS BY SHEET")
    print("="*60)

    worksheets = spreadsheet.worksheets()

    for ws in worksheets:
        try:
            all_values = ws.get_all_values()
            if not all_values:
                continue

            headers = all_values[0]
            data_rows = len(all_values) - 1

            print(f"\n  📋 {ws.title} ({len(headers)} columns, {data_rows} rows)")
            print("  " + "-"*50)

            for i, header in enumerate(headers, 1):
                # Check if column has data
                has_data = "✓" if data_rows > 0 and any(
                    len(row) > i-1 and row[i-1] and str(row[i-1]).strip()
                    for row in all_values[1:]
                ) else "○"

                print(f"    {i:2}. {has_data} {header}")

        except Exception as e:
            print(f"\n  📋 {ws.title}: ERROR - {e}")


def main():
    """Main function."""
    print("\n" + "="*60)
    print("GOOGLE SHEETS RESET AND TEST")
    print("="*60)

    # Connect to Google Sheets
    print("\nConnecting to Google Sheets...")
    client, spreadsheet = get_sheets_client()

    if not spreadsheet:
        print("Failed to connect to Google Sheets!")
        return

    print(f"Connected to: {spreadsheet.title}")
    print(f"URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")

    # Step 1: Clear all sheets
    clear_all_sheets(spreadsheet)

    # Step 2: Reinitialize headers
    reinitialize_sheets(spreadsheet)

    # Wait for sheets to sync
    print("\n  Waiting for Google Sheets to sync...")
    time.sleep(3)

    # Step 3: Run company analyses
    results = run_company_analyses()

    # Wait for async logging to complete
    print("\n  Waiting for logs to be written...")
    time.sleep(5)

    # Step 4: Verify data
    summary = verify_sheet_data(spreadsheet)

    # Step 5: List all columns
    list_all_columns(spreadsheet)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\n  Company Analysis Results:")
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        print(f"    {status} {r['company']}: {r.get('risk_level', 'N/A')}, Score: {r.get('credit_score', 'N/A')}")

    sheets_with_data = sum(1 for s in summary if s.get("data_rows", 0) > 0)
    total_sheets = len(summary)

    print(f"\n  Sheets with data: {sheets_with_data}/{total_sheets}")
    print(f"\n  Spreadsheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")


if __name__ == "__main__":
    main()
