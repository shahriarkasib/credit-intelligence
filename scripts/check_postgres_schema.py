#!/usr/bin/env python3
"""Check PostgreSQL schema and compare with expected columns."""

import os
import sys
sys.path.insert(0, "/Users/shariarsourav/Desktop/credit_intelligence/src")

from dotenv import load_dotenv
load_dotenv("/Users/shariarsourav/Desktop/credit_intelligence/.env")

import psycopg2
from psycopg2.extras import RealDictCursor

# Get database URL
db_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

print(f"Connecting to PostgreSQL...")

# Expected columns from sheets_logger (Google Sheets is source of truth)
EXPECTED_COLUMNS = {
    "wf_runs": ["run_id", "company_name", "node", "agent_name", "master_agent", "model", "temperature",
                "status", "started_at", "completed_at", "risk_level", "credit_score", "confidence",
                "total_time_ms", "total_steps", "total_llm_calls", "tools_used", "evaluation_score",
                "workflow_correct", "output_correct", "timestamp"],
    "wf_tool_calls": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                      "tool_name", "tool_input", "tool_output", "parent_node", "workflow_phase",
                      "call_depth", "parent_tool_id", "execution_time_ms", "status", "error", "timestamp"],
    "wf_llm_calls": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                     "call_type", "model", "temperature", "prompt", "response", "reasoning", "context",
                     "current_task", "prompt_tokens", "completion_tokens", "total_tokens", "input_cost",
                     "output_cost", "total_cost", "execution_time_ms", "status", "error", "timestamp"],
    "wf_assessments": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                       "model", "temperature", "prompt", "risk_level", "credit_score", "confidence",
                       "reasoning", "recommendations", "duration_ms", "status", "timestamp"],
    "wf_plans": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "num_tasks",
                 "plan_summary", "full_plan", "task_1", "task_2", "task_3", "task_4", "task_5",
                 "task_6", "task_7", "task_8", "task_9", "task_10", "created_at", "status", "timestamp"],
    "wf_data_sources": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                        "source_name", "records_found", "data_summary", "execution_time_ms", "status",
                        "error", "timestamp"],
    "wf_state_dumps": ["run_id", "company_name", "node", "master_agent", "step_number", "company_info_json",
                       "plan_json", "plan_size_bytes", "plan_tasks_count", "api_data_summary",
                       "api_data_size_bytes", "api_sources_count", "search_data_summary",
                       "search_data_size_bytes", "risk_level", "credit_score", "confidence",
                       "assessment_json", "coalition_score", "agent_metrics_score", "evaluation_json",
                       "errors_json", "error_count", "total_state_size_bytes", "duration_ms",
                       "status", "timestamp"],
    "lg_events": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                  "event_type", "event_name", "model", "temperature", "tokens", "input_preview",
                  "output_preview", "duration_ms", "status", "error", "timestamp"],
    "meta_prompts": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                     "prompt_id", "prompt_name", "category", "system_prompt", "user_prompt",
                     "variables_json", "model", "temperature", "timestamp"],
    "eval_results": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                     "model", "tool_selection_score", "tool_reasoning", "data_quality_score", "data_reasoning",
                     "synthesis_score", "synthesis_reasoning", "overall_score", "eval_status",
                     "duration_ms", "status", "timestamp"],
    "eval_tool_selection": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent",
                            "step_number", "model", "selected_tools", "expected_tools", "correct_tools",
                            "missing_tools", "extra_tools", "precision", "recall", "f1_score",
                            "reasoning", "duration_ms", "status", "timestamp"],
    "eval_consistency": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                         "model_name", "evaluation_type", "num_runs", "risk_level_consistency",
                         "score_consistency", "score_std", "overall_consistency", "eval_status",
                         "risk_levels", "credit_scores", "duration_ms", "status", "timestamp"],
    "eval_cross_model": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                         "models_compared", "num_models", "risk_level_agreement", "credit_score_mean",
                         "credit_score_std", "credit_score_range", "confidence_agreement", "best_model",
                         "best_model_reasoning", "cross_model_agreement", "eval_status", "llm_judge_analysis",
                         "model_recommendations", "model_results", "pairwise_comparisons",
                         "duration_ms", "status", "timestamp"],
    "eval_llm_judge": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                       "model_used", "temperature", "accuracy_score", "completeness_score", "consistency_score",
                       "actionability_score", "data_utilization_score", "overall_score", "eval_status",
                       "accuracy_reasoning", "completeness_reasoning", "consistency_reasoning",
                       "actionability_reasoning", "data_utilization_reasoning", "overall_reasoning",
                       "benchmark_alignment", "benchmark_comparison", "suggestions", "tokens_used",
                       "evaluation_cost", "duration_ms", "status", "timestamp"],
    "eval_agent_metrics": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent",
                           "step_number", "model", "intent_correctness", "plan_quality",
                           "tool_choice_correctness", "tool_completeness", "trajectory_match",
                           "final_answer_quality", "step_count", "tool_calls", "latency_ms",
                           "overall_score", "eval_status", "intent_details", "plan_details",
                           "tool_details", "trajectory_details", "answer_details", "status", "timestamp"],
    "eval_coalition": ["run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                       "is_correct", "correctness_score", "confidence", "correctness_category",
                       "efficiency_score", "quality_score", "tool_score", "consistency_score",
                       "agreement_score", "num_evaluators", "votes_json", "evaluation_time_ms",
                       "status", "timestamp"],
    "eval_log_tests": ["run_id", "company_name", "runs", "langgraph_events", "llm_calls", "tool_calls",
                       "assessments", "evaluations", "tool_selections", "consistency_scores",
                       "data_sources", "plans", "prompts", "cross_model_eval", "llm_judge_results",
                       "agent_metrics", "coalition", "total_sheets_logged", "verification_status", "timestamp"],
}

try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    print("✅ Connected to PostgreSQL\n")

    # Get all base tables (not partitions)
    cursor.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename NOT LIKE '%_2026_%'
        ORDER BY tablename
    """)
    all_tables = [row['tablename'] for row in cursor.fetchall()]

    print(f"Found {len(all_tables)} base tables in PostgreSQL\n")
    print("=" * 80)

    mismatches = []

    for table_name in sorted(EXPECTED_COLUMNS.keys()):
        expected = set(EXPECTED_COLUMNS[table_name])

        # Get actual columns from PostgreSQL
        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
        """, (table_name,))
        actual = set(row['column_name'] for row in cursor.fetchall())

        # Remove 'id' from actual (auto-generated)
        actual.discard('id')

        missing_in_pg = expected - actual
        extra_in_pg = actual - expected

        if missing_in_pg or extra_in_pg:
            print(f"❌ {table_name}")
            if missing_in_pg:
                print(f"   MISSING: {sorted(missing_in_pg)}")
            if extra_in_pg:
                print(f"   EXTRA:   {sorted(extra_in_pg)}")
            mismatches.append({
                "table": table_name,
                "missing": list(missing_in_pg),
                "extra": list(extra_in_pg)
            })
        else:
            print(f"✅ {table_name} - OK ({len(expected)} columns)")

    print("\n" + "=" * 80)
    if mismatches:
        print(f"\n⚠️  {len(mismatches)} tables need schema updates")
    else:
        print("\n✅ All PostgreSQL tables match expected schema!")

    conn.close()

except Exception as e:
    print(f"❌ Error: {e}")
