"""
PostgreSQL Storage Module for Credit Intelligence Logs.

This module provides PostgreSQL-based storage for all workflow logs,
replacing Google Sheets for better scalability, querying, and retention.

Table Naming Convention (per PRD Appendix B):
- wf_*: Workflow execution logs (wf_runs, wf_llm_calls, wf_tool_calls, etc.)
- eval_*: Evaluation results (eval_results, eval_consistency, eval_coalition, etc.)
- lg_*: LangGraph framework tables (lg_events)
- meta_*: Metadata tables (meta_prompts, meta_api_keys)

All tables are partitioned by timestamp (monthly) for efficient retention management.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    pool = None
    sql = None
    RealDictCursor = None
    execute_values = None


class PostgresStorage:
    """
    PostgreSQL storage for Credit Intelligence logs with partitioning support.

    Features:
    - Connection pooling for efficient connections
    - Monthly partitioned tables for data retention
    - Automatic partition creation
    - Foreign key relationships between tables
    """

    # Table definitions with their columns and types
    # Format: (column_name, column_type, nullable, is_partition_key)
    # Naming convention per PRD: wf_* (workflow), eval_* (evaluation), lg_* (langgraph), meta_* (metadata)
    TABLE_SCHEMAS = {
        # Main runs table - the core table that others reference
        "wf_runs": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("risk_level", "VARCHAR(50)", True, False),
                ("credit_score", "INTEGER", True, False),
                ("confidence", "DECIMAL(5,4)", True, False),
                ("reasoning", "TEXT", True, False),
                ("tool_selection_score", "DECIMAL(5,4)", True, False),
                ("data_quality_score", "DECIMAL(5,4)", True, False),
                ("synthesis_score", "DECIMAL(5,4)", True, False),
                ("overall_score", "DECIMAL(5,4)", True, False),
                ("final_decision", "VARCHAR(50)", True, False),
                ("decision_reasoning", "TEXT", True, False),
                ("errors", "JSONB", True, False),
                ("warnings", "JSONB", True, False),
                ("tools_used", "JSONB", True, False),
                ("agents_used", "JSONB", True, False),
                ("started_at", "TIMESTAMPTZ", True, False),
                ("completed_at", "TIMESTAMPTZ", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("total_tokens", "INTEGER", True, False),
                ("total_cost", "DECIMAL(12,6)", True, False),
                ("llm_calls_count", "INTEGER", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),  # Partition key
            ],
            "primary_key": ["id", "timestamp"],
            "unique_constraints": [("run_id", "timestamp")],
            "indexes": ["run_id", "company_name", "status", "risk_level"],
        },

        # LLM Calls (workflow execution)
        "wf_llm_calls": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("call_type", "VARCHAR(100)", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("provider", "VARCHAR(50)", True, False),
                ("temperature", "DECIMAL(3,2)", True, False),
                ("prompt", "TEXT", True, False),
                ("response", "TEXT", True, False),
                ("reasoning", "TEXT", True, False),
                ("context", "TEXT", True, False),
                ("current_task", "TEXT", True, False),
                ("prompt_tokens", "INTEGER", True, False),
                ("completion_tokens", "INTEGER", True, False),
                ("total_tokens", "INTEGER", True, False),
                ("input_cost", "DECIMAL(12,8)", True, False),
                ("output_cost", "DECIMAL(12,8)", True, False),
                ("total_cost", "DECIMAL(12,8)", True, False),
                ("execution_time_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("error", "TEXT", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "agent_name", "model", "call_type"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Tool Calls (workflow execution)
        "wf_tool_calls": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("tool_name", "VARCHAR(100)", True, False),
                ("tool_input", "JSONB", True, False),
                ("tool_output", "JSONB", True, False),
                ("parent_node", "VARCHAR(100)", True, False),
                ("workflow_phase", "VARCHAR(100)", True, False),
                ("call_depth", "INTEGER", True, False),
                ("parent_tool_id", "VARCHAR(100)", True, False),
                ("execution_time_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("error", "TEXT", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "tool_name", "agent_name"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # LangGraph Events (framework)
        "lg_events": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("event_type", "VARCHAR(100)", True, False),
                ("event_name", "VARCHAR(255)", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("temperature", "DECIMAL(3,2)", True, False),
                ("tokens", "INTEGER", True, False),
                ("input_preview", "TEXT", True, False),
                ("output_preview", "TEXT", True, False),
                ("state_before", "JSONB", True, False),
                ("state_after", "JSONB", True, False),
                ("metadata", "JSONB", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("error", "TEXT", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "event_type", "node", "agent_name"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Plans (workflow execution)
        "wf_plans": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("num_tasks", "INTEGER", True, False),
                ("plan_summary", "TEXT", True, False),
                ("full_plan", "JSONB", True, False),
                ("task_1", "TEXT", True, False),
                ("task_2", "TEXT", True, False),
                ("task_3", "TEXT", True, False),
                ("task_4", "TEXT", True, False),
                ("task_5", "TEXT", True, False),
                ("task_6", "TEXT", True, False),
                ("task_7", "TEXT", True, False),
                ("task_8", "TEXT", True, False),
                ("task_9", "TEXT", True, False),
                ("task_10", "TEXT", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "agent_name"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Prompts (metadata)
        "meta_prompts": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("prompt_id", "VARCHAR(100)", True, False),
                ("prompt_name", "VARCHAR(255)", True, False),
                ("category", "VARCHAR(100)", True, False),
                ("system_prompt", "TEXT", True, False),
                ("user_prompt", "TEXT", True, False),
                ("variables", "JSONB", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("temperature", "DECIMAL(3,2)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "prompt_name", "category"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Data Sources (workflow execution)
        "wf_data_sources": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("source_name", "VARCHAR(100)", True, False),
                ("records_found", "INTEGER", True, False),
                ("data_summary", "TEXT", True, False),
                ("raw_data", "JSONB", True, False),
                ("execution_time_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("error", "TEXT", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "source_name"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Assessments (workflow execution)
        "wf_assessments": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("temperature", "DECIMAL(3,2)", True, False),
                ("prompt", "TEXT", True, False),
                ("risk_level", "VARCHAR(50)", True, False),
                ("credit_score", "INTEGER", True, False),
                ("confidence", "DECIMAL(5,4)", True, False),
                ("reasoning", "TEXT", True, False),
                ("recommendations", "JSONB", True, False),
                ("risk_factors", "JSONB", True, False),
                ("positive_factors", "JSONB", True, False),
                ("financial_summary", "JSONB", True, False),
                ("sources_used", "JSONB", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "risk_level", "company_name"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Evaluations (eval results)
        "eval_results": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("evaluation_type", "VARCHAR(100)", True, False),
                ("tool_selection_score", "DECIMAL(5,4)", True, False),
                ("tool_reasoning", "TEXT", True, False),
                ("data_quality_score", "DECIMAL(5,4)", True, False),
                ("data_reasoning", "TEXT", True, False),
                ("synthesis_score", "DECIMAL(5,4)", True, False),
                ("synthesis_reasoning", "TEXT", True, False),
                ("overall_score", "DECIMAL(5,4)", True, False),
                ("eval_status", "VARCHAR(50)", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "evaluation_type", "overall_score"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Tool Selections (evaluation)
        "eval_tool_selection": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("selected_tools", "JSONB", True, False),
                ("expected_tools", "JSONB", True, False),
                ("correct_tools", "JSONB", True, False),
                ("missing_tools", "JSONB", True, False),
                ("extra_tools", "JSONB", True, False),
                ("precision_score", "DECIMAL(5,4)", True, False),
                ("recall_score", "DECIMAL(5,4)", True, False),
                ("f1_score", "DECIMAL(5,4)", True, False),
                ("reasoning", "TEXT", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "f1_score"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Consistency Scores (evaluation)
        "eval_consistency": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("model_name", "VARCHAR(100)", True, False),
                ("evaluation_type", "VARCHAR(100)", True, False),
                ("num_runs", "INTEGER", True, False),
                ("risk_level_consistency", "DECIMAL(5,4)", True, False),
                ("score_consistency", "DECIMAL(5,4)", True, False),
                ("score_std", "DECIMAL(10,4)", True, False),
                ("overall_consistency", "DECIMAL(5,4)", True, False),
                ("eval_status", "VARCHAR(50)", True, False),
                ("risk_levels", "JSONB", True, False),
                ("credit_scores", "JSONB", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "overall_consistency"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Cross-Model Evaluation
        "eval_cross_model": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("models_compared", "JSONB", True, False),
                ("num_models", "INTEGER", True, False),
                ("risk_level_agreement", "DECIMAL(5,4)", True, False),
                ("credit_score_mean", "DECIMAL(10,4)", True, False),
                ("credit_score_std", "DECIMAL(10,4)", True, False),
                ("credit_score_range", "DECIMAL(10,4)", True, False),
                ("confidence_agreement", "DECIMAL(5,4)", True, False),
                ("best_model", "VARCHAR(100)", True, False),
                ("best_model_reasoning", "TEXT", True, False),
                ("cross_model_agreement", "DECIMAL(5,4)", True, False),
                ("eval_status", "VARCHAR(50)", True, False),
                ("llm_judge_analysis", "TEXT", True, False),
                ("model_recommendations", "JSONB", True, False),
                ("model_results", "JSONB", True, False),
                ("pairwise_comparisons", "JSONB", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "cross_model_agreement"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # LLM Judge Results (evaluation)
        "eval_llm_judge": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("model_used", "VARCHAR(100)", True, False),
                ("temperature", "DECIMAL(3,2)", True, False),
                ("accuracy_score", "DECIMAL(5,4)", True, False),
                ("completeness_score", "DECIMAL(5,4)", True, False),
                ("consistency_score", "DECIMAL(5,4)", True, False),
                ("actionability_score", "DECIMAL(5,4)", True, False),
                ("data_utilization_score", "DECIMAL(5,4)", True, False),
                ("overall_score", "DECIMAL(5,4)", True, False),
                ("eval_status", "VARCHAR(50)", True, False),
                ("accuracy_reasoning", "TEXT", True, False),
                ("completeness_reasoning", "TEXT", True, False),
                ("consistency_reasoning", "TEXT", True, False),
                ("actionability_reasoning", "TEXT", True, False),
                ("data_utilization_reasoning", "TEXT", True, False),
                ("overall_reasoning", "TEXT", True, False),
                ("benchmark_alignment", "DECIMAL(5,4)", True, False),
                ("benchmark_comparison", "JSONB", True, False),
                ("suggestions", "JSONB", True, False),
                ("tokens_used", "INTEGER", True, False),
                ("evaluation_cost", "DECIMAL(12,8)", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "overall_score"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Agent Metrics (evaluation)
        "eval_agent_metrics": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("node_type", "VARCHAR(50)", True, False),
                ("agent_name", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                ("model", "VARCHAR(100)", True, False),
                ("intent_correctness", "DECIMAL(5,4)", True, False),
                ("plan_quality", "DECIMAL(5,4)", True, False),
                ("tool_choice_correctness", "DECIMAL(5,4)", True, False),
                ("tool_completeness", "DECIMAL(5,4)", True, False),
                ("trajectory_match", "DECIMAL(5,4)", True, False),
                ("final_answer_quality", "DECIMAL(5,4)", True, False),
                ("step_count", "INTEGER", True, False),
                ("tool_calls_count", "INTEGER", True, False),
                ("latency_ms", "DECIMAL(15,3)", True, False),
                ("overall_score", "DECIMAL(5,4)", True, False),
                ("eval_status", "VARCHAR(50)", True, False),
                ("intent_details", "JSONB", True, False),
                ("plan_details", "JSONB", True, False),
                ("tool_details", "JSONB", True, False),
                ("trajectory_details", "JSONB", True, False),
                ("answer_details", "JSONB", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "agent_name", "overall_score"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Coalition Results (evaluation)
        "eval_coalition": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("is_correct", "BOOLEAN", True, False),
                ("correctness_score", "DECIMAL(5,4)", True, False),
                ("confidence", "DECIMAL(5,4)", True, False),
                ("correctness_category", "VARCHAR(50)", True, False),
                ("agreement_score", "DECIMAL(5,4)", True, False),
                ("num_evaluators", "INTEGER", True, False),
                ("efficiency_score", "DECIMAL(5,4)", True, False),
                ("quality_score", "DECIMAL(5,4)", True, False),
                ("tool_score", "DECIMAL(5,4)", True, False),
                ("consistency_score", "DECIMAL(5,4)", True, False),
                ("votes", "JSONB", True, False),
                ("evaluation_time_ms", "DECIMAL(15,3)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "correctness_score", "correctness_category"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # Log Tests (evaluation/verification)
        "eval_log_tests": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("runs_logged", "INTEGER", True, False),
                ("langgraph_events_logged", "INTEGER", True, False),
                ("llm_calls_logged", "INTEGER", True, False),
                ("tool_calls_logged", "INTEGER", True, False),
                ("assessments_logged", "INTEGER", True, False),
                ("evaluations_logged", "INTEGER", True, False),
                ("tool_selections_logged", "INTEGER", True, False),
                ("consistency_scores_logged", "INTEGER", True, False),
                ("data_sources_logged", "INTEGER", True, False),
                ("plans_logged", "INTEGER", True, False),
                ("prompts_logged", "INTEGER", True, False),
                ("cross_model_eval_logged", "INTEGER", True, False),
                ("llm_judge_results_logged", "INTEGER", True, False),
                ("agent_metrics_logged", "INTEGER", True, False),
                ("coalition_logged", "INTEGER", True, False),
                ("total_tables_logged", "INTEGER", True, False),
                ("verification_status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "verification_status"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # State Dumps (workflow execution snapshots)
        "wf_state_dumps": {
            "prefix": "",
            "columns": [
                ("id", "BIGSERIAL", False, False),
                ("run_id", "VARCHAR(64)", False, False),
                ("company_name", "VARCHAR(255)", True, False),
                ("node", "VARCHAR(100)", True, False),
                ("step_number", "INTEGER", True, False),
                # Company info
                ("company_info", "JSONB", True, False),
                # Plan
                ("plan", "JSONB", True, False),
                ("plan_size_bytes", "INTEGER", True, False),
                # API data
                ("api_data", "JSONB", True, False),
                ("api_data_size_bytes", "INTEGER", True, False),
                ("api_sources_count", "INTEGER", True, False),
                # Search data
                ("search_data", "JSONB", True, False),
                ("search_data_size_bytes", "INTEGER", True, False),
                # Assessment
                ("assessment", "JSONB", True, False),
                ("risk_level", "VARCHAR(50)", True, False),
                ("credit_score", "INTEGER", True, False),
                ("confidence", "DECIMAL(5,4)", True, False),
                # Evaluation
                ("evaluation", "JSONB", True, False),
                ("evaluation_scores", "JSONB", True, False),
                # Errors
                ("errors", "JSONB", True, False),
                ("error_count", "INTEGER", True, False),
                # Metadata
                ("total_state_size_bytes", "INTEGER", True, False),
                ("duration_ms", "DECIMAL(15,3)", True, False),
                ("status", "VARCHAR(50)", True, False),
                ("timestamp", "TIMESTAMPTZ", False, True),
            ],
            "primary_key": ["id", "timestamp"],
            "indexes": ["run_id", "company_name", "status"],
            "foreign_keys": [("run_id", "wf_runs", "run_id")],
        },

        # API Keys (metadata, not partitioned)
        "meta_api_keys": {
            "prefix": "",
            "columns": [
                ("id", "SERIAL", False, False),
                ("key_name", "VARCHAR(100)", False, False),
                ("key_value", "TEXT", False, False),
                ("created_at", "TIMESTAMPTZ", True, False),
                ("updated_at", "TIMESTAMPTZ", True, False),
            ],
            "primary_key": ["id"],
            "unique_constraints": [("key_name",)],
            "indexes": ["key_name"],
            "partitioned": False,  # Small table, no partitioning needed
        },
    }

    # Table name aliases for backward compatibility (old_name -> new_name)
    TABLE_ALIASES = {
        "runs": "wf_runs",
        "llm_calls": "wf_llm_calls",
        "tool_calls": "wf_tool_calls",
        "langgraph_events": "lg_events",
        "plans": "wf_plans",
        "prompts": "meta_prompts",
        "data_sources": "wf_data_sources",
        "assessments": "wf_assessments",
        "evaluations": "eval_results",
        "tool_selections": "eval_tool_selection",
        "consistency_scores": "eval_consistency",
        "cross_model_eval": "eval_cross_model",
        "llm_judge_results": "eval_llm_judge",
        "agent_metrics": "eval_agent_metrics",
        "coalition": "eval_coalition",
        "log_tests": "eval_log_tests",
        "state_dumps": "wf_state_dumps",
        "api_keys": "meta_api_keys",
    }

    # Reverse mapping (new_name -> old_name) for display
    TABLE_REVERSE_ALIASES = {v: k for k, v in TABLE_ALIASES.items()}

    def __init__(self, database_url: Optional[str] = None, min_conn: int = 1, max_conn: int = 10):
        """
        Initialize PostgreSQL storage.

        Args:
            database_url: PostgreSQL connection URL. If not provided, uses DATABASE_URL env var.
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialized = False

        if not POSTGRES_AVAILABLE:
            logger.warning("psycopg2 not available. Install with: pip install psycopg2-binary")

    def connect(self) -> bool:
        """
        Initialize connection pool.

        Returns:
            True if connection successful
        """
        if not POSTGRES_AVAILABLE:
            logger.error("psycopg2 not available")
            return False

        if not self.database_url:
            logger.error("DATABASE_URL not configured")
            return False

        try:
            # Heroku Postgres uses postgres:// but psycopg2 needs postgresql://
            db_url = self.database_url
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)

            self._pool = pool.ThreadedConnectionPool(
                self.min_conn,
                self.max_conn,
                db_url,
                cursor_factory=RealDictCursor
            )
            logger.info("PostgreSQL connection pool initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to PostgreSQL."""
        return self._pool is not None

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Connection pool not initialized")

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._pool.putconn(conn)

    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    def initialize_schema(self, months_ahead: int = 12) -> bool:
        """
        Initialize database schema with all tables and partitions.

        Args:
            months_ahead: Number of months to create partitions for

        Returns:
            True if successful
        """
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create tables in order (runs first, then others that reference it)
                tables_order = ["runs"] + [t for t in self.TABLE_SCHEMAS.keys() if t != "runs"]

                for table_name in tables_order:
                    schema = self.TABLE_SCHEMAS[table_name]
                    self._create_table(cursor, table_name, schema, months_ahead)

                logger.info("PostgreSQL schema initialized successfully")
                self._initialized = True
                return True

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False

    def _create_table(self, cursor, table_name: str, schema: Dict, months_ahead: int):
        """Create a single table with partitions if needed."""
        columns = schema["columns"]
        primary_key = schema.get("primary_key", [])
        is_partitioned = schema.get("partitioned", True)  # Default to partitioned

        # Build column definitions
        col_defs = []
        partition_col = None

        for col_name, col_type, nullable, is_partition in columns:
            null_str = "" if nullable else " NOT NULL"
            col_defs.append(f"{col_name} {col_type}{null_str}")
            if is_partition:
                partition_col = col_name

        # Add primary key
        if primary_key:
            col_defs.append(f"PRIMARY KEY ({', '.join(primary_key)})")

        columns_sql = ",\n  ".join(col_defs)

        # Create table (partitioned or regular)
        if is_partitioned and partition_col:
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              {columns_sql}
            ) PARTITION BY RANGE ({partition_col});
            """
        else:
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              {columns_sql}
            );
            """

        cursor.execute(create_sql)
        logger.info(f"Created table: {table_name}")

        # Create partitions for partitioned tables
        if is_partitioned and partition_col:
            self._create_partitions(cursor, table_name, months_ahead)

        # Create indexes
        for idx_col in schema.get("indexes", []):
            idx_name = f"idx_{table_name}_{idx_col}"
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({idx_col});
                """)
            except Exception as e:
                # Index on partitioned table might fail, try on each partition
                logger.debug(f"Index {idx_name} creation note: {e}")

        # Create unique constraints
        for unique_cols in schema.get("unique_constraints", []):
            constraint_name = f"uq_{table_name}_{'_'.join(unique_cols)}"
            try:
                cols_str = ", ".join(unique_cols)
                cursor.execute(f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS {constraint_name} ON {table_name} ({cols_str});
                """)
            except Exception as e:
                logger.debug(f"Unique constraint {constraint_name} note: {e}")

    def _create_partitions(self, cursor, table_name: str, months_ahead: int):
        """Create monthly partitions for a table."""
        now = datetime.now(timezone.utc)
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        for i in range(months_ahead):
            p_start = start_of_month + timedelta(days=32 * i)
            p_start = p_start.replace(day=1)

            p_end = p_start + timedelta(days=32)
            p_end = p_end.replace(day=1)

            p_name = f"{table_name}_{p_start.strftime('%Y_%m')}"

            try:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {p_name}
                    PARTITION OF {table_name}
                    FOR VALUES FROM ('{p_start.strftime('%Y-%m-%d')}')
                    TO ('{p_end.strftime('%Y-%m-%d')}');
                """)
                logger.debug(f"Created partition: {p_name}")
            except Exception as e:
                # Partition might already exist
                logger.debug(f"Partition {p_name} note: {e}")

    def _resolve_table_name(self, table_name: str) -> str:
        """
        Resolve table name, supporting both old and new naming conventions.

        Args:
            table_name: Either old name (e.g., 'runs') or new name (e.g., 'wf_runs')

        Returns:
            The actual table name to use (new naming convention)
        """
        # If it's an old name, convert to new name
        if table_name in self.TABLE_ALIASES:
            return self.TABLE_ALIASES[table_name]
        # If it's already a new name or unknown, return as-is
        return table_name

    def insert(self, table_name: str, data: Dict[str, Any]) -> bool:
        """
        Insert a single row into a table.

        Args:
            table_name: Name of the table (supports both old and new naming)
            data: Dictionary of column names to values

        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Not connected to PostgreSQL")
            return False

        # Resolve table name (supports old naming for backward compatibility)
        table_name = self._resolve_table_name(table_name)

        # Ensure timestamp is set
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc)

        # Convert lists/dicts to JSON for JSONB columns
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                try:
                    # Use default=str to handle datetime, ObjectId, etc.
                    processed_data[key] = json.dumps(value, default=str)
                except (TypeError, ValueError) as e:
                    logger.warning(f"JSON serialization failed for {key}: {e}, using empty object")
                    processed_data[key] = "{}"
            elif value is None:
                processed_data[key] = value
            else:
                processed_data[key] = value

        columns = list(processed_data.keys())
        values = list(processed_data.values())

        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                    values
                )
            return True
        except Exception as e:
            logger.error(f"Failed to insert into {table_name}: {e}")
            return False

    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> int:
        """
        Insert multiple rows into a table.

        Args:
            table_name: Name of the table (supports both old and new naming)
            data_list: List of dictionaries

        Returns:
            Number of rows inserted
        """
        if not self.is_connected() or not data_list:
            return 0

        # Resolve table name (supports old naming for backward compatibility)
        table_name = self._resolve_table_name(table_name)

        # Process all rows
        processed_list = []
        for data in data_list:
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc)

            processed = {}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    try:
                        processed[key] = json.dumps(value, default=str)
                    except (TypeError, ValueError):
                        processed[key] = "{}"
                elif value is None:
                    processed[key] = value
                else:
                    processed[key] = value
            processed_list.append(processed)

        # Get columns from first row
        columns = list(processed_list[0].keys())
        columns_str = ", ".join(columns)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                values_list = [tuple(row.get(col) for col in columns) for row in processed_list]
                execute_values(
                    cursor,
                    f"INSERT INTO {table_name} ({columns_str}) VALUES %s",
                    values_list
                )
            return len(data_list)
        except Exception as e:
            logger.error(f"Failed to batch insert into {table_name}: {e}")
            return 0

    def query(
        self,
        table_name: str,
        conditions: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query rows from a table.

        Args:
            table_name: Name of the table (supports both old and new naming)
            conditions: WHERE conditions as dict
            order_by: ORDER BY clause
            limit: Maximum rows to return
            offset: Rows to skip

        Returns:
            List of rows as dictionaries
        """
        if not self.is_connected():
            return []

        # Resolve table name (supports old naming for backward compatibility)
        table_name = self._resolve_table_name(table_name)

        query_parts = [f"SELECT * FROM {table_name}"]
        params = []

        if conditions:
            where_clauses = []
            for col, value in conditions.items():
                where_clauses.append(f"{col} = %s")
                params.append(value)
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        if order_by:
            query_parts.append(f"ORDER BY {order_by}")

        if limit:
            query_parts.append(f"LIMIT {limit}")

        if offset:
            query_parts.append(f"OFFSET {offset}")

        query = " ".join(query_parts)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Query failed on {table_name}: {e}")
            return []

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a run by run_id."""
        results = self.query("runs", {"run_id": run_id}, limit=1)
        return results[0] if results else None

    def get_runs(
        self,
        limit: int = 50,
        company_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent runs."""
        conditions = {}
        if company_name:
            conditions["company_name"] = company_name
        if status:
            conditions["status"] = status

        return self.query(
            "runs",
            conditions if conditions else None,
            order_by="timestamp DESC",
            limit=limit
        )

    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive details for a run."""
        result = {"run_id": run_id, "found": False}

        # Get main run
        run = self.get_run(run_id)
        if run:
            result["found"] = True
            result["summary"] = run

        # Get related data (using new table names with prefixes)
        related_tables = {
            "llm_calls": "wf_llm_calls",
            "tool_calls": "wf_tool_calls",
            "langgraph_events": "lg_events",
            "assessment": "wf_assessments",
            "evaluation": "eval_results",
            "tool_selection": "eval_tool_selection",
            "agent_metrics": "eval_agent_metrics",
            "coalition": "eval_coalition",
        }

        for key, table in related_tables.items():
            data = self.query(table, {"run_id": run_id}, order_by="timestamp ASC")
            if data:
                if key in ["assessment", "evaluation", "tool_selection", "coalition"]:
                    result[key] = data[0] if len(data) == 1 else data
                else:
                    result[key] = data

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        stats = {}

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Total runs (using new table name wf_runs)
                cursor.execute("SELECT COUNT(*) as count FROM wf_runs")
                stats["total_runs"] = cursor.fetchone()["count"]

                # Runs by status
                cursor.execute("""
                    SELECT status, COUNT(*) as count
                    FROM wf_runs
                    GROUP BY status
                """)
                stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

                # Runs by risk level
                cursor.execute("""
                    SELECT risk_level, COUNT(*) as count
                    FROM wf_runs
                    WHERE risk_level IS NOT NULL
                    GROUP BY risk_level
                """)
                stats["by_risk_level"] = {row["risk_level"]: row["count"] for row in cursor.fetchall()}

                # Average scores
                cursor.execute("""
                    SELECT
                        AVG(overall_score) as avg_overall_score,
                        AVG(credit_score) as avg_credit_score,
                        AVG(confidence) as avg_confidence,
                        AVG(duration_ms) as avg_duration_ms
                    FROM wf_runs
                """)
                row = cursor.fetchone()
                stats["averages"] = {
                    "overall_score": float(row["avg_overall_score"]) if row["avg_overall_score"] else 0,
                    "credit_score": float(row["avg_credit_score"]) if row["avg_credit_score"] else 0,
                    "confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0,
                    "duration_ms": float(row["avg_duration_ms"]) if row["avg_duration_ms"] else 0,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")

        return stats

    def drop_old_partitions(self, months_to_keep: int = 3) -> List[str]:
        """
        Drop partitions older than specified months.

        Args:
            months_to_keep: Number of months of data to keep

        Returns:
            List of dropped partition names
        """
        dropped = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months_to_keep)
        cutoff_str = cutoff.strftime('%Y_%m')

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Find all partitions
                cursor.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename ~ '_[0-9]{4}_[0-9]{2}$'
                """)

                for row in cursor.fetchall():
                    partition_name = row["tablename"]
                    # Extract date part
                    date_part = partition_name.split('_')[-2] + '_' + partition_name.split('_')[-1]

                    if date_part < cutoff_str:
                        cursor.execute(f"DROP TABLE IF EXISTS {partition_name}")
                        dropped.append(partition_name)
                        logger.info(f"Dropped old partition: {partition_name}")

        except Exception as e:
            logger.error(f"Failed to drop old partitions: {e}")

        return dropped


# Global instance
_postgres_storage: Optional[PostgresStorage] = None


def get_postgres_storage() -> PostgresStorage:
    """Get or create the global PostgresStorage instance."""
    global _postgres_storage
    if _postgres_storage is None:
        _postgres_storage = PostgresStorage()
        _postgres_storage.connect()
    return _postgres_storage


def init_postgres() -> bool:
    """Initialize PostgreSQL storage and schema."""
    storage = get_postgres_storage()
    if storage.is_connected():
        return storage.initialize_schema()
    return False
