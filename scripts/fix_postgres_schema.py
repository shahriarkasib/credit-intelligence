#!/usr/bin/env python3
"""Fix PostgreSQL schema to match Google Sheets (source of truth)."""

import os
import sys
sys.path.insert(0, "/Users/shariarsourav/Desktop/credit_intelligence/src")

from dotenv import load_dotenv
load_dotenv("/Users/shariarsourav/Desktop/credit_intelligence/.env")

import psycopg2
from datetime import datetime, timezone, timedelta

# Get database URL
db_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Table schemas matching Google Sheets exactly
TABLE_SCHEMAS = {
    "wf_runs": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        model VARCHAR(100),
        temperature DECIMAL(3,2),
        status VARCHAR(50),
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        risk_level VARCHAR(50),
        credit_score INTEGER,
        confidence DECIMAL(5,4),
        total_time_ms DECIMAL(15,3),
        total_steps INTEGER,
        total_llm_calls INTEGER,
        tools_used JSONB,
        evaluation_score DECIMAL(5,4),
        workflow_correct BOOLEAN,
        output_correct BOOLEAN,
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "wf_tool_calls": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        tool_name VARCHAR(100),
        tool_input JSONB,
        tool_output JSONB,
        parent_node VARCHAR(100),
        workflow_phase VARCHAR(100),
        call_depth INTEGER,
        parent_tool_id VARCHAR(100),
        execution_time_ms DECIMAL(15,3),
        status VARCHAR(50),
        error TEXT,
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "wf_llm_calls": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        call_type VARCHAR(100),
        model VARCHAR(100),
        temperature DECIMAL(3,2),
        prompt TEXT,
        response TEXT,
        reasoning TEXT,
        context TEXT,
        current_task TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        input_cost DECIMAL(12,8),
        output_cost DECIMAL(12,8),
        total_cost DECIMAL(12,8),
        execution_time_ms DECIMAL(15,3),
        status VARCHAR(50),
        error TEXT,
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "wf_assessments": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        model VARCHAR(100),
        temperature DECIMAL(3,2),
        prompt TEXT,
        risk_level VARCHAR(50),
        credit_score INTEGER,
        confidence DECIMAL(5,4),
        reasoning TEXT,
        recommendations JSONB,
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "wf_plans": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        num_tasks INTEGER,
        plan_summary TEXT,
        full_plan JSONB,
        task_1 TEXT,
        task_2 TEXT,
        task_3 TEXT,
        task_4 TEXT,
        task_5 TEXT,
        task_6 TEXT,
        task_7 TEXT,
        task_8 TEXT,
        task_9 TEXT,
        task_10 TEXT,
        created_at TIMESTAMPTZ,
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "wf_data_sources": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        source_name VARCHAR(100),
        records_found INTEGER,
        data_summary TEXT,
        execution_time_ms DECIMAL(15,3),
        status VARCHAR(50),
        error TEXT,
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "wf_state_dumps": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        company_info_json JSONB,
        plan_json JSONB,
        plan_size_bytes INTEGER,
        plan_tasks_count INTEGER,
        api_data_summary TEXT,
        api_data_size_bytes INTEGER,
        api_sources_count INTEGER,
        search_data_summary TEXT,
        search_data_size_bytes INTEGER,
        risk_level VARCHAR(50),
        credit_score INTEGER,
        confidence DECIMAL(5,4),
        assessment_json JSONB,
        coalition_score DECIMAL(5,4),
        agent_metrics_score DECIMAL(5,4),
        evaluation_json JSONB,
        errors_json JSONB,
        error_count INTEGER,
        total_state_size_bytes INTEGER,
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "lg_events": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        event_type VARCHAR(100),
        event_name VARCHAR(255),
        model VARCHAR(100),
        temperature DECIMAL(3,2),
        tokens INTEGER,
        input_preview TEXT,
        output_preview TEXT,
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        error TEXT,
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "meta_prompts": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        prompt_id VARCHAR(100),
        prompt_name VARCHAR(255),
        category VARCHAR(100),
        system_prompt TEXT,
        user_prompt TEXT,
        variables_json JSONB,
        model VARCHAR(100),
        temperature DECIMAL(3,2),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_results": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        model VARCHAR(100),
        tool_selection_score DECIMAL(5,4),
        tool_reasoning TEXT,
        data_quality_score DECIMAL(5,4),
        data_reasoning TEXT,
        synthesis_score DECIMAL(5,4),
        synthesis_reasoning TEXT,
        overall_score DECIMAL(5,4),
        eval_status VARCHAR(50),
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_tool_selection": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        model VARCHAR(100),
        selected_tools JSONB,
        expected_tools JSONB,
        correct_tools JSONB,
        missing_tools JSONB,
        extra_tools JSONB,
        precision DECIMAL(5,4),
        recall DECIMAL(5,4),
        f1_score DECIMAL(5,4),
        reasoning TEXT,
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_consistency": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        model_name VARCHAR(100),
        evaluation_type VARCHAR(100),
        num_runs INTEGER,
        risk_level_consistency DECIMAL(5,4),
        score_consistency DECIMAL(5,4),
        score_std DECIMAL(10,4),
        overall_consistency DECIMAL(5,4),
        eval_status VARCHAR(50),
        risk_levels JSONB,
        credit_scores JSONB,
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_cross_model": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        models_compared JSONB,
        num_models INTEGER,
        risk_level_agreement DECIMAL(5,4),
        credit_score_mean DECIMAL(10,4),
        credit_score_std DECIMAL(10,4),
        credit_score_range DECIMAL(10,4),
        confidence_agreement DECIMAL(5,4),
        best_model VARCHAR(100),
        best_model_reasoning TEXT,
        cross_model_agreement DECIMAL(5,4),
        eval_status VARCHAR(50),
        llm_judge_analysis TEXT,
        model_recommendations JSONB,
        model_results JSONB,
        pairwise_comparisons JSONB,
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_llm_judge": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        model_used VARCHAR(100),
        temperature DECIMAL(3,2),
        accuracy_score DECIMAL(5,4),
        completeness_score DECIMAL(5,4),
        consistency_score DECIMAL(5,4),
        actionability_score DECIMAL(5,4),
        data_utilization_score DECIMAL(5,4),
        overall_score DECIMAL(5,4),
        eval_status VARCHAR(50),
        accuracy_reasoning TEXT,
        completeness_reasoning TEXT,
        consistency_reasoning TEXT,
        actionability_reasoning TEXT,
        data_utilization_reasoning TEXT,
        overall_reasoning TEXT,
        benchmark_alignment DECIMAL(5,4),
        benchmark_comparison JSONB,
        suggestions JSONB,
        tokens_used INTEGER,
        evaluation_cost DECIMAL(12,8),
        duration_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_agent_metrics": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        model VARCHAR(100),
        intent_correctness DECIMAL(5,4),
        plan_quality DECIMAL(5,4),
        tool_choice_correctness DECIMAL(5,4),
        tool_completeness DECIMAL(5,4),
        trajectory_match DECIMAL(5,4),
        final_answer_quality DECIMAL(5,4),
        step_count INTEGER,
        tool_calls INTEGER,
        latency_ms DECIMAL(15,3),
        overall_score DECIMAL(5,4),
        eval_status VARCHAR(50),
        intent_details JSONB,
        plan_details JSONB,
        tool_details JSONB,
        trajectory_details JSONB,
        answer_details JSONB,
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_coalition": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        node VARCHAR(100),
        node_type VARCHAR(50),
        agent_name VARCHAR(100),
        master_agent VARCHAR(100),
        step_number INTEGER,
        is_correct BOOLEAN,
        correctness_score DECIMAL(5,4),
        confidence DECIMAL(5,4),
        correctness_category VARCHAR(50),
        efficiency_score DECIMAL(5,4),
        quality_score DECIMAL(5,4),
        tool_score DECIMAL(5,4),
        consistency_score DECIMAL(5,4),
        agreement_score DECIMAL(5,4),
        num_evaluators INTEGER,
        votes_json JSONB,
        evaluation_time_ms DECIMAL(15,3),
        status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
    "eval_log_tests": """
        id BIGSERIAL,
        run_id VARCHAR(64) NOT NULL,
        company_name VARCHAR(255),
        runs INTEGER,
        langgraph_events INTEGER,
        llm_calls INTEGER,
        tool_calls INTEGER,
        assessments INTEGER,
        evaluations INTEGER,
        tool_selections INTEGER,
        consistency_scores INTEGER,
        data_sources INTEGER,
        plans INTEGER,
        prompts INTEGER,
        cross_model_eval INTEGER,
        llm_judge_results INTEGER,
        agent_metrics INTEGER,
        coalition INTEGER,
        total_sheets_logged INTEGER,
        verification_status VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (id, timestamp)
    """,
}

def create_partitions(cursor, table_name, months=12):
    """Create monthly partitions for a table."""
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    for i in range(months):
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
        except Exception as e:
            pass  # Partition might already exist

def main():
    print("Connecting to PostgreSQL...")

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL\n")

        for table_name, schema in TABLE_SCHEMAS.items():
            print(f"🔧 Recreating {table_name}...")

            # Drop existing table and partitions
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

            # Create new table with partitioning
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    {schema}
                ) PARTITION BY RANGE (timestamp);
            """)

            # Create partitions
            create_partitions(cursor, table_name)

            # Create index on run_id
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_run_id
                ON {table_name} (run_id);
            """)

            print(f"   ✅ Created {table_name} with partitions")

        print("\n" + "=" * 60)
        print("✅ All PostgreSQL tables recreated successfully!")
        print("=" * 60)

        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
