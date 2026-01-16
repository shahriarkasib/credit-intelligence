#!/usr/bin/env python3
"""Create meta_agents table in PostgreSQL and Google Sheets with static agent definitions."""

import os
import sys
sys.path.insert(0, "/Users/shariarsourav/Desktop/credit_intelligence/src")

from dotenv import load_dotenv
load_dotenv("/Users/shariarsourav/Desktop/credit_intelligence/.env")

import psycopg2
from datetime import datetime, timezone, timedelta

# Static agent definitions
AGENT_DEFINITIONS = [
    # Workflow Nodes (graph.py)
    {
        "agent_name": "input_parser",
        "node_name": "parse_input",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Parses and normalizes company input, extracts company name and metadata from user request"
    },
    {
        "agent_name": "company_validator",
        "node_name": "validate_company",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Validates company data and checks if company exists, routes to END if validation fails"
    },
    {
        "agent_name": "plan_creator",
        "node_name": "create_plan",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Creates execution plan with prioritized tasks for credit analysis workflow"
    },
    {
        "agent_name": "api_agent",
        "node_name": "fetch_api_data",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Fetches data from external APIs including SEC Edgar, Finnhub, and CourtListener"
    },
    {
        "agent_name": "search_agent",
        "node_name": "search_web",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Performs web search to gather additional company information and news"
    },
    {
        "agent_name": "search_agent_enhanced",
        "node_name": "search_web_enhanced",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Enhanced web search with more queries when API data is limited (conditional route)"
    },
    {
        "agent_name": "llm_analyst",
        "node_name": "synthesize",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Synthesizes all collected data into comprehensive credit risk assessment with score and recommendations"
    },
    {
        "agent_name": "database_agent",
        "node_name": "save_to_database",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Saves assessment results to MongoDB and logs workflow data to PostgreSQL and Google Sheets"
    },
    {
        "agent_name": "evaluator",
        "node_name": "evaluate_assessment",
        "master_agent": "supervisor",
        "agent_type": "node",
        "purpose": "Evaluates assessment quality using multiple evaluation strategies (coalition, LLM judge, agent metrics)"
    },

    # Core Agent Classes
    {
        "agent_name": "supervisor",
        "node_name": "workflow_orchestrator",
        "master_agent": "self",
        "agent_type": "orchestrator",
        "purpose": "Master orchestrator that coordinates all workflow nodes and manages state transitions"
    },
    {
        "agent_name": "tool_supervisor",
        "node_name": "tool_orchestrator",
        "master_agent": "supervisor",
        "agent_type": "orchestrator",
        "purpose": "Manages tool selection and execution, coordinates specialized tool agents"
    },
    {
        "agent_name": "tool_selector",
        "node_name": "tool_selection",
        "master_agent": "tool_supervisor",
        "agent_type": "utility",
        "purpose": "Selects appropriate tools based on task requirements and available data sources"
    },

    # Evaluation Agents
    {
        "agent_name": "coalition_evaluator",
        "node_name": "eval_coalition",
        "master_agent": "evaluator",
        "agent_type": "evaluator",
        "purpose": "Runs coalition-based evaluation with multiple evaluators voting on assessment quality"
    },
    {
        "agent_name": "llm_judge",
        "node_name": "eval_llm_judge",
        "master_agent": "evaluator",
        "agent_type": "evaluator",
        "purpose": "LLM-based judge that scores assessment on accuracy, completeness, consistency, actionability"
    },
    {
        "agent_name": "agent_metrics_evaluator",
        "node_name": "eval_agent_metrics",
        "master_agent": "evaluator",
        "agent_type": "evaluator",
        "purpose": "Evaluates agent performance metrics including tool choice, trajectory, and answer quality"
    },
    {
        "agent_name": "consistency_checker",
        "node_name": "eval_consistency",
        "master_agent": "evaluator",
        "agent_type": "evaluator",
        "purpose": "Checks consistency of assessments across multiple runs of the same company"
    },
    {
        "agent_name": "cross_model_evaluator",
        "node_name": "eval_cross_model",
        "master_agent": "evaluator",
        "agent_type": "evaluator",
        "purpose": "Compares assessment results across different LLM models for cross-validation"
    },
]

# PostgreSQL schema for meta_agents
META_AGENTS_SCHEMA = """
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL UNIQUE,
    node_name VARCHAR(100) NOT NULL,
    master_agent VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    purpose TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
"""

# Google Sheets columns
SHEETS_COLUMNS = ["agent_name", "node_name", "master_agent", "agent_type", "purpose", "created_at"]


def create_postgres_table():
    """Create meta_agents table in PostgreSQL."""
    db_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL\n")

    # Drop existing table
    print("🔧 Recreating meta_agents table...")
    cursor.execute("DROP TABLE IF EXISTS meta_agents CASCADE")

    # Create new table (not partitioned since it's static)
    cursor.execute(f"""
        CREATE TABLE meta_agents (
            {META_AGENTS_SCHEMA}
        );
    """)

    # Create index on agent_name
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_meta_agents_agent_name
        ON meta_agents (agent_name);
    """)

    # Insert static data
    print("📝 Inserting agent definitions...")
    for agent in AGENT_DEFINITIONS:
        cursor.execute("""
            INSERT INTO meta_agents (agent_name, node_name, master_agent, agent_type, purpose)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (agent_name) DO UPDATE SET
                node_name = EXCLUDED.node_name,
                master_agent = EXCLUDED.master_agent,
                agent_type = EXCLUDED.agent_type,
                purpose = EXCLUDED.purpose,
                updated_at = NOW();
        """, (agent["agent_name"], agent["node_name"], agent["master_agent"],
              agent["agent_type"], agent["purpose"]))

    print(f"   ✅ Inserted {len(AGENT_DEFINITIONS)} agent definitions")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM meta_agents")
    count = cursor.fetchone()[0]
    print(f"   ✅ Verified: {count} agents in table\n")

    conn.close()
    return True


def create_google_sheet():
    """Create meta_agents sheet in Google Sheets using SheetsLogger."""
    print("Creating Google Sheets meta_agents sheet...")

    try:
        from run_logging.sheets_logger import SheetsLogger

        # Initialize sheets logger (connects in __init__ if credentials available)
        sheets_logger = SheetsLogger()

        if not sheets_logger.spreadsheet:
            print("❌ Could not connect to Google Sheets")
            return False

        print("✅ Connected to Google Sheets\n")

        sheet_name = "meta_agents"

        # Delete existing sheet if exists
        try:
            existing_sheet = sheets_logger.spreadsheet.worksheet(sheet_name)
            sheets_logger.spreadsheet.del_worksheet(existing_sheet)
            print(f"🔧 Deleted existing {sheet_name} sheet")
        except Exception:
            pass

        # Create new sheet
        print(f"📝 Creating {sheet_name} sheet...")
        sheet = sheets_logger.spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=len(SHEETS_COLUMNS))

        # Write headers
        sheet.update("A1", [SHEETS_COLUMNS])

        # Format header row
        sheet.format("A1:F1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
        })

        # Write data
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for agent in AGENT_DEFINITIONS:
            rows.append([
                agent["agent_name"],
                agent["node_name"],
                agent["master_agent"],
                agent["agent_type"],
                agent["purpose"],
                now
            ])

        if rows:
            sheet.update(f"A2:F{len(rows)+1}", rows)

        print(f"   ✅ Created sheet with {len(rows)} agent definitions\n")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Creating meta_agents table and sheet")
    print("=" * 60 + "\n")

    # Create PostgreSQL table
    try:
        create_postgres_table()
    except Exception as e:
        print(f"❌ PostgreSQL Error: {e}")
        import traceback
        traceback.print_exc()

    # Create Google Sheet
    try:
        create_google_sheet()
    except Exception as e:
        print(f"❌ Google Sheets Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("✅ meta_agents setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
