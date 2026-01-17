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
# These match the actual agent_name values used in graph.py
AGENT_DEFINITIONS = [
    # ===========================================
    # WORKFLOW NODE AGENTS (8 unique agents)
    # ===========================================
    {
        "agent_name": "llm_parser",
        "node_name": "parse_input",
        "master_agent": "supervisor",
        "agent_type": "llm",
        "purpose": "Parses and normalizes company input using LLM, extracts company name, ticker, jurisdiction"
    },
    {
        "agent_name": "supervisor",
        "node_name": "validate_company",
        "master_agent": "supervisor",
        "agent_type": "agent",
        "purpose": "Validates company data, determines company type (public/private), creates execution plan"
    },
    {
        "agent_name": "tool_supervisor",
        "node_name": "create_plan",
        "master_agent": "supervisor",
        "agent_type": "agent",
        "purpose": "Creates task plan with prioritized data sources using LLM-based tool selection"
    },
    {
        "agent_name": "api_agent",
        "node_name": "fetch_api_data",
        "master_agent": "supervisor",
        "agent_type": "tool",
        "purpose": "Fetches data from external APIs: SEC Edgar, Finnhub, CourtListener"
    },
    {
        "agent_name": "search_agent",
        "node_name": "search_web",
        "master_agent": "supervisor",
        "agent_type": "tool",
        "purpose": "Performs web search for company information (also handles search_web_enhanced node)"
    },
    {
        "agent_name": "llm_analyst",
        "node_name": "synthesize",
        "master_agent": "supervisor",
        "agent_type": "llm",
        "purpose": "Synthesizes all data into credit risk assessment (PUBLIC companies only)"
    },
    {
        "agent_name": "db_writer",
        "node_name": "save_to_database",
        "master_agent": "supervisor",
        "agent_type": "storage",
        "purpose": "Saves results to MongoDB, logs to PostgreSQL and Google Sheets"
    },
    {
        "agent_name": "workflow_evaluator",
        "node_name": "evaluate",
        "master_agent": "supervisor",
        "agent_type": "agent",
        "purpose": "Evaluates workflow quality using coalition evaluation (PUBLIC companies only)"
    },

    # ===========================================
    # ORCHESTRATOR AGENTS
    # ===========================================
    {
        "agent_name": "supervisor",
        "node_name": "workflow_orchestrator",
        "master_agent": "self",
        "agent_type": "orchestrator",
        "purpose": "Master orchestrator that creates execution plan and coordinates all workflow agents"
    },

    # ===========================================
    # ROUTER AGENTS (conditional edges)
    # ===========================================
    {
        "agent_name": "validation_router",
        "node_name": "should_continue_after_validation",
        "master_agent": "supervisor",
        "agent_type": "router",
        "purpose": "Routes to create_plan (continue) or END based on validation result"
    },
    {
        "agent_name": "api_data_router",
        "node_name": "route_after_api_data",
        "master_agent": "supervisor",
        "agent_type": "router",
        "purpose": "Routes to search_web (normal) or search_web_enhanced based on API data quality"
    },
    {
        "agent_name": "company_type_router",
        "node_name": "route_after_search_by_company_type",
        "master_agent": "supervisor",
        "agent_type": "router",
        "purpose": "Routes to synthesize (public) or save_to_database (private) based on company type"
    },
    {
        "agent_name": "save_router",
        "node_name": "route_after_save_by_company_type",
        "master_agent": "supervisor",
        "agent_type": "router",
        "purpose": "Routes to evaluate (public) or END (private) based on company type"
    },

    # ===========================================
    # EVALUATION SUB-AGENTS (called by workflow_evaluator)
    # ===========================================
    {
        "agent_name": "coalition_evaluator",
        "node_name": "eval_coalition",
        "master_agent": "workflow_evaluator",
        "agent_type": "evaluator",
        "purpose": "Combines multiple evaluators with weighted voting for robust correctness assessment"
    },
    {
        "agent_name": "llm_judge",
        "node_name": "eval_llm_judge",
        "master_agent": "workflow_evaluator",
        "agent_type": "evaluator",
        "purpose": "LLM-based judge scoring accuracy, completeness, consistency, actionability"
    },
    {
        "agent_name": "agent_metrics_evaluator",
        "node_name": "eval_agent_metrics",
        "master_agent": "workflow_evaluator",
        "agent_type": "evaluator",
        "purpose": "Evaluates agent performance: intent, plan, tools, trajectory, answer quality"
    },
    {
        "agent_name": "consistency_checker",
        "node_name": "eval_consistency",
        "master_agent": "workflow_evaluator",
        "agent_type": "evaluator",
        "purpose": "Checks consistency with historical runs for same company"
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
