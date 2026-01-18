#!/usr/bin/env python3
"""Create meta_nodes, meta_agents, and meta_tools tables in PostgreSQL and Google Sheets."""

import os
import sys
sys.path.insert(0, "/Users/shariarsourav/Desktop/credit_intelligence/src")

from dotenv import load_dotenv
load_dotenv("/Users/shariarsourav/Desktop/credit_intelligence/.env")

import psycopg2
from datetime import datetime, timezone

# =============================================================================
# META_NODES - All workflow nodes
# =============================================================================
NODE_DEFINITIONS = [
    # Workflow nodes (in execution order)
    {
        "node_name": "parse_input",
        "node_type": "llm",
        "agent_name": "llm_parser",
        "step_number": 1,
        "purpose": "Parses and normalizes company input using LLM",
        "inputs": "raw company name/ticker",
        "outputs": "normalized company_info dict",
    },
    {
        "node_name": "validate_company",
        "node_type": "agent",
        "agent_name": "supervisor",
        "step_number": 2,
        "purpose": "Validates company data, determines public/private type",
        "inputs": "company_info",
        "outputs": "validation status, company type",
    },
    {
        "node_name": "create_execution_plan",
        "node_type": "agent",
        "agent_name": "supervisor",
        "step_number": 2,
        "purpose": "Creates execution plan based on company type (skip synthesize/evaluate for private)",
        "inputs": "company_info, company_type",
        "outputs": "execution_plan with planned/skipped agents",
    },
    {
        "node_name": "create_plan",
        "node_type": "agent",
        "agent_name": "tool_supervisor",
        "step_number": 3,
        "purpose": "Creates task plan with prioritized data sources",
        "inputs": "company_info, execution_plan",
        "outputs": "task_plan with tools to call",
    },
    {
        "node_name": "fetch_api_data",
        "node_type": "tool",
        "agent_name": "api_agent",
        "step_number": 4,
        "purpose": "Fetches data from external APIs (SEC, Finnhub, CourtListener)",
        "inputs": "company_info, ticker",
        "outputs": "api_data dict with all source data",
    },
    {
        "node_name": "search_web",
        "node_type": "tool",
        "agent_name": "search_agent",
        "step_number": 5,
        "purpose": "Performs standard web search for company information",
        "inputs": "company_name",
        "outputs": "search_data with web results",
    },
    {
        "node_name": "search_web_enhanced",
        "node_type": "tool",
        "agent_name": "search_agent",
        "step_number": 5,
        "purpose": "Enhanced web search when API data is limited (<2 sources)",
        "inputs": "company_name",
        "outputs": "search_data with extended web results",
    },
    {
        "node_name": "synthesize",
        "node_type": "llm",
        "agent_name": "llm_analyst",
        "step_number": 6,
        "purpose": "Synthesizes all data into credit risk assessment (PUBLIC only)",
        "inputs": "api_data, search_data",
        "outputs": "assessment with risk_level, credit_score, confidence",
    },
    {
        "node_name": "save_to_database",
        "node_type": "storage",
        "agent_name": "db_writer",
        "step_number": 7,
        "purpose": "Saves results to MongoDB and logs to PostgreSQL/Sheets",
        "inputs": "all workflow state",
        "outputs": "storage confirmation",
    },
    {
        "node_name": "evaluate",
        "node_type": "agent",
        "agent_name": "workflow_evaluator",
        "step_number": 8,
        "purpose": "Evaluates workflow quality using coalition evaluation (PUBLIC only)",
        "inputs": "assessment, api_data, search_data",
        "outputs": "evaluation scores",
    },
    # Router nodes (conditional edges)
    {
        "node_name": "should_continue_after_validation",
        "node_type": "router",
        "agent_name": "validation_router",
        "step_number": 2,
        "purpose": "Routes to create_plan or END based on validation result",
        "inputs": "validation_status",
        "outputs": "next node decision",
    },
    {
        "node_name": "route_after_api_data",
        "node_type": "router",
        "agent_name": "api_data_router",
        "step_number": 4,
        "purpose": "Routes to search_web or search_web_enhanced based on API data quality",
        "inputs": "api_data source count",
        "outputs": "search mode decision",
    },
    {
        "node_name": "route_after_search_by_company_type",
        "node_type": "router",
        "agent_name": "company_type_router",
        "step_number": 5,
        "purpose": "Routes to synthesize (public) or save_to_database (private)",
        "inputs": "company_type",
        "outputs": "next node decision",
    },
    {
        "node_name": "route_after_save_by_company_type",
        "node_type": "router",
        "agent_name": "save_router",
        "step_number": 7,
        "purpose": "Routes to evaluate (public) or END (private)",
        "inputs": "company_type",
        "outputs": "next node decision",
    },
]

# =============================================================================
# META_AGENTS - All agents in the system
# =============================================================================
AGENT_DEFINITIONS = [
    # Core workflow agents
    {
        "agent_name": "llm_parser",
        "agent_type": "llm",
        "master_agent": "supervisor",
        "nodes": ["parse_input"],
        "tools": [],
        "purpose": "Parses and normalizes company input using LLM",
    },
    {
        "agent_name": "supervisor",
        "agent_type": "orchestrator",
        "master_agent": "self",
        "nodes": ["validate_company", "create_execution_plan"],
        "tools": [],
        "purpose": "Master orchestrator - validates company, creates execution plan, coordinates workflow",
    },
    {
        "agent_name": "tool_supervisor",
        "agent_type": "agent",
        "master_agent": "supervisor",
        "nodes": ["create_plan"],
        "tools": [],
        "purpose": "Creates task plan with prioritized data sources using LLM-based tool selection",
    },
    {
        "agent_name": "api_agent",
        "agent_type": "tool",
        "master_agent": "supervisor",
        "nodes": ["fetch_api_data"],
        "tools": ["fetch_sec_edgar", "fetch_finnhub", "fetch_court_listener"],
        "purpose": "Fetches data from external APIs: SEC Edgar, Finnhub, CourtListener",
    },
    {
        "agent_name": "search_agent",
        "agent_type": "tool",
        "master_agent": "supervisor",
        "nodes": ["search_web", "search_web_enhanced"],
        "tools": ["web_search", "web_search_enhanced"],
        "purpose": "Performs web search for company information",
    },
    {
        "agent_name": "llm_analyst",
        "agent_type": "llm",
        "master_agent": "supervisor",
        "nodes": ["synthesize"],
        "tools": [],
        "purpose": "Synthesizes all data into credit risk assessment",
    },
    {
        "agent_name": "db_writer",
        "agent_type": "storage",
        "master_agent": "supervisor",
        "nodes": ["save_to_database"],
        "tools": ["mongodb_save", "postgres_log", "sheets_log"],
        "purpose": "Saves results to MongoDB, logs to PostgreSQL and Google Sheets",
    },
    {
        "agent_name": "workflow_evaluator",
        "agent_type": "agent",
        "master_agent": "supervisor",
        "nodes": ["evaluate"],
        "tools": [],
        "purpose": "Evaluates workflow quality using coalition evaluation",
    },
    # Router agents
    {
        "agent_name": "validation_router",
        "agent_type": "router",
        "master_agent": "supervisor",
        "nodes": ["should_continue_after_validation"],
        "tools": [],
        "purpose": "Routes based on validation result",
    },
    {
        "agent_name": "api_data_router",
        "agent_type": "router",
        "master_agent": "supervisor",
        "nodes": ["route_after_api_data"],
        "tools": [],
        "purpose": "Routes based on API data quality",
    },
    {
        "agent_name": "company_type_router",
        "agent_type": "router",
        "master_agent": "supervisor",
        "nodes": ["route_after_search_by_company_type"],
        "tools": [],
        "purpose": "Routes based on company type (public/private)",
    },
    {
        "agent_name": "save_router",
        "agent_type": "router",
        "master_agent": "supervisor",
        "nodes": ["route_after_save_by_company_type"],
        "tools": [],
        "purpose": "Routes to evaluate or END based on company type",
    },
    # Evaluation sub-agents
    {
        "agent_name": "coalition_evaluator",
        "agent_type": "evaluator",
        "master_agent": "workflow_evaluator",
        "nodes": ["eval_coalition"],
        "tools": [],
        "purpose": "Combines multiple evaluators with weighted voting",
    },
    {
        "agent_name": "llm_judge",
        "agent_type": "evaluator",
        "master_agent": "workflow_evaluator",
        "nodes": ["eval_llm_judge"],
        "tools": [],
        "purpose": "LLM-based judge for accuracy, completeness, consistency",
    },
    {
        "agent_name": "agent_metrics_evaluator",
        "agent_type": "evaluator",
        "master_agent": "workflow_evaluator",
        "nodes": ["eval_agent_metrics"],
        "tools": [],
        "purpose": "Evaluates agent performance metrics",
    },
    {
        "agent_name": "consistency_checker",
        "agent_type": "evaluator",
        "master_agent": "workflow_evaluator",
        "nodes": ["eval_consistency"],
        "tools": [],
        "purpose": "Checks consistency with historical runs",
    },
]

# =============================================================================
# META_TOOLS - All tools used in the workflow
# =============================================================================
TOOL_DEFINITIONS = [
    # API Tools (used by api_agent)
    {
        "tool_name": "fetch_sec_edgar",
        "tool_type": "api",
        "agent_name": "api_agent",
        "node_name": "fetch_api_data",
        "endpoint": "SEC EDGAR API",
        "purpose": "Fetches SEC filings, financial statements, and company data",
        "inputs": "ticker, company_name",
        "outputs": "filings, financials, company_info",
    },
    {
        "tool_name": "fetch_finnhub",
        "tool_type": "api",
        "agent_name": "api_agent",
        "node_name": "fetch_api_data",
        "endpoint": "Finnhub API",
        "purpose": "Fetches market data, company profile, and financials",
        "inputs": "ticker",
        "outputs": "profile, quote, financials, news",
    },
    {
        "tool_name": "fetch_court_listener",
        "tool_type": "api",
        "agent_name": "api_agent",
        "node_name": "fetch_api_data",
        "endpoint": "CourtListener API",
        "purpose": "Fetches legal cases, litigation history",
        "inputs": "company_name",
        "outputs": "cases, litigation_summary",
    },
    # Search Tools (used by search_agent)
    {
        "tool_name": "web_search",
        "tool_type": "search",
        "agent_name": "search_agent",
        "node_name": "search_web",
        "endpoint": "DuckDuckGo/Bing",
        "purpose": "Standard web search for company information",
        "inputs": "company_name",
        "outputs": "web_results, news_results",
    },
    {
        "tool_name": "web_search_enhanced",
        "tool_type": "search",
        "agent_name": "search_agent",
        "node_name": "search_web_enhanced",
        "endpoint": "DuckDuckGo/Bing (extended)",
        "purpose": "Enhanced search with additional queries when API data limited",
        "inputs": "company_name",
        "outputs": "web_results, news_results (extended)",
    },
    # Storage Tools (used by db_writer)
    {
        "tool_name": "mongodb_save",
        "tool_type": "storage",
        "agent_name": "db_writer",
        "node_name": "save_to_database",
        "endpoint": "MongoDB Atlas",
        "purpose": "Saves company data and assessment to MongoDB",
        "inputs": "company_data, assessment",
        "outputs": "document_id",
    },
    {
        "tool_name": "postgres_log",
        "tool_type": "storage",
        "agent_name": "db_writer",
        "node_name": "save_to_database",
        "endpoint": "Heroku PostgreSQL",
        "purpose": "Logs run data, steps, metrics to PostgreSQL",
        "inputs": "run_data, metrics",
        "outputs": "log_confirmation",
    },
    {
        "tool_name": "sheets_log",
        "tool_type": "storage",
        "agent_name": "db_writer",
        "node_name": "save_to_database",
        "endpoint": "Google Sheets API",
        "purpose": "Logs run data to Google Sheets for visibility",
        "inputs": "run_data, metrics",
        "outputs": "log_confirmation",
    },
]


def get_db_connection():
    """Get PostgreSQL connection."""
    db_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(db_url)


def create_meta_nodes_postgres():
    """Create meta_nodes table in PostgreSQL."""
    print("\n" + "=" * 60)
    print("Creating meta_nodes table in PostgreSQL")
    print("=" * 60)

    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS meta_nodes CASCADE")
    cursor.execute("""
        CREATE TABLE meta_nodes (
            id SERIAL PRIMARY KEY,
            node_name VARCHAR(100) NOT NULL UNIQUE,
            node_type VARCHAR(50) NOT NULL,
            agent_name VARCHAR(100) NOT NULL,
            step_number INTEGER NOT NULL,
            purpose TEXT NOT NULL,
            inputs TEXT,
            outputs TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cursor.execute("CREATE INDEX idx_meta_nodes_name ON meta_nodes (node_name);")

    # Insert data
    for node in NODE_DEFINITIONS:
        cursor.execute("""
            INSERT INTO meta_nodes (node_name, node_type, agent_name, step_number, purpose, inputs, outputs)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (node["node_name"], node["node_type"], node["agent_name"],
              node["step_number"], node["purpose"], node.get("inputs", ""), node.get("outputs", "")))

    print(f"   Inserted {len(NODE_DEFINITIONS)} node definitions")
    conn.close()


def create_meta_agents_postgres():
    """Create meta_agents table in PostgreSQL."""
    print("\n" + "=" * 60)
    print("Creating meta_agents table in PostgreSQL")
    print("=" * 60)

    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS meta_agents CASCADE")
    cursor.execute("""
        CREATE TABLE meta_agents (
            id SERIAL PRIMARY KEY,
            agent_name VARCHAR(100) NOT NULL UNIQUE,
            agent_type VARCHAR(50) NOT NULL,
            master_agent VARCHAR(100) NOT NULL,
            nodes TEXT NOT NULL,
            tools TEXT,
            purpose TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cursor.execute("CREATE INDEX idx_meta_agents_name ON meta_agents (agent_name);")

    # Insert data
    for agent in AGENT_DEFINITIONS:
        cursor.execute("""
            INSERT INTO meta_agents (agent_name, agent_type, master_agent, nodes, tools, purpose)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (agent["agent_name"], agent["agent_type"], agent["master_agent"],
              ",".join(agent["nodes"]), ",".join(agent.get("tools", [])), agent["purpose"]))

    print(f"   Inserted {len(AGENT_DEFINITIONS)} agent definitions")
    conn.close()


def create_meta_tools_postgres():
    """Create meta_tools table in PostgreSQL."""
    print("\n" + "=" * 60)
    print("Creating meta_tools table in PostgreSQL")
    print("=" * 60)

    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS meta_tools CASCADE")
    cursor.execute("""
        CREATE TABLE meta_tools (
            id SERIAL PRIMARY KEY,
            tool_name VARCHAR(100) NOT NULL UNIQUE,
            tool_type VARCHAR(50) NOT NULL,
            agent_name VARCHAR(100) NOT NULL,
            node_name VARCHAR(100) NOT NULL,
            endpoint VARCHAR(200),
            purpose TEXT NOT NULL,
            inputs TEXT,
            outputs TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cursor.execute("CREATE INDEX idx_meta_tools_name ON meta_tools (tool_name);")

    # Insert data
    for tool in TOOL_DEFINITIONS:
        cursor.execute("""
            INSERT INTO meta_tools (tool_name, tool_type, agent_name, node_name, endpoint, purpose, inputs, outputs)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (tool["tool_name"], tool["tool_type"], tool["agent_name"], tool["node_name"],
              tool.get("endpoint", ""), tool["purpose"], tool.get("inputs", ""), tool.get("outputs", "")))

    print(f"   Inserted {len(TOOL_DEFINITIONS)} tool definitions")
    conn.close()


def create_meta_sheets():
    """Create all meta sheets in Google Sheets."""
    print("\n" + "=" * 60)
    print("Creating meta sheets in Google Sheets")
    print("=" * 60)

    try:
        from run_logging.sheets_logger import SheetsLogger
        sheets_logger = SheetsLogger()

        if not sheets_logger.spreadsheet:
            print("   Could not connect to Google Sheets")
            return False

        now = datetime.now(timezone.utc).isoformat()

        # Create meta_nodes sheet
        _create_sheet(sheets_logger.spreadsheet, "meta_nodes",
                      ["node_name", "node_type", "agent_name", "step_number", "purpose", "inputs", "outputs", "created_at"],
                      [[n["node_name"], n["node_type"], n["agent_name"], n["step_number"],
                        n["purpose"], n.get("inputs", ""), n.get("outputs", ""), now] for n in NODE_DEFINITIONS])

        # Create meta_agents sheet
        _create_sheet(sheets_logger.spreadsheet, "meta_agents",
                      ["agent_name", "agent_type", "master_agent", "nodes", "tools", "purpose", "created_at"],
                      [[a["agent_name"], a["agent_type"], a["master_agent"],
                        ",".join(a["nodes"]), ",".join(a.get("tools", [])), a["purpose"], now] for a in AGENT_DEFINITIONS])

        # Create meta_tools sheet
        _create_sheet(sheets_logger.spreadsheet, "meta_tools",
                      ["tool_name", "tool_type", "agent_name", "node_name", "endpoint", "purpose", "inputs", "outputs", "created_at"],
                      [[t["tool_name"], t["tool_type"], t["agent_name"], t["node_name"],
                        t.get("endpoint", ""), t["purpose"], t.get("inputs", ""), t.get("outputs", ""), now] for t in TOOL_DEFINITIONS])

        return True

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def _create_sheet(spreadsheet, sheet_name, headers, rows):
    """Helper to create a single sheet."""
    # Delete existing
    try:
        existing = spreadsheet.worksheet(sheet_name)
        spreadsheet.del_worksheet(existing)
        print(f"   Deleted existing {sheet_name}")
    except Exception:
        pass

    # Create new
    sheet = spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=len(headers))
    sheet.update("A1", [headers])
    sheet.format(f"A1:{chr(64+len(headers))}1", {
        "textFormat": {"bold": True},
        "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
    })

    if rows:
        sheet.update(f"A2:{chr(64+len(headers))}{len(rows)+1}", rows)

    print(f"   Created {sheet_name} with {len(rows)} rows")


def main():
    print("\n" + "=" * 60)
    print("Creating Meta Tables: meta_nodes, meta_agents, meta_tools")
    print("=" * 60)

    # PostgreSQL
    try:
        create_meta_nodes_postgres()
        create_meta_agents_postgres()
        create_meta_tools_postgres()
        print("\n   PostgreSQL tables created successfully")
    except Exception as e:
        print(f"\n   PostgreSQL Error: {e}")
        import traceback
        traceback.print_exc()

    # Google Sheets
    try:
        create_meta_sheets()
        print("\n   Google Sheets created successfully")
    except Exception as e:
        print(f"\n   Google Sheets Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Meta tables setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
