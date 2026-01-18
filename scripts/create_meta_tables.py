#!/usr/bin/env python3
"""Create simplified meta_nodes, meta_agents, and meta_tools tables."""

import os
import sys
sys.path.insert(0, "/Users/shariarsourav/Desktop/credit_intelligence/src")

from dotenv import load_dotenv
load_dotenv("/Users/shariarsourav/Desktop/credit_intelligence/.env")

import psycopg2
from datetime import datetime, timezone

# =============================================================================
# META_AGENTS - Only actual agents from langgraph_events
# =============================================================================
AGENT_DEFINITIONS = [
    {"agent_name": "workflow", "purpose": "Root workflow orchestrator - represents overall LangGraph execution"},
    {"agent_name": "supervisor", "purpose": "Master orchestrator - validates company, creates execution plan"},
    {"agent_name": "llm_parser", "purpose": "Parses and normalizes company input using LLM"},
    {"agent_name": "tool_supervisor", "purpose": "Creates task plan with prioritized data sources"},
    {"agent_name": "api_agent", "purpose": "Fetches data from external APIs (SEC, Finnhub, CourtListener)"},
    {"agent_name": "search_agent", "purpose": "Performs web search for company information"},
    {"agent_name": "llm_analyst", "purpose": "Synthesizes all data into credit risk assessment"},
    {"agent_name": "db_writer", "purpose": "Saves results to MongoDB and logs to PostgreSQL/Sheets"},
    {"agent_name": "workflow_evaluator", "purpose": "Evaluates workflow quality using coalition evaluation"},
]

# =============================================================================
# META_NODES - Only actual nodes from langgraph_events
# =============================================================================
NODE_DEFINITIONS = [
    {"node_name": "parse_input", "purpose": "Parses and normalizes company input"},
    {"node_name": "validate_company", "purpose": "Validates company data and determines type"},
    {"node_name": "create_execution_plan", "purpose": "Creates execution plan based on company type"},
    {"node_name": "should_continue_after_validation", "purpose": "Routes based on validation result"},
    {"node_name": "create_plan", "purpose": "Creates task plan with tools to call"},
    {"node_name": "fetch_api_data", "purpose": "Orchestrates API data fetching"},
    {"node_name": "fetch_sec_edgar", "purpose": "Fetches SEC Edgar filings"},
    {"node_name": "fetch_finnhub", "purpose": "Fetches Finnhub market data"},
    {"node_name": "fetch_court_listener", "purpose": "Fetches CourtListener legal data"},
    {"node_name": "search_web_enhanced", "purpose": "Enhanced web search for company info"},
    {"node_name": "web_search_enhanced", "purpose": "Web search tool execution"},
    {"node_name": "synthesize", "purpose": "Synthesizes data into credit assessment"},
    {"node_name": "save_to_database", "purpose": "Saves results to MongoDB"},
    {"node_name": "evaluate", "purpose": "Evaluates workflow quality"},
]

# =============================================================================
# META_TOOLS - Only actual tools from tool_calls
# =============================================================================
TOOL_DEFINITIONS = [
    {"tool_name": "fetch_sec_edgar", "purpose": "Fetches SEC filings and financial data"},
    {"tool_name": "fetch_finnhub", "purpose": "Fetches market data and company profile"},
    {"tool_name": "fetch_court_listener", "purpose": "Fetches legal cases and litigation"},
    {"tool_name": "web_search_enhanced", "purpose": "Enhanced web search for company info"},
]


def get_db_connection():
    """Get PostgreSQL connection."""
    db_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(db_url)


def create_postgres_tables():
    """Create all meta tables in PostgreSQL."""
    print("\nCreating meta tables in PostgreSQL...")

    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # meta_agents
    cursor.execute("DROP TABLE IF EXISTS meta_agents CASCADE")
    cursor.execute("""
        CREATE TABLE meta_agents (
            agent_name VARCHAR(100) PRIMARY KEY,
            purpose TEXT NOT NULL
        );
    """)
    for agent in AGENT_DEFINITIONS:
        cursor.execute("INSERT INTO meta_agents (agent_name, purpose) VALUES (%s, %s)",
                      (agent["agent_name"], agent["purpose"]))
    print(f"  meta_agents: {len(AGENT_DEFINITIONS)} rows")

    # meta_nodes
    cursor.execute("DROP TABLE IF EXISTS meta_nodes CASCADE")
    cursor.execute("""
        CREATE TABLE meta_nodes (
            node_name VARCHAR(100) PRIMARY KEY,
            purpose TEXT NOT NULL
        );
    """)
    for node in NODE_DEFINITIONS:
        cursor.execute("INSERT INTO meta_nodes (node_name, purpose) VALUES (%s, %s)",
                      (node["node_name"], node["purpose"]))
    print(f"  meta_nodes: {len(NODE_DEFINITIONS)} rows")

    # meta_tools
    cursor.execute("DROP TABLE IF EXISTS meta_tools CASCADE")
    cursor.execute("""
        CREATE TABLE meta_tools (
            tool_name VARCHAR(100) PRIMARY KEY,
            purpose TEXT NOT NULL
        );
    """)
    for tool in TOOL_DEFINITIONS:
        cursor.execute("INSERT INTO meta_tools (tool_name, purpose) VALUES (%s, %s)",
                      (tool["tool_name"], tool["purpose"]))
    print(f"  meta_tools: {len(TOOL_DEFINITIONS)} rows")

    conn.close()
    print("PostgreSQL done")


def create_sheets_tables():
    """Create all meta sheets in Google Sheets."""
    print("\nCreating meta sheets in Google Sheets...")

    from run_logging.sheets_logger import SheetsLogger
    sheets_logger = SheetsLogger()

    if not sheets_logger.spreadsheet:
        print("  Could not connect to Google Sheets")
        return

    # Helper to create sheet
    def create_sheet(name, headers, rows):
        try:
            existing = sheets_logger.spreadsheet.worksheet(name)
            sheets_logger.spreadsheet.del_worksheet(existing)
        except:
            pass

        sheet = sheets_logger.spreadsheet.add_worksheet(title=name, rows=100, cols=len(headers))
        sheet.update("A1", [headers])
        sheet.format(f"A1:{chr(64+len(headers))}1", {"textFormat": {"bold": True}})
        if rows:
            sheet.update(f"A2:{chr(64+len(headers))}{len(rows)+1}", rows)
        print(f"  {name}: {len(rows)} rows")

    # Create sheets
    create_sheet("meta_agents", ["agent_name", "purpose"],
                 [[a["agent_name"], a["purpose"]] for a in AGENT_DEFINITIONS])

    create_sheet("meta_nodes", ["node_name", "purpose"],
                 [[n["node_name"], n["purpose"]] for n in NODE_DEFINITIONS])

    create_sheet("meta_tools", ["tool_name", "purpose"],
                 [[t["tool_name"], t["purpose"]] for t in TOOL_DEFINITIONS])

    print("Google Sheets done")


def main():
    print("=" * 50)
    print("Creating Simplified Meta Tables")
    print("=" * 50)

    try:
        create_postgres_tables()
    except Exception as e:
        print(f"PostgreSQL Error: {e}")

    try:
        create_sheets_tables()
    except Exception as e:
        print(f"Google Sheets Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
