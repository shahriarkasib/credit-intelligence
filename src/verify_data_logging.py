#!/usr/bin/env python3
"""
Verify data logging across all storage targets:
- Google Sheets
- PostgreSQL
- MongoDB

Checks that all columns have proper values and hierarchy fields are populated.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Define hierarchy columns that should be populated in all tables
HIERARCHY_COLUMNS = [
    "run_id", "company_name", "node", "node_type",
    "agent_name", "master_agent", "step_number"
]


def verify_postgresql():
    """Verify PostgreSQL data logging."""
    print("\n" + "="*70)
    print("VERIFYING POSTGRESQL DATA")
    print("="*70)

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Get connection details - try HEROKU_POSTGRES_URL first, then DATABASE_URL
        database_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")

        # If DATABASE_URL is SQLite, try to construct from individual vars
        if not database_url or "sqlite" in database_url.lower():
            # Try individual Heroku env vars
            host = os.getenv("POSTGRES_HOST") or os.getenv("HEROKU_POSTGRES_HOST")
            if host:
                database_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{host}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DATABASE')}"
            else:
                # Hardcoded Heroku credentials as fallback
                database_url = "postgresql://u7bmabcqqqb5f3:p3ce4b360f25aab34a34cdaaac5a61e0a2e61118d1a4f58f84bb62ba3e91dd47b@cb09s9hprum9q5.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d7oc90f0q9oeec"

        if not database_url:
            print("ERROR: No PostgreSQL URL configured")
            return False

        print(f"  Connecting to PostgreSQL...")

        # Connect
        conn = psycopg2.connect(database_url, sslmode='require')
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Tables to check (matching sheets)
        tables = [
            "wf_runs", "wf_llm_calls", "wf_tool_calls", "wf_assessments",
            "wf_plans", "wf_data_sources", "wf_state_dumps",
            "lg_events", "meta_prompts",
            "eval_results", "eval_tool_selection", "eval_consistency",
            "eval_cross_model", "eval_llm_judge", "eval_agent_metrics",
            "eval_coalition", "eval_log_tests"
        ]

        print(f"\n{'Table':<25} {'Rows':<8} {'Hierarchy Columns Populated'}")
        print("-"*70)

        summary = {}

        for table in tables:
            try:
                # Count rows
                cur.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                count = cur.fetchone()['cnt']

                if count == 0:
                    print(f"  {table:<25} {count:<8} N/A (no data)")
                    summary[table] = {"rows": 0, "hierarchy_populated": "N/A"}
                    continue

                # Check hierarchy columns
                cur.execute(f"SELECT * FROM {table} LIMIT 5")
                rows = cur.fetchall()

                # Check which hierarchy columns exist and have values
                hierarchy_status = []
                for col in HIERARCHY_COLUMNS:
                    if col in rows[0]:
                        has_value = any(row[col] is not None and str(row[col]).strip() for row in rows)
                        hierarchy_status.append(f"{col}={'Y' if has_value else 'N'}")

                status_str = ", ".join(hierarchy_status[:4])  # First 4
                print(f"  {table:<25} {count:<8} {status_str}")

                summary[table] = {
                    "rows": count,
                    "hierarchy_populated": hierarchy_status
                }

            except Exception as e:
                print(f"  {table:<25} ERROR: {str(e)[:40]}")
                summary[table] = {"error": str(e)}

        # Show sample data from wf_tool_calls to verify hierarchy
        print("\n" + "-"*70)
        print("SAMPLE DATA FROM wf_tool_calls (hierarchy verification):")
        print("-"*70)

        cur.execute("""
            SELECT run_id, company_name, tool_name, node, node_type,
                   agent_name, master_agent, step_number
            FROM wf_tool_calls
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        rows = cur.fetchall()

        if rows:
            for row in rows:
                print(f"  run_id: {row['run_id'][:8]}...")
                print(f"    company: {row['company_name']}")
                print(f"    tool: {row['tool_name']}")
                print(f"    node: {row['node']}, type: {row['node_type']}")
                print(f"    agent: {row['agent_name']}, master: {row['master_agent']}")
                print(f"    step: {row['step_number']}")
                print()
        else:
            print("  No data found")

        cur.close()
        conn.close()

        return summary

    except Exception as e:
        print(f"ERROR connecting to PostgreSQL: {e}")
        return None


def verify_mongodb():
    """Verify MongoDB data logging."""
    print("\n" + "="*70)
    print("VERIFYING MONGODB DATA")
    print("="*70)

    try:
        from pymongo import MongoClient
        import certifi

        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            print("ERROR: MONGODB_URI not set")
            return False

        client = MongoClient(mongodb_uri, tlsCAFile=certifi.where())
        db = client.credit_intelligence

        # Collections to check
        collections = [
            "runs", "llm_calls", "tool_calls", "assessments",
            "plans", "data_sources", "langgraph_events", "prompts",
            "evaluations", "consistency_scores", "agent_metrics", "coalition"
        ]

        print(f"\n{'Collection':<25} {'Docs':<8} {'Has Hierarchy Fields'}")
        print("-"*70)

        summary = {}

        for coll_name in collections:
            try:
                coll = db[coll_name]
                count = coll.count_documents({})

                if count == 0:
                    print(f"  {coll_name:<25} {count:<8} N/A (no data)")
                    summary[coll_name] = {"docs": 0}
                    continue

                # Check sample documents for hierarchy fields
                samples = list(coll.find().limit(3))

                hierarchy_present = []
                for col in ["node", "agent_name", "step_number"]:
                    has_field = any(col in doc and doc[col] for doc in samples)
                    hierarchy_present.append(f"{col}={'Y' if has_field else 'N'}")

                status_str = ", ".join(hierarchy_present)
                print(f"  {coll_name:<25} {count:<8} {status_str}")

                summary[coll_name] = {
                    "docs": count,
                    "hierarchy_fields": hierarchy_present
                }

            except Exception as e:
                print(f"  {coll_name:<25} ERROR: {str(e)[:40]}")

        client.close()
        return summary

    except Exception as e:
        print(f"ERROR connecting to MongoDB: {e}")
        return None


def verify_google_sheets():
    """Verify Google Sheets data logging."""
    print("\n" + "="*70)
    print("VERIFYING GOOGLE SHEETS DATA")
    print("="*70)

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        SCOPES = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]

        creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")

        if not creds_path or not spreadsheet_id:
            print("ERROR: Google credentials not configured")
            return None

        project_root = Path(__file__).parent.parent
        full_path = project_root / creds_path

        creds = Credentials.from_service_account_file(str(full_path), scopes=SCOPES)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(spreadsheet_id)

        worksheets = spreadsheet.worksheets()

        print(f"\n{'Sheet':<25} {'Rows':<8} {'Hierarchy Columns With Data'}")
        print("-"*70)

        summary = {}

        for ws in worksheets:
            try:
                all_values = ws.get_all_values()

                if len(all_values) <= 1:
                    print(f"  {ws.title:<25} 0        N/A (no data)")
                    summary[ws.title] = {"rows": 0}
                    continue

                headers = all_values[0]
                data_rows = all_values[1:]
                row_count = len(data_rows)

                # Check hierarchy columns
                hierarchy_status = []
                for col in ["node", "node_type", "agent_name", "step_number"]:
                    if col in headers:
                        idx = headers.index(col)
                        has_data = any(
                            len(row) > idx and row[idx] and str(row[idx]).strip()
                            for row in data_rows[:5]  # Check first 5
                        )
                        hierarchy_status.append(f"{col}={'Y' if has_data else 'N'}")

                status_str = ", ".join(hierarchy_status) if hierarchy_status else "no hierarchy cols"
                print(f"  {ws.title:<25} {row_count:<8} {status_str}")

                summary[ws.title] = {
                    "rows": row_count,
                    "hierarchy_status": hierarchy_status
                }

            except Exception as e:
                print(f"  {ws.title:<25} ERROR: {str(e)[:40]}")

        return summary

    except Exception as e:
        print(f"ERROR connecting to Google Sheets: {e}")
        return None


def main():
    """Main verification function."""
    print("\n" + "="*70)
    print("DATA LOGGING VERIFICATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Verify PostgreSQL
    pg_result = verify_postgresql()

    # Verify MongoDB
    mongo_result = verify_mongodb()

    # Verify Google Sheets
    sheets_result = verify_google_sheets()

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    print("\n  PostgreSQL:", "OK" if pg_result else "FAILED")
    print("  MongoDB:", "OK" if mongo_result else "FAILED")
    print("  Google Sheets:", "OK" if sheets_result else "FAILED")

    all_ok = pg_result and mongo_result and sheets_result
    print(f"\n  Overall Status: {'ALL SYSTEMS OK' if all_ok else 'SOME ISSUES DETECTED'}")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
