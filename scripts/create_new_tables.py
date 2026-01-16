#!/usr/bin/env python3
"""
Create the 8 new PostgreSQL tables to match Google Sheets.

Run this script from an environment that can connect to Heroku PostgreSQL:
  - Heroku dyno
  - VPN with access to AWS RDS
  - After updating DATABASE_URL in .env

Tables to create:
  1. wf_step_logs
  2. lg_langsmith_traces
  3. wf_llm_calls_detailed
  4. wf_run_summaries
  5. eval_unified_metrics
  6. eval_model_consistency
  7. eval_deepeval_metrics
  8. eval_openevals_metrics
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def main():
    import psycopg2

    # Try to get database URL
    database_url = os.getenv("HEROKU_POSTGRES_URL") or os.getenv("DATABASE_URL")

    # Fallback to hardcoded Heroku URL
    if not database_url or "sqlite" in database_url.lower():
        database_url = "postgresql://u7bmabcqqqb5f3:p3ce4b360f25aab34a34cdaaac5a61e0a2e61118d1a4f58f84bb62ba3e91dd47b@cb09s9hprum9q5.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d7oc90f0q9oeec"

    print(f"Connecting to PostgreSQL...")

    try:
        conn = psycopg2.connect(database_url, sslmode='require', connect_timeout=15)
        cur = conn.cursor()
        print("Connected successfully!")

        # New tables to create
        new_tables = [
            "wf_step_logs",
            "lg_langsmith_traces",
            "wf_llm_calls_detailed",
            "wf_run_summaries",
            "eval_unified_metrics",
            "eval_model_consistency",
            "eval_deepeval_metrics",
            "eval_openevals_metrics",
        ]

        # Import storage module to get table definitions
        from storage.postgres import PostgresStorage

        storage = PostgresStorage(database_url)

        # Check which tables already exist
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        existing_tables = {row[0] for row in cur.fetchall()}
        print(f"\nExisting tables: {len(existing_tables)}")

        # Create each new table if it doesn't exist
        created = 0
        for table_name in new_tables:
            if table_name in existing_tables:
                print(f"  {table_name}: Already exists")
            else:
                try:
                    # Use the storage class to create the table
                    if storage.connect():
                        storage._create_table(table_name)
                        print(f"  {table_name}: Created")
                        created += 1
                except Exception as e:
                    print(f"  {table_name}: Error - {e}")

        # Verify final count
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        final_tables = [row[0] for row in cur.fetchall()]

        print(f"\n=== RESULT ===")
        print(f"Tables created: {created}")
        print(f"Total tables now: {len(final_tables)}")

        cur.close()
        conn.close()

        return created

    except Exception as e:
        print(f"Connection failed: {e}")
        print("\nTry running this script from:")
        print("  1. Heroku dyno: heroku run python scripts/create_new_tables.py")
        print("  2. Or set HEROKU_POSTGRES_URL in your .env file")
        return -1


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result >= 0 else 1)
