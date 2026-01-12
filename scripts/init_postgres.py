#!/usr/bin/env python3
"""
PostgreSQL Database Initialization Script.

This script initializes the PostgreSQL database schema for Credit Intelligence,
creating all tables with proper partitioning for data retention.

Usage:
    python scripts/init_postgres.py [--months 12] [--retention 3]

Options:
    --months: Number of months to create partitions for (default: 12)
    --retention: Months of data to keep when running retention (default: 3)
    --drop-old: Drop partitions older than retention period
    --status: Show database status and statistics
"""

import os
import sys
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_database(months_ahead: int = 12) -> bool:
    """
    Initialize the PostgreSQL database schema.

    Args:
        months_ahead: Number of months to create partitions for

    Returns:
        True if successful
    """
    from storage.postgres import PostgresStorage

    logger.info("Initializing PostgreSQL database...")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set!")
        logger.info("Set DATABASE_URL to your PostgreSQL connection string.")
        logger.info("For Heroku: heroku config:get DATABASE_URL")
        return False

    storage = PostgresStorage(database_url)

    if not storage.connect():
        logger.error("Failed to connect to PostgreSQL")
        return False

    logger.info(f"Connected to PostgreSQL. Creating schema with {months_ahead} months of partitions...")

    if storage.initialize_schema(months_ahead=months_ahead):
        logger.info("Database schema initialized successfully!")
        return True
    else:
        logger.error("Failed to initialize schema")
        return False


def apply_retention(months_to_keep: int = 3) -> bool:
    """
    Apply data retention by dropping old partitions.

    Args:
        months_to_keep: Number of months of data to keep

    Returns:
        True if successful
    """
    from storage.postgres import PostgresStorage

    logger.info(f"Applying data retention (keeping {months_to_keep} months)...")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set!")
        return False

    storage = PostgresStorage(database_url)

    if not storage.connect():
        logger.error("Failed to connect to PostgreSQL")
        return False

    dropped = storage.drop_old_partitions(months_to_keep)

    if dropped:
        logger.info(f"Dropped {len(dropped)} old partitions:")
        for p in dropped:
            logger.info(f"  - {p}")
    else:
        logger.info("No old partitions to drop")

    return True


def show_status() -> bool:
    """
    Show database status and statistics.

    Returns:
        True if successful
    """
    from storage.postgres import PostgresStorage

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set!")
        return False

    storage = PostgresStorage(database_url)

    if not storage.connect():
        logger.error("Failed to connect to PostgreSQL")
        return False

    logger.info("Database Status:")
    logger.info("-" * 50)

    # Get statistics
    stats = storage.get_statistics()
    if stats:
        logger.info(f"Total runs: {stats.get('total_runs', 0)}")

        by_status = stats.get('by_status', {})
        if by_status:
            logger.info("Runs by status:")
            for status, count in by_status.items():
                logger.info(f"  - {status}: {count}")

        by_risk = stats.get('by_risk_level', {})
        if by_risk:
            logger.info("Runs by risk level:")
            for risk, count in by_risk.items():
                logger.info(f"  - {risk}: {count}")

        avgs = stats.get('averages', {})
        if avgs:
            logger.info("Averages:")
            for key, value in avgs.items():
                logger.info(f"  - {key}: {value:.4f}")

    # List tables and partitions
    try:
        with storage.get_connection() as conn:
            cursor = conn.cursor()

            # Count partitions
            cursor.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename ~ '_[0-9]{4}_[0-9]{2}$'
                ORDER BY tablename
            """)
            partitions = cursor.fetchall()

            logger.info(f"\nTotal partitions: {len(partitions)}")

            # Group by base table
            partition_counts = {}
            for row in partitions:
                name = row['tablename']
                # Extract base table name (everything before _YYYY_MM)
                base = '_'.join(name.split('_')[:-2])
                partition_counts[base] = partition_counts.get(base, 0) + 1

            logger.info("Partitions by table:")
            for table, count in sorted(partition_counts.items()):
                logger.info(f"  - {table}: {count} partitions")

    except Exception as e:
        logger.error(f"Failed to get partition info: {e}")

    return True


def migrate_from_mongodb() -> bool:
    """
    Migrate existing data from MongoDB to PostgreSQL.

    This is a one-time migration helper.
    """
    from storage.postgres import PostgresStorage
    from storage.mongodb import MongoDBStorage

    logger.info("Starting MongoDB to PostgreSQL migration...")

    database_url = os.getenv("DATABASE_URL")
    mongodb_uri = os.getenv("MONGODB_URI")

    if not database_url:
        logger.error("DATABASE_URL not set")
        return False

    if not mongodb_uri:
        logger.error("MONGODB_URI not set")
        return False

    # Connect to both databases
    pg = PostgresStorage(database_url)
    mongo = MongoDBStorage()

    if not pg.connect():
        logger.error("Failed to connect to PostgreSQL")
        return False

    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return False

    # Initialize PostgreSQL schema
    pg.initialize_schema()

    # Migrate run_summaries -> runs
    logger.info("Migrating run_summaries...")
    summaries = mongo.get_run_summaries(limit=1000)
    migrated = 0
    for summary in summaries:
        # Remove MongoDB _id
        if '_id' in summary:
            del summary['_id']
        if pg.insert("runs", summary):
            migrated += 1
    logger.info(f"Migrated {migrated} runs")

    # Migrate llm_calls -> wf_llm_calls
    logger.info("Migrating llm_calls...")
    llm_calls = list(mongo.db.llm_calls.find().limit(5000))
    migrated = 0
    for call in llm_calls:
        if '_id' in call:
            del call['_id']
        if pg.insert("wf_llm_calls", call):
            migrated += 1
    logger.info(f"Migrated {migrated} LLM calls")

    # Migrate langgraph_events -> wf_langgraph_events
    logger.info("Migrating langgraph_events...")
    events = list(mongo.db.langgraph_events.find().limit(10000))
    migrated = 0
    for event in events:
        if '_id' in event:
            del event['_id']
        if pg.insert("wf_langgraph_events", event):
            migrated += 1
    logger.info(f"Migrated {migrated} LangGraph events")

    # Migrate assessments -> wf_assessments
    logger.info("Migrating assessments...")
    assessments = list(mongo.db.assessments.find().limit(1000))
    migrated = 0
    for assessment in assessments:
        if '_id' in assessment:
            del assessment['_id']
        if pg.insert("wf_assessments", assessment):
            migrated += 1
    logger.info(f"Migrated {migrated} assessments")

    logger.info("Migration complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL Database Management for Credit Intelligence"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months to create partitions for (default: 12)"
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=3,
        help="Months of data to keep when applying retention (default: 3)"
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help="Drop partitions older than retention period"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show database status and statistics"
    )
    parser.add_argument(
        "--migrate-mongodb",
        action="store_true",
        help="Migrate data from MongoDB to PostgreSQL"
    )

    args = parser.parse_args()

    if args.status:
        success = show_status()
    elif args.drop_old:
        success = apply_retention(args.retention)
    elif args.migrate_mongodb:
        success = migrate_from_mongodb()
    else:
        success = init_database(args.months)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
