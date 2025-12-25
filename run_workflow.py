#!/usr/bin/env python3
"""
Credit Intelligence Workflow Runner
Runs the LangGraph-based credit assessment workflow.
"""

import os
import sys

# Set up the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(levelname)s: %(message)s",
)

def run_workflow(company_name: str, ticker: str = None):
    """Run the credit intelligence workflow."""

    print("\n" + "=" * 60)
    print("  CREDIT INTELLIGENCE WORKFLOW")
    print("=" * 60)
    print(f"\nCompany: {company_name}")
    if ticker:
        print(f"Ticker: {ticker}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Import here after path is set
    from agents.workflow import CreditIntelligenceWorkflow

    print("\nInitializing workflow...")
    workflow = CreditIntelligenceWorkflow()

    print("Running credit assessment...")
    print("  - Fetching SEC EDGAR data")
    print("  - Fetching Finnhub market data")
    print("  - Checking court records")
    print("  - Searching web for news")
    print("  - Synthesizing assessment")

    result = workflow.run(company_name)

    # Print report
    report = workflow.get_assessment_report(result)
    print("\n" + report)

    # Summary
    print("\n" + "-" * 60)
    print("WORKFLOW COMPLETE")
    print("-" * 60)
    print(f"Status: {result.get('status', 'unknown')}")

    errors = result.get('errors', [])
    if errors:
        print(f"Warnings: {len(errors)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_workflow.py <company_name> [ticker]")
        print("\nExamples:")
        print("  python run_workflow.py 'Apple Inc' AAPL")
        print("  python run_workflow.py 'Teva Pharmaceutical' TEVA")
        print("  python run_workflow.py 'Microsoft' MSFT")
        sys.exit(1)

    company = sys.argv[1]
    ticker = sys.argv[2] if len(sys.argv) > 2 else None

    run_workflow(company, ticker)
