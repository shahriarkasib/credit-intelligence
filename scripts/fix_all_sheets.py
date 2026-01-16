#!/usr/bin/env python3
"""Fix all sheets with header mismatches - recreate with correct columns."""

import sys
sys.path.insert(0, "/Users/shariarsourav/Desktop/credit_intelligence/src")

from run_logging.sheets_logger import get_sheets_logger

def main():
    logger = get_sheets_logger()

    if not logger.is_connected():
        print("❌ Not connected to Google Sheets")
        return

    print("Connected to Google Sheets")
    print(f"Spreadsheet: {logger.get_spreadsheet_url()}")
    print()

    # Sheets to fix (excluding 'runs' which was already fixed)
    sheets_to_fix = [
        "assessments",
        "evaluations",
        "tool_selections",
        "consistency_scores",
        "data_sources",
        "cross_model_eval",
        "llm_judge_results",
        "agent_metrics",
    ]

    results = {}

    for sheet_name in sheets_to_fix:
        print(f"🔧 Fixing '{sheet_name}'...")
        result = logger.recreate_sheet(sheet_name)

        if result.get("success"):
            print(f"   ✅ Created with {result['columns']} columns")
            print(f"   Headers: {result['headers']}")
            results[sheet_name] = "success"
        else:
            print(f"   ❌ Failed: {result.get('error')}")
            results[sheet_name] = f"failed: {result.get('error')}"
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for v in results.values() if v == "success")
    print(f"Fixed: {success_count}/{len(sheets_to_fix)} sheets")

    for sheet, status in results.items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {sheet}: {status}")

if __name__ == "__main__":
    main()
