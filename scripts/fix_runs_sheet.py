#!/usr/bin/env python3
"""Fix the runs sheet - recreate with correct 22 columns."""

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

    # Recreate the runs sheet with correct headers
    print("\n🔧 Recreating 'runs' sheet with correct 22 columns...")
    result = logger.recreate_sheet("runs")

    if result.get("success"):
        print(f"✅ Success! Created 'runs' sheet with {result['columns']} columns")
        print(f"Headers: {result['headers']}")
    else:
        print(f"❌ Failed: {result.get('error')}")

if __name__ == "__main__":
    main()
