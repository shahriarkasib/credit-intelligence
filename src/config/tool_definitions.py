"""Tool Definitions - Static sheet with all tool names and descriptions.

This file serves as the single source of truth for all tools used in the
Credit Intelligence workflow.
"""

from typing import Dict, List, Any


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = {
    # -------------------------------------------------------------------------
    # SEC EDGAR Tool
    # -------------------------------------------------------------------------
    "fetch_sec_data": {
        "name": "fetch_sec_data",
        "display_name": "SEC EDGAR",
        "description": "Fetches US public company filings from SEC EDGAR database",
        "api_source": "SEC EDGAR (https://www.sec.gov/cgi-bin/browse-edgar)",
        "file_location": "src/tools/sec_tool.py",
        "class_name": "SECTool",

        "when_to_use": """
- Company is a US public company (has SEC filings)
- Need official financial statements (10-K, 10-Q)
- Need revenue, assets, liabilities, cash flow data
- Verifying company is legitimate and registered""",

        "parameters": {
            "company_identifier": {
                "type": "str",
                "required": True,
                "description": "Company name or CIK number",
            },
            "filing_types": {
                "type": "List[str]",
                "required": False,
                "default": ["10-K", "10-Q"],
                "description": "Types of SEC filings to fetch",
            },
        },

        "returns": {
            "company_name": "Official company name from SEC",
            "cik": "Central Index Key (unique SEC identifier)",
            "sic_description": "Industry classification",
            "filings": "List of recent filings with dates",
            "financials": {
                "revenue": "Annual revenue",
                "total_assets": "Total assets",
                "total_liabilities": "Total liabilities",
                "operating_cash_flow": "Cash flow from operations",
                "net_income": "Net income/loss",
                "debt_to_assets_ratio": "Debt ratio",
            },
        },

        "example_input": {
            "company_identifier": "Tesla",
        },

        "example_output": {
            "company_name": "TESLA, INC.",
            "cik": "0001318605",
            "sic_description": "Motor Vehicles & Passenger Car Bodies",
            "financials": {
                "revenue": {"value": 96773000000, "unit": "USD"},
                "total_assets": {"value": 106618000000, "unit": "USD"},
            },
        },

        "limitations": [
            "Only works for US public companies",
            "Data may be delayed (quarterly filings)",
            "No real-time stock prices",
        ],
    },

    # -------------------------------------------------------------------------
    # Finnhub Tool
    # -------------------------------------------------------------------------
    "fetch_market_data": {
        "name": "fetch_market_data",
        "display_name": "Finnhub Market Data",
        "description": "Fetches real-time stock prices and company profile from Finnhub",
        "api_source": "Finnhub (https://finnhub.io)",
        "file_location": "src/tools/finnhub_tool.py",
        "class_name": "FinnhubTool",

        "when_to_use": """
- Need real-time or recent stock price
- Need company profile (industry, market cap)
- Need financial metrics (P/E ratio, debt/equity)
- Company is publicly traded (any exchange)""",

        "parameters": {
            "ticker": {
                "type": "str",
                "required": True,
                "description": "Stock ticker symbol (e.g., AAPL, TSLA)",
            },
            "company_name": {
                "type": "str",
                "required": False,
                "description": "Company name for logging",
            },
        },

        "returns": {
            "profile": {
                "name": "Company name",
                "industry": "Industry sector",
                "market_cap": "Market capitalization",
                "exchange": "Stock exchange",
                "ipo_date": "IPO date",
            },
            "quote": {
                "current_price": "Current stock price",
                "open": "Opening price",
                "high": "Day high",
                "low": "Day low",
                "percent_change": "Daily % change",
            },
            "metrics": {
                "pe_ratio": "Price to earnings ratio",
                "debt_equity": "Debt to equity ratio",
                "current_ratio": "Current ratio (liquidity)",
                "52_week_high": "52-week high price",
                "52_week_low": "52-week low price",
            },
        },

        "example_input": {
            "ticker": "AAPL",
            "company_name": "Apple Inc",
        },

        "example_output": {
            "profile": {
                "name": "Apple Inc",
                "industry": "Technology",
                "market_cap": 2800000,
                "exchange": "NASDAQ",
            },
            "quote": {
                "current_price": 178.50,
                "percent_change": 1.25,
            },
        },

        "limitations": [
            "Requires FINNHUB_API_KEY environment variable",
            "Rate limited on free tier",
            "Some metrics may be null for smaller companies",
        ],
    },

    # -------------------------------------------------------------------------
    # CourtListener Tool
    # -------------------------------------------------------------------------
    "fetch_legal_data": {
        "name": "fetch_legal_data",
        "display_name": "CourtListener Legal Records",
        "description": "Searches US court records for lawsuits, bankruptcies, and legal history",
        "api_source": "CourtListener (https://www.courtlistener.com)",
        "file_location": "src/tools/court_tool.py",
        "class_name": "CourtListenerTool",

        "when_to_use": """
- Need to check for bankruptcy filings
- Need to find civil lawsuits
- Assessing legal risk / litigation history
- Part of fraud/legitimacy check""",

        "parameters": {
            "company_name": {
                "type": "str",
                "required": True,
                "description": "Company name to search for",
            },
            "case_types": {
                "type": "List[str]",
                "required": False,
                "default": ["bankruptcy", "civil"],
                "description": "Types of cases to search",
            },
        },

        "returns": {
            "total_dockets": "Total number of court cases found",
            "bankruptcy_cases": "List of bankruptcy filings",
            "civil_cases": "List of civil lawsuits",
            "risk_indicators": {
                "risk_level": "low/medium/high",
                "has_bankruptcy": "Boolean",
                "civil_case_count": "Number of civil cases",
            },
        },

        "example_input": {
            "company_name": "Enron Corporation",
        },

        "example_output": {
            "total_dockets": 15,
            "bankruptcy_cases": [{"case_name": "In re: Enron Corp", "date_filed": "2001-12-02"}],
            "risk_indicators": {
                "risk_level": "high",
                "has_bankruptcy": True,
                "civil_case_count": 14,
            },
        },

        "limitations": [
            "US courts only",
            "May not find all cases (name variations)",
            "Historical data may be incomplete",
        ],
    },

    # -------------------------------------------------------------------------
    # Web Search Tool
    # -------------------------------------------------------------------------
    "web_search": {
        "name": "web_search",
        "display_name": "Web Search",
        "description": "Searches the web for news, articles, and recent information about a company",
        "api_source": "Tavily Search API / Web Search",
        "file_location": "src/tools/web_search_tool.py",
        "class_name": "WebSearchTool",

        "when_to_use": """
- Need recent news and articles
- Company is private (no SEC/stock data)
- Looking for reputation/sentiment information
- Gathering general company information
- Always useful as supplementary data""",

        "parameters": {
            "company_name": {
                "type": "str",
                "required": True,
                "description": "Company name to search for",
            },
            "search_queries": {
                "type": "List[str]",
                "required": False,
                "description": "Custom search queries",
            },
        },

        "returns": {
            "results": "List of web search results with titles and snippets",
            "news": "List of recent news articles",
            "sentiment": {
                "sentiment": "positive/neutral/negative",
                "score": "Sentiment score -1 to 1",
            },
            "key_findings": "List of notable findings",
        },

        "example_input": {
            "company_name": "SpaceX",
        },

        "example_output": {
            "results": [
                {"title": "SpaceX launches Starship", "snippet": "..."},
            ],
            "news": [
                {"title": "SpaceX valuation reaches $180B", "date": "2024-01-15"},
            ],
            "sentiment": {
                "sentiment": "positive",
                "score": 0.7,
            },
        },

        "limitations": [
            "Results depend on search API availability",
            "May include irrelevant results",
            "Sentiment analysis is approximate",
        ],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tool_definition(tool_name: str) -> Dict[str, Any]:
    """Get full definition for a tool."""
    return TOOLS.get(tool_name, {})


def get_tool_description(tool_name: str) -> str:
    """Get short description for a tool."""
    tool = TOOLS.get(tool_name, {})
    return tool.get("description", "Unknown tool")


def get_tool_when_to_use(tool_name: str) -> str:
    """Get when_to_use guidance for a tool."""
    tool = TOOLS.get(tool_name, {})
    return tool.get("when_to_use", "")


def get_all_tool_names() -> List[str]:
    """Get list of all tool names."""
    return list(TOOLS.keys())


def get_tools_summary() -> str:
    """Get formatted summary of all tools for LLM prompts."""
    lines = []
    for i, (name, tool) in enumerate(TOOLS.items(), 1):
        lines.append(f"{i}. **{tool['display_name']}** (`{name}`)")
        lines.append(f"   Description: {tool['description']}")
        lines.append(f"   When to use:{tool['when_to_use']}")
        lines.append("")
    return "\n".join(lines)


def get_tools_table() -> str:
    """Get tools as markdown table."""
    lines = [
        "| # | Tool Name | Display Name | Description | API Source |",
        "|---|-----------|--------------|-------------|------------|",
    ]
    for i, (name, tool) in enumerate(TOOLS.items(), 1):
        lines.append(
            f"| {i} | `{name}` | {tool['display_name']} | "
            f"{tool['description'][:50]}... | {tool['api_source'].split('(')[0].strip()} |"
        )
    return "\n".join(lines)


def export_to_csv(filepath: str):
    """Export tool definitions to CSV."""
    import csv

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Tool Name",
            "Display Name",
            "Description",
            "API Source",
            "File Location",
            "When to Use",
            "Limitations",
        ])
        for name, tool in TOOLS.items():
            writer.writerow([
                name,
                tool["display_name"],
                tool["description"],
                tool["api_source"],
                tool["file_location"],
                tool["when_to_use"].strip(),
                "; ".join(tool["limitations"]),
            ])


def export_to_json(filepath: str):
    """Export tool definitions to JSON."""
    import json
    with open(filepath, "w") as f:
        json.dump(TOOLS, f, indent=2)


# =============================================================================
# PRINT TOOLS TABLE (for reference)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CREDIT INTELLIGENCE - TOOL DEFINITIONS")
    print("=" * 80 + "\n")

    for name, tool in TOOLS.items():
        print(f"📦 {tool['display_name']} ({name})")
        print(f"   {tool['description']}")
        print(f"   Source: {tool['api_source']}")
        print(f"   File: {tool['file_location']}")
        print()
