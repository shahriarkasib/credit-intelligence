# Credit Intelligence Agentic Workflow

An enterprise-grade demo environment that transforms manual credit intelligence processes into an autonomous agentic system for **B2B credit assessment** (company creditworthiness evaluation).

## Project Overview

This project demonstrates three key capabilities:

1. **Tool-Based Agentic Workflow**: LLM-powered supervisor that autonomously selects and executes tools for credit data collection
2. **Comprehensive Evaluation Framework**: Evaluate tool selection accuracy, synthesis quality, and cross-run consistency
3. **Full Observability**: Step-by-step logging with metrics, token usage, and cost estimation

### Target Use Case

**Customer = Company (B2B)**. This system evaluates the creditworthiness of businesses, not individuals. It answers:

| Question | Category | What We Assess |
|----------|----------|----------------|
| What is the company's credit score? | Credit Assessment | Financial health indicators |
| What is their ability to pay? | Financial Capacity | Revenue, cash flow, assets |
| What is their willingness to pay? | Payment Behavior | Litigation history, judgments |
| Is the application legitimate? | Fraud Detection | Sanctions, company status |

---

## Architecture Overview

```
                         ┌──────────────────────────────────────────────┐
                         │           TOOL SUPERVISOR (LLM)              │
                         │  - Receives company name                     │
                         │  - Analyzes company type (public/private)    │
                         │  - Selects appropriate tools                 │
                         │  - Synthesizes credit assessment             │
                         └─────────────────────┬────────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
          ┌─────────▼─────────┐    ┌───────────▼───────────┐   ┌──────────▼──────────┐
          │   SEC EDGAR TOOL  │    │   MARKET DATA TOOL    │   │   WEB SEARCH TOOL   │
          │   (fetch_sec_data)│    │  (fetch_market_data)  │   │    (web_search)     │
          │                   │    │                       │   │                     │
          │ • 10-K filings    │    │ • Stock quotes        │   │ • Company info      │
          │ • Financial data  │    │ • Company profile     │   │ • News articles     │
          │ • CIK lookup      │    │ • Fundamentals        │   │ • Sentiment         │
          └───────────────────┘    └───────────────────────┘   └─────────────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               │
                         ┌─────────────────────▼────────────────────────┐
                         │           EVALUATION FRAMEWORK               │
                         │  - Tool selection accuracy (precision/recall)│
                         │  - Data quality assessment                   │
                         │  - Synthesis quality scoring                 │
                         │  - Multi-run consistency                     │
                         └──────────────────────────────────────────────┘
```

---

## Key Features

### Tool-Based Agent Architecture

The LLM Supervisor dynamically selects which tools to use based on company type:

| Company Type | Tools Selected | Reasoning |
|--------------|----------------|-----------|
| **Public US** (Apple, Microsoft) | SEC + Market Data | Official filings available |
| **Public Non-US** (Toyota, Samsung) | Market Data + Web Search | No SEC filings |
| **Private** (Local LLC) | Web Search + Legal Data | Limited public data |

### Evaluation Metrics

Every workflow run is evaluated on:

| Metric | Description | Target |
|--------|-------------|--------|
| **Tool Selection F1** | Did LLM choose correct tools? | ≥ 0.85 |
| **Data Completeness** | How much data was collected? | ≥ 0.80 |
| **Synthesis Quality** | Is assessment well-reasoned? | ≥ 0.75 |
| **Cross-Run Consistency** | Same results across runs? | ≥ 0.90 |

---

## Project Structure

```
credit_intelligence/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── langgraph.json               # LangGraph Studio configuration
├── docs/
│   ├── SYSTEM_OVERVIEW.md       # Detailed system documentation
│   ├── IMPLEMENTATION_PLAN.md   # Implementation details
│   └── MONGODB_SCHEMA.md        # Database schema documentation
├── src/
│   ├── tools/                   # Tool-based agents
│   │   ├── base_tool.py         # Base tool class with metrics
│   │   ├── tool_executor.py     # Tool management and execution
│   │   ├── sec_tool.py          # SEC EDGAR data tool
│   │   ├── finnhub_tool.py      # Market data tool
│   │   ├── court_tool.py        # Legal records tool
│   │   └── web_search_tool.py   # Web search tool
│   ├── agents/                  # Agent implementations
│   │   ├── tool_supervisor.py   # LLM-based tool selection
│   │   ├── supervisor.py        # Original supervisor
│   │   ├── search_agent.py      # Web search agent
│   │   ├── api_agent.py         # External API agent
│   │   ├── llm_analyst.py       # LLM analysis agent
│   │   └── workflow.py          # LangGraph workflow
│   ├── evaluation/              # Evaluation framework
│   │   ├── tool_selection_evaluator.py  # Tool choice evaluation
│   │   ├── workflow_evaluator.py        # Full workflow evaluation
│   │   ├── consistency_scorer.py        # Cross-run consistency
│   │   ├── correctness_scorer.py        # Output correctness
│   │   └── analyzer.py                  # Correlation analysis
│   ├── run_logging/             # Logging infrastructure
│   │   ├── run_logger.py        # MongoDB logging
│   │   └── metrics_collector.py # Metrics collection
│   ├── data_sources/            # Data source connectors
│   │   ├── sec_edgar.py         # SEC EDGAR API
│   │   ├── finnhub.py           # Finnhub stock data
│   │   ├── court_listener.py    # Court records
│   │   └── web_search.py        # DuckDuckGo search
│   ├── test_tool_workflow.py    # Test suite
│   └── run_evaluation.py        # Evaluation runner
└── data/
    └── evaluation_results/      # Saved evaluation results
```

---

## Installation

```bash
# Clone/navigate to project
cd credit_intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

| Key | Required | Cost | How to Get |
|-----|----------|------|------------|
| `GROQ_API_KEY` | **Yes** | Free | [console.groq.com](https://console.groq.com) |
| `FINNHUB_API_KEY` | Yes | Free tier | [finnhub.io](https://finnhub.io) |
| `COURTLISTENER_API_KEY` | Optional | Free | [courtlistener.com](https://www.courtlistener.com) |
| `MONGODB_URI` | Optional | Free tier | [mongodb.com/atlas](https://www.mongodb.com/atlas) |

---

## Usage

### Run Tests

```bash
cd src
source ../venv/bin/activate
python test_tool_workflow.py
```

Expected output:
```
============================================================
TEST SUMMARY
============================================================
  Tool Executor: PASS
  Tool Supervisor: PASS
  Full Assessment: PASS
  Evaluation Framework: PASS
  Logging Infrastructure: PASS

  Total: 5/5 tests passed
```

### Run Single Company Assessment

```bash
python run_evaluation.py --company "Apple Inc"
```

Output includes:
- **Risk Level**: low / medium / high / critical
- **Credit Score**: 0-100
- **Confidence**: 0.0-1.0
- **Tools Used**: Which data sources were queried
- **Evaluation Scores**: Tool selection accuracy, synthesis quality

### Run Consistency Evaluation (Multiple Runs)

```bash
# Run 3 assessments for consistency check
python run_evaluation.py --company "Microsoft Corporation" --runs 3
```

### Run Batch Evaluation

```bash
# Evaluate multiple companies
python run_evaluation.py --companies "Apple Inc,Microsoft,Tesla"
```

### Run with LangGraph Studio

```bash
# Start LangGraph Studio (visual workflow)
cd credit_intelligence
langgraph dev
```

Then open http://localhost:2024 in your browser.

---

## Sample Output

```json
{
  "company_name": "Microsoft Corporation",
  "run_id": "b164c57d-bd80-4ec7-85a7-b5138032833b",
  "total_execution_time_ms": 19469.90,
  "tool_selection": {
    "tools_selected": ["fetch_sec_data", "fetch_market_data", "fetch_legal_data"],
    "reasoning": "Microsoft is a US public company, SEC filings are available..."
  },
  "assessment": {
    "risk_level": "low",
    "credit_score": 92,
    "confidence": 0.8,
    "reasoning": "Strong financial position with consistent revenue growth...",
    "risk_factors": ["High market concentration in cloud services"],
    "positive_factors": ["Strong cash reserves", "Market leader", "Consistent growth"]
  },
  "evaluation": {
    "tool_selection_score": 1.0,
    "data_completeness": 0.85,
    "synthesis_consistency": 0.92
  }
}
```

---

## Data Sources

All data sources are **free** for demo purposes:

| Source | Tool Name | Free Tier | Data Available |
|--------|-----------|-----------|----------------|
| SEC EDGAR | `fetch_sec_data` | Unlimited | US public company financials |
| Finnhub | `fetch_market_data` | 60 calls/min | Stock data, fundamentals |
| CourtListener | `fetch_legal_data` | 5000/hour | Federal/state court records |
| DuckDuckGo | `web_search` | Unlimited | Web search, news |

---

## Evaluation Framework

### Tool Selection Evaluation

Measures if the LLM chose appropriate tools:

```
Expected tools for Apple Inc: [fetch_sec_data, fetch_market_data]
Selected tools:               [fetch_sec_data, fetch_market_data]

Precision: 1.00 (all selected tools were correct)
Recall:    1.00 (all expected tools were selected)
F1 Score:  1.00
```

### Consistency Evaluation

Run the same company multiple times to measure consistency:

```
Run 1: risk_level=low, credit_score=92
Run 2: risk_level=low, credit_score=90
Run 3: risk_level=low, credit_score=91

Risk Level Consistency: 1.0 (all agree)
Credit Score Range: 2 points
Overall Consistency: 0.95
```

---

## Technology Stack

| Component | Technology | Cost |
|-----------|------------|------|
| LLM | Groq (Llama 3.3 70B) | Free |
| Agent Framework | LangGraph | Free (OSS) |
| Web Search | DuckDuckGo | Free |
| Database | MongoDB Atlas | Free tier |
| Embeddings | Sentence-BERT | Free (OSS) |

### Groq Models Used

| Model | Use Case | Speed |
|-------|----------|-------|
| `llama-3.3-70b-versatile` | Tool selection, synthesis | Primary |
| `llama-3.1-8b-instant` | Fast assessments | Fast |
| `mixtral-8x7b-32768` | Cross-model validation | Balanced |

---

## MongoDB Schema

See `docs/MONGODB_SCHEMA.md` for full documentation. Key collections:

| Collection | Purpose |
|------------|---------|
| `runs` | Complete workflow run summaries |
| `steps` | Individual step logs with metrics |
| `tool_calls` | Tool execution logs |
| `assessments` | Final credit assessments |
| `evaluations` | Evaluation results |

---

## References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Groq API](https://console.groq.com) - Free LLM inference
- [SEC EDGAR API](https://www.sec.gov/developer)
- [Finnhub API](https://finnhub.io/docs/api)
- [CourtListener API](https://www.courtlistener.com/help/api/)

---

## License

MIT License
