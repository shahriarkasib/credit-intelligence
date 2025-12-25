# Credit Intelligence Agentic Workflow

An enterprise-grade demo environment that transforms manual credit intelligence processes into an autonomous agentic system for **B2B credit assessment** (company creditworthiness evaluation).

## Project Overview

This project demonstrates two key capabilities:

1. **Agentic Workflow Transformation**: Automate credit intelligence data collection from multiple public sources to assess company creditworthiness
2. **Consistency-as-Correctness Evaluation (Part 3)**: Validate the hypothesis that inter-model LLM consistency can serve as a reliable proxy for output correctness

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
                    ┌────────────────────────────────────────┐
                    │           SUPERVISOR AGENT             │
                    │  - Receives company details            │
                    │  - Orchestrates workflow               │
                    │  - Synthesizes credit assessment       │
                    └──────────────┬─────────────┬───────────┘
                                   │             │
                    ┌──────────────▼─────┐ ┌─────▼──────────────┐
                    │    SEARCH AGENT    │ │     API AGENT      │
                    │  - Web search      │ │  - SEC EDGAR       │
                    │  - News gathering  │ │  - OpenCorporates  │
                    │  - Sentiment       │ │  - Finnhub         │
                    └────────────────────┘ │  - CourtListener   │
                                           │  - OpenSanctions   │
                                           └────────────────────┘
```

---

## Project Structure

```
credit_intelligence/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── config/
│   ├── config.yaml              # Main configuration
│   └── models.yaml              # LLM model configurations
├── src/
│   ├── data_sources/            # Part 1: Data Source Connectors
│   │   ├── base.py              # Base connector class
│   │   ├── sec_edgar.py         # SEC EDGAR (US public companies)
│   │   ├── opencorporates.py    # Global company registry
│   │   ├── finnhub.py           # Stock/market data
│   │   ├── court_listener.py    # Court records
│   │   ├── opensanctions.py     # Sanctions/PEP checks
│   │   └── web_search.py        # DuckDuckGo web search
│   ├── agents/                  # Part 2: Agentic Workflow
│   │   ├── supervisor.py        # Main orchestrator
│   │   ├── search_agent.py      # Web search agent
│   │   ├── api_agent.py         # External API agent
│   │   └── workflow.py          # LangGraph workflow
│   └── evaluation/              # Part 3: Consistency Evaluation
│       ├── execution_wrapper.py # Multi-LLM router
│       ├── consistency_scorer.py# Consistency scoring
│       ├── correctness_scorer.py# Correctness validation
│       └── analyzer.py          # Correlation analysis
├── data/
│   ├── sample_companies/        # Demo company data
│   └── golden_test_sets/        # Ground truth test data
├── tests/
└── docs/
    └── data_source_mapping.md   # Detailed field mapping
```

---

## Part 1: Data Sources

### Free Data Sources Used

All data sources are **free** for demo purposes:

| Source | URL | Free Tier | Data Available |
|--------|-----|-----------|----------------|
| SEC EDGAR | sec.gov/developer | Unlimited | US public company financials |
| OpenCorporates | api.opencorporates.com | 500/month | Global company registry (140+ jurisdictions) |
| Finnhub | finnhub.io | 60 calls/min | Stock data, fundamentals |
| CourtListener | courtlistener.com | 5000/hour | Federal/state court records |
| OpenSanctions | opensanctions.org | Unlimited | Sanctions, PEPs, watchlists |
| DuckDuckGo | duckduckgo.com | N/A | Web search (no API key needed) |

### Data Fields (20 Key Fields)

| # | Field | Source | Credit Question |
|---|-------|--------|-----------------|
| 1 | Company Name | OpenCorporates | Identity/Fraud |
| 2 | Registration Number | OpenCorporates | Identity/Fraud |
| 3 | Company Status | OpenCorporates | Fraud Check |
| 4 | Incorporation Date | OpenCorporates | Ability to Pay |
| 5 | Directors/Officers | OpenCorporates | Fraud Check |
| 6 | Annual Revenue | SEC EDGAR | Ability to Pay |
| 7 | Net Income | SEC EDGAR | Ability to Pay |
| 8 | Total Assets | SEC EDGAR | Ability to Pay |
| 9 | Total Liabilities | SEC EDGAR | Ability to Pay |
| 10 | Operating Cash Flow | SEC EDGAR | Ability to Pay |
| 11 | Current Stock Price | Finnhub | Ability to Pay |
| 12 | Market Capitalization | Finnhub | Ability to Pay |
| 13 | Industry | Finnhub | Risk Assessment |
| 14 | Federal Court Cases | CourtListener | Willingness to Pay |
| 15 | Bankruptcy Filings | CourtListener | Willingness to Pay |
| 16 | Judgments/Liens | CourtListener | Willingness to Pay |
| 17 | Sanctions Status | OpenSanctions | Fraud Check |
| 18 | PEP Status | OpenSanctions | Fraud Check |
| 19 | Recent News | Web Search | Risk Assessment |
| 20 | News Sentiment | Web Search + LLM | Risk Assessment |

---

## Part 2: Agentic Workflow

### Agent Architecture

**Platform**: LangGraph (open source)

| Agent | Role | Tools |
|-------|------|-------|
| **Supervisor** | Receives company name, orchestrates workflow, synthesizes final credit report | Routes to sub-agents |
| **Search Agent** | Gathers public web information | DuckDuckGo search |
| **API Agent** | Fetches structured data from external APIs | SEC, OpenCorp, Finnhub, CourtListener, OpenSanctions |

### Workflow Steps

1. **Input**: Company name and optional jurisdiction
2. **Supervisor**: Analyzes input, creates task plan
3. **API Agent**: Fetches structured data (parallel API calls)
4. **Search Agent**: Gathers unstructured web data
5. **Supervisor**: Synthesizes all data into credit assessment
6. **Output**: Structured credit intelligence report with risk scores

---

## Part 3: Consistency-as-Correctness Evaluation

### Hypothesis

> **If multiple diverse LLMs produce consistent outputs for a given prompt, that output is likely correct.**

### Evaluation Framework

```
┌─────────────┐
│   Prompt    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│       EXECUTION WRAPPER              │
│  Routes prompt to 3+ LLMs            │
└──────┬───────────┬───────────┬───────┘
       │           │           │
       ▼           ▼           ▼
   ┌───────┐   ┌───────┐   ┌───────┐
   │GPT-4o │   │Claude │   │Ollama │
   │ mini  │   │Haiku  │   │Llama  │
   └───┬───┘   └───┬───┘   └───┬───┘
       │           │           │
       ▼           ▼           ▼
┌──────────────────────────────────────┐
│       CONSISTENCY SCORER             │
│  - Semantic similarity (Sentence-BERT)│
│  - Output: Consistent/Inconsistent   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│       CORRECTNESS SCORER             │
│  - Compare against golden truth      │
│  - Output: Correct/Incorrect         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│       CORRELATION ANALYZER           │
│  - Calculate consistency↔correctness │
│  - Target: ≥85% correlation          │
└──────────────────────────────────────┘
```

### Success Criteria

- **Correlation ≥ 85%** between inter-model consistency and correctness
- **OR Near-100% precision**: when system says "correct," it actually is correct

### Logged Metrics (per evaluation)

- Prompt and raw input
- Context (if applicable)
- Raw output from each LLM
- Golden answer
- Consistency score
- Correctness score
- Execution stats (tokens, latency, errors)

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

| Key | Required | How to Get |
|-----|----------|------------|
| `OPENAI_API_KEY` | Yes (for agents) | platform.openai.com |
| `FINNHUB_API_KEY` | Yes | finnhub.io (free tier) |
| `ANTHROPIC_API_KEY` | Optional | For Claude models in evaluation |
| `OPENCORPORATES_API_KEY` | Optional | Higher rate limits |

---

## Usage

### Run Credit Intelligence Workflow

```bash
# Analyze a public US company (uses SEC EDGAR)
python -m src.main analyze --company "Apple Inc"

# Analyze with specific jurisdiction
python -m src.main analyze --company "Microsoft Corporation" --jurisdiction US

# Analyze any company (uses OpenCorporates)
python -m src.main analyze --company "BMW AG" --jurisdiction DE
```

### Run Consistency Evaluation (Part 3)

```bash
# Run evaluation on golden test set
python -m src.evaluation.run --test-set data/golden_test_sets/finance_v1.json

# Generate correlation report
python -m src.evaluation.analyzer --results data/results/

# Run with specific models
python -m src.evaluation.run --models gpt-4o-mini,claude-3-haiku
```

---

## Demo Companies

For demonstration, we use these publicly-traded companies with available data:

| Company | Ticker | Jurisdiction | Data Available |
|---------|--------|--------------|----------------|
| Apple Inc | AAPL | US | Full (SEC + all sources) |
| Microsoft Corporation | MSFT | US | Full (SEC + all sources) |
| Tesla Inc | TSLA | US | Full (SEC + all sources) |
| Alphabet Inc | GOOGL | US | Full (SEC + all sources) |

---

## Key Deliverables

1. **Data Source Mapping** - `docs/data_source_mapping.md`
2. **Working Agentic Workflow** - `src/agents/`
3. **Consistency Evaluation Framework** - `src/evaluation/`
4. **Demo Results** - `data/results/`

---

## Technology Stack

| Component | Technology | Cost |
|-----------|------------|------|
| Agent Framework | LangGraph | Free (OSS) |
| LLM (Agents) | GPT-4o-mini | Pay per use |
| LLM (Local option) | Ollama + Llama | Free |
| Web Search | DuckDuckGo | Free |
| Database | SQLite | Free |
| Embeddings | Sentence-BERT | Free (OSS) |

---

## References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Karpathy's LLM Council](https://github.com/karpathy/llm-council) - Inspiration for consistency evaluation
- [SEC EDGAR API](https://www.sec.gov/developer)
- [OpenCorporates API](https://api.opencorporates.com/documentation)
- [Finnhub API](https://finnhub.io/docs/api)
- [CourtListener API](https://www.courtlistener.com/help/api/)
- [OpenSanctions API](https://www.opensanctions.org/docs/api/)

---

## License

MIT License
