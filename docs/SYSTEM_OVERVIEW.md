# Credit Intelligence System

> An AI-powered system that assesses company creditworthiness by gathering data from multiple sources. The final credit decision is made by an LLM (AI), not rules.

---

## How It Works

```
You enter a company name (e.g., "Apple")
                ↓
    System gathers data from 5 sources (in parallel)
                ↓
    All data is sent to LLM (LLaMA 3.3 70B)
                ↓
    LLM analyzes and makes credit decision
                ↓
    Output: Risk Level + Credit Score + Reasoning + Recommendations
```

---

## Data Sources & APIs

| Source | What We Get | API Link | Cost |
|--------|-------------|----------|------|
| **SEC EDGAR** | Financial reports (revenue, profit, debt) | https://www.sec.gov/edgar/sec-api-documentation | Free |
| **Finnhub** | Stock price, market value, company profile | https://finnhub.io/docs/api | Free (60 calls/min) |
| **CourtListener** | Bankruptcy & lawsuit history | https://www.courtlistener.com/api/rest-info/ | Free |
| **DuckDuckGo** | Recent news & public sentiment | https://duckduckgo.com | Free |
| **Groq (LLM)** | AI-powered credit decision | https://console.groq.com/docs | Free |

---

## The 4 Agents

### 1. Supervisor Agent
**Role:** The Manager

- Receives company name from user
- Decides what data to collect
- Assigns tasks to other agents
- Sends all data to LLM for final decision

### 2. API Agent
**Role:** The Data Collector

- Fetches financial statements from SEC EDGAR
- Gets stock/market data from Finnhub
- Searches CourtListener for legal issues
- Runs all API calls in parallel (fast!)

### 3. Search Agent
**Role:** The News Reader

- Searches web for recent company news
- Collects news articles
- Analyzes sentiment (positive/negative/neutral)

### 4. LLM Analyst Agent (Decision Maker)
**Role:** The Expert - **Makes the Final Credit Decision**

- Receives all collected data
- Analyzes patterns and relationships
- **Makes the final credit decision** (not rule-based)
- Provides reasoning and confidence score
- Generates actionable recommendations

**Model Used:** `llama-3.3-70b-versatile` via Groq (FREE)

---

## How the AI Makes Decisions

The LLM analyzes all data holistically - no fixed rules or weights.

| Factor | What the AI Looks At |
|--------|---------------------|
| **Financial Health** | Revenue, cash flow, debt levels, profitability trends |
| **Legal History** | Lawsuits, bankruptcies, regulatory issues |
| **Fraud/Legitimacy** | Sanctions, company status, age, registered officers |
| **Market Position** | Stock performance, market cap, industry context |
| **News Sentiment** | Recent news, public perception, media coverage |

The AI weighs factors based on context. For example:
- A startup won't be penalized for low revenue like an established company
- A tech company's high debt might be acceptable if growth is strong
- Recent negative news is weighted more heavily than old news

---

## Output Format

The system returns the LLM's analysis:

```json
{
  "company_name": "Apple",
  "risk_level": "low",
  "credit_score": 85,
  "confidence": 0.92,
  "reasoning": "Apple demonstrates strong financial health with...",
  "risk_factors": ["High market concentration in smartphones"],
  "positive_factors": ["Strong cash reserves", "Consistent revenue growth"],
  "recommendations": ["Standard credit terms apply", "May qualify for preferred terms"]
}
```

| Field | Example | Meaning |
|-------|---------|---------|
| **risk_level** | low / medium / high / critical | Overall risk category |
| **credit_score** | 85 | 0-100, higher = safer |
| **confidence** | 0.92 | How confident the AI is (0-1) |
| **reasoning** | "Apple demonstrates..." | AI's explanation |
| **recommendations** | ["Standard terms..."] | Suggested actions |

---

## Visualization & Monitoring

| Tool | Purpose | How to Access |
|------|---------|---------------|
| **LangGraph Studio** | Visual workflow debugging | `langgraph dev` then open browser |
| **LangSmith** | Execution traces & history | https://smith.langchain.com |

**To run LangGraph Studio:**
```bash
cd credit_intelligence
source venv/bin/activate
langgraph dev
```
Then open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

---

## Project Structure

```
credit_intelligence/
├── src/
│   ├── agents/
│   │   ├── graph.py          # LangGraph workflow (entry point)
│   │   ├── supervisor.py     # Orchestrates agents
│   │   ├── api_agent.py      # Fetches from APIs
│   │   ├── search_agent.py   # Web search & news
│   │   └── llm_analyst.py    # AI decision maker (Groq)
│   ├── data_sources/
│   │   ├── sec_edgar.py      # SEC financial data
│   │   ├── finnhub.py        # Stock market data
│   │   ├── court_listener.py # Legal/court data
│   │   └── web_search.py     # DuckDuckGo search
│   └── evaluation/           # Part 3: Evaluation scripts
├── langgraph.json            # LangGraph Studio config
└── .env                      # API keys
```

---

## Next Steps: Evaluation Plan

### Goal
Evaluate the quality and consistency of the AI's credit decisions.

### Approach: Consistency-as-Correctness

Since we don't have ground truth credit scores, we evaluate by checking if the AI is **consistent**.

---

### Evaluation Method 1: Same Model Consistency

**Question:** Does the same LLM give the same answer when asked multiple times?

| Run | Model | Company | Risk Level | Score |
|-----|-------|---------|------------|-------|
| 1 | LLaMA 70B | Apple | low | 85 |
| 2 | LLaMA 70B | Apple | low | 87 |
| 3 | LLaMA 70B | Apple | low | 84 |

**Metrics:**
- Score variance (should be < 5 points)
- Risk level consistency (should be 100% same)
- Reasoning similarity (key factors should match)

---

### Evaluation Method 2: Cross-Model Validation

**Question:** Do different LLMs agree on the same company?

| Model | Company | Risk Level | Score |
|-------|---------|------------|-------|
| LLaMA 3.3 70B | Apple | low | 85 |
| Mixtral 8x7B | Apple | low | 82 |
| LLaMA 3.1 8B | Apple | low | 80 |

**Available Models (all FREE via Groq):**
| Model | Size | Speed | Use For |
|-------|------|-------|---------|
| `llama-3.3-70b-versatile` | 70B | Slower | Primary (best quality) |
| `mixtral-8x7b-32768` | 8x7B | Medium | Cross-validation |
| `llama-3.1-8b-instant` | 8B | Fast | Cross-validation |

**Metrics:**
- Model agreement rate (% of companies where all models agree on risk level)
- Score deviation across models (should be < 10 points)

---

### Combined Evaluation Process

```
For each company (10-20 test companies):
    │
    ├── Run with LLaMA 70B (3 times)
    │       └── Check: Are all 3 runs consistent?
    │
    ├── Run with Mixtral 8x7B (2 times)
    │       └── Check: Are runs consistent?
    │
    └── Run with LLaMA 8B (2 times)
            └── Check: Are runs consistent?

    Then compare across all models:
        └── Do all 3 models agree on risk level?
```

---

### Evaluation Metrics Summary

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Intra-Model Consistency** | Same model, multiple runs | Score variance < 5 |
| **Inter-Model Agreement** | Different models, same input | Risk level match > 80% |
| **Reasoning Overlap** | Key risk factors identified | > 70% overlap |
| **Confidence Calibration** | High confidence = correct? | Correlation > 0.7 |

---

### Test Dataset

| Company | Type | Expected Risk | Notes |
|---------|------|---------------|-------|
| Apple | Large Cap | Low | Strong financials |
| Tesla | Growth | Medium | Volatile |
| Enron | Bankrupt | Critical | Known fraud |
| Microsoft | Large Cap | Low | Stable |
| WeWork | Startup | High | Known issues |
| ... | ... | ... | ... |

---

### Implementation Steps

```
Step 1: Select 15-20 test companies (mix of risk levels)
            ↓
Step 2: Run each company with 3 models × 2-3 runs each
            ↓
Step 3: Store all results in MongoDB (evaluations collection)
            ↓
Step 4: Calculate consistency metrics
            ↓
Step 5: Identify problematic cases (high variance)
            ↓
Step 6: Analyze and improve prompts if needed
            ↓
Step 7: Re-run evaluation to confirm improvement
```

---

### Files for Evaluation

```
src/evaluation/
├── run_evaluation.py       # Main runner - executes all tests
├── consistency_scorer.py   # Intra-model consistency
├── cross_model_scorer.py   # Inter-model agreement
├── analyzer.py             # Analyzes & visualizes results
└── report_generator.py     # Creates evaluation report
```

---

## Data Storage (MongoDB)

All assessments and raw data are automatically saved to MongoDB Atlas.

**Collections:**
| Collection | What's Stored |
|------------|---------------|
| `companies` | Company profiles and metadata |
| `assessments` | LLM credit decisions (risk level, score, reasoning) |
| `raw_data` | Raw API responses for auditing |
| `evaluations` | Consistency evaluation results |

**Connection:** MongoDB Atlas (cloud-hosted, free tier)

**Workflow:**
```
LLM makes decision → Assessment saved to MongoDB → Raw data archived
```

**Benefits:**
- Historical tracking of all assessments
- Audit trail of raw API data
- Query and analyze past decisions
- Data persists across sessions

---

## API Keys Required

| API | Environment Variable | Where to Get |
|-----|---------------------|--------------|
| Groq (LLM) | `GROQ_API_KEY` | https://console.groq.com |
| Finnhub | `FINNHUB_API_KEY` | https://finnhub.io |
| LangSmith | `LANGCHAIN_API_KEY` | https://smith.langchain.com |
| CourtListener | `COURTLISTENER_API_KEY` | https://www.courtlistener.com |
| MongoDB | `MONGODB_URI` | https://cloud.mongodb.com |

All APIs have free tiers sufficient for development and testing.
