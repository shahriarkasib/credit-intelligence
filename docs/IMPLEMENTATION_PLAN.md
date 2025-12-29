# Implementation Plan: Tool-Based Agents + Comprehensive Evaluation

## Overview

Transform the current workflow into a **tool-based agent system** where the LLM Supervisor intelligently chooses which tools to use, with **comprehensive evaluation** at every step.

---

## Part 1: Tool-Based Agent Architecture

### Current vs New Architecture

**Current:**
```
Supervisor → API Agent → Search Agent → LLM Analysis → Result
(Fixed order, no choice)
```

**New (Tool-Based):**
```
Supervisor (LLM) decides:
  ├── "I need financial data" → Call API Tool
  ├── "I need recent news" → Call Web Search Tool
  ├── "I need both" → Call both tools
  └── "Company not found" → Try alternative approach
```

### Tools to Implement

| Tool | Purpose | When LLM Should Choose It |
|------|---------|---------------------------|
| `fetch_sec_data` | Get SEC EDGAR financials | Public US companies, need official filings |
| `fetch_market_data` | Get Finnhub stock data | Public companies, need stock/market info |
| `fetch_legal_data` | Get CourtListener records | Any company, check for lawsuits |
| `web_search` | DuckDuckGo search | Private companies, recent news, general info |
| `analyze_credit` | LLM credit analysis | After data collection, make final decision |

### Supervisor Agent Decision Logic

The LLM Supervisor should:
1. Understand the company type (public/private, US/international)
2. Choose appropriate tools based on context
3. Handle cases where tools return no data
4. Combine results intelligently

**Example Decision Flow:**
```
Input: "Apple"
LLM thinks: "Apple is a public US company (AAPL), I should use:
  1. fetch_sec_data (for financial filings)
  2. fetch_market_data (for stock price)
  3. fetch_legal_data (for any lawsuits)
  4. web_search (for recent news)"

Input: "Joe's Pizza Shop"
LLM thinks: "This is likely a private local business, I should use:
  1. web_search (to find information)
  2. fetch_legal_data (check for lawsuits)
  (Skip SEC/Finnhub - not a public company)"
```

---

## Part 2: Comprehensive Evaluation Framework

### Evaluation Categories

#### A. Execution Metrics (Per Step)
| Metric | What We Measure | How |
|--------|-----------------|-----|
| Step Execution Time | Time for each node | `time.time()` before/after |
| Token Usage | Prompt + completion tokens | Groq API response |
| Cost | Per call and per run | Token count × rate |
| Error Rate | % of failed steps | Error count / total |
| Retries | Number of retry attempts | Counter |
| Tool Calls | Which tools were called | Log each tool invocation |

#### B. Tool Selection Evaluation
| Metric | What We Measure | How |
|--------|-----------------|-----|
| Tool Selection Accuracy | Did LLM choose right tools? | Compare to expected tools |
| Tool Completeness | Were all necessary tools called? | Check coverage |
| Tool Efficiency | No unnecessary tool calls? | Count redundant calls |
| Tool Order | Logical execution order? | Sequence analysis |

#### C. Data Quality Evaluation
| Metric | What We Measure | How |
|--------|-----------------|-----|
| Company Name Understanding | Correct company identified? | Verify ticker/CIK match |
| API Data Accuracy | Did we get the right company? | Cross-validate sources |
| Data Completeness | All expected fields populated? | Field coverage % |
| Search Relevance | Web results about the company? | Keyword matching |

#### D. Synthesis Evaluation
| Metric | What We Measure | How |
|--------|-----------------|-----|
| Intent Fulfillment | Output matches user request? | LLM judge comparison |
| Reasoning Integrity | Logic is sound and consistent? | Chain-of-thought check |
| Grounded in Context | Claims supported by data? | Citation verification |
| Instruction Following | Followed output format? | Schema validation |

#### E. Consistency Evaluation
| Metric | What We Measure | How |
|--------|-----------------|-----|
| Intra-Model Consistency | Same model, same answer? | Run 3x, compare |
| Inter-Model Consistency | Different models agree? | Run with 3 models |
| Score Variance | Credit scores stable? | Standard deviation |
| Risk Level Agreement | Same risk classification? | % agreement |

#### F. Output Quality Metrics
| Metric | What We Measure | How |
|--------|-----------------|-----|
| Expected Output Similarity | Close to expected answer? | Cosine similarity |
| Confidence Calibration | High confidence = correct? | Correlation analysis |
| Avoided Answer | Did it refuse when appropriate? | Edge case handling |

---

## Part 3: Logging Architecture

### MongoDB Collections

```
credit_intelligence (database)
├── runs                    # Complete run logs
│   ├── run_id
│   ├── company_name
│   ├── timestamp
│   ├── steps[]             # Each step with metrics
│   ├── tool_calls[]        # All tool invocations
│   ├── final_result
│   └── total_metrics
│
├── step_logs               # Individual step logs
│   ├── run_id
│   ├── step_name
│   ├── input
│   ├── output
│   ├── execution_time_ms
│   ├── tokens_used
│   ├── errors
│   └── metadata
│
├── evaluations             # Evaluation results
│   ├── run_id
│   ├── evaluation_type
│   ├── metrics{}
│   ├── scores{}
│   └── timestamp
│
├── tool_logs               # Tool-specific logs
│   ├── run_id
│   ├── tool_name
│   ├── input
│   ├── output
│   ├── success
│   ├── execution_time_ms
│   └── selection_reason
│
└── assessments             # Final credit assessments
    ├── company_name
    ├── risk_level
    ├── credit_score
    ├── reasoning
    └── data_sources_used
```

### Log Entry Structure

```json
{
  "run_id": "uuid",
  "company_name": "Apple",
  "timestamp": "2024-12-25T10:00:00Z",
  "steps": [
    {
      "step_name": "parse_input",
      "execution_time_ms": 45,
      "tokens_used": 0,
      "success": true
    },
    {
      "step_name": "tool_selection",
      "execution_time_ms": 1200,
      "tokens_used": {"prompt": 500, "completion": 100},
      "tools_selected": ["fetch_sec_data", "fetch_market_data", "web_search"],
      "selection_reasoning": "Public company, need financials and news"
    },
    {
      "step_name": "fetch_sec_data",
      "execution_time_ms": 2500,
      "data_fields_retrieved": 15,
      "success": true
    }
  ],
  "evaluation": {
    "tool_selection_accuracy": 1.0,
    "data_completeness": 0.85,
    "reasoning_integrity": 0.92,
    "final_score_confidence": 0.88
  }
}
```

---

## Part 4: Google Sheets Export

### What You Need to Provide

1. **Google Cloud Service Account JSON** (for API access)
   - Go to: https://console.cloud.google.com
   - Create a project
   - Enable Google Sheets API
   - Create service account & download JSON key

2. **Google Sheet ID** (where to write data)
   - Create a new Google Sheet
   - Share it with the service account email
   - Copy the Sheet ID from URL

### Sheets Structure

| Sheet | Contents |
|-------|----------|
| `Runs` | All run summaries (company, risk, score, time) |
| `Step Metrics` | Detailed step-by-step metrics |
| `Evaluations` | All evaluation scores |
| `Tool Calls` | Tool selection and results |
| `Errors` | Error logs and retry info |

---

## Part 5: Relevant Evaluation Metrics (From Appendix)

### High Priority (Implement First)
| Metric | Relevance | Why |
|--------|-----------|-----|
| **Step Execution Time** | High | Track performance |
| **Token Usage** | High | Cost tracking |
| **Tool Calls** | High | Core to tool-based architecture |
| **Intent Fulfillment** | High | Did we answer the credit question? |
| **Tools Completeness** | High | Did we use all needed tools? |
| **Reasoning Integrity** | High | Is the credit logic sound? |
| **Expected Output Similarity** | High | Consistency check |

### Medium Priority (Implement Second)
| Metric | Relevance | Why |
|--------|-----------|-----|
| **Plan Efficiency** | Medium | Optimal tool selection |
| **Tool Coverage** | Medium | Comprehensive data collection |
| **Instruction Following** | Medium | Output format compliance |
| **Error Rates** | Medium | Reliability tracking |
| **Retries** | Medium | Resilience metrics |

### Lower Priority (Nice to Have)
| Metric | Relevance | Why |
|--------|-----------|-----|
| **Cost per run** | Low | Free APIs currently |
| **Avoided Answer** | Low | Edge cases |
| **Retrieval Relevance** | Low | We don't do RAG |
| **Grounded in Context** | Low | Already using source data |

---

## Implementation Steps

### Phase 1: Tool-Based Agents (Days 1-2)
```
1. Refactor agents into tools with clear interfaces
2. Create ToolExecutor wrapper with logging
3. Update Supervisor to use LLM for tool selection
4. Add tool selection reasoning capture
5. Test tool selection accuracy
```

### Phase 2: Step-by-Step Logging (Days 2-3)
```
1. Create StepLogger class
2. Wrap each node with timing/token tracking
3. Store logs in MongoDB
4. Create run summary aggregation
```

### Phase 3: Evaluation Framework (Days 3-4)
```
1. Implement execution metrics collectors
2. Create tool selection evaluator
3. Build consistency checker (multi-run)
4. Add reasoning integrity scorer
5. Create evaluation report generator
```

### Phase 4: Google Sheets Export (Day 5)
```
1. Set up Google Sheets API
2. Create export functions
3. Build auto-sync mechanism
4. Add real-time dashboard sheet
```

### Phase 5: Testing & Refinement (Days 5-6)
```
1. Run evaluation on 20 test companies
2. Analyze results
3. Tune tool selection prompts
4. Document findings
```

---

## Files to Create

```
src/
├── tools/
│   ├── __init__.py
│   ├── base_tool.py           # Tool interface with logging
│   ├── sec_tool.py            # SEC EDGAR tool
│   ├── finnhub_tool.py        # Finnhub tool
│   ├── court_tool.py          # CourtListener tool
│   ├── web_search_tool.py     # Web search tool
│   └── tool_executor.py       # Executes tools with metrics
│
├── logging/
│   ├── __init__.py
│   ├── step_logger.py         # Logs each step
│   ├── run_logger.py          # Aggregates run logs
│   └── metrics_collector.py   # Collects all metrics
│
├── evaluation/
│   ├── tool_selection_eval.py # Evaluates tool choices
│   ├── consistency_eval.py    # Multi-run consistency
│   ├── reasoning_eval.py      # Reasoning quality
│   ├── metrics_calculator.py  # Computes all metrics
│   └── report_generator.py    # Creates evaluation reports
│
├── export/
│   ├── __init__.py
│   ├── google_sheets.py       # Google Sheets export
│   ├── csv_export.py          # CSV export
│   └── sync_manager.py        # Sync MongoDB → Sheets
│
└── agents/
    ├── tool_supervisor.py     # LLM-based tool selection
    └── (existing files...)
```

---

## Summary

| Component | Status | Priority |
|-----------|--------|----------|
| Tool-Based Agents | To Build | High |
| Step Logging | To Build | High |
| MongoDB Logging | Partial | High |
| Evaluation Framework | To Build | High |
| Google Sheets Export | To Build | Medium |

**Next Action:** Start with Phase 1 - Converting agents to tools with the Supervisor making LLM-based tool selection decisions.

---

## What You Need to Provide

1. **For Google Sheets:**
   - Google Cloud Project with Sheets API enabled
   - Service Account JSON key file
   - Sheet ID (create and share with service account)

2. **For Evaluation:**
   - List of 15-20 test companies (mix of public/private)
   - Expected outcomes for each (for accuracy validation)

Ready to start implementation?
