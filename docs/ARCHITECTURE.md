# Credit Intelligence System Architecture

## Overview

Credit Intelligence is a multi-agent system for automated credit risk assessment. It uses LangGraph for workflow orchestration, multiple LLM providers for analysis, and various data sources for information gathering.

---

## Canonical Agent Names

**IMPORTANT:** These are the exact `agent_name` values logged to Google Sheets.

| Node | agent_name | Description |
|------|------------|-------------|
| parse_input | `llm_parser` | Parses company input |
| validate_company | `supervisor` | Validates company |
| create_plan | `tool_supervisor` | LLM tool selection |
| fetch_api_data | `api_agent` | Fetches API data |
| search_web | `search_agent` | Web search |
| synthesize | `llm_analyst` | Credit synthesis |
| save_to_database | `db_writer` | Database storage |
| evaluate_assessment | `workflow_evaluator` | All evaluation tasks |

### LLM Call Types (logged in llm_calls sheet)

| call_type | Description |
|-----------|-------------|
| `company_parser` | Parse company input |
| `tool_selection` | Select tools to use |
| `credit_synthesis` | Synthesize assessment |
| `credit_analysis` | Full credit analysis |
| `validation` | Validate assessment |
| `tool_selection_evaluation` | Evaluate tool selection |

---

## Entity Hierarchy

```
WORKFLOW (LangGraph StateGraph)
│
├── RUN
│   ├── run_id: UUID (unique identifier)
│   ├── company_name: string
│   ├── started_at: timestamp
│   ├── completed_at: timestamp
│   └── status: pending | running | completed | failed
│
├── NODES (Graph Steps)
│   │
│   ├── parse_input
│   │   ├── Agent: llm_parser
│   │   ├── LLM Call: company_parser prompt
│   │   └── Output: company_info (ticker, jurisdiction, is_public)
│   │
│   ├── validate_company
│   │   ├── Agent: supervisor
│   │   ├── Human-in-the-loop: optional approval
│   │   └── Output: validation_status
│   │
│   ├── create_plan
│   │   ├── Agent: tool_supervisor
│   │   ├── LLM Call: tool_selection prompt
│   │   └── Output: task_plan (list of tools to execute)
│   │
│   ├── fetch_api_data
│   │   ├── Agent: api_agent
│   │   ├── Tools: SECTool, FinnhubTool, CourtTool
│   │   ├── Data Sources: SEC EDGAR, Finnhub, CourtListener
│   │   └── Output: api_data
│   │
│   ├── search_web
│   │   ├── Agent: search_agent
│   │   ├── Tools: WebSearchTool, TavilySearch
│   │   ├── Data Sources: Tavily API, Web Scraper
│   │   └── Output: search_data
│   │
│   ├── synthesize
│   │   ├── Agent: llm_analyst
│   │   ├── LLM Calls: credit_synthesis prompt (primary + secondary model)
│   │   ├── Cross-Model Evaluation: compare results
│   │   ├── Same-Model Consistency: multiple runs
│   │   └── Output: assessment
│   │
│   ├── save_to_database
│   │   ├── Agent: db_writer
│   │   ├── Storage: MongoDB
│   │   └── Output: stored run record
│   │
│   └── evaluate_assessment
│       ├── Agent: workflow_evaluator (for ALL evaluation tasks)
│       ├── Evaluators: Tool selection, Data quality, Synthesis, LLM Judge, Agent efficiency
│       └── Output: evaluation scores
│
└── STATE (CreditWorkflowState)
    ├── Input: company_name, jurisdiction, ticker
    ├── Intermediate: company_info, task_plan, api_data, search_data, tool_selection
    ├── Output: assessment, evaluation
    └── Metadata: errors, status, execution_time_ms
```

---

## Detailed Entity Descriptions

### 1. WORKFLOW

**Location:** `src/agents/graph.py`

The workflow is a LangGraph StateGraph that orchestrates the entire credit assessment process.

| Property | Type | Description |
|----------|------|-------------|
| `graph` | StateGraph | LangGraph compiled graph |
| `state_class` | CreditWorkflowState | TypedDict defining all state fields |
| `entry_point` | string | First node to execute ("parse_input") |
| `nodes` | Dict[str, Callable] | Map of node names to functions |
| `edges` | List[Tuple] | Transitions between nodes |
| `conditionals` | Dict | Conditional routing logic |

---

### 2. RUN

**Location:** State managed across all nodes

A run represents a single execution of the workflow for one company.

| Property | Type | Description |
|----------|------|-------------|
| `run_id` | UUID string | Unique identifier for this run |
| `company_name` | string | Company being analyzed |
| `started_at` | ISO timestamp | When run started |
| `completed_at` | ISO timestamp | When run completed |
| `status` | string | Current status |
| `duration_ms` | float | Total execution time |
| `errors` | List[string] | Any errors encountered |

---

### 3. NODES

Each node is a function that transforms the workflow state.

#### 3.1 parse_input

**Purpose:** Parse company name and extract metadata using LLM

| Property | Value |
|----------|-------|
| step_number | 1 |
| node | "parse_input" |
| agent_name | **`llm_parser`** |
| prompt_id | "company_parser" |
| llm_provider | groq |
| llm_model | fast (llama-3.1-8b-instant) |
| temperature | 0.1 |

**Output:**
```python
{
    "run_id": "uuid",
    "company_info": {
        "is_public_company": bool,
        "ticker": str | None,
        "industry": str,
        "sector": str,
        "jurisdiction": str,
        "confidence": float
    }
}
```

#### 3.2 validate_company

**Purpose:** Validate parsed company info, optional human approval

| Property | Value |
|----------|-------|
| step_number | 2 |
| node | "validate_company" |
| agent_name | **`supervisor`** |
| human_in_loop | optional |

**Output:**
```python
{
    "human_approved": bool,
    "validation_message": str,
    "requires_review": bool
}
```

#### 3.3 create_plan

**Purpose:** LLM decides which tools to use for data collection

| Property | Value |
|----------|-------|
| step_number | 3 |
| node | "create_plan" |
| agent_name | **`tool_supervisor`** |
| prompt_id | "tool_selection" |
| llm_provider | groq |
| llm_model | primary (llama-3.3-70b-versatile) |
| temperature | 0.1 |

**Output:**
```python
{
    "task_plan": [
        {
            "tool": "sec_edgar",
            "params": {"ticker": "AAPL"},
            "reason": "Public company, need SEC filings"
        },
        {
            "tool": "finnhub",
            "params": {"ticker": "AAPL"},
            "reason": "Get market data and financials"
        }
    ],
    "tool_selection": {
        "model": "llama-3.3-70b-versatile",
        "reasoning": "Selected tools based on company type..."
    }
}
```

#### 3.4 fetch_api_data

**Purpose:** Execute API tools to collect structured data

| Property | Value |
|----------|-------|
| step_number | 4 |
| node | "fetch_api_data" |
| agent_name | **`api_agent`** |
| tools | SEC, Finnhub, Court tools |
| parallel | true |

**Tools Used:**
- `SECTool` -> SEC EDGAR API
- `FinnhubTool` -> Finnhub API
- `CourtListenerTool` -> CourtListener API

**Output:**
```python
{
    "api_data": {
        "sec_edgar": {...},
        "finnhub": {...},
        "court_listener": {...}
    }
}
```

#### 3.5 search_web

**Purpose:** Search web for additional company information

| Property | Value |
|----------|-------|
| step_number | 5 |
| node | "search_web" |
| agent_name | **`search_agent`** |
| tools | WebSearch, Tavily |

**Tools Used:**
- `WebSearchTool` -> General web search
- `TavilySearchDataSource` -> Tavily AI search

**Output:**
```python
{
    "search_data": {
        "news": [...],
        "web_results": [...],
        "scraped_content": {...}
    }
}
```

#### 3.6 synthesize

**Purpose:** LLM analyzes all data and produces credit assessment

| Property | Value |
|----------|-------|
| step_number | 6 |
| node | "synthesize" |
| agent_name | **`llm_analyst`** |
| prompt_id | "credit_synthesis" |
| llm_provider | groq |
| llm_model | primary (llama-3.3-70b-versatile) |
| temperature | 0.0 |
| cross_model | true (compares with secondary model) |

**Output:**
```python
{
    "assessment": {
        "overall_risk_level": "low" | "medium" | "high" | "critical",
        "credit_score_estimate": 300-850,
        "confidence_score": 0.0-1.0,
        "reasoning": str,
        "risk_factors": [...],
        "positive_factors": [...],
        "recommendations": [...]
    }
}
```

#### 3.7 save_to_database

**Purpose:** Persist run results to MongoDB

| Property | Value |
|----------|-------|
| step_number | 7 |
| node | "save_to_database" |
| agent_name | **`db_writer`** |
| storage | MongoDB |
| collection | "runs" |

#### 3.8 evaluate_assessment

**Purpose:** Evaluate the quality of the assessment

| Property | Value |
|----------|-------|
| step_number | 8 |
| node | "evaluate" |
| agent_name | **`workflow_evaluator`** (for ALL evaluation tasks) |

**Evaluation Types:**
| Evaluation | Description |
|------------|-------------|
| Tool Selection | Precision, recall, F1 of selected tools |
| Data Quality | Completeness of collected data |
| Synthesis Quality | Quality of credit assessment |
| LLM Judge | LLM evaluates assessment quality |
| Agent Efficiency | Intent, plan, trajectory metrics |
| Cross-Model | Compare primary vs secondary model |
| Consistency | Same model across multiple runs |

**Output:**
```python
{
    "evaluation": {
        "overall_score": 0.0-1.0,
        "tool_selection_score": 0.0-1.0,
        "data_quality_score": 0.0-1.0,
        "synthesis_score": 0.0-1.0,
        "agent_metrics": {...},
        "llm_judge": {...}
    }
}
```

---

### 4. AGENTS

#### 4.1 LLMParser (llm_parser)

**Location:** `src/agents/llm_parser.py`

Parses company input using LLM.

| Property | Type | Description |
|----------|------|-------------|
| `model` | string | LLM model to use |
| `prompt_id` | string | "company_parser" |

#### 4.2 SupervisorAgent (supervisor)

**Location:** `src/agents/supervisor.py`

Orchestrates the workflow and makes high-level decisions.

| Property | Type | Description |
|----------|------|-------------|
| `config` | Dict | Configuration options |
| `analysis_mode` | string | "rule_based" | "llm" | "hybrid" |

#### 4.3 ToolSupervisor (tool_supervisor)

**Location:** `src/agents/tool_supervisor.py`

LLM-based tool selection agent.

| Property | Type | Description |
|----------|------|-------------|
| `model` | string | LLM model to use |
| `tool_executor` | ToolExecutor | Executes selected tools |

**LLM Calls:**
1. Tool Selection (prompt: "tool_selection")
2. Credit Synthesis (prompt: "credit_synthesis")

#### 4.4 APIAgent (api_agent)

**Location:** `src/agents/api_agent.py`

Fetches structured data from external APIs.

| Property | Type | Description |
|----------|------|-------------|
| `sec_edgar` | SECEdgarDataSource | SEC EDGAR connector |
| `finnhub` | FinnhubDataSource | Finnhub connector |
| `court_listener` | CourtListenerDataSource | Court data connector |

#### 4.5 SearchAgent (search_agent)

**Location:** `src/agents/search_agent.py`

Web search and content scraping.

| Property | Type | Description |
|----------|------|-------------|
| `web_search` | WebSearchDataSource | Web search |
| `tavily` | TavilySearchDataSource | Tavily AI search |
| `scraper` | WebScraper | Content scraper |

#### 4.6 LLMAnalystAgent (llm_analyst)

**Location:** `src/agents/llm_analyst.py`

LLM-powered credit analysis.

| Property | Type | Description |
|----------|------|-------------|
| `model` | string | LLM model |
| `temperature` | float | Sampling temperature |

#### 4.7 WorkflowEvaluator (workflow_evaluator)

**Location:** `src/evaluation/workflow_evaluator.py`

Handles ALL evaluation tasks with a single canonical agent name.

| Evaluation | Description |
|------------|-------------|
| Tool Selection | Evaluates tool selection precision/recall |
| Data Quality | Checks data completeness |
| Synthesis | Evaluates assessment quality |
| LLM Judge | LLM-as-a-judge evaluation |
| Agent Metrics | Efficiency metrics |
| Cross-Model | Compares multiple models |
| Consistency | Same-model consistency |

---

### 5. TOOLS

**Location:** `src/tools/`

Tools are executable units that fetch data from external sources.

#### 5.1 SECTool (sec_edgar)

| Property | Value |
|----------|-------|
| name | "sec_edgar" |
| data_source | SECEdgarDataSource |
| parameters | ticker (required), filing_type (optional) |

#### 5.2 FinnhubTool (finnhub)

| Property | Value |
|----------|-------|
| name | "finnhub" |
| data_source | FinnhubDataSource |
| parameters | ticker (required) |

#### 5.3 CourtListenerTool (court_listener)

| Property | Value |
|----------|-------|
| name | "court_listener" |
| data_source | CourtListenerDataSource |
| parameters | company_name (required) |

#### 5.4 WebSearchTool (web_search)

| Property | Value |
|----------|-------|
| name | "web_search" |
| data_source | WebSearchDataSource |
| parameters | query (required), num_results (optional) |

---

### 6. DATA SOURCES

**Location:** `src/data_sources/`

Data sources are API connectors that handle authentication and data transformation.

| Data Source | API | Methods |
|-------------|-----|---------|
| SECEdgarDataSource | SEC EDGAR | get_company_info, get_filings, get_financials |
| FinnhubDataSource | Finnhub | get_profile, get_quote, get_financials, get_news |
| CourtListenerDataSource | CourtListener | search_opinions, search_dockets |
| TavilySearchDataSource | Tavily | search, company_search |
| WebScraper | HTTP | scrape_url, scrape_company_website |

---

### 7. LLM CALLS

**Location:** Various agents

#### 7.1 Available Prompts

| Prompt ID | Category | Agent | Purpose |
|-----------|----------|-------|---------|
| `company_parser` | input | llm_parser | Parse company name |
| `tool_selection` | planning | tool_supervisor | Select tools to use |
| `tool_selection_evaluation` | evaluation | workflow_evaluator | Evaluate tool selection |
| `credit_synthesis` | synthesis | llm_analyst | Produce credit assessment |
| `credit_analysis` | analysis | llm_analyst | Full credit analysis |
| `validation` | validation | supervisor | Validate assessment |

#### 7.2 LLM Providers & Models

**Groq (Default):**
| Alias | Model ID | Use Case |
|-------|----------|----------|
| primary | llama-3.3-70b-versatile | Complex reasoning |
| fast | llama-3.1-8b-instant | Simple parsing |
| balanced | llama3-70b-8192 | General use |

**OpenAI (Optional):**
| Alias | Model ID | Use Case |
|-------|----------|----------|
| primary | gpt-4o-mini | Complex reasoning |
| fast | gpt-4o-mini | Quick tasks |

**Anthropic (Optional):**
| Alias | Model ID | Use Case |
|-------|----------|----------|
| primary | claude-3-5-sonnet-20241022 | Complex reasoning |
| fast | claude-3-haiku-20240307 | Quick tasks |

---

### 8. LOGGING

**Location:** `src/run_logging/`

#### 8.1 SheetsLogger

Logs to Google Sheets for analysis.

**15 Sheets:**
| Sheet | Content |
|-------|---------|
| runs | Run summaries |
| langgraph_events | LangGraph execution events |
| llm_calls | LLM API calls |
| tool_calls | Tool executions |
| assessments | Credit assessments |
| evaluations | Evaluation results |
| tool_selections | Tool selection details |
| consistency_scores | Model consistency |
| data_sources | Data fetch results |
| plans | Task plans |
| prompts | Prompts used |
| cross_model_eval | Cross-model comparison |
| llm_judge_results | LLM judge evaluation |
| agent_metrics | Agent efficiency metrics |
| log_tests | Verification |

---

### 9. STORAGE

**Location:** `src/storage/`

#### MongoDB

| Collection | Content |
|------------|---------|
| runs | Complete run records |
| companies | Company data cache |
| assessments | Credit assessments |

---

## Data Flow

```
INPUT: company_name
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. PARSE_INPUT (agent_name: llm_parser)                         │
│    ├── LLM Call: company_parser                                 │
│    │   ├── Provider: groq                                       │
│    │   ├── Model: fast (llama-3.1-8b-instant)                  │
│    │   └── Output: {is_public, ticker, industry, jurisdiction} │
│    └── Generate: run_id                                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. VALIDATE_COMPANY (agent_name: supervisor)                    │
│    ├── Check: company_info validity                             │
│    ├── Optional: Human approval                                 │
│    └── Decision: continue or stop                               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CREATE_PLAN (agent_name: tool_supervisor)                    │
│    ├── LLM Call: tool_selection                                 │
│    │   ├── Provider: groq                                       │
│    │   ├── Model: primary (llama-3.3-70b-versatile)            │
│    │   └── Output: task_plan [{tool, params, reason}, ...]     │
│    └── Stores: tool_selection in state for evaluation           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. FETCH_API_DATA (agent_name: api_agent)                       │
│    ├── Tool Execution (parallel):                               │
│    │   ├── SECTool → SEC EDGAR API → filings, financials       │
│    │   ├── FinnhubTool → Finnhub API → profile, quote, news    │
│    │   └── CourtTool → CourtListener API → cases, opinions     │
│    └── Output: api_data {sec_edgar, finnhub, court_listener}   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. SEARCH_WEB (agent_name: search_agent)                        │
│    ├── Tool Execution:                                          │
│    │   ├── WebSearchTool → web search results                  │
│    │   ├── TavilySearch → AI-powered search                    │
│    │   └── WebScraper → company website content                │
│    └── Output: search_data {news, web_results, scraped}        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SYNTHESIZE (agent_name: llm_analyst)                         │
│    ├── Primary LLM Call: credit_synthesis                       │
│    │   ├── Provider: groq                                       │
│    │   ├── Model: primary (llama-3.3-70b-versatile)            │
│    │   ├── Temperature: 0.0                                     │
│    │   └── Input: api_data + search_data                        │
│    │                                                            │
│    ├── Secondary LLM Call (cross-model):                        │
│    │   ├── Model: llama-3.1-8b-instant                         │
│    │   └── Cross-model evaluation logged                        │
│    │                                                            │
│    └── Output: assessment {risk_level, credit_score, ...}      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. SAVE_TO_DATABASE (agent_name: db_writer)                     │
│    └── MongoDB: store full run record                           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. EVALUATE_ASSESSMENT (agent_name: workflow_evaluator)         │
│    ├── Tool Selection Eval → precision, recall, f1             │
│    ├── Data Quality Eval → completeness score                  │
│    ├── Synthesis Eval → output quality score                   │
│    ├── Agent Efficiency → intent, plan, trajectory             │
│    ├── LLM Judge → accuracy, completeness, consistency         │
│    ├── Cross-Model Eval → compare 2 models                     │
│    ├── Consistency Eval → same model stability                 │
│    └── LogVerification → verify all sheets logged              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT: {
    assessment: CreditAssessment,
    evaluation: EvaluationResults,
    run_id: string
}
```

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (required) |
| `OPENAI_API_KEY` | OpenAI API key (optional) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `FINNHUB_API_KEY` | Finnhub API key |
| `TAVILY_API_KEY` | Tavily API key |
| `MONGODB_URI` | MongoDB connection string |
| `GOOGLE_SPREADSHEET_ID` | Google Sheets ID |
| `LANGSMITH_API_KEY` | LangSmith tracing key |

### Prompt Configuration

Each prompt can specify:
```python
{
    "llm_config": {
        "provider": "groq" | "openai" | "anthropic",
        "model": "primary" | "fast" | "balanced",
        "temperature": 0.0 - 1.0,
        "max_tokens": int
    }
}
```

---

## File Structure

```
src/
├── agents/
│   ├── graph.py           # LangGraph workflow
│   ├── supervisor.py      # SupervisorAgent (supervisor)
│   ├── tool_supervisor.py # ToolSupervisor (tool_supervisor)
│   ├── api_agent.py       # APIAgent (api_agent)
│   ├── search_agent.py    # SearchAgent (search_agent)
│   ├── llm_analyst.py     # LLMAnalystAgent (llm_analyst)
│   └── llm_parser.py      # LLM company parser (llm_parser)
│
├── tools/
│   ├── base_tool.py       # BaseTool class
│   ├── tool_executor.py   # Tool execution
│   ├── sec_tool.py        # SEC EDGAR tool
│   ├── finnhub_tool.py    # Finnhub tool
│   ├── court_tool.py      # CourtListener tool
│   ├── web_search_tool.py # Web search tool
│   └── langchain_tools.py # LangChain wrappers
│
├── data_sources/
│   ├── sec_edgar.py       # SEC EDGAR API
│   ├── finnhub.py         # Finnhub API
│   ├── court_listener.py  # CourtListener API
│   ├── tavily_search.py   # Tavily search
│   └── web_scraper.py     # Web scraper
│
├── config/
│   ├── prompts.py         # Centralized prompts
│   ├── langchain_llm.py   # LLM factory
│   ├── output_parsers.py  # Response parsers
│   └── external_config.py # YAML config loader
│
├── evaluation/
│   └── workflow_evaluator.py # All evaluation (workflow_evaluator)
│
├── run_logging/
│   ├── workflow_logger.py # Main logger
│   ├── sheets_logger.py   # Google Sheets logger (15 sheets)
│   └── langgraph_logger.py
│
└── storage/
    └── mongodb.py         # MongoDB storage (db_writer)
```

---

## Summary

The Credit Intelligence system is a hierarchical multi-agent workflow:

1. **Workflow** orchestrates the entire process
2. **Nodes** are discrete steps in the workflow
3. **Agents** execute within nodes (8 canonical agent names)
4. **Tools** are used by agents to fetch data
5. **Data Sources** connect to external APIs
6. **LLM Calls** power analysis and decision-making
7. **Evaluators** measure output quality (all use `workflow_evaluator`)
8. **Loggers** record everything to 15 Google Sheets

Each entity has well-defined inputs, outputs, and relationships, enabling comprehensive tracing and evaluation of the credit assessment process.
