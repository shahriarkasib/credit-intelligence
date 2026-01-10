# Credit Intelligence System Architecture

## Overview

Credit Intelligence is a multi-agent system for automated credit risk assessment. It uses LangGraph for workflow orchestration, multiple LLM providers for analysis, and various data sources for information gathering.

---

## Agent Names Reference (Canonical)

**IMPORTANT:** These are the exact `agent_name` values logged to Google Sheets.

### Current State (Actual Values in Code)

| Node | agent_name (in Sheets) | Status | Class Name |
|------|------------------------|--------|------------|
| parse_input | `llm_parser` | OK | LLMParser |
| validate_company | `supervisor` | OK | SupervisorAgent |
| create_plan | `tool_supervisor` | OK | ToolSupervisor |
| fetch_api_data | `api_agent` | OK | APIAgent |
| search_web | `search_agent` | OK | SearchAgent |
| synthesize | `llm_analyst` | OK | LLMAnalystAgent |
| synthesize (cross) | `cross_model_evaluator` | OK | - |
| synthesize (consistency) | `consistency_evaluator` | OK | - |
| save_to_database | `db_writer` | OK | MongoDB |
| evaluate_assessment | `workflow_evaluator` | OK | WorkflowEvaluator |
| evaluate_assessment | `agent_efficiency_evaluator` | OK | AgentEfficiencyEvaluator |
| evaluate_assessment | `llm_judge` | OK | LLMJudgeEvaluator |
| evaluate_assessment | `unified_evaluator` | OK | UnifiedAgentEvaluator |
| evaluate_assessment | `llm_tool_selection_judge` | OK | - |

### Recommended Standard Names

| Node | agent_name (Standard) | Description |
|------|----------------------|-------------|
| parse_input | `llm_parser` | Parses company input |
| validate_company | `supervisor` | Validates company |
| create_plan | `tool_supervisor` | LLM tool selection |
| fetch_api_data | `api_agent` | Fetches API data |
| search_web | `search_agent` | Web search |
| synthesize | `llm_analyst` | Credit synthesis |
| save_to_database | `db_writer` | Database storage |
| evaluate_assessment | `workflow_evaluator` | Overall evaluation |

### LLM Call Types (logged as agent_name in llm_calls sheet)

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
│   │   ├── Agent: LLMParser
│   │   ├── LLM Call: company_parser prompt
│   │   └── Output: company_info (ticker, jurisdiction, is_public)
│   │
│   ├── validate_company
│   │   ├── Agent: SupervisorAgent
│   │   ├── Human-in-the-loop: optional approval
│   │   └── Output: validation_status
│   │
│   ├── create_plan
│   │   ├── Agent: ToolSupervisor
│   │   ├── LLM Call: tool_selection prompt
│   │   └── Output: task_plan (list of tools to execute)
│   │
│   ├── fetch_api_data
│   │   ├── Agent: APIAgent
│   │   ├── Tools: SECTool, FinnhubTool, CourtTool
│   │   ├── Data Sources: SEC EDGAR, Finnhub, CourtListener
│   │   └── Output: api_data
│   │
│   ├── search_web
│   │   ├── Agent: SearchAgent
│   │   ├── Tools: WebSearchTool, TavilySearch
│   │   ├── Data Sources: Tavily API, Web Scraper
│   │   └── Output: search_data
│   │
│   ├── synthesize
│   │   ├── Agent: LLMAnalystAgent
│   │   ├── LLM Calls: credit_synthesis prompt (primary + secondary model)
│   │   ├── Cross-Model Evaluation: compare results
│   │   ├── Same-Model Consistency: multiple runs
│   │   └── Output: assessment
│   │
│   ├── save_to_database
│   │   ├── Storage: MongoDB
│   │   └── Output: stored run record
│   │
│   └── evaluate_assessment
│       ├── Evaluators: Multiple evaluation frameworks
│       ├── LLM Calls: LLM-as-Judge, DeepEval
│       └── Output: evaluation scores
│
└── STATE (CreditWorkflowState)
    ├── Input: company_name, jurisdiction, ticker
    ├── Intermediate: company_info, task_plan, api_data, search_data
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

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 1 | Execution order |
| `node` | "parse_input" | Node name in graph |
| `agent_name` | **"llm_parser"** | Logged to sheets |
| `agent_class` | LLMParser | Python class |
| `prompt_id` | "company_parser" | Prompt used |
| `llm_provider` | groq | Default provider |
| `llm_model` | fast | llama-3.1-8b-instant |
| `temperature` | 0.1 | Low for consistency |

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

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 2 | Execution order |
| `node` | "validate_company" | Node name in graph |
| `agent_name` | `""` (empty - NEEDS FIX to `"supervisor"`) | Logged to sheets |
| `agent_class` | SupervisorAgent | Python class |
| `human_in_loop` | optional | May require approval |

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

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 3 | Execution order |
| `node` | "create_plan" | Node name in graph |
| `agent_name` | **"tool_supervisor"** | Logged to sheets |
| `agent_class` | ToolSupervisor | Python class |
| `prompt_id` | "tool_selection" | Prompt used |
| `llm_provider` | groq | Default provider |
| `llm_model` | primary | llama-3.3-70b-versatile |
| `temperature` | 0.1 | Low for consistency |

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
    ]
}
```

#### 3.4 fetch_api_data

**Purpose:** Execute API tools to collect structured data

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 4 | Execution order |
| `node` | "fetch_api_data" | Node name in graph |
| `agent_name` | **"api_agent"** | Logged to sheets |
| `agent_class` | APIAgent | Python class |
| `tools` | List[BaseTool] | SEC, Finnhub, Court tools |
| `parallel` | bool | Execute tools in parallel |

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

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 5 | Execution order |
| `node` | "search_web" | Node name in graph |
| `agent_name` | `""` (empty - NEEDS FIX to `"search_agent"`) | Logged to sheets |
| `agent_class` | SearchAgent | Python class |
| `tools` | List[BaseTool] | WebSearch, Tavily |

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

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 6 | Execution order |
| `node` | "synthesize" | Node name in graph |
| `agent_name` | **"llm_analyst"** | Logged to sheets (primary) |
| `agent_class` | LLMAnalystAgent | Python class |
| `prompt_id` | "credit_synthesis" | Prompt used |
| `llm_provider` | groq | Default provider |
| `llm_model` | primary | llama-3.3-70b-versatile |
| `temperature` | 0.0 | Deterministic |
| `cross_model` | bool | Compare with secondary model |
| `consistency_runs` | int | Multiple runs for consistency |

**Sub-processes and their agent_names:**
| Sub-process | agent_name | Description |
|-------------|------------|-------------|
| Primary Analysis | `llm_analyst` | Main LLM call with collected data |
| Secondary Analysis | `llm_analyst` | Different model for comparison |
| Cross-Model Evaluation | `cross_model_evaluator` | Compare primary vs secondary |
| Same-Model Consistency | `consistency_evaluator` | Run primary model 3 times |

**Output:**
```python
{
    "assessment": {
        "overall_risk_level": "low" | "medium" | "high" | "critical",
        "credit_score_estimate": 0-100,
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

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 7 | Execution order |
| `node` | "save_to_database" | Node name in graph |
| `agent_name` | `""` (empty - NEEDS FIX to `"db_writer"`) | Logged to sheets |
| `storage` | MongoDB | Database backend |
| `collection` | "runs" | MongoDB collection |

**Stored Fields:**
- Full state snapshot
- Assessment results
- Execution metrics

#### 3.8 evaluate_assessment

**Purpose:** Evaluate the quality of the assessment

| Property | Type | Description |
|----------|------|-------------|
| `step_number` | 8 | Execution order |
| `node` | "evaluate" | Node name in graph |

**Evaluators and their agent_names:**
| Evaluator | agent_name | Description |
|-----------|------------|-------------|
| ToolSelectionEvaluator | `llm_tool_selection_judge` | Tool selection accuracy |
| WorkflowEvaluator | `workflow_evaluator` | Data quality & synthesis |
| AgentEfficiencyEvaluator | `agent_efficiency_evaluator` | Agent performance |
| LLMJudgeEvaluator | `llm_judge` | LLM-as-Judge evaluation |
| UnifiedAgentEvaluator | `unified_evaluator` | Combined evaluation |
| DeepEvalEvaluator | `deepeval_evaluator` | DeepEval metrics |

**Output:**
```python
{
    "evaluation": {
        "overall_score": 0.0-1.0,
        "tool_selection_score": 0.0-1.0,
        "data_quality_score": 0.0-1.0,
        "synthesis_score": 0.0-1.0,
        "agent_metrics": {...},
        "llm_judge": {...},
        "deepeval_metrics": {...}
    }
}
```

---

### 4. AGENTS

#### 4.1 SupervisorAgent

**Location:** `src/agents/supervisor.py`

Orchestrates the workflow and makes high-level decisions.

| Property | Type | Description |
|----------|------|-------------|
| `config` | Dict | Configuration options |
| `analysis_mode` | string | "rule_based" | "llm" | "hybrid" |
| `llm_analyst` | LLMAnalystAgent | For LLM mode |

**Methods:**
- `parse_company()` - Extract company info
- `assess_credit()` - Produce credit assessment
- `synthesize()` - Combine all data

#### 4.2 ToolSupervisor

**Location:** `src/agents/tool_supervisor.py`

LLM-based tool selection agent.

| Property | Type | Description |
|----------|------|-------------|
| `model` | string | LLM model to use |
| `tool_executor` | ToolExecutor | Executes selected tools |
| `decision_log` | List | History of decisions |

**Methods:**
- `select_tools()` - LLM decides which tools to use
- `execute_plan()` - Execute the tool plan
- `synthesize_assessment()` - Final LLM synthesis

**LLM Calls:**
1. Tool Selection (prompt: "tool_selection")
2. Credit Synthesis (prompt: "credit_synthesis")

#### 4.3 APIAgent

**Location:** `src/agents/api_agent.py`

Fetches structured data from external APIs.

| Property | Type | Description |
|----------|------|-------------|
| `sec_edgar` | SECEdgarDataSource | SEC EDGAR connector |
| `finnhub` | FinnhubDataSource | Finnhub connector |
| `court_listener` | CourtListenerDataSource | Court data connector |

**Methods:**
- `fetch_all_data()` - Fetch from all sources
- `_fetch_parallel()` - Parallel execution
- `_fetch_sequential()` - Sequential execution

#### 4.4 SearchAgent

**Location:** `src/agents/search_agent.py`

Web search and content scraping.

| Property | Type | Description |
|----------|------|-------------|
| `web_search` | WebSearchDataSource | Web search |
| `tavily` | TavilySearchDataSource | Tavily AI search |
| `scraper` | WebScraper | Content scraper |

**Methods:**
- `search()` - Execute web search
- `scrape_url()` - Scrape specific URL

#### 4.5 LLMAnalystAgent

**Location:** `src/agents/llm_analyst.py`

LLM-powered credit analysis.

| Property | Type | Description |
|----------|------|-------------|
| `model` | string | LLM model |
| `temperature` | float | Sampling temperature |
| `use_langchain` | bool | Use LangChain or raw API |

**Methods:**
- `analyze()` - Analyze company data
- `_call_llm()` - Make LLM API call
- `_parse_response()` - Parse LLM response

---

### 5. TOOLS

**Location:** `src/tools/`

Tools are executable units that fetch data from external sources.

#### 5.1 BaseTool

**Location:** `src/tools/base_tool.py`

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Tool identifier |
| `description` | string | What the tool does |
| `parameters` | Dict | Expected parameters |

**Methods:**
- `execute()` - Run the tool
- `get_spec()` - Get tool specification

#### 5.2 SECTool

**Location:** `src/tools/sec_tool.py`

| Property | Type | Description |
|----------|------|-------------|
| `name` | "sec_edgar" | Tool name |
| `data_source` | SECEdgarDataSource | API connector |

**Parameters:**
- `ticker` (required): Stock ticker symbol
- `filing_type` (optional): 10-K, 10-Q, 8-K

**API Call:**
- Endpoint: SEC EDGAR API
- Returns: Company filings, financials

#### 5.3 FinnhubTool

**Location:** `src/tools/finnhub_tool.py`

| Property | Type | Description |
|----------|------|-------------|
| `name` | "finnhub" | Tool name |
| `data_source` | FinnhubDataSource | API connector |
| `api_key` | string | Finnhub API key |

**Parameters:**
- `ticker` (required): Stock ticker symbol

**API Calls:**
- Company profile
- Stock quote
- Financials
- News

#### 5.4 CourtListenerTool

**Location:** `src/tools/court_tool.py`

| Property | Type | Description |
|----------|------|-------------|
| `name` | "court_listener" | Tool name |
| `data_source` | CourtListenerDataSource | API connector |

**Parameters:**
- `company_name` (required): Company to search

**API Call:**
- Endpoint: CourtListener API
- Returns: Court cases, opinions

#### 5.5 WebSearchTool

**Location:** `src/tools/web_search_tool.py`

| Property | Type | Description |
|----------|------|-------------|
| `name` | "web_search" | Tool name |
| `data_source` | WebSearchDataSource | Search connector |

**Parameters:**
- `query` (required): Search query
- `num_results` (optional): Number of results

---

### 6. DATA SOURCES

**Location:** `src/data_sources/`

Data sources are API connectors that handle authentication and data transformation.

#### 6.1 SECEdgarDataSource

| Property | Type | Description |
|----------|------|-------------|
| `base_url` | string | SEC EDGAR API URL |
| `user_agent` | string | Required header |

**Methods:**
- `get_company_info()` - Get CIK and basic info
- `get_filings()` - Get SEC filings
- `get_financials()` - Extract financial data

#### 6.2 FinnhubDataSource

| Property | Type | Description |
|----------|------|-------------|
| `api_key` | string | Finnhub API key |
| `base_url` | string | Finnhub API URL |

**Methods:**
- `get_profile()` - Company profile
- `get_quote()` - Stock quote
- `get_financials()` - Financial statements
- `get_news()` - Company news

#### 6.3 CourtListenerDataSource

| Property | Type | Description |
|----------|------|-------------|
| `base_url` | string | CourtListener API URL |

**Methods:**
- `search_opinions()` - Search court opinions
- `search_dockets()` - Search court dockets

#### 6.4 TavilySearchDataSource

| Property | Type | Description |
|----------|------|-------------|
| `api_key` | string | Tavily API key |

**Methods:**
- `search()` - AI-powered search
- `company_search()` - Company-specific search

#### 6.5 WebScraper

| Property | Type | Description |
|----------|------|-------------|
| `timeout` | int | Request timeout |
| `headers` | Dict | HTTP headers |

**Methods:**
- `scrape_url()` - Scrape single URL
- `scrape_company_website()` - Scrape company site

---

### 7. LLM CALLS

**Location:** Various agents

LLM calls are API requests to language model providers.

#### 7.1 LLM Call Structure

| Property | Type | Description |
|----------|------|-------------|
| `prompt_id` | string | Prompt identifier |
| `provider` | string | groq | openai | anthropic |
| `model` | string | Model identifier |
| `temperature` | float | 0.0 - 1.0 |
| `max_tokens` | int | Maximum response tokens |
| `system_prompt` | string | System instructions |
| `user_prompt` | string | User message |

#### 7.2 Available Prompts

| Prompt ID | Category | Purpose |
|-----------|----------|---------|
| `company_parser` | input | Parse company name |
| `tool_selection` | planning | Select tools to use |
| `tool_selection_evaluation` | evaluation | Evaluate tool selection |
| `credit_synthesis` | synthesis | Produce credit assessment |
| `credit_analysis` | analysis | Full credit analysis |
| `validation` | validation | Validate assessment |

#### 7.3 LLM Providers & Models

**Groq (Default):**
| Alias | Model ID | Use Case |
|-------|----------|----------|
| primary | llama-3.3-70b-versatile | Complex reasoning |
| fast | llama-3.1-8b-instant | Simple parsing |
| balanced | llama3-70b-8192 | General use |

**OpenAI:**
| Alias | Model ID | Use Case |
|-------|----------|----------|
| primary | gpt-4o-mini | Complex reasoning |
| fast | gpt-4o-mini | Quick tasks |

**Anthropic:**
| Alias | Model ID | Use Case |
|-------|----------|----------|
| primary | claude-3-5-sonnet-20241022 | Complex reasoning |
| fast | claude-3-haiku-20240307 | Quick tasks |

---

### 8. EVALUATORS

**Location:** `src/evaluation/`

Evaluators measure the quality of workflow outputs.

#### 8.1 ToolSelectionEvaluator

| Metric | Range | Description |
|--------|-------|-------------|
| precision | 0-1 | Correct tools / Selected tools |
| recall | 0-1 | Correct tools / Expected tools |
| f1_score | 0-1 | Harmonic mean |

#### 8.2 WorkflowEvaluator

| Metric | Range | Description |
|--------|-------|-------------|
| tool_selection_score | 0-1 | Tool selection quality |
| data_quality_score | 0-1 | Data completeness |
| synthesis_score | 0-1 | Synthesis quality |
| overall_score | 0-1 | Combined score |

#### 8.3 AgentEfficiencyEvaluator

| Metric | Range | Description |
|--------|-------|-------------|
| intent_correctness | 0-1 | Understood the task |
| plan_quality | 0-1 | Plan was good |
| tool_choice_correctness | 0-1 | Right tools chosen |
| tool_completeness | 0-1 | All needed tools used |
| trajectory_match | 0-1 | Execution matched plan |
| final_answer_quality | 0-1 | Output quality |

#### 8.4 LLMJudgeEvaluator

| Metric | Range | Description |
|--------|-------|-------------|
| accuracy_score | 0-1 | Factual accuracy |
| completeness_score | 0-1 | Information completeness |
| consistency_score | 0-1 | Internal consistency |
| actionability_score | 0-1 | Actionable recommendations |
| data_utilization_score | 0-1 | Data usage quality |

#### 8.5 DeepEvalEvaluator

| Metric | Range | Description |
|--------|-------|-------------|
| answer_relevancy | 0-1 | Answer is relevant |
| faithfulness | 0-1 | Grounded in context |
| hallucination | 0-1 | Contains hallucinations (lower=better) |
| contextual_relevancy | 0-1 | Context is relevant |
| bias | 0-1 | Contains bias (lower=better) |

---

### 9. LOGGING

**Location:** `src/run_logging/`

#### 9.1 WorkflowLogger

Central logger for all workflow events.

**Methods:**
- `start_run()` - Initialize run
- `log_step()` - Log node execution
- `log_llm_call()` - Log LLM API call
- `log_tool_call()` - Log tool execution
- `complete_run()` - Finalize run

#### 9.2 SheetsLogger

Logs to Google Sheets for analysis.

**Sheets:**
| Sheet | Content |
|-------|---------|
| runs | Run summaries |
| step_logs | Step-by-step execution |
| llm_calls | LLM API calls |
| tool_calls | Tool executions |
| assessments | Credit assessments |
| evaluations | Evaluation results |
| log_tests | Verification of logging |
| ... | 22 sheets total |

---

### 10. STORAGE

**Location:** `src/storage/`

#### 10.1 MongoDB

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
│ 1. PARSE_INPUT                                                  │
│    ├── LLM Call: company_parser                                 │
│    │   ├── Provider: groq                                       │
│    │   ├── Model: fast (llama-3.1-8b-instant)                  │
│    │   └── Output: {is_public, ticker, industry, jurisdiction} │
│    └── Generate: run_id                                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. VALIDATE_COMPANY                                             │
│    ├── Check: company_info validity                             │
│    ├── Optional: Human approval                                 │
│    └── Decision: continue or stop                               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CREATE_PLAN                                                  │
│    ├── LLM Call: tool_selection                                 │
│    │   ├── Provider: groq                                       │
│    │   ├── Model: primary (llama-3.3-70b-versatile)            │
│    │   └── Output: task_plan [{tool, params, reason}, ...]     │
│    └── Decision: which tools to execute                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. FETCH_API_DATA                                               │
│    ├── Tool Execution (parallel):                               │
│    │   ├── SECTool → SEC EDGAR API → filings, financials       │
│    │   ├── FinnhubTool → Finnhub API → profile, quote, news    │
│    │   └── CourtTool → CourtListener API → cases, opinions     │
│    └── Output: api_data {sec_edgar, finnhub, court_listener}   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. SEARCH_WEB                                                   │
│    ├── Tool Execution:                                          │
│    │   ├── WebSearchTool → web search results                  │
│    │   ├── TavilySearch → AI-powered search                    │
│    │   └── WebScraper → company website content                │
│    └── Output: search_data {news, web_results, scraped}        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SYNTHESIZE                                                   │
│    ├── Primary LLM Call: credit_synthesis                       │
│    │   ├── Provider: groq                                       │
│    │   ├── Model: primary (llama-3.3-70b-versatile)            │
│    │   ├── Temperature: 0.0                                     │
│    │   └── Input: api_data + search_data                        │
│    │                                                            │
│    ├── Secondary LLM Call (cross-model):                        │
│    │   ├── Different model for comparison                       │
│    │   └── Cross-model evaluation logged                        │
│    │                                                            │
│    ├── Consistency Runs (same-model):                           │
│    │   ├── Run primary model 3x                                 │
│    │   └── Calculate consistency metrics                        │
│    │                                                            │
│    └── Output: assessment {risk_level, credit_score, ...}      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. SAVE_TO_DATABASE                                             │
│    └── MongoDB: store full run record                           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. EVALUATE_ASSESSMENT                                          │
│    ├── ToolSelectionEvaluator → precision, recall, f1          │
│    ├── WorkflowEvaluator → data_quality, synthesis_score       │
│    ├── AgentEfficiencyEvaluator → intent, plan, trajectory     │
│    ├── LLMJudgeEvaluator → accuracy, completeness, consistency │
│    ├── UnifiedAgentEvaluator → combined metrics                │
│    ├── DeepEvalEvaluator → faithfulness, hallucination         │
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
│   ├── supervisor.py      # SupervisorAgent
│   ├── tool_supervisor.py # ToolSupervisor (LLM tool selection)
│   ├── api_agent.py       # APIAgent
│   ├── search_agent.py    # SearchAgent
│   ├── llm_analyst.py     # LLMAnalystAgent
│   └── llm_parser.py      # LLM company parser
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
│   ├── tool_selection_evaluator.py
│   ├── workflow_evaluator.py
│   ├── agent_efficiency_evaluator.py
│   ├── llm_judge_evaluator.py
│   ├── unified_agent_evaluator.py
│   └── deepeval_evaluator.py
│
├── run_logging/
│   ├── workflow_logger.py # Main logger
│   ├── sheets_logger.py   # Google Sheets logger
│   └── langgraph_logger.py
│
└── storage/
    └── mongodb.py         # MongoDB storage
```

---

## Summary

The Credit Intelligence system is a hierarchical multi-agent workflow:

1. **Workflow** orchestrates the entire process
2. **Nodes** are discrete steps in the workflow
3. **Agents** execute within nodes
4. **Tools** are used by agents to fetch data
5. **Data Sources** connect to external APIs
6. **LLM Calls** power analysis and decision-making
7. **Evaluators** measure output quality
8. **Loggers** record everything for analysis

Each entity has well-defined inputs, outputs, and relationships, enabling comprehensive tracing and evaluation of the credit assessment process.
