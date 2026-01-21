# Credit Intelligence - System Architecture

**Version:** 2.0
**Last Updated:** January 2025
**System Version:** v129 (Heroku)

---

## Overview

Credit Intelligence is an autonomous agentic workflow system for B2B credit assessment. It leverages LangGraph for orchestration, multiple LLM providers (Groq, OpenAI, Anthropic), external financial APIs, and a comprehensive evaluation framework with LLM-as-judge node scoring.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Workflow Engine](#workflow-engine)
3. [Agent System](#agent-system)
4. [Tool Framework](#tool-framework)
5. [External Data Sources](#external-data-sources)
6. [Storage Layer](#storage-layer)
7. [Logging Infrastructure](#logging-infrastructure)
8. [Evaluation Framework](#evaluation-framework)
9. [Configuration System](#configuration-system)
10. [API & Frontend](#api--frontend)
11. [Data Flow](#data-flow)
12. [Deployment](#deployment)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React/TypeScript)                     │
│                              /frontend/                                      │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ WebSocket / REST
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND API                                │
│                         /backend/api/main.py                                 │
│                    (REST Endpoints + WebSocket Streaming)                    │
└──────────┬──────────────────────┬──────────────────────┬────────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────────┐
│  WORKFLOW ENGINE │   │  CONFIG SYSTEM   │   │      STORAGE LAYER           │
│  /src/agents/    │   │  /src/config/    │   │      /src/storage/           │
│                  │   │                  │   │                              │
│ • graph.py       │   │ • prompts.py     │   │ • postgres.py (27 tables)    │
│ • workflow.py    │   │ • langchain_llm  │   │ • mongodb.py                 │
│ • 7 Agent Types  │   │ • settings.yaml  │   │ • Google Sheets integration  │
└────────┬─────────┘   └──────────────────┘   └──────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TOOL FRAMEWORK                                     │
│                            /src/tools/                                        │
│                                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ SEC EDGAR   │  │  Finnhub    │  │CourtListener│  │ Web Search  │          │
│  │   Tool      │  │   Tool      │  │   Tool      │  │   Tool      │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                                 │
│                         /src/data_sources/                                    │
│                                                                               │
│    SEC EDGAR API    Finnhub.io API    CourtListener API    DuckDuckGo/Tavily │
│    (US Financials)  (Market Data)     (Legal Records)      (Web/News)        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Core Components Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| Workflow Engine | `/src/agents/` | LangGraph-based orchestration |
| Agent System | `/src/agents/*.py` | 7 specialized AI agents |
| Tool Framework | `/src/tools/` | 5 data collection tools |
| Data Sources | `/src/data_sources/` | 8+ external API integrations |
| Storage | `/src/storage/` | PostgreSQL (27 tables), MongoDB, Sheets |
| Logging | `/src/run_logging/` | Comprehensive audit trail |
| Evaluation | `/src/evaluation/` | 8 evaluators + LLM judge node scoring |
| Configuration | `/src/config/` | Prompts, LLM settings, node definitions |
| Backend API | `/backend/api/` | FastAPI REST + WebSocket |
| Frontend | `/frontend/` | React UI with real-time updates |

---

## Workflow Engine

### LangGraph Workflow with PUBLIC/PRIVATE Routing

**File:** `src/agents/graph.py`

The workflow uses LangGraph's StateGraph for orchestration with **conditional routing based on company type** (PUBLIC vs PRIVATE).

```
┌─────────────┐
│ parse_input │  ← Parse company name, identify type/ticker
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ validate_company │  ← Validate and enrich company info
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│ create_plan │  ← Select tools based on company type
└──────┬──────┘
       │
       ├────────────────────────────────┐
       │ PUBLIC                         │ PRIVATE
       ▼                                ▼
┌────────────────┐               ┌─────────────────────┐
│ fetch_api_data │               │ search_web_enhanced │
│ (SEC, Finnhub, │               │ (Deep web search    │
│  CourtListener)│               │  for private cos)   │
└──────┬─────────┘               └──────────┬──────────┘
       │                                    │
       ▼                                    │
┌────────────┐                              │
│ search_web │                              │
└──────┬─────┘                              │
       │                                    │
       └──────────────┬─────────────────────┘
                      ▼
               ┌────────────┐
               │ synthesize │  ← LLM credit analysis
               └──────┬─────┘
                      │
                      ▼
            ┌─────────────────┐
            │ save_to_database│
            └──────┬──────────┘
                   │
                   ▼
               ┌──────────┐
               │ evaluate │  ← Node scoring with LLM judge
               └──────────┘
```

### Conditional Routing Logic

```python
def route_after_plan_by_company_type(state: CreditWorkflowState) -> str:
    """Route based on PUBLIC vs PRIVATE company type."""
    company_info = state.get("company_info", {})
    is_public = company_info.get("is_public_company", False)

    if is_public:
        return "PUBLIC"   # → fetch_api_data → search_web → synthesize
    else:
        return "PRIVATE"  # → search_web_enhanced → synthesize
```

### Workflow State

```python
class CreditWorkflowState(TypedDict):
    # Input
    company_name: str
    jurisdiction: Optional[str]
    ticker: Optional[str]

    # Processing
    company_info: Dict[str, Any]      # Parsed company details
    task_plan: Dict[str, Any]         # Execution plan with selected tools
    api_data: Dict[str, Any]          # SEC, Finnhub, CourtListener data
    search_data: Dict[str, Any]       # Web search results

    # Output
    assessment: Dict[str, Any]        # Credit assessment result
    evaluation: Dict[str, Any]        # Quality evaluation scores
    node_scores: List[Dict]           # Per-node LLM judge scores
    errors: List[str]
    status: str
```

---

## Agent System

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                          │
│                    (Orchestration)                           │
└────────────────────────────┬────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ LLM Parser   │     │Tool Supervisor│    │ LLM Analyst  │
│ Agent        │     │ Agent         │    │ Agent        │
└──────────────┘     └───────┬───────┘    └──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │API Agent │  │Search    │  │Workflow  │
        │          │  │Agent     │  │Evaluator │
        └──────────┘  └──────────┘  └──────────┘
```

### Agent Descriptions

| Agent | File | `agent_name` | Purpose |
|-------|------|--------------|---------|
| **SupervisorAgent** | `supervisor.py` | `supervisor` | Master orchestrator, workflow coordination |
| **LLMParserAgent** | `llm_parser.py` | `llm_parser` | Parse company input, identify type/ticker/industry |
| **ToolSupervisorAgent** | `tool_supervisor.py` | `tool_supervisor` | Dynamic tool selection based on company type |
| **APIAgent** | `api_agent.py` | `api_agent` | Fetch structured data from external APIs |
| **SearchAgent** | `search_agent.py` | `search_agent` | Web search and news gathering |
| **LLMAnalystAgent** | `llm_analyst.py` | `llm_analyst` | Credit analysis synthesis with LLM |
| **WorkflowEvaluator** | `workflow_evaluator.py` | `workflow_evaluator` | All evaluation and scoring tasks |

### Credit Assessment Output

```python
@dataclass
class CreditAssessment:
    overall_risk_level: str        # low, medium, high, critical
    credit_score_estimate: int     # 0-100
    confidence: float              # 0.0-1.0

    # Component Assessments
    ability_to_pay: str
    willingness_to_pay: str
    fraud_risk: str

    # Analysis Sections
    financial_summary: str
    legal_summary: str
    market_summary: str
    news_summary: str

    # Factors
    risk_factors: List[str]
    positive_factors: List[str]
    recommendations: List[str]

    # Metadata
    analysis_method: str           # rule_based, llm, hybrid
    llm_model_used: Optional[str]
    data_sources_used: List[str]
```

---

## Tool Framework

### Tool Architecture

**Base Class:** `src/tools/base_tool.py`

```python
class BaseTool(ABC):
    @abstractmethod
    def _get_name(self) -> str: ...

    @abstractmethod
    def _get_description(self) -> str: ...

    @abstractmethod
    def _get_when_to_use(self) -> str: ...

    @abstractmethod
    def _execute(self, **kwargs) -> Dict[str, Any]: ...
```

### Available Tools

| Tool Name | File | Data Source | Use Case |
|-----------|------|-------------|----------|
| `fetch_sec_edgar` | `sec_tool.py` | SEC EDGAR | US public company financials (10-K, 10-Q) |
| `fetch_finnhub` | `finnhub_tool.py` | Finnhub.io | Stock/market data, company profiles |
| `fetch_court_listener` | `court_tool.py` | CourtListener | Legal records, bankruptcies, court cases |
| `web_search` | `web_search_tool.py` | DuckDuckGo | General company info and news |
| `web_search_enhanced` | `web_search_tool.py` | Tavily | Deep web search for private companies |

### Tool Routing by Company Type

```python
TOOL_ROUTING = {
    "PUBLIC": [
        "fetch_sec_edgar",      # SEC filings
        "fetch_finnhub",        # Market data
        "fetch_court_listener", # Legal records
        "web_search"            # News and web info
    ],
    "PRIVATE": [
        "web_search_enhanced",  # Deep web search
        "fetch_court_listener"  # Legal records
    ],
}
```

### Node Task Definitions (for LLM Judge Scoring)

```python
NODE_TASK_DEFINITIONS = {
    # === AGENTS (high-level nodes) ===
    "parse_input": {
        "task": "Parse company input and identify company type, ticker, industry",
        "success_criteria": "Correctly identified company type (public/private) and extracted relevant metadata",
        "agent_name": "llm_parser",
        "node_type": "agent",
    },
    "create_plan": {
        "task": "Create execution plan with appropriate tools for company type",
        "success_criteria": "Selected tools appropriate for company type (PUBLIC: SEC/Finnhub APIs + web search, PRIVATE: web search only)",
        "agent_name": "tool_supervisor",
        "node_type": "agent",
    },
    "synthesize": {
        "task": "Analyze all collected data and produce credit assessment",
        "success_criteria": "Produced valid risk_level (low/medium/high/critical), credit_score in range 0-100, confidence score 0-1",
        "agent_name": "llm_analyst",
        "node_type": "agent",
    },

    # === INDIVIDUAL TOOLS ===
    "fetch_sec_edgar": {
        "task": "Fetch SEC EDGAR filings for the company (10-K, 10-Q, 8-K forms)",
        "success_criteria": "Retrieved SEC filings if company is public, or correctly identified company is not in SEC database",
        "agent_name": "api_agent",
        "node_type": "tool",
        "parent_agent": "fetch_api_data",
    },
    "fetch_finnhub": {
        "task": "Fetch market data from Finnhub (stock price, company profile, metrics)",
        "success_criteria": "Retrieved market data with stock price and company profile, or correctly handled non-public company",
        "agent_name": "api_agent",
        "node_type": "tool",
        "parent_agent": "fetch_api_data",
    },
    "fetch_court_listener": {
        "task": "Search for legal records and court cases involving the company",
        "success_criteria": "Searched court records and returned relevant cases, or confirmed no cases found",
        "agent_name": "api_agent",
        "node_type": "tool",
        "parent_agent": "fetch_api_data",
    },
    "web_search": {
        "task": "Search the web for company news, articles, and general information",
        "success_criteria": "Retrieved relevant news articles and web content about the company",
        "agent_name": "search_agent",
        "node_type": "tool",
        "parent_agent": "search_web",
    },
    "web_search_enhanced": {
        "task": "Perform enhanced web search with multiple query strategies for private companies",
        "success_criteria": "Retrieved comprehensive web data using multiple search strategies",
        "agent_name": "search_agent",
        "node_type": "tool",
        "parent_agent": "search_web_enhanced",
    },
}
```

---

## External Data Sources

### Data Source Architecture

**Base Class:** `src/data_sources/base.py`

All data sources implement:
- Rate limiting with configurable limits
- Retry logic with exponential backoff
- Response caching (configurable TTL)
- Comprehensive error handling

### Integrated APIs

| Source | File | Rate Limit | Auth | Data Types |
|--------|------|------------|------|------------|
| **SEC EDGAR** | `sec_edgar.py` | 10 req/sec | User-Agent | 10-K, 10-Q, 8-K filings, financials |
| **Finnhub** | `finnhub.py` | 60 req/min | API Key | Stock quotes, profiles, financials |
| **CourtListener** | `court_listener.py` | 5000 req/hr | API Key (optional) | Court records, bankruptcies |
| **DuckDuckGo** | `web_search.py` | Reasonable | None | Web search results |
| **Tavily** | `tavily_search.py` | Per plan | API Key | AI-optimized search |
| **OpenCorporates** | `opencorporates.py` | Per plan | API Key | Company registrations |
| **OpenSanctions** | `opensanctions.py` | Per plan | API Key | Sanctions data |

### Data Source Result

```python
@dataclass
class DataSourceResult:
    source: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str]
    execution_time_ms: float
    records_found: int
```

---

## Storage Layer

### Multi-Storage Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                             │
├──────────────────┬──────────────────┬───────────────────────┤
│   PostgreSQL     │    MongoDB       │   Google Sheets       │
│   (Primary)      │    (Flexible)    │   (Monitoring)        │
├──────────────────┼──────────────────┼───────────────────────┤
│ 27 Tables        │ Document Store   │ Human-readable logs   │
│ Monthly Partitions│ Flexible Schema │ Real-time dashboards  │
│ Full SQL Support │ GridFS for large │ Team collaboration    │
│ ACID Compliance  │ Quick prototyping│                       │
└──────────────────┴──────────────────┴───────────────────────┘
```

### PostgreSQL Schema (27 Tables)

**Naming Convention:**
- `wf_*` - Workflow execution tables (10)
- `eval_*` - Evaluation result tables (10)
- `lg_*` - LangGraph framework tables (2)
- `meta_*` - Metadata tables (2)

#### Key Tables

| Table | Columns | Purpose |
|-------|---------|---------|
| `wf_runs` | 25 | Run summaries with performance scores |
| `wf_llm_calls` | 24 | LLM API call logs with token/cost tracking |
| `wf_tool_calls` | 19 | Tool execution logs |
| `wf_assessments` | 22 | Credit assessment results |
| `wf_plans` | 23 | Execution plans with task breakdown |
| `wf_data_sources` | 16 | Data source fetch results |
| `wf_state_dumps` | 28 | Complete state snapshots |
| `eval_node_scoring` | 16 | **LLM judge node quality scores** |
| `eval_llm_judge` | 28 | Overall quality evaluation |
| `eval_coalition` | 22 | Multi-evaluator consensus |
| `eval_agent_metrics` | 27 | Agent efficiency metrics |
| `lg_events` | 22 | LangGraph framework events |

#### wf_runs Table Schema

```sql
CREATE TABLE wf_runs (
    id BIGSERIAL,
    run_id VARCHAR(64) NOT NULL,
    company_name VARCHAR(255),
    node VARCHAR(100),
    agent_name VARCHAR(100),
    master_agent VARCHAR(100),
    model VARCHAR(100),
    temperature DECIMAL(3,2),
    status VARCHAR(50),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    risk_level VARCHAR(50),
    credit_score INTEGER,
    confidence DECIMAL(5,4),
    total_time_ms DECIMAL(15,3),
    total_steps INTEGER,
    total_llm_calls INTEGER,
    tools_used JSONB,
    evaluation_score DECIMAL(5,4),
    workflow_correct BOOLEAN,
    output_correct BOOLEAN,
    tool_overall_score DECIMAL(5,4),      -- NEW: Tool scoring
    agent_overall_score DECIMAL(5,4),     -- NEW: Agent scoring
    workflow_overall_score DECIMAL(5,4),  -- NEW: Workflow scoring
    timestamp TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);
```

#### eval_node_scoring Table Schema

```sql
CREATE TABLE eval_node_scoring (
    id BIGSERIAL,
    run_id VARCHAR(64) NOT NULL,
    company_name VARCHAR(255),
    node VARCHAR(100),              -- Node name (e.g., "synthesize", "fetch_sec_edgar")
    node_type VARCHAR(50),          -- "agent" or "tool"
    agent_name VARCHAR(100),
    master_agent VARCHAR(100),
    step_number INTEGER,
    task_description TEXT,          -- What the node should do
    task_completed BOOLEAN,         -- Did it complete successfully?
    quality_score DECIMAL(5,4),     -- LLM judge score (0.0-1.0)
    quality_reasoning TEXT,         -- LLM judge explanation
    input_summary TEXT,
    output_summary TEXT,
    judge_model VARCHAR(100),       -- Model used for judging
    timestamp TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);
```

---

## Logging Infrastructure

### Logging Architecture

**Directory:** `/src/run_logging/`

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW LOGGER                           │
│                 workflow_logger.py                           │
└────────────────────────────┬────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PostgreSQL  │     │   MongoDB    │     │Google Sheets │
│   Logger     │     │   Logger     │     │   Logger     │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Logger Components

| Logger | File | Destination | Features |
|--------|------|-------------|----------|
| **WorkflowLogger** | `workflow_logger.py` | All backends | Central orchestration, dual-write |
| **PostgresLogger** | `postgres_logger.py` | PostgreSQL | Partitioned tables, JSONB support |
| **SheetsLogger** | `sheets_logger.py` | Google Sheets | Thread-safe, concurrent logging |
| **RunLogger** | `run_logger.py` | MongoDB | Document storage |
| **LangGraphLogger** | `langgraph_logger.py` | Framework events | Step tracking |
| **MetricsCollector** | `metrics_collector.py` | All | Token/cost tracking |

### Logged Events

```python
# Every workflow step logs:
{
    "run_id": "uuid",
    "company_name": "Apple Inc",
    "node": "synthesize",
    "node_type": "agent",
    "agent_name": "llm_analyst",
    "master_agent": "supervisor",
    "step_number": 5,
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "prompt": "...",
    "response": "...",
    "prompt_tokens": 1500,
    "completion_tokens": 800,
    "total_cost": 0.0023,
    "execution_time_ms": 3200,
    "status": "success",
    "timestamp": "2025-01-21T10:30:00Z"
}
```

---

## Evaluation Framework

### Evaluation Architecture

**Directory:** `/src/evaluation/` (20 files)

```
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION BRAIN                           │
│                 evaluation_brain.py                          │
└────────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │           │            │            │           │
    ▼           ▼            ▼            ▼           ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│  Tool  │ │Workflow│ │LLM Judge │ │Coalition│ │Node      │
│Selection│ │Evaluator│ │Evaluator│ │Evaluator│ │Scoring   │
└────────┘ └────────┘ └──────────┘ └─────────┘ └──────────┘
```

### Evaluator Types

| Evaluator | File | Metrics | Purpose |
|-----------|------|---------|---------|
| **ToolSelectionEvaluator** | `tool_selection_evaluator.py` | Precision, Recall, F1 | Did agent select correct tools? |
| **WorkflowEvaluator** | `workflow_evaluator.py` | Composite score | End-to-end workflow quality |
| **LLMJudgeEvaluator** | `llm_judge_evaluator.py` | 5 dimensions | LLM-as-a-judge scoring |
| **AgentEfficiencyEvaluator** | `agent_efficiency_evaluator.py` | 6 metrics | Agent performance |
| **CoalitionEvaluator** | `coalition_evaluator.py` | Agreement score | Multi-evaluator consensus |
| **ConsistencyScorer** | `consistency_scorer.py` | Variance metrics | Cross-run stability |
| **UnifiedAgentEvaluator** | `unified_agent_evaluator.py` | Combined | All metrics unified |
| **NodeScoringEvaluator** | (in graph.py) | Per-node quality | LLM judge for each node |

### LLM Judge Node Scoring

Each workflow node (both agents AND individual tools) is scored by an LLM judge:

```python
def evaluate_all_nodes_with_llm_judge(state: CreditWorkflowState) -> List[Dict]:
    """Evaluate all nodes using LLM-as-judge pattern."""
    node_scores = []

    for node_name, definition in NODE_TASK_DEFINITIONS.items():
        # Get node input/output from state
        node_input = get_node_input(state, node_name)
        node_output = get_node_output(state, node_name)

        # Call LLM judge
        score_result = llm_judge.evaluate(
            task=definition["task"],
            success_criteria=definition["success_criteria"],
            input_data=node_input,
            output_data=node_output
        )

        node_scores.append({
            "node": node_name,
            "node_type": definition["node_type"],
            "agent_name": definition["agent_name"],
            "task_completed": score_result["completed"],
            "quality_score": score_result["score"],  # 0.0-1.0
            "quality_reasoning": score_result["reasoning"]
        })

    return node_scores
```

### LLM Judge Dimensions (for synthesize node)

```python
class LLMJudgeResult:
    accuracy_score: float         # Risk assessment reasonableness (0-1)
    completeness_score: float     # Covers all relevant factors (0-1)
    consistency_score: float      # Reasoning aligns with conclusion (0-1)
    actionability_score: float    # Recommendations are actionable (0-1)
    data_utilization_score: float # Data well-utilized (0-1)
    overall_score: float          # Weighted average (0-1)
```

---

## Configuration System

### Configuration Files

```
config/
├── config.yaml       # Data sources, agent config
├── models.yaml       # Multi-model evaluation settings
└── settings.yaml     # Application settings, credentials

src/config/
├── prompts.py           # Centralized prompt management
├── langchain_llm.py     # LLM factory (Groq/OpenAI/Anthropic)
├── node_definitions.py  # Node metadata for logging
├── cost_tracker.py      # Token cost calculation
├── output_parsers.py    # LLM output parsing
└── output_schemas.py    # Pydantic validation schemas
```

### LLM Provider Configuration

**File:** `src/config/langchain_llm.py`

```python
# Supported providers
PROVIDERS = ["groq", "openai", "anthropic"]

# Default provider (configurable via LLM_PROVIDER env var)
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Model aliases
GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
}

OPENAI_MODELS = {
    "primary": "gpt-4o-mini",
    "fast": "gpt-4o-mini",
}

ANTHROPIC_MODELS = {
    "primary": "claude-3-5-sonnet-20241022",
    "fast": "claude-3-haiku-20240307",
}

def get_chat_llm(model="primary", temperature=0.1, provider=None):
    """Get LLM instance with automatic provider selection."""
    provider = provider or DEFAULT_PROVIDER
    # Normalize model alias and return appropriate LLM
    ...
```

### Prompt Management

**File:** `src/config/prompts.py`

```python
DEFAULT_PROMPTS = {
    "company_parser": "Parse the company name and identify...",
    "tool_selection": "Select appropriate tools for data collection...",
    "credit_analysis": "Analyze the data and produce credit assessment...",
    "credit_synthesis": "Synthesize into final assessment...",
    "node_scoring_judge": "Evaluate the quality of node execution...",
    "validation": "Validate the assessment against criteria...",
}

# Key prompt: node_scoring_judge
# Success criteria use credit_score range 0-100 (not 300-850)
# create_plan success: "Selected tools appropriate for company type"
```

---

## API & Frontend

### Backend API

**File:** `/backend/api/main.py`

```python
# FastAPI application
app = FastAPI(title="Credit Intelligence API")

# Endpoints
@app.post("/analyze")              # Start credit analysis
@app.get("/runs/{run_id}")         # Get run details
@app.get("/runs")                  # List all runs
@app.websocket("/ws/analyze")      # Real-time streaming
@app.get("/prompts")               # Get all prompts
@app.put("/prompts/{prompt_id}")   # Update prompt
@app.get("/erd")                   # ERD visualization (static)
```

### WebSocket Streaming

```javascript
// Frontend connects via WebSocket
const ws = new WebSocket("ws://localhost:8000/ws/analyze");

ws.send(JSON.stringify({
    company_name: "Apple Inc",
    jurisdiction: "US"
}));

// Receives real-time updates for each node
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    // { node: "parse_input", status: "completed", ... }
};
```

### Static Assets

- **ERD Visualization:** `/backend/static/erd.html` - Interactive D3.js ERD

---

## Data Flow

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. USER REQUEST                                                              │
│    Company: "Apple Inc"                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. PARSE INPUT (agent_name: llm_parser)                                      │
│    → Identify: PUBLIC company, ticker: AAPL, industry: Technology           │
│    → Log to: PostgreSQL, MongoDB, Sheets                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CREATE PLAN (agent_name: tool_supervisor)                                 │
│    → Select tools: [SEC EDGAR, Finnhub, CourtListener, WebSearch]           │
│    → Route: PUBLIC path (company is public)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. FETCH API DATA (agent_name: api_agent) - Parallel Execution               │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│    │ SEC EDGAR   │  │  Finnhub    │  │CourtListener│                        │
│    │ $394B rev   │  │ $178 price  │  │ 0 cases     │                        │
│    └─────────────┘  └─────────────┘  └─────────────┘                        │
│    → Each tool scored by LLM judge                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. SEARCH WEB (agent_name: search_agent)                                     │
│    → News: 15 articles                                                       │
│    → Sentiment: Positive                                                     │
│    → Key findings: Strong Q4, AI investments                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. SYNTHESIZE (agent_name: llm_analyst)                                      │
│    → Risk Level: LOW                                                         │
│    → Credit Score: 85/100                                                    │
│    → Confidence: 0.92                                                        │
│    → Reasoning: Strong financials, market leader, no legal issues           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. EVALUATE (agent_name: workflow_evaluator)                                 │
│    → Node Scores (LLM Judge):                                               │
│       • parse_input: 0.95                                                   │
│       • create_plan: 0.90                                                   │
│       • fetch_sec_edgar: 0.92                                               │
│       • fetch_finnhub: 0.88                                                 │
│       • synthesize: 0.87                                                    │
│    → Tool Selection F1: 1.0                                                  │
│    → Coalition Agreement: 0.92                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. STORE & RETURN                                                            │
│    → PostgreSQL: wf_runs, wf_assessments, eval_node_scoring (27 tables)     │
│    → Google Sheets: Real-time dashboard update                              │
│    → Response: Credit assessment with full audit trail                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### PRIVATE Company Flow (Alternative)

```
Company: "Private Tech Solutions LLC"
    │
    ▼
parse_input → identify as PRIVATE (no ticker)
    │
    ▼
create_plan → select tools: [web_search_enhanced, court_listener]
    │
    ▼
search_web_enhanced (skips fetch_api_data)
    │   → Deep web search with multiple query strategies
    │   → Financial performance queries
    │   → Legal issues queries
    │   → Industry competitor queries
    │
    ▼
synthesize → produce assessment from web data only
    │
    ▼
evaluate → score all nodes with LLM judge
```

---

## Deployment

### Environment Variables

```bash
# LLM Providers
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openai  # groq, openai, or anthropic

# Data Sources
FINNHUB_API_KEY=...
COURTLISTENER_API_KEY=...
TAVILY_API_KEY=...

# Storage
HEROKU_POSTGRES_URL=postgres://...
MONGODB_URI=mongodb+srv://...
GOOGLE_SHEETS_CREDENTIALS_JSON=...
GOOGLE_SPREADSHEET_ID=...

# Observability
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

### Heroku Deployment

```bash
# Deploy
git push heroku main

# Scale
heroku ps:scale web=1

# Logs
heroku logs --tail

# Current version: v129
# URL: https://credit-intelligence-096cc99c71eb.herokuapp.com/
```

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Python Files | 100+ |
| Agent Types | 7 |
| Tools | 5 |
| External APIs | 8+ |
| Evaluators | 8 |
| PostgreSQL Tables | 27 |
| Google Sheets Tabs | 16 |
| Configuration Files | 6 |
| Logging Destinations | 4 (PostgreSQL, MongoDB, Sheets, LangSmith) |

---

## File Structure

```
credit_intelligence/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   └── static/
│       └── erd.html             # Interactive ERD visualization
│
├── frontend/                     # React/TypeScript UI
│
├── src/
│   ├── agents/
│   │   ├── graph.py             # LangGraph workflow + node scoring
│   │   ├── workflow.py          # Alternative workflow implementation
│   │   ├── supervisor.py        # SupervisorAgent
│   │   ├── tool_supervisor.py   # ToolSupervisorAgent
│   │   ├── api_agent.py         # APIAgent
│   │   ├── search_agent.py      # SearchAgent
│   │   ├── llm_analyst.py       # LLMAnalystAgent
│   │   └── llm_parser.py        # LLMParserAgent
│   │
│   ├── tools/
│   │   ├── base_tool.py         # BaseTool abstract class
│   │   ├── tool_executor.py     # Tool execution orchestration
│   │   ├── sec_tool.py          # SEC EDGAR tool
│   │   ├── finnhub_tool.py      # Finnhub tool
│   │   ├── court_tool.py        # CourtListener tool
│   │   └── web_search_tool.py   # Web search tools
│   │
│   ├── data_sources/
│   │   ├── base.py              # BaseDataSource with rate limiting
│   │   ├── sec_edgar.py         # SEC EDGAR API
│   │   ├── finnhub.py           # Finnhub API
│   │   ├── court_listener.py    # CourtListener API
│   │   ├── tavily_search.py     # Tavily search
│   │   └── web_scraper.py       # Web scraper
│   │
│   ├── config/
│   │   ├── prompts.py           # Centralized prompt management
│   │   ├── langchain_llm.py     # LLM factory (multi-provider)
│   │   ├── node_definitions.py  # Node metadata
│   │   ├── output_parsers.py    # LLM output parsing
│   │   └── cost_tracker.py      # Token cost tracking
│   │
│   ├── evaluation/
│   │   ├── workflow_evaluator.py      # Main evaluator
│   │   ├── tool_selection_evaluator.py
│   │   ├── llm_judge_evaluator.py
│   │   ├── agent_efficiency_evaluator.py
│   │   ├── coalition_evaluator.py
│   │   ├── consistency_scorer.py
│   │   └── unified_agent_evaluator.py
│   │
│   ├── run_logging/
│   │   ├── workflow_logger.py   # Central logger (dual-write)
│   │   ├── sheets_logger.py     # Google Sheets logger
│   │   ├── postgres_logger.py   # PostgreSQL logger
│   │   └── run_logger.py        # MongoDB logger
│   │
│   └── storage/
│       ├── postgres.py          # PostgreSQL (27 tables)
│       └── mongodb.py           # MongoDB storage
│
├── config/
│   ├── config.yaml              # Data source config
│   ├── models.yaml              # Multi-model settings
│   └── settings.yaml            # Application settings
│
└── docs/
    ├── ARCHITECTURE.md          # This document
    ├── POSTGRES_SETUP.md        # Database setup
    └── GOOGLE_SHEETS_TABS.md    # Sheets schema
```

---

## Summary

The Credit Intelligence system is a production-grade multi-agent workflow for B2B credit assessment:

1. **LangGraph Orchestration** with PUBLIC/PRIVATE conditional routing
2. **7 Specialized Agents** for parsing, planning, data collection, analysis, evaluation
3. **5 Data Collection Tools** (SEC, Finnhub, CourtListener, Web Search)
4. **8+ External API Integrations** with rate limiting and caching
5. **Comprehensive Evaluation** with LLM-as-judge node scoring
6. **27-Table PostgreSQL Schema** with monthly partitioning
7. **Dual-Write Logging** to PostgreSQL and Google Sheets
8. **Configurable LLM Provider** (Groq, OpenAI, Anthropic)

**Key Features:**
- Credit score range: **0-100** (not 300-850)
- Node scoring evaluates **both agents AND individual tools**
- Tool selection success criteria: **"Selected tools appropriate for company type"**
- Real-time WebSocket streaming for frontend updates
- Full audit trail for every workflow execution
