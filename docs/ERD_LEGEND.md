# Credit Intelligence ERD Legend

## Generated Files
- `credit_intelligence_erd.png` - Horizontal layout ERD
- `credit_intelligence_erd_vertical.png` - Vertical layout ERD (best for Google Docs)
- `credit_intelligence_erd_pdf.pdf` / `credit_intelligence_erd_vertical_pdf.pdf` - PDF versions
- `credit_intelligence_erd_svg.svg` / `credit_intelligence_erd_vertical_svg.svg` - Scalable SVG versions

## Color Legend

| Color | Meaning |
|-------|---------|
| **Blue Header** (#1565C0) | Workflow Tables (`wf_*`) |
| **Green Header** (#2E7D32) | LangGraph Tables (`lg_*`) |
| **Orange Header** (#E65100) | Evaluation Tables (`eval_*`) |
| **Purple Header** (#6A1B9A) | Metadata Tables (`meta_*`) |
| **Yellow Row** (#FFEB3B) | Primary Key (PK) |
| **Light Blue Row** (#BBDEFB) | Foreign Key (FK) |
| **Pale Blue Row** (#E1F5FE) | Join Column (company_name, agent_name, node, etc.) |

## Relationship Types

### Primary Key / Foreign Key Relationships (Solid Blue Lines)
All tables connect to `wf_runs` via `run_id`:
- `wf_runs.run_id` (PK) → All other tables (FK)
- Cardinality: 1:N (one run has many records in child tables)

### Logical Join Relationships (Dashed Gray Lines)
All tables now have **consistent hierarchy columns** for full traceability:

**Universal Join Columns** (available in ALL tables):
- `run_id` - Primary FK to wf_runs
- `company_name` - Company being analyzed
- `node` - Workflow node name (e.g., parse_input, create_plan, synthesize)
- `node_type` - Type of node (agent, tool, llm, evaluator)
- `agent_name` - Agent that performed the action
- `master_agent` - Parent supervisor agent (typically "supervisor")
- `step_number` - Sequential step in workflow (1-8)

**Recommended Join Patterns:**

| Table A | Table B | Join Columns |
|---------|---------|--------------|
| `wf_llm_calls` | `lg_events` | `run_id` + `node` + `agent_name` + `step_number` |
| `wf_tool_calls` | `lg_events` | `run_id` + `node` + `agent_name` + `step_number` |
| `wf_tool_calls` | `wf_llm_calls` | `run_id` + `agent_name` + `step_number` |
| `wf_assessments` | `eval_results` | `run_id` + `node` + `agent_name` |
| `eval_agent_metrics` | `eval_coalition` | `run_id` + `agent_name` |
| `wf_llm_calls` | `meta_prompts` | `run_id` + `node` + `agent_name` |
| `wf_data_sources` | `wf_tool_calls` | `run_id` + `node` + `step_number` |
| `wf_plans` | `wf_tool_calls` | `run_id` (plans at step 3, tools at step 4+) |

## Common Join Columns (across all tables)

| Column | Description | Used For |
|--------|-------------|----------|
| `run_id` | Unique workflow run identifier | Primary FK relationship |
| `company_name` | Company being analyzed | Filter by company |
| `agent_name` | Agent that performed action | Track agent activities |
| `master_agent` | Parent supervisor agent | Agent hierarchy |
| `node` | Workflow node name | Track workflow position |
| `node_type` | Type of node (agent, tool, etc.) | Filter by node type |
| `step_number` | Step in workflow sequence | Order events |
| `timestamp` | When event occurred | Time-based queries |

## Table Categories

### Workflow Tables (`wf_*`) - 7 Tables
| Table | Purpose | Records Per Run |
|-------|---------|-----------------|
| `wf_runs` | Run summaries & final results | 1 |
| `wf_llm_calls` | All LLM API calls | 10-50+ |
| `wf_tool_calls` | Tool/API executions | 5-20+ |
| `wf_assessments` | Credit assessments | 1-5 |
| `wf_plans` | Task plans created | 1 |
| `wf_data_sources` | Data source results | 3-6 |
| `wf_state_dumps` | Workflow state snapshots | 1 |

### LangGraph Tables (`lg_*`) - 1 Table
| Table | Purpose | Records Per Run |
|-------|---------|-----------------|
| `lg_events` | LangGraph framework events | 50-200+ |

### Evaluation Tables (`eval_*`) - 8 Tables
| Table | Purpose | Records Per Run |
|-------|---------|-----------------|
| `eval_results` | Overall evaluation results | 1 |
| `eval_tool_selection` | Tool selection accuracy | 1 |
| `eval_consistency` | LLM consistency scores | 1-3 |
| `eval_cross_model` | Cross-model comparisons | 1 |
| `eval_llm_judge` | LLM-as-judge results | 1 |
| `eval_agent_metrics` | Agent efficiency metrics | 1 |
| `eval_coalition` | Coalition voting results | 1 |
| `eval_log_tests` | Logging verification | 1 |

### Metadata Tables (`meta_*`) - 2 Tables
| Table | Purpose | Records Per Run |
|-------|---------|-----------------|
| `meta_prompts` | Prompts used in workflow | 5-15 |
| `meta_api_keys` | API key storage | N/A (global) |

## Sample JOIN Queries

### Get all LLM calls with their LangGraph events (full hierarchy)
```sql
SELECT l.*, e.event_type, e.duration_ms
FROM wf_llm_calls l
LEFT JOIN lg_events e ON l.run_id = e.run_id
    AND l.node = e.node
    AND l.agent_name = e.agent_name
    AND l.step_number = e.step_number
WHERE l.run_id = 'your-run-id';
```

### Get tool calls with their data source results
```sql
SELECT t.*, d.records_found, d.data_summary
FROM wf_tool_calls t
LEFT JOIN wf_data_sources d ON t.run_id = d.run_id
    AND t.node = d.node
    AND t.step_number = d.step_number
WHERE t.run_id = 'your-run-id';
```

### Get tool calls linked with LangGraph events
```sql
SELECT t.tool_name, t.execution_time_ms, t.status,
       e.event_type, e.duration_ms as event_duration
FROM wf_tool_calls t
LEFT JOIN lg_events e ON t.run_id = e.run_id
    AND t.node = e.node
    AND t.agent_name = e.agent_name
    AND t.step_number = e.step_number
WHERE t.run_id = 'your-run-id'
ORDER BY t.step_number;
```

### Get full evaluation results for a run
```sql
SELECT r.*, c.correctness_score, c.is_correct,
       a.overall_score as agent_score
FROM wf_runs r
LEFT JOIN eval_coalition c ON r.run_id = c.run_id
LEFT JOIN eval_agent_metrics a ON r.run_id = a.run_id
WHERE r.run_id = 'your-run-id';
```

### Get complete workflow timeline with hierarchy
```sql
SELECT
    timestamp,
    step_number,
    node,
    node_type,
    agent_name,
    'llm_call' as activity_type,
    model as detail
FROM wf_llm_calls WHERE run_id = 'your-run-id'
UNION ALL
SELECT
    timestamp,
    step_number,
    node,
    node_type,
    agent_name,
    'tool_call' as activity_type,
    tool_name as detail
FROM wf_tool_calls WHERE run_id = 'your-run-id'
UNION ALL
SELECT
    timestamp,
    step_number,
    node,
    node_type,
    agent_name,
    'langgraph_event' as activity_type,
    event_type as detail
FROM lg_events WHERE run_id = 'your-run-id'
ORDER BY step_number, timestamp;
```

### Trace a specific agent's activities across tables
```sql
-- Find all activities for a specific agent in a run
SELECT 'llm_call' as source, node, step_number, timestamp
FROM wf_llm_calls WHERE run_id = 'your-run-id' AND agent_name = 'planner'
UNION ALL
SELECT 'tool_call' as source, node, step_number, timestamp
FROM wf_tool_calls WHERE run_id = 'your-run-id' AND agent_name = 'planner'
UNION ALL
SELECT 'lg_event' as source, node, step_number, timestamp
FROM lg_events WHERE run_id = 'your-run-id' AND agent_name = 'planner'
ORDER BY step_number, timestamp;
```
