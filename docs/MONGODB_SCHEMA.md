# MongoDB Schema for Credit Intelligence

This document describes the MongoDB collections used for logging and evaluation.

## Database: `credit_intelligence`

### Collections Overview

| Collection | Purpose |
|------------|---------|
| `runs` | Complete workflow run summaries |
| `steps` | Individual step logs with metrics |
| `tool_calls` | Tool execution logs |
| `tool_selections` | LLM tool selection decisions |
| `assessments` | Final credit assessments |
| `evaluations` | Evaluation results |

---

## Collection: `runs`

Stores complete run summaries for each credit assessment workflow.

```javascript
{
  "_id": ObjectId,
  "run_id": "uuid-string",
  "company_name": "Apple Inc",
  "context": {
    // Additional context provided
    "ticker": "AAPL",
    "jurisdiction": "US"
  },
  "status": "started" | "completed" | "failed",
  "started_at": ISODate,
  "completed_at": ISODate,

  // Embedded summary of steps
  "steps": [
    {
      "step_name": "tool_selection",
      "execution_time_ms": 1234.5,
      "success": true
    }
  ],

  // Embedded summary of tool calls
  "tool_calls": [
    {
      "tool_name": "fetch_sec_data",
      "success": true,
      "execution_time_ms": 500
    }
  ],

  // Aggregated metrics
  "metrics": {
    "total_execution_time_ms": 5000,
    "total_tokens": 2500,
    "llm_calls": 2
  },

  // Final result (when completed)
  "final_result": {
    // Complete assessment result
  },

  // Error (when failed)
  "error": "Error message if failed",

  // Evaluation scores
  "evaluation": {
    "tool_selection": {
      "precision": 0.85,
      "recall": 0.90,
      "f1": 0.87
    },
    "consistency": {
      "score": 0.92
    }
  }
}
```

**Indexes:**
- `run_id` (unique)
- `company_name`
- `started_at` (descending)
- `status`

---

## Collection: `steps`

Detailed logs for each workflow step.

```javascript
{
  "_id": ObjectId,
  "run_id": "uuid-string",
  "step_name": "tool_selection" | "tool_execution" | "synthesis",
  "input_data": {
    // Input to this step
    "company_name": "Apple Inc",
    "context": {}
  },
  "output_data": {
    // Output from this step
    "tools_selected": ["fetch_sec_data", "fetch_market_data"]
  },
  "execution_time_ms": 1234.5,
  "tokens_used": {
    "prompt_tokens": 500,
    "completion_tokens": 200,
    "total_tokens": 700
  },
  "success": true,
  "error": null,
  "timestamp": ISODate
}
```

**Indexes:**
- `run_id`
- `step_name`
- `timestamp`

---

## Collection: `tool_calls`

Logs for individual tool executions.

```javascript
{
  "_id": ObjectId,
  "run_id": "uuid-string",
  "tool_name": "fetch_sec_data",
  "input_params": {
    "company_identifier": "Apple Inc"
  },
  "output_data": {
    "found": true,
    "cik": "0000320193",
    "company_name": "APPLE INC",
    // ... tool-specific data
  },
  "execution_time_ms": 500,
  "success": true,
  "selection_reason": "Company is a US public company, SEC filings needed",
  "timestamp": ISODate
}
```

**Indexes:**
- `run_id`
- `tool_name`
- `success`
- `timestamp`

---

## Collection: `tool_selections`

LLM tool selection decisions and reasoning.

```javascript
{
  "_id": ObjectId,
  "run_id": "uuid-string",
  "company_name": "Apple Inc",
  "tools_selected": ["fetch_sec_data", "fetch_market_data"],
  "selection_reasoning": {
    "company_analysis": {
      "is_likely_public": true,
      "reasoning": "Apple is a well-known US public company"
    },
    "tools_to_use": [
      {
        "name": "fetch_sec_data",
        "params": {"company_identifier": "Apple Inc"},
        "reason": "Need SEC filings for public company"
      }
    ],
    "execution_order_reasoning": "SEC first for official data, then market data"
  },
  "llm_metrics": {
    "model": "llama-3.3-70b-versatile",
    "execution_time_ms": 800,
    "prompt_tokens": 400,
    "completion_tokens": 150,
    "total_tokens": 550
  },
  "timestamp": ISODate
}
```

**Indexes:**
- `run_id`
- `company_name`
- `timestamp`

---

## Collection: `assessments`

Final credit risk assessments.

```javascript
{
  "_id": ObjectId,
  "run_id": "uuid-string",
  "company_name": "Apple Inc",
  "risk_level": "low" | "medium" | "high" | "critical",
  "credit_score": 85,
  "confidence": 0.92,
  "reasoning": "Detailed explanation of assessment...",
  "risk_factors": [
    "High market concentration in smartphone sector"
  ],
  "positive_factors": [
    "Strong cash reserves",
    "Consistent revenue growth",
    "Market leader position"
  ],
  "recommendations": [
    "Approve credit line up to $X"
  ],
  "data_quality_assessment": {
    "data_completeness": 0.95,
    "sources_used": ["SEC", "Finnhub"],
    "missing_data": []
  },
  "llm_metrics": {
    "model": "llama-3.3-70b-versatile",
    "execution_time_ms": 1200,
    "prompt_tokens": 1500,
    "completion_tokens": 400,
    "total_tokens": 1900
  },
  "timestamp": ISODate
}
```

**Indexes:**
- `run_id`
- `company_name`
- `risk_level`
- `credit_score`
- `timestamp`

---

## Collection: `evaluations`

Evaluation results for workflow runs.

```javascript
{
  "_id": ObjectId,
  "run_id": "uuid-string",
  "evaluation_type": "tool_selection" | "consistency" | "synthesis",
  "metrics": {
    // Type-specific metrics
    "expected_tools": ["fetch_sec_data", "fetch_market_data"],
    "selected_tools": ["fetch_sec_data", "fetch_market_data"],
    "true_positives": 2,
    "false_positives": 0,
    "false_negatives": 0
  },
  "scores": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "timestamp": ISODate
}
```

**Indexes:**
- `run_id`
- `evaluation_type`
- `timestamp`

---

## Queries

### Get Recent Runs
```javascript
db.runs.find({})
  .sort({started_at: -1})
  .limit(20)
```

### Get Runs for Company
```javascript
db.runs.find({company_name: "Apple Inc"})
  .sort({started_at: -1})
```

### Get All Tool Calls for a Run
```javascript
db.tool_calls.find({run_id: "uuid-string"})
  .sort({timestamp: 1})
```

### Get High-Risk Assessments
```javascript
db.assessments.find({risk_level: {$in: ["high", "critical"]}})
  .sort({timestamp: -1})
```

### Get Evaluation Statistics
```javascript
db.evaluations.aggregate([
  {$match: {evaluation_type: "tool_selection"}},
  {$group: {
    _id: null,
    avg_precision: {$avg: "$scores.precision"},
    avg_recall: {$avg: "$scores.recall"},
    avg_f1: {$avg: "$scores.f1"}
  }}
])
```

### Get Consistency Trends
```javascript
db.evaluations.aggregate([
  {$match: {evaluation_type: "consistency"}},
  {$group: {
    _id: {$dateToString: {format: "%Y-%m-%d", date: "$timestamp"}},
    avg_consistency: {$avg: "$scores.overall"}
  }},
  {$sort: {_id: 1}}
])
```

---

## Data Retention

Recommended retention policies:

| Collection | Retention |
|------------|-----------|
| `runs` | 90 days |
| `steps` | 30 days |
| `tool_calls` | 30 days |
| `tool_selections` | 90 days |
| `assessments` | Indefinite (for audit) |
| `evaluations` | 90 days |

Use TTL indexes for automatic cleanup:
```javascript
db.steps.createIndex(
  {timestamp: 1},
  {expireAfterSeconds: 2592000}  // 30 days
)
```
