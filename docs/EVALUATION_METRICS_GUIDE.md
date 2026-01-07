# Evaluation Metrics Guide

This document explains all evaluation metrics used in the Credit Intelligence system, including how they are calculated and what they measure.

---

## Table of Contents

1. [Overview](#overview)
2. [Google Sheets Structure](#google-sheets-structure)
3. [DeepEval Metrics](#deepeval-metrics)
4. [OpenEvals / AgentEvals Metrics](#openevals--agentevals-metrics)
5. [LLM Judge Metrics](#llm-judge-metrics)
6. [Consistency Metrics](#consistency-metrics)
7. [Unified Metrics](#unified-metrics)
8. [Metric Calculation Examples](#metric-calculation-examples)

---

## Overview

The evaluation system uses three main frameworks:

| Framework | Source | Purpose |
|-----------|--------|---------|
| **DeepEval** | `deepeval` library | LLM-powered quality evaluation (hallucination, faithfulness, etc.) |
| **OpenEvals/AgentEvals** | LangSmith, custom | Agent efficiency metrics (intent, tools, trajectory) |
| **LLM-as-Judge** | OpenAI GPT-4 | Human-like evaluation with reasoning |

All metrics are logged to dedicated Google Sheets for analysis and tracking.

---

## Google Sheets Structure

### Sheet 19: `deepeval_metrics`
DeepEval LLM-powered evaluation results.

### Sheet 20: `openevals_metrics`
Agent efficiency metrics from OpenEvals/AgentEvals.

### Sheet 14: `agent_metrics`
Detailed agent efficiency with breakdowns.

### Sheet 15: `unified_metrics`
Combined view of all metrics (DeepEval + OpenEvals + Built-in).

### Sheet 16: `llm_judge_results`
LLM-as-a-judge evaluation results.

---

## DeepEval Metrics

**Source:** DeepEval library (https://docs.confident-ai.com/)

**Sheet:** `deepeval_metrics`

### Core Metrics

| Metric | Range | Direction | Description |
|--------|-------|-----------|-------------|
| `answer_relevancy` | 0-1 | Higher is better | Is the answer relevant to the question? |
| `faithfulness` | 0-1 | Higher is better | Is the answer grounded in the provided context? |
| `hallucination` | 0-1 | **Lower is better** | Does the answer contain hallucinated information? |
| `contextual_relevancy` | 0-1 | Higher is better | Is the retrieval context relevant to the question? |
| `bias` | 0-1 | **Lower is better** | Does the answer contain bias? |
| `toxicity` | 0-1 | **Lower is better** | Does the answer contain toxic content? |

### Overall Score Calculation

```python
overall_score = (
    answer_relevancy * 0.25 +      # 25% weight
    faithfulness * 0.30 +          # 30% weight
    (1 - hallucination) * 0.25 +   # 25% weight (inverted)
    contextual_relevancy * 0.10 +  # 10% weight
    (1 - bias) * 0.10              # 10% weight (inverted)
)
```

### What Each Metric Measures

#### Answer Relevancy (0-1, higher is better)
**Question:** "Is the generated answer relevant to the input query?"

**How it's calculated:**
1. DeepEval extracts key statements from the answer
2. For each statement, it checks if it addresses the original query
3. Score = (relevant statements) / (total statements)

**Example:**
- Query: "Analyze credit risk for Apple"
- Answer: "Apple's revenue is $394B. The company was founded by Steve Jobs."
- Score: 0.5 (revenue relevant, founding history not relevant to credit risk)

#### Faithfulness (0-1, higher is better)
**Question:** "Is the answer grounded in the provided context?"

**How it's calculated:**
1. DeepEval extracts claims from the answer
2. For each claim, it checks if it's supported by the context
3. Score = (supported claims) / (total claims)

**Example:**
- Context: "Apple has $166B in cash reserves"
- Answer: "Apple has strong liquidity with over $150B in cash"
- Score: 1.0 (claim is supported by context)

#### Hallucination (0-1, LOWER is better)
**Question:** "Does the answer contain information not in the context?"

**How it's calculated:**
1. DeepEval identifies facts in the answer
2. For each fact, it checks if it contradicts or fabricates beyond context
3. Score = (hallucinated facts) / (total facts)

**Example:**
- Context: "Apple's debt-to-equity ratio is 1.8"
- Answer: "Apple has a debt-to-equity ratio of 0.5"
- Score: 1.0 (hallucinated - contradicts context)

#### Contextual Relevancy (0-1, higher is better)
**Question:** "Is the retrieved context relevant to answering the query?"

**How it's calculated:**
1. DeepEval analyzes each context chunk
2. Checks if each chunk contains information useful for answering
3. Score = (relevant chunks) / (total chunks)

#### Bias (0-1, LOWER is better)
**Question:** "Does the answer contain biased statements?"

**How it's calculated:**
1. DeepEval identifies opinions and judgments in the answer
2. Checks for unfair treatment of groups, leading language, etc.
3. Score = (biased statements) / (total statements)

---

## OpenEvals / AgentEvals Metrics

**Source:** LangSmith evaluators, custom implementation

**Sheet:** `openevals_metrics`

### Agent Efficiency Metrics

| Metric | Range | Description | Calculation |
|--------|-------|-------------|-------------|
| `intent_correctness` | 0-1 | Did the agent understand the task? | See below |
| `plan_quality` | 0-1 | How good was the execution plan? | See below |
| `tool_choice_correctness` | 0-1 | Did agent choose correct tools? (Precision) | correct / selected |
| `tool_completeness` | 0-1 | Did agent use all needed tools? (Recall) | used / expected |
| `trajectory_match` | 0-1 | Did agent follow expected path? | Jaccard + order bonus |
| `final_answer_quality` | 0-1 | Is the output correct and complete? | See below |

### Execution Stats

| Metric | Type | Description |
|--------|------|-------------|
| `step_count` | int | Number of workflow steps executed |
| `tool_calls` | int | Number of tool/API invocations |
| `latency_ms` | float | Total execution time in milliseconds |

### Overall Score Calculation

```python
overall_score = (
    intent_correctness * 0.15 +        # 15% weight
    plan_quality * 0.15 +              # 15% weight
    tool_choice_correctness * 0.20 +   # 20% weight
    tool_completeness * 0.15 +         # 15% weight
    trajectory_match * 0.15 +          # 15% weight
    final_answer_quality * 0.20        # 20% weight
)
```

### Detailed Calculations

#### Intent Correctness

**Question:** "Did the agent correctly understand what was asked?"

```python
score = 0.0

# 1. Company name parsing (40%)
if input_company.lower() in parsed_company.lower():
    score += 0.4

# 2. Company type identification (30%)
if company_type is not None:  # e.g., "public_us", "private"
    score += 0.3

# 3. Parse confidence (30%)
if confidence > 0.7:
    score += 0.3
elif confidence > 0.5:
    score += 0.2
elif confidence > 0.3:
    score += 0.1
```

#### Plan Quality

**Question:** "Did the agent create a good execution plan?"

```python
score = 0.0

# 1. Plan size appropriate (30%)
if 3 <= len(task_plan) <= 10:
    score += 0.3

# 2. Has data gathering step (35%)
if any("data" in step or "fetch" in step or "search" in step for step in plan):
    score += 0.35

# 3. Has analysis step (35%)
if any("analy" in step or "synthe" in step for step in plan):
    score += 0.35
```

#### Tool Choice Correctness (Precision)

**Question:** "Of the tools selected, how many were correct?"

```python
# Expected tools by company type
EXPECTED_TOOLS = {
    "public_us": {"fetch_sec_data", "fetch_market_data", "web_search"},
    "public_non_us": {"fetch_market_data", "web_search"},
    "private": {"web_search", "fetch_legal_data"},
}

expected = EXPECTED_TOOLS[company_type]
selected = set(tools_used)

true_positives = len(expected & selected)
precision = true_positives / len(selected) if selected else 0.0
```

**Example:**
- Company type: `public_us`
- Expected tools: {SEC, Market, Web}
- Selected tools: {SEC, Market, Legal}
- Correct: {SEC, Market} = 2
- Precision: 2/3 = 0.67

#### Tool Completeness (Recall)

**Question:** "Of the tools expected, how many were used?"

```python
true_positives = len(expected & selected)
recall = true_positives / len(expected) if expected else 0.0
```

**Example:**
- Expected tools: {SEC, Market, Web} = 3
- Selected tools: {SEC, Market} = 2
- Used expected: {SEC, Market} = 2
- Recall: 2/3 = 0.67

#### Trajectory Match

**Question:** "Did the agent follow the expected execution path?"

```python
EXPECTED_TRAJECTORY = [
    "parse_input", "validate_company", "create_plan",
    "fetch_api_data", "search_web", "synthesize",
    "save_to_database", "evaluate"
]

# 1. Jaccard Similarity (60%)
expected_set = set(EXPECTED_TRAJECTORY)
actual_set = set(actual_trajectory)
jaccard = len(expected_set & actual_set) / len(expected_set | actual_set)

# 2. Order Bonus (40%)
# Check if consecutive steps maintain expected order
order_score = correct_order_pairs / total_pairs

# Combined score
trajectory_match = jaccard * 0.6 + order_score * 0.4
```

#### Final Answer Quality

**Question:** "Is the output correct and complete?"

```python
REQUIRED_FIELDS = {"risk_level", "credit_score", "confidence", "reasoning", "recommendations"}

score = 0.0

# 1. Field completeness (50%)
fields_present = sum(1 for f in REQUIRED_FIELDS if f in assessment)
completeness = fields_present / len(REQUIRED_FIELDS)
score += completeness * 0.5

# 2. Valid risk level (15%)
if risk_level in ["low", "moderate", "high", "critical"]:
    score += 0.15

# 3. Valid credit score (15%)
if 0 <= credit_score <= 100:
    score += 0.15

# 4. Valid confidence (10%)
if 0 <= confidence <= 1:
    score += 0.1

# 5. Substantial reasoning (10%)
if len(reasoning) > 50:
    score += 0.1
```

---

## LLM Judge Metrics

**Source:** OpenAI GPT-4 as evaluator

**Sheet:** `llm_judge_results`

### Dimension Scores

| Metric | Range | Description |
|--------|-------|-------------|
| `accuracy_score` | 0-1 | Factual correctness of the assessment |
| `completeness_score` | 0-1 | Coverage of required analysis areas |
| `consistency_score` | 0-1 | Internal consistency of conclusions |
| `actionability_score` | 0-1 | Usefulness of recommendations |
| `data_utilization_score` | 0-1 | How well data sources were used |
| `overall_score` | 0-1 | Weighted average of all dimensions |

### How It Works

1. The assessment is sent to GPT-4 with a structured prompt
2. GPT-4 evaluates each dimension and provides:
   - A score (0-1)
   - Reasoning for the score
   - Specific suggestions for improvement
3. Benchmark comparison (if reference data available)

---

## Consistency Metrics

**Sheet:** `consistency_scores`, `model_consistency`, `cross_model_eval`

### Same-Model Consistency

**Question:** "Does the same model give consistent results across multiple runs?"

| Metric | Description |
|--------|-------------|
| `risk_level_consistency` | % of runs with same risk level |
| `score_consistency` | How close are credit scores (normalized) |
| `score_std` | Standard deviation of credit scores |
| `overall_consistency` | Combined consistency score |

### Cross-Model Consistency

**Question:** "Do different models agree on the assessment?"

| Metric | Description |
|--------|-------------|
| `risk_level_agreement` | % of models agreeing on risk level |
| `credit_score_range` | Max - Min credit score across models |
| `confidence_agreement` | Similarity of confidence levels |
| `cross_model_agreement` | Overall agreement score |

---

## Unified Metrics

**Sheet:** `unified_metrics`

Combines all evaluation frameworks into a single view:

### Categories

| Category | Metrics Included |
|----------|------------------|
| **Accuracy** | faithfulness, hallucination, answer_relevancy, factual_accuracy |
| **Consistency** | same_model, cross_model, risk_agreement, semantic_similarity |
| **Agent Efficiency** | intent, plan, tools, trajectory, final_answer |

### Overall Quality Score

```python
overall_quality_score = (
    accuracy_score * 0.35 +      # 35% weight
    consistency_score * 0.25 +    # 25% weight
    agent_efficiency_score * 0.40 # 40% weight
)
```

---

## Metric Calculation Examples

### Example 1: Public US Company (Apple)

**Input:** "Analyze credit risk for Apple Inc"

**Expected Tools:** {SEC, Market, Web}
**Selected Tools:** {SEC, Market, Web, Legal}

**Tool Choice Correctness (Precision):**
- Correct: {SEC, Market, Web} = 3
- Selected: 4
- Score: 3/4 = **0.75**

**Tool Completeness (Recall):**
- Expected: 3
- Used from expected: 3
- Score: 3/3 = **1.0**

### Example 2: Private Company

**Input:** "Analyze credit risk for XYZ Private Corp"

**Expected Tools:** {Web, Legal}
**Selected Tools:** {Web, SEC}

**Tool Choice Correctness:**
- Correct: {Web} = 1
- Selected: 2
- Score: 1/2 = **0.5**

**Tool Completeness:**
- Expected: 2
- Used from expected: 1
- Score: 1/2 = **0.5**

### Example 3: DeepEval Score

**Context:** "Apple has $166B cash, revenue $394B, debt ratio 1.8"
**Answer:** "Apple has excellent liquidity with cash reserves around $160B and stable revenue streams. The moderate debt ratio suggests responsible leverage."

**Scores:**
- answer_relevancy: **0.95** (highly relevant to credit analysis)
- faithfulness: **0.90** (numbers slightly rounded but accurate)
- hallucination: **0.05** (minor approximation)
- contextual_relevancy: **0.85** (context covers key metrics)
- bias: **0.10** (slight positive bias in "excellent")

**Overall:**
```
0.95*0.25 + 0.90*0.30 + (1-0.05)*0.25 + 0.85*0.10 + (1-0.10)*0.10
= 0.2375 + 0.27 + 0.2375 + 0.085 + 0.09
= 0.92
```

---

## Interpreting Scores

### Score Ranges

| Range | Quality | Action |
|-------|---------|--------|
| 0.9 - 1.0 | Excellent | No action needed |
| 0.7 - 0.89 | Good | Minor improvements possible |
| 0.5 - 0.69 | Fair | Review and improve |
| 0.3 - 0.49 | Poor | Significant issues |
| 0.0 - 0.29 | Critical | Requires immediate attention |

### Common Issues and Fixes

| Low Metric | Likely Cause | Fix |
|------------|--------------|-----|
| Low faithfulness | Answer contains unsupported claims | Improve context retrieval |
| High hallucination | Model generating facts not in context | Add stricter grounding |
| Low tool_completeness | Missing required data sources | Update tool selection logic |
| Low trajectory_match | Steps out of order or missing | Review workflow graph |
| Low final_answer_quality | Missing required output fields | Update synthesis prompt |

---

## Installation

To use DeepEval metrics, install the library:

```bash
pip install deepeval
```

Set your OpenAI API key for DeepEval evaluations:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage

### Log DeepEval Metrics

```python
from run_logging.sheets_logger import get_sheets_logger

logger = get_sheets_logger()
logger.log_deepeval_metrics(
    run_id="run_123",
    company_name="Apple Inc",
    model_used="gpt-4",
    answer_relevancy=0.95,
    faithfulness=0.90,
    hallucination=0.05,
    contextual_relevancy=0.85,
    bias=0.10,
    overall_score=0.92,
)
```

### Log OpenEvals Metrics

```python
logger.log_openevals_metrics(
    run_id="run_123",
    company_name="Apple Inc",
    model_used="llama-3.3-70b",
    intent_correctness=0.95,
    plan_quality=0.85,
    tool_choice_correctness=0.75,
    tool_completeness=1.0,
    trajectory_match=0.90,
    final_answer_quality=0.88,
    step_count=8,
    tool_calls=4,
    latency_ms=15000,
    overall_score=0.87,
)
```

---

## Related Files

- `/src/evaluation/deepeval_evaluator.py` - DeepEval integration
- `/src/evaluation/agent_efficiency_evaluator.py` - OpenEvals metrics
- `/src/evaluation/llm_judge_evaluator.py` - LLM-as-Judge
- `/src/evaluation/unified_agent_evaluator.py` - Combined evaluator
- `/src/run_logging/sheets_logger.py` - Google Sheets logging
