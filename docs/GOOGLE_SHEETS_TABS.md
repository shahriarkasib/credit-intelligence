# Google Sheets Tab Documentation

This document describes all 18 tabs in the Credit Intelligence Google Sheets logging system.

**Spreadsheet ID:** `1E8unWiaAEvxeCby_s_r2Y7g9iW4PznukuC1tpWG58s0`

---

## Quick Reference

| Tab # | Sheet Name | Purpose | Data Source |
|-------|------------|---------|-------------|
| 1 | `runs` | Run summaries | Internal |
| 2 | `tool_calls` | Tool execution logs | Internal |
| 3 | `assessments` | Credit assessment outputs | LLM |
| 4 | `evaluations` | Main evaluation scores | Rule-Based |
| 5 | `tool_selections` | Tool selection analysis | Rule-Based |
| 6 | `llm_calls` | All LLM API calls | Measured |
| 7 | `consistency_scores` | Model consistency | Rule-Based |
| 8 | `data_sources` | API data fetching | Internal |
| 9 | `langgraph_events` | LangGraph events | Framework |
| 10 | `plans` | Task plans | Internal |
| 11 | `prompts` | Prompts used | Internal |
| 12 | `deepeval_metrics` | DeepEval scores | DeepEval (Groq) |
| 13 | `openevals_metrics` | OpenEvals scores | OpenEvals (OpenAI) |
| 14 | `cross_model_eval` | Cross-model comparison | Internal + LLM |
| 15 | `llm_judge_results` | LLM-as-Judge results | LLM |
| 16 | `agent_metrics` | Agent efficiency | Rule-Based |
| 17 | `log_tests` | Logging verification | Internal |
| 18 | `node_scoring` | LLM judge node quality | LLM |

---

## Score Thresholds (All Evaluation Sheets)

| eval_status | Score Range | Color |
|-------------|-------------|-------|
| `good` | >= 0.80 | Green |
| `average` | 0.60 - 0.79 | Yellow |
| `bad` | < 0.60 | Red |

---

## Tab 1: `runs`

**Purpose:** High-level summary of each credit analysis run, including 3 key performance metrics.

**Key Columns:**
- `run_id` - Unique identifier for the run
- `company_name` - Company being analyzed
- `risk_level` - Final risk assessment (LOW/MODERATE/HIGH/CRITICAL)
- `credit_score` - Credit score 300-850
- `confidence` - LLM confidence 0-1
- `evaluation_score` - Overall evaluation score
- `total_time_ms` - Total execution time
- `tools_used` - List of tools executed
- `tool_overall_score` - Overall tool performance (0-1)
- `agent_overall_score` - Overall agent performance (0-1)
- `workflow_overall_score` - Overall workflow performance (0-1)

**When Logged:** Once per run, at completion. Performance scores updated after evaluation.

---

## Tab 2: `tool_calls`

**Purpose:** Detailed log of every tool execution.

**Key Columns:**
- `tool_name` - Tool executed (e.g., fetch_sec_edgar, fetch_finnhub)
- `tool_input` - Input parameters (JSON)
- `tool_output` - Output data (JSON)
- `execution_time_ms` - How long the tool took
- `status` - ok/fail
- `call_depth` - Nesting level (0 = top-level)

**When Logged:** After each tool execution.

---

## Tab 3: `assessments`

**Purpose:** Credit assessment outputs from the LLM synthesis step.

**Key Columns:**
- `risk_level` - LOW/MODERATE/HIGH/CRITICAL
- `credit_score` - 300-850
- `confidence` - 0-1
- `reasoning` - LLM's explanation
- `recommendations` - Action items
- `model` - LLM model used
- `duration_ms` - LLM call duration

**When Logged:** After synthesis node completes.

---

## Tab 4: `evaluations`

**Purpose:** Main evaluation scores combining tool selection, data quality, and synthesis quality.

**Key Columns:**
- `tool_selection_score` - F1 score for tool selection (0-1)
- `data_quality_score` - Data completeness (0-1)
- `synthesis_score` - Output completeness (0-1)
- `overall_score` - Weighted average: `(tool*0.3) + (data*0.3) + (synthesis*0.4)`
- `eval_status` - good/average/bad

**Calculation:** Rule-based, no LLM calls.

**When Logged:** After evaluation node completes.

---

## Tab 5: `tool_selections`

**Purpose:** Detailed analysis of which tools were selected vs expected.

**Key Columns:**
- `selected_tools` - Tools the LLM chose
- `expected_tools` - Tools it should have chosen
- `correct_tools` - Intersection
- `missing_tools` - Expected but not selected
- `extra_tools` - Selected but not expected
- `precision` - `correct / selected`
- `recall` - `correct / expected`
- `f1_score` - `2 * (P * R) / (P + R)`

**Expected Tools by Company Type:**
- Public US: fetch_sec_edgar, fetch_finnhub
- Public non-US: fetch_finnhub, web_search
- Private: web_search, fetch_court_listener
- Unknown: web_search

**When Logged:** After tool selection decision.

---

## Tab 6: `llm_calls`

**Purpose:** Every LLM API call with tokens and costs.

**Key Columns:**
- `call_type` - Purpose (tool_selection, synthesis, evaluation)
- `model` - Model used
- `prompt` - Full prompt text
- `response` - Full response text
- `prompt_tokens` - Input tokens
- `completion_tokens` - Output tokens
- `total_cost` - USD cost
- `execution_time_ms` - API latency

**Cost Tracking:** Automatically calculated based on model pricing.

**When Logged:** After every LLM call.

---

## Tab 7: `consistency_scores`

**Purpose:** Measures consistency across multiple runs.

**Key Columns:**
- `evaluation_type` - "same_model" or "cross_model"
- `num_runs` - Number of runs compared
- `risk_level_consistency` - Do runs agree on risk? (0-1)
- `score_consistency` - Credit score stability (0-1)
- `score_std` - Standard deviation of scores
- `overall_consistency` - Combined score

**When Logged:** After consistency evaluation (requires multiple runs).

---

## Tab 8: `data_sources`

**Purpose:** Results from each data source API.

**Key Columns:**
- `source_name` - API name (sec_edgar, finnhub, court_listener, etc.)
- `records_found` - Number of records returned
- `data_summary` - Preview of data (up to 50k chars)
- `execution_time_ms` - API response time
- `status` - ok/fail

**When Logged:** After each API call.

---

## Tab 9: `langgraph_events`

**Purpose:** Raw LangGraph execution events from astream_events.

**Key Columns:**
- `event_type` - on_chain_start, on_chain_end, on_llm_start, etc.
- `event_name` - Event identifier
- `node` - Graph node
- `tokens` - Token count (for LLM events)
- `duration_ms` - Event duration

**Source:** Automatically captured by LangGraph framework.

**When Logged:** During workflow execution.

---

## Tab 10: `plans`

**Purpose:** Task plans created for each run.

**Key Columns:**
- `num_tasks` - Number of tasks in plan
- `plan_summary` - Brief description
- `full_plan` - Complete plan as JSON
- `task_1` through `task_10` - Individual task details

**When Logged:** After plan creation.

---

## Tab 11: `prompts`

**Purpose:** All prompts used during a run.

**Key Columns:**
- `prompt_id` - Identifier (company_parser, tool_selection, credit_synthesis)
- `prompt_name` - Human-readable name
- `category` - input/planning/synthesis/evaluation
- `system_prompt` - Full system prompt text
- `user_prompt` - Full user prompt text
- `variables_json` - Variables substituted

**When Logged:** Each time a prompt is used.

---

## Tab 12: `deepeval_metrics`

**Purpose:** DeepEval LLM-powered evaluation metrics.

**Provider:** Groq (llama-3.3-70b-versatile) - **FREE**

**Key Columns:**
- `answer_relevancy` - Is assessment relevant to query? (0-1)
- `faithfulness` - Is assessment grounded in data? (0-1)
- `hallucination` - Contains fabricated info? (0-1, **lower is better**)
- `contextual_relevancy` - Is context data relevant? (0-1)
- `bias` - Shows bias? (0-1, **lower is better**)
- `overall_score` - Weighted combination

**When Logged:** After DeepEval evaluation.

---

## Tab 13: `openevals_metrics`

**Purpose:** OpenEvals LLM-as-Judge evaluation metrics.

**Provider:** OpenAI (gpt-4o-mini) - **~$0.001/eval**

**Key Columns:**
- `correctness` - Is output correct vs expected? (0-1)
- `helpfulness` - Is output helpful for user? (0-1)
- `coherence` - Is output well-structured? (0-1)
- `relevance` - Is output relevant to query? (0-1)
- `overall_score` - Weighted combination
- `has_expected_output` - Whether reference was provided

**When Logged:** After OpenEvals evaluation.

---

## Tab 14: `cross_model_eval`

**Purpose:** Compare outputs across different LLM models.

**Key Columns:**
- `models_compared` - List of models (e.g., "llama-3.3-70b, llama-3.1-8b")
- `risk_level_agreement` - Do models agree on risk? (0-1)
- `credit_score_mean` - Average score across models
- `credit_score_std` - Score variation
- `best_model` - LLM picks best performer
- `best_model_reasoning` - LLM explains why

**When Logged:** After cross-model evaluation (requires multiple model runs).

---

## Tab 15: `llm_judge_results`

**Purpose:** Internal LLM-as-Judge evaluation of assessment quality.

**Key Columns:**
- `accuracy_score` - Is risk level reasonable? (0-1)
- `completeness_score` - All risk factors covered? (0-1)
- `consistency_score` - Logic coherent? (0-1)
- `actionability_score` - Recommendations clear? (0-1)
- `data_utilization_score` - Data well used? (0-1)
- `overall_score` - Average of all scores
- `*_reasoning` - LLM explanations for each score
- `suggestions` - Improvement recommendations

**Provider:** Groq (llama-3.3-70b-versatile) - **FREE**

**When Logged:** After LLM judge evaluation.

---

## Tab 16: `agent_metrics`

**Purpose:** Agent efficiency evaluation (all rule-based, no LLM).

**Key Columns:**
- `intent_correctness` - Did agent understand the task? (0-1)
- `plan_quality` - How good was the plan? (0-1)
- `tool_choice_correctness` - Precision of tool selection (0-1)
- `tool_completeness` - Recall of tool selection (0-1)
- `trajectory_match` - Did agent follow expected path? (0-1)
- `final_answer_quality` - Is output valid? (0-1)
- `step_count` - Number of steps taken
- `latency_ms` - Total execution time

**Formula:**
```
overall_score = (intent*0.15) + (plan*0.15) + (tool_prec*0.20) +
                (tool_rec*0.15) + (trajectory*0.15) + (answer*0.20)
```

**When Logged:** After agent evaluation.

---

## Tab 17: `log_tests`

**Purpose:** Verify that all sheets received data for each run.

**Key Columns:**
- `runs` through `agent_metrics` - Row count for each sheet
- `total_sheets_logged` - Number of sheets with data
- `verification_status` - pass/partial/fail

**Thresholds:**
- `pass`: >= 5 sheets have data
- `partial`: 1-4 sheets have data
- `fail`: 0 sheets have data

**When Logged:** After run completion.

---

## Tab 18: `node_scoring`

**Purpose:** LLM-as-Judge quality scores for each node's execution.

**Key Columns:**
- `node` - Graph node name (parse_input, validate_company, etc.)
- `node_type` - Type of node (llm, tool, supervisor, etc.)
- `agent_name` - Canonical agent name executing the node
- `master_agent` - Top-level agent (typically "supervisor")
- `step_number` - Order in workflow execution
- `task_description` - What the node was supposed to do
- `task_completed` - Did the node complete its task? (true/false)
- `quality_score` - LLM judge quality score (0.0-1.0)
- `quality_reasoning` - LLM explanation for the score
- `input_summary` - Summary of node input
- `output_summary` - Summary of node output
- `judge_model` - Model used for evaluation

**Evaluation Criteria by Node:**
| Node | Success Criteria |
|------|------------------|
| parse_input | Correctly identified public/private, valid ticker, confidence > 0.5 |
| validate_company | Confirmed company exists, appropriate validation status |
| create_plan | Plan has 3-10 tasks, tools relevant for company type |
| fetch_api_data | Made API calls, retrieved data from ≥1 source (PUBLIC only) |
| search_web | Executed searches, found relevant results |
| search_web_enhanced | Enhanced search with additional queries (PRIVATE companies) |
| synthesize | Valid risk_level, credit_score (300-850), confidence (0-1) |
| save_to_database | Data saved without errors |
| evaluate_assessment | Produced evaluation scores |

**Provider:** Groq (llama-3.3-70b-versatile) - **FREE**

**When Logged:** After all evaluations complete (one entry per node per run).

---

## Evaluation Framework Summary

| Framework | Tab | Provider | Model | Cost | Metrics |
|-----------|-----|----------|-------|------|---------|
| Rule-Based | evaluations, tool_selections, agent_metrics | Local | N/A | Free | F1, precision, recall |
| DeepEval | deepeval_metrics | Groq | llama-3.3-70b | Free | hallucination, faithfulness |
| OpenEvals | openevals_metrics | OpenAI | gpt-4o-mini | $0.001 | helpfulness, coherence |
| LLM Judge | llm_judge_results | Groq | llama-3.3-70b | Free | accuracy, completeness |
| Node Scoring | node_scoring | Groq | llama-3.3-70b | Free | task_completed, quality_score |

---

## Common Columns (All Sheets)

| Column | Description |
|--------|-------------|
| `run_id` | UUID identifying the run |
| `company_name` | Company being analyzed |
| `node` | Current LangGraph node |
| `agent_name` | Canonical agent name |
| `timestamp` | ISO timestamp when logged |
| `status` | ok/error |
| `generated_by` | Data source (Us/FW/DeepEval/OpenEvals) |

---

## Canonical Agent Names

| Node | agent_name | Description |
|------|------------|-------------|
| parse_input | `llm_parser` | Parses company input |
| validate_company | `supervisor` | Validates company |
| create_plan | `tool_supervisor` | LLM tool selection |
| fetch_api_data | `api_agent` | Fetches API data (PUBLIC companies only) |
| search_web | `search_agent` | Web search (normal mode) |
| search_web_enhanced | `search_agent` | Enhanced web search (PRIVATE companies or low API data) |
| synthesize | `llm_analyst` | Credit synthesis |
| save_to_database | `db_writer` | Database storage |
| evaluate_assessment | `workflow_evaluator` | All evaluations + node scoring |

---

## generated_by Values

| Value | Meaning |
|-------|---------|
| `Us` | Generated by our custom code |
| `FW` | From Framework (LangGraph/LangChain) |
| `Mixed` | Row logged by us, some values from Framework |
| `DeepEval` | From DeepEval library |
| `OpenEvals` | From OpenEvals library |
