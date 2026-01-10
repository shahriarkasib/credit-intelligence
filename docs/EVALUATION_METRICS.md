# Evaluation Metrics Documentation (Sheet-wise)

---

## Score Thresholds (All Sheets)

| eval_status | Score Range |
|-------------|-------------|
| `good` | >= 0.80 |
| `average` | 0.60 - 0.79 |
| `bad` | < 0.60 |

---

## Canonical Agent Names

All logging uses these standardized agent names:

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

---

## Sheet 1: `runs`

Run summaries with final assessment results.

| Column | Type | Description | Calculation/Source |
|--------|------|-------------|-------------------|
| `run_id` | UUID | Unique run identifier | Generated at start |
| `company_name` | Text | Company analyzed | User input |
| `node` | Text | Final node | Usually "evaluate" |
| `agent_name` | Text | Agent that logged | `llm_analyst` |
| `model` | Text | Primary LLM model | e.g., "llama-3.3-70b-versatile" |
| `temperature` | Float | LLM temperature | e.g., 0.1 |
| `status` | Text | Run status | completed/failed |
| `started_at` | ISO | Start timestamp | When run began |
| `completed_at` | ISO | End timestamp | When run finished |
| `risk_level` | LLM | Risk assessment | LOW/MODERATE/HIGH/CRITICAL |
| `credit_score` | LLM | Credit score 300-850 | LLM estimates from data |
| `confidence` | LLM | Confidence 0-1 | LLM self-assessment |
| `total_time_ms` | Measured | Total duration | Execution time |
| `total_steps` | Count | Steps executed | Count of nodes |
| `total_llm_calls` | Count | LLM API calls | Count from llm_calls |
| `tools_used` | List | Tools executed | From task_plan actions |
| `evaluation_score` | Formula | Overall eval | Weighted avg of scores |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 2: `tool_calls`

Tool execution logs with hierarchy tracking.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | e.g., "fetch_api_data" |
| `node_type` | Text | Node type | "tool" |
| `agent_name` | Text | Executing agent | `api_agent` |
| `step_number` | Int | Step in workflow | Sequential |
| `tool_name` | Text | Tool executed | e.g., "sec_edgar" |
| `tool_input` | JSON | Input parameters | Tool params |
| `tool_output` | JSON | Output data | API response |
| `parent_node` | Text | Parent node | For hierarchy |
| `workflow_phase` | Text | Current phase | e.g., "data_collection" |
| `call_depth` | Int | Nesting level | 0=top-level |
| `parent_tool_id` | Text | Parent tool | For nested calls |
| `execution_time_ms` | Measured | Duration | Tool execution time |
| `status` | Text | Result | ok/fail |
| `error` | Text | Error message | If failed |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 3: `assessments`

Credit assessment outputs from LLM synthesis.

| Column | Type | Description | Calculation/Source |
|--------|------|-------------|-------------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "synthesize" |
| `node_type` | Text | Node type | "agent" |
| `agent_name` | Text | Agent | `llm_analyst` |
| `step_number` | Int | Step number | Sequential |
| `model` | Text | LLM model | Model used |
| `temperature` | Float | Temperature | LLM setting |
| `prompt` | Text | Prompt used | credit_synthesis |
| `risk_level` | LLM | Risk level | LOW/MODERATE/HIGH/CRITICAL |
| `credit_score` | LLM | Score 300-850 | LLM estimation |
| `confidence` | LLM | Confidence 0-1 | LLM self-assessment |
| `reasoning` | LLM | Explanation | LLM generated |
| `recommendations` | LLM | Action items | LLM generated |
| `duration_ms` | Measured | LLM call time | Execution duration |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 4: `evaluations`

Main evaluation scores for each run.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "evaluate" |
| `node_type` | Text | Node type | "agent" |
| `agent_name` | Text | Agent | `workflow_evaluator` |
| `step_number` | Int | Step number | Sequential |
| `model` | Text | Model used | For LLM evals |
| `tool_selection_score` | Rule | Tool F1 | `2×(P×R)/(P+R)` |
| `tool_reasoning` | Text | Explanation | Why these tools |
| `data_quality_score` | Rule | Completeness | % expected fields present |
| `data_reasoning` | Text | Explanation | Data quality notes |
| `synthesis_score` | Rule | Output quality | % required output fields |
| `synthesis_reasoning` | Text | Explanation | Synthesis notes |
| `overall_score` | Formula | Combined | `(tool×0.3)+(data×0.3)+(synthesis×0.4)` |
| `eval_status` | Formula | Category | good/average/bad |
| `duration_ms` | Measured | Duration | Evaluation time |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 5: `tool_selections`

Detailed tool selection analysis.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "create_plan" |
| `node_type` | Text | Node type | "agent" |
| `agent_name` | Text | Agent | `tool_supervisor` |
| `step_number` | Int | Step number | Sequential |
| `model` | Text | LLM model | e.g., "llama-3.3-70b-versatile" |
| `selected_tools` | List | Tools chosen | From LLM selection |
| `expected_tools` | Rule | Expected tools | Based on company type |
| `correct_tools` | Rule | Intersection | `selected ∩ expected` |
| `missing_tools` | Rule | Not used | `expected - selected` |
| `extra_tools` | Rule | Unnecessary | `selected - expected` |
| `precision` | Formula | Correctness | `correct / selected` |
| `recall` | Formula | Completeness | `correct / expected` |
| `f1_score` | Formula | Balanced | `2×(P×R)/(P+R)` |
| `reasoning` | Text | Explanation | Why tools selected |
| `duration_ms` | Measured | Duration | Selection time |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 6: `llm_calls`

All LLM API calls with full details and costs.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | e.g., "create_plan" |
| `node_type` | Text | Node type | "llm" |
| `agent_name` | Text | Calling agent | e.g., `tool_supervisor` |
| `step_number` | Int | Step number | Sequential |
| `call_type` | Text | Call purpose | e.g., "tool_selection" |
| `model` | Text | LLM model | Model identifier |
| `temperature` | Float | Temperature | LLM setting |
| `prompt` | Text | Full prompt | System + User |
| `response` | Text | Full response | LLM output |
| `reasoning` | Text | Reasoning | LLM explanation |
| `context` | Text | Context data | Input context |
| `current_task` | Text | Task | What triggered call |
| `prompt_tokens` | Measured | Input tokens | From LLM response |
| `completion_tokens` | Measured | Output tokens | From LLM response |
| `total_tokens` | Formula | Total | `prompt + completion` |
| `input_cost` | Formula | Input USD | `tokens × rate` |
| `output_cost` | Formula | Output USD | `tokens × rate` |
| `total_cost` | Formula | Total USD | `input + output` |
| `execution_time_ms` | Measured | Duration | API call time |
| `status` | Text | Result | ok/error |
| `error` | Text | Error | If failed |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Mixed" |

---

## Sheet 7: `consistency_scores`

Same-model and cross-model consistency evaluation.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "evaluate" |
| `node_type` | Text | Node type | "agent" |
| `agent_name` | Text | Agent | `workflow_evaluator` |
| `step_number` | Int | Step number | Sequential |
| `model_name` | Text | Model evaluated | e.g., "llama-3.3-70b-versatile" |
| `evaluation_type` | Label | Type | "same_model" or "cross_model" |
| `num_runs` | Count | Runs compared | Usually 3 |
| `risk_level_consistency` | Formula | Risk agreement | `count(mode) / total_runs` |
| `score_consistency` | Formula | Score stability | `1 - (std / mean)` |
| `score_std` | Formula | Std deviation | `std(credit_scores)` |
| `overall_consistency` | Formula | Combined | `(risk×0.5) + (score×0.5)` |
| `eval_status` | Formula | Category | good/average/bad |
| `risk_levels` | Data | All risks | e.g., "LOW, LOW, MODERATE" |
| `credit_scores` | Data | All scores | e.g., "750, 745, 720" |
| `duration_ms` | Measured | Duration | Evaluation time |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 8: `data_sources`

Data fetching results from APIs.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "fetch_api_data" |
| `node_type` | Text | Node type | "tool" |
| `agent_name` | Text | Agent | `api_agent` |
| `step_number` | Int | Step number | Sequential |
| `source_name` | Text | Data source | e.g., "sec_edgar" |
| `records_found` | Count | Records | Count from API |
| `data_summary` | JSON | Data preview | Full data (up to 50k) |
| `execution_time_ms` | Measured | Duration | API call time |
| `status` | Text | Result | ok/fail |
| `error` | Text | Error | If failed |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 9: `langgraph_events`

LangGraph execution events from astream_events.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | Graph node |
| `node_type` | Text | Node type | agent/tool/llm |
| `agent_name` | Text | Agent | Agent executing |
| `step_number` | Int | Step number | Sequential |
| `event_type` | Text | Event type | e.g., "on_chain_start" |
| `event_name` | Text | Event name | Event identifier |
| `model` | Text | Model | If LLM event |
| `temperature` | Float | Temperature | If LLM event |
| `tokens` | Int | Token count | If LLM event |
| `input_preview` | Text | Input preview | First 10k chars |
| `output_preview` | Text | Output preview | First 10k chars |
| `duration_ms` | Measured | Duration | Event duration |
| `status` | Text | Result | ok/error |
| `error` | Text | Error | If failed |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "FW" (Framework) |

---

## Sheet 10: `plans`

Task plans created for each run.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "create_plan" |
| `agent_name` | Text | Agent | `tool_supervisor` |
| `num_tasks` | Count | Task count | Number of tasks |
| `plan_summary` | Text | Summary | Brief description |
| `full_plan` | JSON | Full plan | All task details |
| `task_1` - `task_10` | JSON | Individual tasks | Up to 10 tasks |
| `created_at` | ISO | Created | When plan made |
| `status` | Text | Result | ok/error |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 11: `prompts`

All prompts used in runs.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | Node using prompt |
| `agent_name` | Text | Agent | Agent using prompt |
| `step_number` | Int | Step number | Sequential |
| `prompt_id` | Text | Prompt ID | e.g., "company_parser" |
| `prompt_name` | Text | Name | Human-readable |
| `category` | Text | Category | input/planning/synthesis |
| `system_prompt` | Text | System | Full system prompt |
| `user_prompt` | Text | User | Full user prompt |
| `variables_json` | JSON | Variables | Variables used |
| `model` | Text | Model | LLM model |
| `temperature` | Float | Temperature | LLM setting |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 12: `cross_model_eval`

Cross-model comparison results.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `run_id` | UUID | Evaluation ID | Unique ID |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "synthesize" |
| `node_type` | Text | Node type | "agent" |
| `agent_name` | Text | Agent | `workflow_evaluator` |
| `step_number` | Int | Step number | Sequential |
| `models_compared` | List | Models | e.g., "llama-3.3-70b, llama-3.1-8b" |
| `num_models` | Count | Model count | Number compared |
| `risk_level_agreement` | Formula | Risk match | 1.0 if all agree |
| `credit_score_mean` | Formula | Average | `mean(all_scores)` |
| `credit_score_std` | Formula | Variation | `std(all_scores)` |
| `credit_score_range` | Formula | Spread | `max - min` |
| `confidence_agreement` | Formula | Confidence | 1 - normalized variance |
| `best_model` | LLM | Best model | LLM picks best |
| `best_model_reasoning` | LLM | Explanation | LLM explains |
| `cross_model_agreement` | Formula | Overall | Weighted average |
| `eval_status` | Formula | Category | good/average/bad |
| `llm_judge_analysis` | LLM | Analysis | LLM comparison |
| `model_recommendations` | LLM | Suggestions | LLM recommends |
| `model_results` | JSON | Full results | All model outputs |
| `pairwise_comparisons` | JSON | Pairwise | Model pairs |
| `duration_ms` | Measured | Duration | Evaluation time |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 13: `llm_judge_results`

LLM-as-a-Judge evaluation results.

| Column | Type | Description | LLM Prompt/Calculation |
|--------|------|-------------|------------------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "evaluate" |
| `node_type` | Text | Node type | "llm" |
| `agent_name` | Text | Agent | `workflow_evaluator` |
| `step_number` | Int | Step number | Sequential |
| `model_used` | Text | Judge model | LLM doing evaluation |
| `temperature` | Float | Temperature | Usually 0.0 |
| `accuracy_score` | LLM | Risk reasonable? | "Is risk level reasonable?" 0-1 |
| `completeness_score` | LLM | All factors? | "All risk factors covered?" 0-1 |
| `consistency_score` | LLM | Logic coherent? | "Reasoning leads to conclusion?" 0-1 |
| `actionability_score` | LLM | Clear actions? | "Recommendations actionable?" 0-1 |
| `data_utilization_score` | LLM | Data well used? | "Data effectively used?" 0-1 |
| `overall_score` | Formula | Average | `(all scores) / 5` |
| `eval_status` | Formula | Category | good/average/bad |
| `accuracy_reasoning` | LLM | Explanation | LLM explains accuracy |
| `completeness_reasoning` | LLM | Explanation | LLM explains completeness |
| `consistency_reasoning` | LLM | Explanation | LLM explains consistency |
| `actionability_reasoning` | LLM | Explanation | LLM explains actionability |
| `data_utilization_reasoning` | LLM | Explanation | LLM explains data use |
| `overall_reasoning` | LLM | Summary | LLM overall analysis |
| `benchmark_alignment` | LLM | Benchmark | Comparison to expected |
| `benchmark_comparison` | LLM | Details | Benchmark details |
| `suggestions` | LLM | Improvements | LLM suggests fixes |
| `tokens_used` | Measured | Tokens | Judge token usage |
| `evaluation_cost` | Formula | Cost USD | Judge cost |
| `duration_ms` | Measured | Duration | Evaluation time |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

---

## Sheet 14: `agent_metrics`

Agent efficiency evaluation metrics.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `node` | Text | Current node | "evaluate" |
| `node_type` | Text | Node type | "agent" |
| `agent_name` | Text | Agent | `workflow_evaluator` |
| `step_number` | Int | Step number | Sequential |
| `model` | Text | Model | Model evaluated |
| `intent_correctness` | Rule | Task understanding | `(name×0.4)+(type×0.3)+(conf×0.3)` |
| `plan_quality` | Rule | Plan quality | `(size×0.3)+(data×0.35)+(analysis×0.35)` |
| `tool_choice_correctness` | Rule | Precision | `correct / selected` |
| `tool_completeness` | Rule | Recall | `used / expected` |
| `trajectory_match` | Rule | Path following | `(jaccard×0.6)+(order×0.4)` |
| `final_answer_quality` | Rule | Output validity | Field presence + validity |
| `step_count` | Measured | Steps taken | Count of steps |
| `tool_calls` | Measured | Tool calls | Count of tool calls |
| `latency_ms` | Measured | Duration | Total time |
| `overall_score` | Formula | Combined | Weighted average |
| `eval_status` | Formula | Category | good/average/bad |
| `intent_details` | JSON | Details | Intent analysis |
| `plan_details` | JSON | Details | Plan analysis |
| `tool_details` | JSON | Details | Tool analysis |
| `trajectory_details` | JSON | Details | Trajectory analysis |
| `answer_details` | JSON | Details | Answer analysis |
| `status` | Text | Result | ok/error |
| `timestamp` | ISO | Log timestamp | When logged |
| `generated_by` | Label | Data source | "Us" |

**All metrics are Rule-Based (no LLM calls)**

---

## Sheet 15: `log_tests`

Verification of sheet logging per run.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `run_id` | UUID | Run identifier | From state |
| `company_name` | Text | Company | From state |
| `runs` | Count | Rows in runs | Count for run_id |
| `langgraph_events` | Count | Rows | Count for run_id |
| `llm_calls` | Count | Rows | Count for run_id |
| `tool_calls` | Count | Rows | Count for run_id |
| `assessments` | Count | Rows | Count for run_id |
| `evaluations` | Count | Rows | Count for run_id |
| `tool_selections` | Count | Rows | Count for run_id |
| `consistency_scores` | Count | Rows | Count for run_id |
| `data_sources` | Count | Rows | Count for run_id |
| `plans` | Count | Rows | Count for run_id |
| `prompts` | Count | Rows | Count for run_id |
| `cross_model_eval` | Count | Rows | Count for run_id |
| `llm_judge_results` | Count | Rows | Count for run_id |
| `agent_metrics` | Count | Rows | Count for run_id |
| `total_sheets_logged` | Count | Total | Sheets with data |
| `verification_status` | Rule | Status | pass/partial/fail |
| `timestamp` | ISO | Log timestamp | When verified |
| `generated_by` | Label | Data source | "Us" |

---

## Summary: Metric Types

| Type | Description | Example |
|------|-------------|---------|
| `LLM` | Generated by LLM | risk_level, credit_score |
| `Rule` | Calculated by rules | precision, recall, f1_score |
| `Formula` | Mathematical calculation | overall_score, total_cost |
| `Measured` | Direct measurement | execution_time_ms, tokens |
| `Count` | Simple count | step_count, records_found |
| `Data` | Raw data | risk_levels list |
| `Label` | Category label | eval_status, generated_by |

---

## Summary: LLM vs Rule-Based by Sheet

| Sheet | LLM Metrics | Rule/Formula Metrics |
|-------|-------------|---------------------|
| `runs` | 3 (risk, score, confidence) | 4 |
| `assessments` | 5 (all assessment fields) | 0 |
| `evaluations` | 0 | 5 |
| `tool_selections` | 0 | 4 |
| `llm_calls` | 0 | 3 (cost calculations) |
| `consistency_scores` | 0 | 4 |
| `cross_model_eval` | 4 (best_model, analysis) | 6 |
| `llm_judge_results` | 11 (all judge scores) | 1 |
| `agent_metrics` | 0 | 7 |

---

## LLM Providers Used

| Evaluation | Provider | Model |
|------------|----------|-------|
| Credit Analysis | Groq | llama-3.3-70b-versatile |
| LLM Judge | Groq | llama-3.3-70b-versatile |
| Secondary Model | Groq | llama-3.1-8b-instant |
| Tool Selection | Groq | llama-3.3-70b-versatile |

---

## generated_by Values

| Value | Meaning |
|-------|---------|
| `Us` | Data generated by our custom code |
| `FW` | Data from Framework (LangGraph/LangChain) |
| `Mixed` | Row logged by us, some values from Framework |
