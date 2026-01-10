# Google Sheets Cleanup Plan

## Issues Identified

### 1. Sheets to DELETE (redundant)
- `step_logs` → redundant with `langgraph_events`
- `run_summaries` → redundant with `runs`

### 2. Agent Name Inconsistencies
| Sheet | Current | Should Be |
|-------|---------|-----------|
| data_sources | `data_fetcher` | `api_agent` |
| agent_metrics | `agent_efficiency_evaluator` | `workflow_evaluator` |
| llm_judge_results | `llm_judge` | `workflow_evaluator` |

### 3. Missing Data Issues
| Sheet | Issue | Fix |
|-------|-------|-----|
| runs | model is empty | Add model field when logging |
| consistency_scores | step_number=0, only 3 runs | Fix to log 6 runs (2 models × 3 runs) |
| model_consistency | no data | Investigate why not logging |
| cross_model_eval | no data | Investigate why not logging |
| deepeval_metrics | no data | Check if deepeval is configured |
| prompts | only 1 prompt | Log ALL prompts used |

### 4. Consolidation
- Keep `llm_calls` (comprehensive)
- DELETE `llm_calls_detailed` (redundant)
- Ensure `llm_calls` captures: parse_input, tool_selection, credit_analysis

### 5. log_tests Sheet
- Redesign to show: sheet_name, has_data (yes/no), row_count

## Canonical Agent Names (Final)
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

## Final Sheets (After Cleanup)
1. `runs` - Run summaries with model info
2. `langgraph_events` - All workflow events
3. `llm_calls` - All LLM calls with full details
4. `tool_calls` - Tool execution logs
5. `assessments` - Credit assessments
6. `evaluations` - Evaluation results
7. `tool_selections` - Tool selection decisions
8. `consistency_scores` - Model consistency (6 runs)
9. `data_sources` - Data source results
10. `plans` - Execution plans
11. `prompts` - ALL prompts used
12. `log_tests` - Verification (sheet_name, has_data, count)

## Sheets to Delete
- `step_logs`
- `run_summaries`
- `llm_calls_detailed`
- `agent_metrics` (merge into evaluations)
- `llm_judge_results` (merge into evaluations)
- `model_consistency` (merge into consistency_scores)
- `cross_model_eval` (merge into consistency_scores)
- `deepeval_metrics` (if not using deepeval)
