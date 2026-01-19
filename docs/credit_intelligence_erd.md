# Credit Intelligence ERD (Mermaid)

## Legend
- **PK** = Primary Key
- **FK** = Foreign Key (links to wf_runs.run_id)
- **J** = Join Column (hierarchy fields for cross-table joins)

## ERD Diagram

```mermaid
erDiagram
    %% LEGEND:
    %% PK = Primary Key | FK = Foreign Key | J = Join Column
    %% Workflow (wf_*) | LangGraph (lg_*) | Evaluation (eval_*) | Metadata (meta_*)

    %% Workflow tables
    wf_runs ||--o{ wf_llm_calls : "run_id"
    wf_runs ||--o{ wf_tool_calls : "run_id"
    wf_runs ||--o{ wf_assessments : "run_id"
    wf_runs ||--o{ wf_plans : "run_id"
    wf_runs ||--o{ wf_data_sources : "run_id"
    wf_runs ||--o{ wf_state_dumps : "run_id"

    %% LangGraph tables
    wf_runs ||--o{ lg_events : "run_id"

    %% Evaluation tables
    wf_runs ||--o{ eval_results : "run_id"
    wf_runs ||--o{ eval_tool_selection : "run_id"
    wf_runs ||--o{ eval_consistency : "run_id"
    wf_runs ||--o{ eval_coalition : "run_id"
    wf_runs ||--o{ eval_agent_metrics : "run_id"
    wf_runs ||--o{ eval_llm_judge : "run_id"
    wf_runs ||--o{ eval_cross_model : "run_id"
    wf_runs ||--o{ eval_log_tests : "run_id"

    %% Metadata tables
    wf_runs ||--o{ meta_prompts : "run_id"

    wf_runs {
        bigint id PK
        varchar run_id UK
        varchar company_name J
        varchar node J
        varchar agent_name J
        varchar master_agent J
        varchar status
        varchar risk_level
        int credit_score
        decimal confidence
        decimal evaluation_score
        decimal total_time_ms
        int total_steps
        int total_llm_calls
        decimal tool_overall_score
        decimal agent_overall_score
        decimal workflow_overall_score
        timestamptz timestamp
    }

    wf_llm_calls {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar call_type
        varchar model
        int prompt_tokens
        int completion_tokens
        int total_tokens
        decimal total_cost
        decimal execution_time_ms
        varchar status
        timestamptz timestamp
    }

    wf_tool_calls {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar tool_name
        jsonb tool_input
        jsonb tool_output
        decimal execution_time_ms
        varchar status
        int call_depth
        timestamptz timestamp
    }

    wf_assessments {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar risk_level
        int credit_score
        decimal confidence
        text reasoning
        text recommendations
        decimal duration_ms
        timestamptz timestamp
    }

    wf_plans {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int num_tasks
        text plan_summary
        jsonb full_plan
        timestamptz timestamp
    }

    wf_data_sources {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar source_name
        int records_found
        decimal execution_time_ms
        varchar status
        timestamptz timestamp
    }

    wf_state_dumps {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        int step_number J
        jsonb company_info_json
        jsonb plan_json
        jsonb assessment_json
        jsonb evaluation_json
        int total_state_size_bytes
        decimal duration_ms
        timestamptz timestamp
    }

    lg_events {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar event_type
        varchar event_name
        int tokens
        decimal duration_ms
        varchar status
        timestamptz timestamp
    }

    eval_results {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        decimal tool_selection_score
        decimal data_quality_score
        decimal synthesis_score
        decimal overall_score
        varchar eval_status
        timestamptz timestamp
    }

    eval_tool_selection {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        jsonb selected_tools
        jsonb expected_tools
        jsonb correct_tools
        jsonb missing_tools
        decimal precision
        decimal recall
        decimal f1_score
        timestamptz timestamp
    }

    eval_consistency {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar model_name
        varchar evaluation_type
        int num_runs
        decimal risk_level_consistency
        decimal score_consistency
        decimal score_std
        decimal overall_consistency
        varchar eval_status
        timestamptz timestamp
    }

    eval_coalition {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        boolean is_correct
        decimal correctness_score
        decimal efficiency_score
        decimal quality_score
        decimal agreement_score
        int num_evaluators
        jsonb votes_json
        timestamptz timestamp
    }

    eval_agent_metrics {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        decimal intent_correctness
        decimal plan_quality
        decimal tool_choice_correctness
        decimal tool_completeness
        decimal trajectory_match
        decimal final_answer_quality
        int step_count
        int tool_calls
        decimal latency_ms
        decimal overall_score
        varchar eval_status
        timestamptz timestamp
    }

    eval_llm_judge {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar model_used
        decimal accuracy_score
        decimal completeness_score
        decimal consistency_score
        decimal actionability_score
        decimal data_utilization_score
        decimal overall_score
        decimal benchmark_alignment
        varchar eval_status
        int tokens_used
        decimal evaluation_cost
        timestamptz timestamp
    }

    eval_cross_model {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        jsonb models_compared
        int num_models
        decimal risk_level_agreement
        decimal credit_score_mean
        decimal credit_score_std
        decimal credit_score_range
        decimal confidence_agreement
        decimal cross_model_agreement
        varchar eval_status
        timestamptz timestamp
    }

    eval_log_tests {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        int runs
        int langgraph_events
        int llm_calls
        int tool_calls
        int assessments
        int evaluations
        int tool_selections
        int consistency_scores
        int data_sources
        int plans
        int prompts
        int total_sheets_logged
        varchar verification_status
        timestamptz timestamp
    }

    meta_prompts {
        bigint id PK
        varchar run_id FK
        varchar company_name J
        varchar node J
        varchar agent_name J
        int step_number J
        varchar prompt_id
        varchar prompt_name
        varchar category
        text system_prompt
        text user_prompt
        jsonb variables_json
        timestamptz timestamp
    }
```

## Table Summary

| Category | Prefix | Count | Tables |
|----------|--------|-------|--------|
| Workflow | `wf_*` | 7 | wf_runs, wf_llm_calls, wf_tool_calls, wf_assessments, wf_plans, wf_data_sources, wf_state_dumps |
| LangGraph | `lg_*` | 1 | lg_events |
| Evaluation | `eval_*` | 8 | eval_results, eval_tool_selection, eval_consistency, eval_coalition, eval_agent_metrics, eval_llm_judge, eval_cross_model, eval_log_tests |
| Metadata | `meta_*` | 1 | meta_prompts |
| **Total** | | **17** | |

## Storage Mapping

| Google Sheet | PostgreSQL Table |
|--------------|------------------|
| runs | wf_runs |
| llm_calls | wf_llm_calls |
| tool_calls | wf_tool_calls |
| assessments | wf_assessments |
| plans | wf_plans |
| data_sources | wf_data_sources |
| state_dumps | wf_state_dumps |
| langgraph_events | lg_events |
| evaluations | eval_results |
| tool_selections | eval_tool_selection |
| consistency_scores | eval_consistency |
| coalition | eval_coalition |
| agent_metrics | eval_agent_metrics |
| llm_judge_results | eval_llm_judge |
| cross_model_eval | eval_cross_model |
| log_tests | eval_log_tests |
| prompts | meta_prompts |

## Database Connection

```
Host: c5cnr847jq0fj3.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com
Port: 5432
Database: dfau9ra4slsntc
SSL: require
```

## Storage Summary

- **Google Sheets**: 17 sheets (runs sheet includes 3 performance metrics columns)
- **PostgreSQL**: 17 base tables + monthly partitions
- **MongoDB**: 17 collections (mirrors Google Sheets)

### Partitioning
All PostgreSQL tables are partitioned by `timestamp`:
- Pattern: `{table_name}_2026_{MM}` (e.g., wf_runs_2026_01)
- 16 partitioned tables × 12 months = 192 partitions
- meta_prompts has 12 partitions
