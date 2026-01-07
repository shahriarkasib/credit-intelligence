# Evaluation Metrics Documentation (Sheet-wise)

---

## Score Thresholds (All Sheets)

| eval_status | Score Range |
|-------------|-------------|
| `good` | >= 0.80 |
| `average` | 0.60 - 0.79 |
| `bad` | < 0.60 |

---

## Sheet 1: `runs`

Basic run summary with final assessment results.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `risk_level` | LLM | Final risk assessment | LLM synthesizes from collected data |
| `credit_score` | LLM | Credit score 300-850 | LLM estimates based on data |
| `confidence` | LLM | Confidence 0-1 | LLM self-assessed confidence |
| `evaluation_score` | Formula | Overall evaluation | Weighted avg of all eval scores |

---

## Sheet 2: `tool_calls`

Logs each tool execution. No metrics - just execution data.

| Column | Type | Description |
|--------|------|-------------|
| `execution_time_ms` | Measured | Time taken by tool |
| `status` | Measured | ok/fail |

---

## Sheet 3: `assessments`

Credit assessment outputs from LLM.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `risk_level` | LLM | LOW/MODERATE/HIGH/CRITICAL | LLM judgment from data |
| `credit_score` | LLM | 300-850 range | LLM estimation |
| `confidence` | LLM | 0-1 confidence | LLM self-assessment |
| `reasoning` | LLM | Explanation text | LLM generated |
| `recommendations` | LLM | Action items | LLM generated |

---

## Sheet 4: `evaluations`

Main evaluation scores for each run.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `tool_selection_score` | Rule | Did agent pick right tools? | F1 score of selected vs expected tools |
| `tool_reasoning` | Text | Why these tools | Logged explanation |
| `data_quality_score` | Rule | Data completeness | % of expected data fields present |
| `data_reasoning` | Text | Data quality notes | Logged explanation |
| `synthesis_score` | Rule | Assessment completeness | % of required output fields present |
| `synthesis_reasoning` | Text | Synthesis notes | Logged explanation |
| `overall_score` | Formula | Combined score | `(tool×0.3 + data×0.3 + synthesis×0.4)` |
| `eval_status` | Formula | good/average/bad | Based on overall_score threshold |

---

## Sheet 5: `tool_selections`

Detailed tool selection analysis.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `selected_tools` | Measured | Tools agent chose | List from execution |
| `expected_tools` | Rule | Tools it should use | Based on company type |
| `correct_tools` | Rule | Intersection | `selected ∩ expected` |
| `missing_tools` | Rule | Not used but needed | `expected - selected` |
| `extra_tools` | Rule | Used but not needed | `selected - expected` |
| `precision` | Formula | Correctness ratio | `correct / selected` |
| `recall` | Formula | Completeness ratio | `correct / expected` |
| `f1_score` | Formula | Balanced score | `2 × (P × R) / (P + R)` |

---

## Sheet 6: `step_logs`

Step-by-step execution log. No metrics.

---

## Sheet 7: `llm_calls`

LLM call details and costs.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `prompt_tokens` | Measured | Input tokens | From LLM response |
| `completion_tokens` | Measured | Output tokens | From LLM response |
| `total_tokens` | Formula | Sum | `prompt + completion` |
| `input_cost` | Formula | Input cost USD | `tokens × rate` |
| `output_cost` | Formula | Output cost USD | `tokens × rate` |
| `total_cost` | Formula | Total cost | `input + output` |

---

## Sheet 8: `consistency_scores`

Same-model consistency across multiple runs.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `evaluation_type` | Label | same_model or cross_model | - |
| `num_runs` | Count | Number of runs compared | Usually 3 |
| `risk_level_consistency` | Formula | Risk agreement % | `count(mode) / total_runs` |
| `score_consistency` | Formula | Score stability | `1 - (std_dev / mean)` |
| `score_std` | Formula | Score std deviation | `std(credit_scores)` |
| `overall_consistency` | Formula | Combined | `(risk×0.5 + score×0.5)` |
| `eval_status` | Formula | good/average/bad | Based on overall_consistency |
| `risk_levels` | Data | All risk levels | e.g., "LOW, LOW, MODERATE" |
| `credit_scores` | Data | All scores | e.g., "750, 745, 720" |

---

## Sheet 9: `data_sources`

Data fetching results. No metrics.

| Column | Type | Description |
|--------|------|-------------|
| `records_found` | Count | Records retrieved |
| `status` | Measured | ok/fail |

---

## Sheet 10: `langsmith_traces`

LangSmith trace data. No metrics.

---

## Sheet 11: `langgraph_events`

LangGraph execution events. No metrics.

---

## Sheet 12: `llm_calls_detailed`

Detailed LLM calls with full prompts/responses. Same cost metrics as `llm_calls`.

---

## Sheet 13: `run_summaries`

Comprehensive run summary.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `risk_level` | LLM | Final risk | LLM output |
| `credit_score` | LLM | Final score | LLM output |
| `confidence` | LLM | LLM confidence | LLM self-assessment |
| `tool_selection_score` | Rule | Tool F1 | From evaluations |
| `data_quality_score` | Rule | Data completeness | From evaluations |
| `synthesis_score` | Rule | Output completeness | From evaluations |
| `overall_score` | Formula | Combined | Weighted average |
| `total_tokens` | Sum | All tokens used | Sum of all LLM calls |
| `total_cost` | Sum | Total USD | Sum of all costs |

---

## Sheet 14: `agent_metrics`

Agent efficiency evaluation.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `intent_correctness` | Rule | Task understanding | `(name_parsed×0.4) + (type_id×0.3) + (confidence×0.3)` |
| `plan_quality` | Rule | Plan quality | `(size_ok×0.3) + (has_data×0.35) + (has_analysis×0.35)` |
| `tool_choice_correctness` | Rule | Precision | `correct_tools / selected_tools` |
| `tool_completeness` | Rule | Recall | `used_tools / expected_tools` |
| `trajectory_match` | Rule | Path following | `(jaccard×0.6) + (order×0.4)` |
| `final_answer_quality` | Rule | Output validity | `(fields×0.5) + (risk×0.15) + (score×0.15) + (conf×0.1) + (reason×0.1)` |
| `step_count` | Count | Steps taken | Measured |
| `tool_calls` | Count | Tool invocations | Measured |
| `latency_ms` | Measured | Execution time | Measured |
| `overall_score` | Formula | Agent efficiency | `(intent×0.15) + (plan×0.15) + (tool_prec×0.20) + (tool_rec×0.15) + (traj×0.15) + (answer×0.20)` |
| `eval_status` | Formula | good/average/bad | Based on overall_score |

**All metrics are Rule-Based (no LLM calls)**

---

## Sheet 15: `unified_metrics`

Combined evaluation from all sources.

### Accuracy Metrics

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `faithfulness` | **LLM** | Grounded in context? | DeepEval LLM judges 0-1 |
| `hallucination` | **LLM** | Made-up content? | DeepEval LLM judges 0-1 (lower=better) |
| `answer_relevancy` | **LLM** | Relevant to question? | DeepEval LLM judges 0-1 |
| `factual_accuracy` | **LLM** | Facts correct? | OpenAI verifies claims 0-1 |
| `final_answer_quality` | Rule | Output valid? | Rule-based field checks |
| `accuracy_score` | Formula | Combined accuracy | `(faith×0.25) + ((1-halluc)×0.20) + (factual×0.20) + (relevancy×0.15) + (answer×0.20)` |

### Consistency Metrics

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `same_model_consistency` | Formula | Same LLM stability | From consistency_scores |
| `cross_model_consistency` | Formula | Cross-LLM agreement | From cross_model_eval |
| `risk_level_agreement` | Formula | Risk consensus | 1.0 if all agree, else partial |
| `semantic_similarity` | **ML** | Reasoning similarity | Sentence embeddings cosine |
| `consistency_score` | Formula | Combined | `(same×0.30) + (cross×0.30) + (risk×0.20) + (semantic×0.20)` |

### Agent Efficiency Metrics

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `intent_correctness` | Rule | Task understanding | From agent_metrics |
| `plan_quality` | Rule | Plan quality | From agent_metrics |
| `tool_choice_correctness` | Rule | Precision | From agent_metrics |
| `tool_completeness` | Rule | Recall | From agent_metrics |
| `trajectory_match` | Rule | Path following | From agent_metrics |
| `agent_final_answer` | Rule | Output quality | From agent_metrics |
| `agent_efficiency_score` | Formula | Combined | Weighted average |

### Overall

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `overall_quality_score` | Formula | Master score | `(accuracy×0.40) + (consistency×0.30) + (efficiency×0.30)` |
| `eval_status` | Formula | good/average/bad | Based on overall_quality_score |

---

## Sheet 16: `llm_judge_results`

LLM-as-a-Judge evaluation (all LLM-powered).

| Column | Type | Description | LLM Prompt |
|--------|------|-------------|------------|
| `accuracy_score` | **LLM** | Risk reasonable? | "Is this risk assessment reasonable given the data?" |
| `completeness_score` | **LLM** | All factors covered? | "Does this cover all important risk factors?" |
| `consistency_score` | **LLM** | Logic coherent? | "Does the reasoning lead to the conclusion?" |
| `actionability_score` | **LLM** | Clear actions? | "Are recommendations specific and actionable?" |
| `data_utilization_score` | **LLM** | Data well used? | "How well was the collected data utilized?" |
| `overall_score` | Formula | Average | `(acc + comp + cons + act + data) / 5` |
| `eval_status` | Formula | good/average/bad | Based on overall_score |
| `accuracy_reasoning` | **LLM** | Explanation | LLM explains score |
| `completeness_reasoning` | **LLM** | Explanation | LLM explains score |
| `consistency_reasoning` | **LLM** | Explanation | LLM explains score |
| `actionability_reasoning` | **LLM** | Explanation | LLM explains score |
| `data_utilization_reasoning` | **LLM** | Explanation | LLM explains score |
| `overall_reasoning` | **LLM** | Summary | LLM overall analysis |
| `benchmark_alignment` | **LLM** | Benchmark match | LLM compares to expected |
| `suggestions` | **LLM** | Improvements | LLM suggests fixes |

---

## Sheet 17: `model_consistency`

Single model consistency analysis.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `num_runs` | Count | Runs compared | Usually 3 |
| `risk_level_consistency` | Formula | Risk agreement | `count(mode) / total` |
| `credit_score_mean` | Formula | Average score | `mean(scores)` |
| `credit_score_std` | Formula | Score variation | `std(scores)` |
| `confidence_variance` | Formula | Confidence spread | `variance(confidences)` |
| `reasoning_similarity` | **ML** | Text similarity | Sentence embeddings cosine |
| `risk_factors_overlap` | Formula | Factor agreement | Jaccard similarity |
| `recommendations_overlap` | Formula | Rec agreement | Jaccard similarity |
| `overall_consistency` | Formula | Combined | Weighted average |
| `is_consistent` | Rule | Pass/Fail | Yes if overall >= 0.8 |
| `consistency_grade` | Rule | A/B/C/D/F | Based on overall |
| `eval_status` | Formula | good/average/bad | Based on overall |
| `llm_judge_analysis` | **LLM** | LLM analysis | LLM reviews consistency |
| `llm_judge_concerns` | **LLM** | Issues found | LLM lists concerns |

---

## Sheet 18: `cross_model_eval`

Cross-model comparison.

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `models_compared` | List | LLMs compared | e.g., "groq, openai" |
| `num_models` | Count | Model count | Number of models |
| `risk_level_agreement` | Formula | All same risk? | 1.0 if unanimous, else partial |
| `credit_score_mean` | Formula | Average | `mean(all_scores)` |
| `credit_score_std` | Formula | Variation | `std(all_scores)` |
| `credit_score_range` | Formula | Spread | `max - min` |
| `confidence_agreement` | Formula | Confidence match | 1 - normalized variance |
| `best_model` | **LLM** | Recommended model | LLM picks best |
| `best_model_reasoning` | **LLM** | Why best | LLM explains |
| `cross_model_agreement` | Formula | Overall agreement | Weighted average |
| `eval_status` | Formula | good/average/bad | Based on agreement |
| `llm_judge_analysis` | **LLM** | LLM comparison | LLM analyzes differences |
| `model_recommendations` | **LLM** | Suggestions | LLM recommends |

---

## Sheet 19: `deepeval_metrics`

DeepEval library metrics (all LLM-powered).

| Column | Type | Description | LLM Prompt |
|--------|------|-------------|------------|
| `answer_relevancy` | **LLM** | Relevant answer? | "Is this answer relevant to the question?" |
| `faithfulness` | **LLM** | Grounded in data? | "Is every claim supported by context?" |
| `hallucination` | **LLM** | Made-up info? | "Contains info not in context?" (lower=better) |
| `contextual_relevancy` | **LLM** | Context useful? | "Is the context relevant?" |
| `bias` | **LLM** | Shows bias? | "Does answer show unfair bias?" (lower=better) |
| `toxicity` | **LLM** | Harmful content? | "Contains toxic content?" (lower=better) |
| `overall_score` | Formula | Combined | `(rel×0.25) + (faith×0.30) + ((1-hal)×0.25) + (ctx×0.10) + ((1-bias)×0.10)` |
| `eval_status` | Formula | good/average/bad | Based on overall_score |
| `answer_relevancy_reason` | **LLM** | Explanation | LLM reasoning |
| `faithfulness_reason` | **LLM** | Explanation | LLM reasoning |
| `hallucination_reason` | **LLM** | Explanation | LLM reasoning |
| `contextual_relevancy_reason` | **LLM** | Explanation | LLM reasoning |
| `bias_reason` | **LLM** | Explanation | LLM reasoning |

---

## Sheet 20: `openevals_metrics`

Agent efficiency metrics (all Rule-based).

| Column | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `intent_correctness` | Rule | Understood task? | `(name×0.4) + (type×0.3) + (conf×0.3)` |
| `plan_quality` | Rule | Good plan? | `(size×0.3) + (data×0.35) + (analysis×0.35)` |
| `tool_choice_correctness` | Rule | Right tools? | `correct / selected` (Precision) |
| `tool_completeness` | Rule | All tools used? | `used / expected` (Recall) |
| `trajectory_match` | Rule | Followed path? | `(jaccard×0.6) + (order×0.4)` |
| `final_answer_quality` | Rule | Valid output? | Field presence + validity checks |
| `step_count` | Count | Steps taken | Measured |
| `tool_calls` | Count | Tool uses | Measured |
| `latency_ms` | Measured | Time taken | Measured |
| `overall_score` | Formula | Combined | `(int×0.15) + (plan×0.15) + (prec×0.20) + (rec×0.15) + (traj×0.15) + (ans×0.20)` |
| `eval_status` | Formula | good/average/bad | Based on overall_score |

---

## Summary: LLM vs Rule-Based

| Sheet | LLM Metrics | Rule/Formula Metrics |
|-------|-------------|---------------------|
| `evaluations` | 0 | 4 |
| `consistency_scores` | 0 | 4 |
| `agent_metrics` | 0 | 7 |
| `unified_metrics` | 4 (DeepEval) + 1 (factual) | 12 |
| `llm_judge_results` | 5 scores + 6 reasoning | 1 |
| `model_consistency` | 2 | 9 |
| `cross_model_eval` | 3 | 7 |
| `deepeval_metrics` | 6 scores + 5 reasoning | 1 |
| `openevals_metrics` | 0 | 7 |

---

## LLM Providers Used

| Evaluation | Provider | Model |
|------------|----------|-------|
| DeepEval | Groq (free) | llama-3.3-70b via LiteLLM |
| LLM Judge | Groq | llama-3.3-70b-versatile |
| Factual Accuracy | OpenAI | gpt-4o-mini |
| Semantic Similarity | Local | all-MiniLM-L6-v2 (embeddings) |
