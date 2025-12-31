"""LangGraph entry point for LangGraph Studio visualization.

This module exports the compiled graph for use with:
1. LangGraph Studio (desktop app)
2. LangSmith (cloud tracing)
3. Programmatic visualization

Now includes integrated evaluation framework for:
- Tool selection accuracy
- Data quality assessment
- Synthesis quality scoring
- Consistency metrics
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

# Import agents
from agents.supervisor import SupervisorAgent
from agents.api_agent import APIAgent
from agents.search_agent import SearchAgent
from agents.llm_analyst import LLMAnalystAgent

# Import LLM parser for company input
try:
    from agents.llm_parser import get_llm_parser, parse_company_input as llm_parse_company
    LLM_PARSER_AVAILABLE = True
except ImportError:
    LLM_PARSER_AVAILABLE = False
    llm_parse_company = None

# Import LangSmith configuration
try:
    from config.langsmith_config import setup_langsmith, is_langsmith_enabled, get_langsmith_url
    from config.step_logs import StepLogger, PROMPTS
    LANGSMITH_CONFIG_AVAILABLE = True
    # Setup LangSmith tracing
    setup_langsmith(project_name="credit-intelligence")
except ImportError:
    LANGSMITH_CONFIG_AVAILABLE = False
    StepLogger = None
    PROMPTS = {}

# Import MongoDB storage
try:
    from storage.mongodb import CreditIntelligenceDB
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Import evaluation framework
try:
    from evaluation import ToolSelectionEvaluator, WorkflowEvaluator
    from run_logging import RunLogger, MetricsCollector, get_workflow_logger
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    ToolSelectionEvaluator = None
    WorkflowEvaluator = None
    get_workflow_logger = None

# Initialize workflow logger
wf_logger = get_workflow_logger() if get_workflow_logger else None

# Configure logging
logger = logging.getLogger(__name__)

# Results directory for saving evaluations
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class CreditWorkflowInput(TypedDict):
    """Simple input - just the company name."""
    company_name: str


class CreditWorkflowOutput(TypedDict):
    """Output from the workflow."""
    company_name: str
    risk_level: str
    credit_score: int
    confidence: float
    reasoning: str
    recommendations: List[str]
    # Evaluation scores
    evaluation_score: float
    data_quality_score: float


class CreditWorkflowState(TypedDict):
    """State for the credit intelligence workflow."""
    # Input (required)
    company_name: str

    # Optional inputs with defaults
    jurisdiction: Optional[str]
    ticker: Optional[str]

    # Run tracking
    run_id: str

    # Parsed company info
    company_info: Dict[str, Any]

    # Task plan
    task_plan: List[Dict[str, Any]]

    # Agent results
    api_data: Dict[str, Any]
    search_data: Dict[str, Any]

    # Final output
    assessment: Optional[Dict[str, Any]]

    # Multi-LLM results for consistency evaluation
    llm_results: List[Dict[str, Any]]

    # Evaluation results
    evaluation: Optional[Dict[str, Any]]
    tool_selection_score: float
    data_quality_score: float
    synthesis_score: float

    # Metadata
    errors: List[str]
    status: str
    execution_time_ms: float

    # Human-in-the-loop fields
    human_approved: bool
    human_feedback: Optional[str]
    validation_message: str
    requires_review: bool


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

# Initialize agents
supervisor = SupervisorAgent({"analysis_mode": "llm"})
api_agent = APIAgent()
search_agent = SearchAgent()

# Initialize MongoDB (optional)
db = CreditIntelligenceDB() if MONGODB_AVAILABLE else None


def parse_input(state: CreditWorkflowState) -> Dict[str, Any]:
    """Parse and enrich company input using LLM.

    This step now uses an LLM to intelligently parse company names and determine:
    - Whether the company is public or private
    - Stock ticker symbol (if applicable)
    - Jurisdiction and industry
    - Confidence in the determination
    """
    import time
    start_time = time.time()

    company_name = state["company_name"].strip()

    # Start run in workflow logger and use its run_id
    if wf_logger:
        run_id = wf_logger.start_run(company_name, {"source": "langgraph"})
    else:
        run_id = str(uuid.uuid4())

    logger.info(f"Starting run {run_id} for company: {company_name}")

    # Initialize step logger for tracking prompts
    step_logger = StepLogger(run_id) if StepLogger else None

    # Validate input
    if not company_name:
        result = {
            "run_id": run_id,
            "company_info": {},
            "status": "error",
            "errors": ["Company name cannot be empty"],
            "validation_message": "ERROR: Please provide a valid company name.",
            "requires_review": False,
        }
        # Log step
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="parse_input",
                input_data={"company_name": company_name},
                output_data={"error": "Company name empty"},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error="Company name cannot be empty",
            )
        return result

    # Use LLM parser if available, otherwise fallback to rule-based
    llm_metrics = {}
    if LLM_PARSER_AVAILABLE and llm_parse_company:
        logger.info(f"Using LLM parser for: {company_name}")
        parser = get_llm_parser()
        company_info, llm_metrics = parser.parse_company_llm(
            company_name,
            context={"jurisdiction": state.get("jurisdiction")},
            step_logger=step_logger,
        )
    else:
        logger.info(f"Using rule-based parser for: {company_name}")
        company_info = supervisor.parse_company_input(
            company_name,
            state.get("jurisdiction"),
        )

    # Check if company was recognized
    is_known = company_info.get("is_public_company", False)
    ticker = company_info.get("ticker")
    confidence = company_info.get("confidence", 0.5)
    parsed_by = company_info.get("parsed_by", "rule-based")
    reasoning = company_info.get("reasoning", "")

    if is_known and ticker:
        validation_msg = f"Found: {company_name} (Ticker: {ticker}) - Public company, full data available."
        if reasoning:
            validation_msg += f" [{reasoning}]"
        requires_review = False
    else:
        validation_msg = f"Note: '{company_name}' identified as private company. Will search with limited data sources."
        if reasoning:
            validation_msg += f" [{reasoning}]"
        requires_review = confidence < 0.7  # Review if low confidence

    result = {
        "run_id": run_id,
        "company_info": company_info,
        "ticker": ticker,
        "status": "input_parsed",
        "validation_message": validation_msg,
        "requires_review": requires_review,
    }

    # Log step with LLM metrics (include LLM info in output_data)
    if wf_logger:
        wf_logger.log_step(
            run_id=run_id,
            company_name=company_name,
            step_name="parse_input",
            input_data={"company_name": company_name, "jurisdiction": state.get("jurisdiction")},
            output_data={
                "is_public": is_known,
                "ticker": ticker,
                "confidence": confidence,
                "parsed_by": parsed_by,
                "reasoning": reasoning,
                "requires_review": requires_review,
                "llm_metrics": llm_metrics,
                "llm_model": llm_metrics.get("model"),
                "tokens_used": llm_metrics.get("total_tokens"),
            },
            execution_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    return result


def validate_company(state: CreditWorkflowState) -> Dict[str, Any]:
    """
    Human-in-the-loop checkpoint: Validate company before proceeding.
    This node pauses for human approval if the company is unknown.
    """
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")

    validation_msg = state.get("validation_message", "")
    requires_review = state.get("requires_review", False)
    company_info = state.get("company_info", {})

    if not company_info:
        result = {
            "status": "validation_failed",
            "errors": state.get("errors", []) + ["Company validation failed"],
        }
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="validate_company",
                input_data={"validation_message": validation_msg},
                output_data={"status": "validation_failed"},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error="Company validation failed",
            )
        return result

    result = {
        "status": "validated",
        "human_approved": True,
    }

    if wf_logger:
        wf_logger.log_step(
            run_id=run_id,
            company_name=company_name,
            step_name="validate_company",
            input_data={"validation_message": validation_msg, "requires_review": requires_review},
            output_data={"status": "validated", "human_approved": True},
            execution_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    return result


def create_plan(state: CreditWorkflowState) -> Dict[str, Any]:
    """Create task plan based on company info."""
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")

    task_plan = supervisor.create_task_plan(state["company_info"])

    # Show plan summary for human review
    plan_summary = f"Plan: Fetching data from {len(task_plan)} sources for {state['company_name']}"

    result = {
        "task_plan": task_plan,
        "status": "plan_created",
        "validation_message": plan_summary,
    }

    if wf_logger:
        wf_logger.log_step(
            run_id=run_id,
            company_name=company_name,
            step_name="create_plan",
            input_data={"company_info": state.get("company_info", {})},
            output_data={
                "num_tasks": len(task_plan),
                "tasks": [{"agent": t.get("agent"), "action": t.get("action")} for t in task_plan],
            },
            execution_time_ms=(time.time() - start_time) * 1000,
            success=True,
        )

    return result


def fetch_api_data(state: CreditWorkflowState) -> Dict[str, Any]:
    """Fetch data from external APIs."""
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")
    company_info = state["company_info"]

    try:
        api_result = api_agent.fetch_all_data(
            company_name=company_info["company_name"],
            ticker=company_info.get("ticker"),
            jurisdiction=company_info.get("jurisdiction"),
            parallel=True,
        )

        result = {
            "api_data": api_result.to_dict(),
            "errors": state.get("errors", []) + api_result.errors,
            "status": "api_data_fetched",
        }

        # Log main step
        if wf_logger:
            api_data = api_result.to_dict()
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="fetch_api_data",
                input_data={"ticker": company_info.get("ticker"), "jurisdiction": company_info.get("jurisdiction")},
                output_data={
                    "sources_fetched": list(api_data.keys()),
                    "has_sec": bool(api_data.get("sec_edgar")),
                    "has_finnhub": bool(api_data.get("finnhub")),
                    "has_court": bool(api_data.get("court_listener")),
                    "errors": len(api_result.errors),
                },
                execution_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )

            # Log individual data sources AND tool calls
            for source_name, source_data in api_data.items():
                records = 0
                if isinstance(source_data, dict):
                    records = len(source_data.get("filings", [])) or len(source_data.get("data", [])) or (1 if source_data else 0)
                elif isinstance(source_data, list):
                    records = len(source_data)

                # Log as data source
                wf_logger.log_data_source(
                    run_id=run_id,
                    company_name=company_name,
                    source_name=source_name,
                    success=bool(source_data),
                    records_found=records,
                    data_summary=source_data if source_data else {},  # Full data
                    execution_time_ms=0,  # Individual timing not available
                )

                # Log as tool call
                wf_logger.log_tool_call(
                    run_id=run_id,
                    company_name=company_name,
                    tool_name=f"fetch_{source_name}",
                    tool_input={"ticker": company_info.get("ticker"), "company": company_name},
                    tool_output=source_data,
                    execution_time_ms=0,
                    success=bool(source_data),
                )

        return result

    except Exception as e:
        result = {
            "api_data": {},
            "errors": state.get("errors", []) + [f"API fetch error: {str(e)}"],
            "status": "api_data_error",
        }

        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="fetch_api_data",
                input_data={"ticker": company_info.get("ticker")},
                output_data={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )

        return result


def search_web(state: CreditWorkflowState) -> Dict[str, Any]:
    """Search for web information."""
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state["company_info"]["company_name"]

    try:
        search_result = search_agent.search_company(company_name)
        result = {
            "search_data": search_result.to_dict(),
            "errors": state.get("errors", []) + search_result.errors,
            "status": "search_complete",
        }

        if wf_logger:
            search_data = search_result.to_dict()
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="search_web",
                input_data={"company_name": company_name},
                output_data={
                    "num_results": len(search_data.get("results", [])) if isinstance(search_data, dict) else 0,
                    "has_data": bool(search_data),
                },
                execution_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )
            # Log as data source
            wf_logger.log_data_source(
                run_id=run_id,
                company_name=company_name,
                source_name="web_search",
                success=bool(search_data),
                records_found=len(search_data.get("results", [])) if isinstance(search_data, dict) else 0,
                data_summary=search_data if search_data else {},  # Full data
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        return result

    except Exception as e:
        result = {
            "search_data": {},
            "errors": state.get("errors", []) + [f"Search error: {str(e)}"],
            "status": "search_error",
        }

        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="search_web",
                input_data={"company_name": company_name},
                output_data={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )

        return result


def synthesize(state: CreditWorkflowState) -> Dict[str, Any]:
    """
    Synthesize final credit assessment with multi-LLM consistency evaluation.

    Runs multiple LLM calls for consistency:
    1. Primary model (llama-3.3-70b) - 2 runs for same-model consistency
    2. Fast model (llama-3.1-8b) - 1 run for cross-model consistency
    3. Balanced model (gemma2-9b) - 1 run for cross-model consistency
    """
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")

    # Models to use for consistency evaluation (3 runs each for proper consistency)
    # Using 2 working Groq models (many others are decommissioned)
    MODELS_CONFIG = [
        # Primary model (llama-3.3-70b) - 3 runs for same-model consistency
        ("primary", "llama-3.3-70b-versatile", "synthesis_primary_1"),
        ("primary", "llama-3.3-70b-versatile", "synthesis_primary_2"),
        ("primary", "llama-3.3-70b-versatile", "synthesis_primary_3"),
        # Fast model (llama-3.1-8b) - 3 runs for cross-model consistency
        ("fast", "llama-3.1-8b-instant", "synthesis_fast_1"),
        ("fast", "llama-3.1-8b-instant", "synthesis_fast_2"),
        ("fast", "llama-3.1-8b-instant", "synthesis_fast_3"),
    ]

    llm_results = []  # Store all LLM results for consistency calculation

    try:
        # First, get the rule-based assessment from supervisor
        assessment = supervisor.synthesize_assessment(
            company_info=state["company_info"],
            api_data=state.get("api_data", {}),
            search_data=state.get("search_data", {}),
        )
        assessment_dict = assessment.to_dict()

        # Log the primary synthesis step
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="synthesize",
                input_data={
                    "has_api_data": bool(state.get("api_data")),
                    "has_search_data": bool(state.get("search_data")),
                },
                output_data={
                    "risk_level": assessment_dict.get("overall_risk_level"),
                    "credit_score": assessment_dict.get("credit_score_estimate"),
                    "confidence": assessment_dict.get("confidence_score"),
                    "has_reasoning": bool(assessment_dict.get("llm_reasoning")),
                    "num_recommendations": len(assessment_dict.get("recommendations", [])),
                },
                execution_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )

        # Run multiple LLM analyses for consistency evaluation
        logger.info(f"Running multi-LLM consistency evaluation with {len(MODELS_CONFIG)} calls...")

        for model_key, model_id, call_type in MODELS_CONFIG:
            llm_start = time.time()
            try:
                # Create LLM analyst with specific model
                llm_analyst = LLMAnalystAgent(model=model_key)

                if llm_analyst.is_available():
                    # Run LLM analysis
                    llm_result = llm_analyst.analyze_company(
                        company_info=state["company_info"],
                        api_data=state.get("api_data", {}),
                        search_data=state.get("search_data", {}),
                    )

                    llm_exec_time = (time.time() - llm_start) * 1000

                    if llm_result.success:
                        llm_results.append({
                            "model": model_id,
                            "call_type": call_type,
                            "risk_level": llm_result.risk_level,
                            "credit_score": llm_result.credit_score_estimate,
                            "confidence": llm_result.confidence,
                            "reasoning": llm_result.reasoning,
                            "success": True,
                            # Token and cost tracking
                            "prompt_tokens": llm_result.prompt_tokens,
                            "completion_tokens": llm_result.completion_tokens,
                            "total_tokens": llm_result.total_tokens,
                            "input_cost": llm_result.input_cost,
                            "output_cost": llm_result.output_cost,
                            "total_cost": llm_result.total_cost,
                        })

                        # Log each LLM call with token usage
                        if wf_logger:
                            wf_logger.log_llm_call(
                                run_id=run_id,
                                company_name=company_name,
                                call_type=call_type,
                                model=model_id,
                                prompt=f"Credit analysis for {company_name}",
                                response=json.dumps({
                                    "risk_level": llm_result.risk_level,
                                    "credit_score": llm_result.credit_score_estimate,
                                    "confidence": llm_result.confidence,
                                    "reasoning": llm_result.reasoning,
                                    "key_findings": llm_result.key_findings,
                                    "risk_factors": llm_result.risk_factors,
                                    "positive_factors": llm_result.positive_factors,
                                    "recommendations": llm_result.recommendations,
                                }, default=str),
                                prompt_tokens=llm_result.prompt_tokens,
                                completion_tokens=llm_result.completion_tokens,
                                execution_time_ms=llm_exec_time,
                            )

                        logger.info(f"  [{call_type}] {model_id}: {llm_result.risk_level} (score: {llm_result.credit_score_estimate}, tokens: {llm_result.total_tokens}, cost: ${llm_result.total_cost:.6f})")
                    else:
                        logger.warning(f"  [{call_type}] {model_id}: Failed - {llm_result.error}")
                        llm_results.append({
                            "model": model_id,
                            "call_type": call_type,
                            "success": False,
                            "error": llm_result.error,
                        })
                else:
                    logger.warning(f"  [{call_type}] {model_id}: LLM not available")

            except Exception as llm_error:
                logger.error(f"  [{call_type}] {model_id}: Error - {llm_error}")
                llm_results.append({
                    "model": model_id,
                    "call_type": call_type,
                    "success": False,
                    "error": str(llm_error),
                })

        # Calculate consistency metrics from LLM results
        successful_results = [r for r in llm_results if r.get("success")]

        if len(successful_results) >= 2:
            # Calculate per-model consistency
            model_consistency = {}
            for model_type in ["primary", "fast", "balanced"]:
                model_results = [r for r in successful_results if model_type in r.get("call_type", "")]
                if len(model_results) >= 2:
                    risk_match = len(set(r["risk_level"] for r in model_results)) == 1
                    score_std = _calculate_std([r["credit_score"] for r in model_results])
                    consistency = 1.0 if risk_match else 0.5
                    if score_std < 5:
                        consistency = min(1.0, consistency + 0.3)
                    model_consistency[model_type] = {
                        "consistency": consistency,
                        "risk_levels": [r["risk_level"] for r in model_results],
                        "credit_scores": [r["credit_score"] for r in model_results],
                        "score_std": score_std,
                    }

            # Overall same-model consistency (average of all models)
            if model_consistency:
                same_model_consistency = sum(m["consistency"] for m in model_consistency.values()) / len(model_consistency)
            else:
                same_model_consistency = 1.0

            # Cross-model consistency (compare across different models)
            all_risk_levels = [r["risk_level"] for r in successful_results]
            all_scores = [r["credit_score"] for r in successful_results]

            cross_model_risk_match = len(set(all_risk_levels)) == 1
            cross_model_score_std = _calculate_std(all_scores)
            cross_model_consistency = 1.0 if cross_model_risk_match else 0.5
            if cross_model_score_std < 10:
                cross_model_consistency = min(1.0, cross_model_consistency + 0.3)

            # Calculate total tokens and cost across all LLM calls
            total_tokens = sum(r.get("total_tokens", 0) for r in successful_results)
            total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in successful_results)
            total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful_results)
            total_cost = sum(r.get("total_cost", 0) for r in successful_results)

            # Store consistency data in assessment
            assessment_dict["llm_consistency"] = {
                "num_llm_calls": len(successful_results),
                "same_model_consistency": same_model_consistency,
                "cross_model_consistency": cross_model_consistency,
                "per_model_consistency": model_consistency,
                "risk_levels": all_risk_levels,
                "credit_scores": all_scores,
                "llm_results": successful_results,
                # Token and cost summary
                "total_tokens": total_tokens,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_cost": round(total_cost, 6),
                "total_cost_formatted": f"${total_cost:.4f}",
            }

            logger.info(f"Consistency: same_model={same_model_consistency:.2f}, cross_model={cross_model_consistency:.2f}")
            logger.info(f"Total tokens: {total_tokens}, Total cost: ${total_cost:.4f}")
            for model_type, data in model_consistency.items():
                logger.info(f"  {model_type}: {data['consistency']:.2f} (std={data['score_std']:.1f})")

            # Log per-model consistency to sheets (one row per model)
            if wf_logger:
                # Map model_type to actual model name
                MODEL_NAMES = {
                    "primary": "llama-3.3-70b-versatile",
                    "fast": "llama-3.1-8b-instant",
                    "balanced": "llama3-70b-8192",
                }

                for model_type, data in model_consistency.items():
                    model_name = MODEL_NAMES.get(model_type, model_type)
                    wf_logger.log_consistency(
                        run_id=run_id,
                        company_name=company_name,
                        model_name=model_name,
                        evaluation_type="same_model",
                        num_runs=len(data["risk_levels"]),
                        consistency_data={
                            "risk_level_consistency": 1.0 if len(set(data["risk_levels"])) == 1 else 0.5,
                            "score_consistency": 1.0 if data["score_std"] < 5 else 0.5,
                            "score_std": data["score_std"],
                            "overall_consistency": data["consistency"],
                        },
                        risk_levels=data["risk_levels"],
                        credit_scores=data["credit_scores"],
                    )

                # Log cross-model consistency (overall)
                wf_logger.log_consistency(
                    run_id=run_id,
                    company_name=company_name,
                    model_name="overall",
                    evaluation_type="cross_model",
                    num_runs=len(successful_results),
                    consistency_data={
                        "risk_level_consistency": 1.0 if cross_model_risk_match else 0.5,
                        "score_consistency": 1.0 if cross_model_score_std < 10 else 0.5,
                        "score_std": cross_model_score_std,
                        "overall_consistency": cross_model_consistency,
                    },
                    risk_levels=all_risk_levels,
                    credit_scores=all_scores,
                )

        # Log the primary assessment
        if wf_logger:
            wf_logger.log_assessment(
                run_id=run_id,
                company_name=company_name,
                risk_level=assessment_dict.get('overall_risk_level', 'unknown'),
                credit_score=assessment_dict.get('credit_score_estimate', 0),
                confidence=assessment_dict.get('confidence_score', 0),
                reasoning=assessment_dict.get('llm_reasoning', ''),
                recommendations=assessment_dict.get('recommendations', []),
                risk_factors=assessment_dict.get('risk_factors', []),
                positive_factors=assessment_dict.get('positive_factors', []),
            )

        result = {
            "assessment": assessment_dict,
            "run_id": run_id,
            "status": "synthesized",
            "company_name": company_name,
            "risk_level": assessment_dict.get("overall_risk_level", "unknown"),
            "credit_score": assessment_dict.get("credit_score_estimate", 0),
            "confidence": assessment_dict.get("confidence_score", 0.0),
            "reasoning": assessment_dict.get("llm_reasoning", ""),
            "recommendations": assessment_dict.get("recommendations", []),
            "evaluation_score": 0.0,
            "data_quality_score": assessment_dict.get("data_quality_score", 0.0),
            "llm_results": llm_results,  # Include all LLM results
        }

        return result

    except Exception as e:
        result = {
            "assessment": None,
            "errors": state.get("errors", []) + [f"Synthesis error: {str(e)}"],
            "status": "synthesis_error",
            "company_name": company_name,
            "risk_level": "error",
            "credit_score": 0,
            "confidence": 0.0,
            "reasoning": str(e),
            "recommendations": [],
            "evaluation_score": 0.0,
            "data_quality_score": 0.0,
        }

        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="synthesize",
                input_data={"has_api_data": bool(state.get("api_data"))},
                output_data={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )

        return result


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def save_to_database(state: CreditWorkflowState) -> Dict[str, Any]:
    """Save assessment and raw data to MongoDB."""
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")

    if not db or not db.is_connected():
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="save_to_database",
                input_data={"db_connected": False},
                output_data={"status": "skipped_no_db"},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )
        return {"status": "complete_no_db"}

    try:
        assessment = state.get("assessment", {})
        api_data = state.get("api_data", {})
        search_data = state.get("search_data", {})

        # Save assessment
        if assessment:
            db.save_assessment(assessment)

        # Save raw data for auditing
        db.save_all_raw_data(company_name, api_data, search_data)

        # Save/update company profile
        company_info = state.get("company_info", {})
        if company_info:
            db.save_company(company_info)

        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="save_to_database",
                input_data={"has_assessment": bool(assessment), "has_api_data": bool(api_data)},
                output_data={"status": "saved", "saved_assessment": bool(assessment)},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )

        return {"status": "complete_saved"}

    except Exception as e:
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="save_to_database",
                input_data={"db_connected": True},
                output_data={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )
        return {
            "status": "complete_db_error",
            "errors": state.get("errors", []) + [f"DB save error: {str(e)}"],
        }


def evaluate_assessment(state: CreditWorkflowState) -> Dict[str, Any]:
    """
    Evaluate the assessment quality and save results.

    This node:
    1. Evaluates tool selection accuracy
    2. Scores data quality
    3. Scores synthesis quality
    4. Saves evaluation to file and MongoDB
    """
    import time
    start_time = time.time()

    if not EVALUATION_AVAILABLE:
        logger.warning("Evaluation framework not available")
        return {
            "evaluation": {"error": "Evaluation framework not available"},
            "tool_selection_score": 0.0,
            "data_quality_score": 0.0,
            "synthesis_score": 0.0,
            "status": "complete_no_eval",
        }

    try:
        run_id = state.get("run_id", str(uuid.uuid4()))
        company_name = state.get("company_name", "")
        assessment = state.get("assessment", {})
        api_data = state.get("api_data", {})
        task_plan = state.get("task_plan", [])

        # Initialize evaluators
        tool_evaluator = ToolSelectionEvaluator()
        workflow_evaluator = WorkflowEvaluator()

        # 1. Evaluate tool selection
        # Extract tools that were planned to be used based on task actions
        planned_tools = []
        for task in task_plan:
            action = task.get("action", "").lower()
            if "sec" in action or "edgar" in action:
                planned_tools.append("fetch_sec_data")
            elif "finnhub" in action or "market" in action:
                planned_tools.append("fetch_market_data")
            elif "court" in action or "sanction" in action or "opencorporates" in action:
                planned_tools.append("fetch_legal_data")
            elif "search" in action or "web" in action:
                planned_tools.append("web_search")

        tool_eval = tool_evaluator.evaluate(
            company_name=company_name,
            selected_tools=planned_tools,
        )
        tool_selection_score = tool_eval.f1_score

        # 2. Evaluate data quality
        data_sources_used = []
        if api_data.get("sec_edgar"):
            data_sources_used.append("SEC")
        if api_data.get("finnhub"):
            data_sources_used.append("Finnhub")
        if api_data.get("court_listener"):
            data_sources_used.append("CourtListener")
        if state.get("search_data"):
            data_sources_used.append("WebSearch")

        data_quality_score = len(data_sources_used) / 4.0  # Max 4 sources

        # 3. Evaluate synthesis quality
        synthesis_score = 0.0
        synthesis_present = []
        synthesis_missing = []
        if assessment:
            # Check for required fields
            required_fields = ["overall_risk_level", "credit_score_estimate", "recommendations"]
            for field in required_fields:
                if assessment.get(field):
                    synthesis_present.append(field)
                else:
                    synthesis_missing.append(field)
            present = len(synthesis_present)
            synthesis_score = present / len(required_fields)

            # Bonus for LLM reasoning
            if assessment.get("llm_reasoning"):
                synthesis_score = min(1.0, synthesis_score + 0.2)
                synthesis_present.append("llm_reasoning (+0.2 bonus)")

        # Generate human-readable reasoning for each score
        # Tool selection reasoning
        correct_tools = tool_eval.details.get("true_positives", [])
        missing_tools = tool_eval.details.get("false_negatives", [])
        extra_tools = tool_eval.details.get("false_positives", [])
        company_type = tool_eval.details.get("company_type", "unknown")

        tool_reasoning_parts = [f"Company classified as '{company_type}'."]
        if correct_tools:
            tool_reasoning_parts.append(f"Correctly selected: {', '.join(correct_tools)}.")
        if missing_tools:
            tool_reasoning_parts.append(f"Missing expected: {', '.join(missing_tools)}.")
        if extra_tools:
            tool_reasoning_parts.append(f"Extra (not expected): {', '.join(extra_tools)}.")
        tool_reasoning_parts.append(f"Precision={tool_eval.precision:.2f}, Recall={tool_eval.recall:.2f}.")
        tool_reasoning = " ".join(tool_reasoning_parts)

        # Data quality reasoning
        all_possible_sources = ["SEC", "Finnhub", "CourtListener", "WebSearch"]
        missing_sources = [s for s in all_possible_sources if s not in data_sources_used]
        data_reasoning_parts = [f"Used {len(data_sources_used)}/4 data sources."]
        if data_sources_used:
            data_reasoning_parts.append(f"Active: {', '.join(data_sources_used)}.")
        if missing_sources:
            data_reasoning_parts.append(f"Not used: {', '.join(missing_sources)}.")
        data_reasoning = " ".join(data_reasoning_parts)

        # Synthesis reasoning
        synthesis_reasoning_parts = []
        if synthesis_present:
            synthesis_reasoning_parts.append(f"Present: {', '.join(synthesis_present)}.")
        if synthesis_missing:
            synthesis_reasoning_parts.append(f"Missing: {', '.join(synthesis_missing)}.")
        if not synthesis_present and not synthesis_missing:
            synthesis_reasoning_parts.append("No assessment data available.")
        synthesis_reasoning = " ".join(synthesis_reasoning_parts)

        # 4. Build evaluation result
        evaluation = {
            "run_id": run_id,
            "company_name": company_name,
            "timestamp": datetime.utcnow().isoformat(),
            "tool_selection": {
                "planned_tools": planned_tools,
                "expected_tools": tool_eval.expected_tools,
                "correct_tools": correct_tools,
                "missing_tools": missing_tools,
                "extra_tools": extra_tools,
                "precision": tool_eval.precision,
                "recall": tool_eval.recall,
                "f1_score": tool_eval.f1_score,
                "is_correct": tool_eval.is_correct,
                "reasoning": tool_reasoning,
            },
            "data_quality": {
                "sources_used": data_sources_used,
                "missing_sources": missing_sources,
                "completeness": data_quality_score,
                "reasoning": data_reasoning,
            },
            "synthesis": {
                "score": synthesis_score,
                "present_fields": synthesis_present,
                "missing_fields": synthesis_missing,
                "has_risk_level": bool(assessment.get("overall_risk_level")),
                "has_credit_score": bool(assessment.get("credit_score_estimate")),
                "has_recommendations": bool(assessment.get("recommendations")),
                "has_llm_reasoning": bool(assessment.get("llm_reasoning")),
                "reasoning": synthesis_reasoning,
            },
            "overall_score": (tool_selection_score + data_quality_score + synthesis_score) / 3,
        }

        # 5. Save evaluation to file
        eval_filename = f"eval_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        eval_filepath = os.path.join(RESULTS_DIR, eval_filename)

        full_result = {
            "evaluation": evaluation,
            "assessment_summary": {
                "company": company_name,
                "risk_level": assessment.get("overall_risk_level"),
                "credit_score": assessment.get("credit_score_estimate"),
                "confidence": assessment.get("confidence_score"),
            },
        }

        with open(eval_filepath, "w") as f:
            json.dump(full_result, f, indent=2, default=str)

        logger.info(f"Evaluation saved to: {eval_filepath}")
        logger.info(f"Scores - Tool: {tool_selection_score:.2f}, Data: {data_quality_score:.2f}, Synthesis: {synthesis_score:.2f}")

        # Log evaluation step
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="evaluate",
                input_data={
                    "has_assessment": bool(assessment),
                    "num_data_sources": len(data_sources_used),
                },
                output_data={
                    "tool_selection_score": tool_selection_score,
                    "data_quality_score": data_quality_score,
                    "synthesis_score": synthesis_score,
                    "overall_score": evaluation["overall_score"],
                },
                execution_time_ms=(time.time() - start_time) * 1000,
                success=True,
            )

            # Log tool selection with reasoning
            wf_logger.log_tool_selection(
                run_id=run_id,
                company_name=company_name,
                selected_tools=planned_tools,
                expected_tools=tool_eval.expected_tools,
                precision=tool_eval.precision,
                recall=tool_eval.recall,
                f1_score=tool_eval.f1_score,
                correct_tools=correct_tools,
                missing_tools=missing_tools,
                extra_tools=extra_tools,
                reasoning=tool_reasoning,
            )

            # Log evaluation with reasoning
            wf_logger.log_evaluation(
                run_id=run_id,
                company_name=company_name,
                tool_selection_score=tool_selection_score,
                data_quality_score=data_quality_score,
                synthesis_score=synthesis_score,
                overall_score=evaluation["overall_score"],
                tool_reasoning=tool_reasoning,
                data_reasoning=data_reasoning,
                synthesis_reasoning=synthesis_reasoning,
            )

            # Log consistency scores from multi-LLM evaluation
            llm_consistency = assessment.get("llm_consistency", {})
            llm_results = state.get("llm_results", [])

            if llm_consistency and llm_results:
                # We have multi-LLM consistency results
                successful_results = [r for r in llm_results if r.get("success")]
                all_risk_levels = llm_consistency.get("risk_levels", [])
                all_credit_scores = llm_consistency.get("credit_scores", [])

                # Consistency already logged in synthesize with per-model breakdown
                # Just log summary here for evaluation reporting
                same_model_consistency = llm_consistency.get("same_model_consistency", 1.0)
                cross_model_consistency = llm_consistency.get("cross_model_consistency", 1.0)

                logger.info(f"Multi-LLM Consistency - Same: {same_model_consistency:.2f}, Cross: {cross_model_consistency:.2f}")
            else:
                # Fallback to single-run consistency
                risk_level = assessment.get("overall_risk_level", "unknown")
                credit_score = assessment.get("credit_score_estimate", 0)
                confidence = assessment.get("confidence_score", 0)

                # Internal consistency: how well does the assessment align with data?
                internal_consistency = min(1.0, (data_quality_score + synthesis_score + confidence) / 3)

                wf_logger.log_consistency(
                    run_id=run_id,
                    company_name=company_name,
                    model_name="single_run",
                    evaluation_type="single_run",
                    num_runs=1,
                    consistency_data={
                        "risk_level_consistency": 1.0,
                        "score_consistency": 1.0,
                        "score_std": 0.0,
                        "overall_consistency": internal_consistency,
                    },
                    risk_levels=[risk_level],
                    credit_scores=[credit_score],
                )

            # Complete the run in workflow logger
            wf_logger.complete_run(
                run_id=run_id,
                final_result={
                    "risk_level": assessment.get("overall_risk_level"),
                    "credit_score": assessment.get("credit_score_estimate"),
                    "evaluation_score": evaluation["overall_score"],
                },
                total_metrics={
                    "tool_selection_score": tool_selection_score,
                    "data_quality_score": data_quality_score,
                    "synthesis_score": synthesis_score,
                },
            )

        return {
            "evaluation": evaluation,
            "tool_selection_score": tool_selection_score,
            "data_quality_score": data_quality_score,
            "synthesis_score": synthesis_score,
            "status": "complete_evaluated",
        }

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

        if wf_logger:
            wf_logger.log_step(
                run_id=state.get("run_id", "unknown"),
                company_name=state.get("company_name", ""),
                step_name="evaluate",
                input_data={},
                output_data={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )
            wf_logger.fail_run(state.get("run_id", "unknown"), str(e))

        return {
            "evaluation": {"error": str(e)},
            "tool_selection_score": 0.0,
            "data_quality_score": 0.0,
            "synthesis_score": 0.0,
            "errors": state.get("errors", []) + [f"Evaluation error: {str(e)}"],
            "status": "complete_eval_error",
        }


# =============================================================================
# BUILD GRAPH
# =============================================================================

def should_continue_after_validation(state: CreditWorkflowState) -> str:
    """Decide whether to continue or stop based on validation."""
    if state.get("status") == "error":
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow with evaluation integrated."""
    workflow = StateGraph(
        CreditWorkflowState,
        input=CreditWorkflowInput,
        output=CreditWorkflowOutput,
    )

    # Add nodes with descriptive names
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("validate_company", validate_company)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("fetch_api_data", fetch_api_data)
    workflow.add_node("search_web", search_web)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("save_to_database", save_to_database)
    workflow.add_node("evaluate", evaluate_assessment)  # NEW: Evaluation node

    # Add edges with human-in-the-loop checkpoints
    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "validate_company")

    # Conditional: stop if validation failed
    workflow.add_conditional_edges(
        "validate_company",
        should_continue_after_validation,
        {
            "continue": "create_plan",
            "end": END,
        }
    )

    workflow.add_edge("create_plan", "fetch_api_data")
    workflow.add_edge("fetch_api_data", "search_web")
    workflow.add_edge("search_web", "synthesize")
    workflow.add_edge("synthesize", "save_to_database")
    workflow.add_edge("save_to_database", "evaluate")  # NEW: Run evaluation after save
    workflow.add_edge("evaluate", END)

    return workflow


def build_graph_with_hitl() -> StateGraph:
    """Build graph with Human-in-the-Loop interrupts at key points."""
    return build_graph()


# Compiled graph for LangGraph Studio
# Two options:
# 1. graph - fully automatic, no interrupts
# 2. graph_hitl - pauses at validate_company for human approval

# Default: Fully automatic graph (no interrupts)
graph = build_graph().compile()

# Alternative: Human-in-the-Loop graph (pauses for approval)
graph_hitl = build_graph().compile(
    interrupt_before=["validate_company"]
)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def get_mermaid_diagram() -> str:
    """Generate a Mermaid diagram of the workflow."""
    return """
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4F46E5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#3730A3', 'lineColor': '#6366F1', 'secondaryColor': '#818CF8', 'tertiaryColor': '#C7D2FE'}}}%%

graph TD
    subgraph CREDIT_INTELLIGENCE_WORKFLOW["Credit Intelligence Workflow"]
        START((Start)) --> PARSE[/"📝 Parse Input<br/>(Supervisor)"/]
        PARSE --> PLAN["📋 Create Plan<br/>(Supervisor)"]
        PLAN --> API["🔌 Fetch API Data<br/>(API Agent)"]

        subgraph DATA_SOURCES["Data Sources"]
            API --> SEC["SEC EDGAR<br/>Financials"]
            API --> FH["Finnhub<br/>Market Data"]
            API --> CL["CourtListener<br/>Legal Records"]
            API --> OC["OpenCorporates<br/>Registry"]
            API --> OS["OpenSanctions<br/>Compliance"]
        end

        SEC --> SEARCH
        FH --> SEARCH
        CL --> SEARCH
        OC --> SEARCH
        OS --> SEARCH

        SEARCH["🔍 Search Web<br/>(Search Agent)"]
        SEARCH --> SYNTH

        subgraph SYNTHESIS["AI-Powered Synthesis"]
            SYNTH["🧠 Synthesize<br/>(Supervisor + LLM)"]
            SYNTH --> RULES["Rule-Based<br/>Scoring"]
            SYNTH --> LLM["LLM Analysis<br/>(Groq)"]
            RULES --> HYBRID["Hybrid<br/>Assessment"]
            LLM --> HYBRID
        end

        HYBRID --> RESULT[/"📊 Credit Assessment"/]
        RESULT --> END_NODE((End))
    end

    style START fill:#10B981,stroke:#059669,color:#fff
    style END_NODE fill:#EF4444,stroke:#DC2626,color:#fff
    style PARSE fill:#6366F1,stroke:#4F46E5,color:#fff
    style PLAN fill:#6366F1,stroke:#4F46E5,color:#fff
    style API fill:#F59E0B,stroke:#D97706,color:#fff
    style SEARCH fill:#8B5CF6,stroke:#7C3AED,color:#fff
    style SYNTH fill:#EC4899,stroke:#DB2777,color:#fff
    style RESULT fill:#10B981,stroke:#059669,color:#fff
    style HYBRID fill:#14B8A6,stroke:#0D9488,color:#fff
```
"""


def print_graph_ascii():
    """Print an ASCII representation of the workflow."""
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CREDIT INTELLIGENCE WORKFLOW                              ║
║                         (LangGraph Visualization)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║    ┌─────────────┐                                                            ║
║    │   START     │                                                            ║
║    └──────┬──────┘                                                            ║
║           │                                                                   ║
║           ▼                                                                   ║
║    ┌─────────────────┐                                                        ║
║    │  PARSE INPUT    │  ◄── Supervisor: Parse company name, lookup ticker     ║
║    │  (Supervisor)   │                                                        ║
║    └────────┬────────┘                                                        ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐                                                        ║
║    │  CREATE PLAN    │  ◄── Supervisor: Decide which data sources to query    ║
║    │  (Supervisor)   │                                                        ║
║    └────────┬────────┘                                                        ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐      ┌─────────────────────────────────────────┐       ║
║    │  FETCH API DATA │      │  Data Sources:                          │       ║
║    │  (API Agent)    │ ───► │  • SEC EDGAR (Financials)               │       ║
║    │                 │      │  • Finnhub (Market Data)                │       ║
║    │                 │      │  • CourtListener (Legal)                │       ║
║    │                 │      │  • OpenCorporates (Registry)            │       ║
║    │                 │      │  • OpenSanctions (Compliance)           │       ║
║    └────────┬────────┘      └─────────────────────────────────────────┘       ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐      ┌─────────────────────────────────────────┐       ║
║    │  SEARCH WEB     │      │  Web Intelligence:                      │       ║
║    │  (Search Agent) │ ───► │  • Company information search           │       ║
║    │                 │      │  • News articles & sentiment            │       ║
║    └────────┬────────┘      └─────────────────────────────────────────┘       ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐      ┌─────────────────────────────────────────┐       ║
║    │  SYNTHESIZE     │      │  Hybrid Analysis:                       │       ║
║    │  (Supervisor    │ ───► │  • Rule-based scoring (35% Ability,     │       ║
║    │   + LLM)        │      │    25% Willingness, 40% Fraud)          │       ║
║    │                 │      │  • LLM Analysis (Groq LLaMA 70B)        │       ║
║    │                 │      │  • Confidence-weighted combination      │       ║
║    └────────┬────────┘      └─────────────────────────────────────────┘       ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐                                                        ║
║    │  ASSESSMENT     │  ◄── Final credit report with risk level & score      ║
║    │  (Output)       │                                                        ║
║    └────────┬────────┘                                                        ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────┐                                                            ║
║    │    END      │                                                            ║
║    └─────────────┘                                                            ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)


def save_graph_image(output_path: str = "workflow_graph.png"):
    """
    Save the graph as a PNG image (requires graphviz).

    Args:
        output_path: Path to save the image
    """
    try:
        # Try to use LangGraph's built-in visualization
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("Try installing: pip install pygraphviz")
        return None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Intelligence Graph Visualization")
    parser.add_argument("--mermaid", action="store_true", help="Print Mermaid diagram")
    parser.add_argument("--ascii", action="store_true", help="Print ASCII diagram")
    parser.add_argument("--save", type=str, help="Save graph as PNG image")
    parser.add_argument("--run", type=str, help="Run workflow for a company")

    args = parser.parse_args()

    if args.mermaid:
        print(get_mermaid_diagram())
    elif args.ascii:
        print_graph_ascii()
    elif args.save:
        save_graph_image(args.save)
    elif args.run:
        # Run the workflow
        initial_state = {
            "company_name": args.run,
            "jurisdiction": None,
            "ticker": None,
            "company_info": {},
            "task_plan": [],
            "api_data": {},
            "search_data": {},
            "assessment": None,
            "errors": [],
            "status": "started",
        }
        result = graph.invoke(initial_state)
        print(f"\nAssessment for {args.run}:")
        print(f"  Risk Level: {result['assessment'].get('overall_risk_level', 'N/A')}")
        print(f"  Credit Score: {result['assessment'].get('credit_score_estimate', 'N/A')}/100")
    else:
        # Default: print ASCII diagram
        print_graph_ascii()
