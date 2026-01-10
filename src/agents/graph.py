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

# Import ToolSupervisor for LLM-based tool selection
try:
    from agents.tool_supervisor import ToolSupervisor
    TOOL_SUPERVISOR_AVAILABLE = True
except ImportError:
    TOOL_SUPERVISOR_AVAILABLE = False
    ToolSupervisor = None

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

# Import agent efficiency evaluator (Task 4)
try:
    from evaluation.agent_efficiency_evaluator import (
        AgentEfficiencyEvaluator,
        evaluate_agent_run,
        get_agent_evaluator,
    )
    AGENT_EVALUATOR_AVAILABLE = True
except ImportError:
    AGENT_EVALUATOR_AVAILABLE = False
    AgentEfficiencyEvaluator = None
    evaluate_agent_run = None
    get_agent_evaluator = None

# Import LLM-as-a-judge evaluator (Task 21)
try:
    from evaluation.llm_judge_evaluator import (
        LLMJudgeEvaluator,
        evaluate_with_llm_judge,
        get_llm_judge,
    )
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False
    LLMJudgeEvaluator = None
    evaluate_with_llm_judge = None
    get_llm_judge = None

# Import unified agent evaluator (combines DeepEval + OpenEvals + Built-in)
try:
    from evaluation.unified_agent_evaluator import (
        UnifiedAgentEvaluator,
        evaluate_workflow,
        get_unified_evaluator,
        UnifiedEvaluationResult,
    )
    UNIFIED_EVALUATOR_AVAILABLE = True
except ImportError:
    UNIFIED_EVALUATOR_AVAILABLE = False
    UnifiedAgentEvaluator = None
    evaluate_workflow = None
    get_unified_evaluator = None
    UnifiedEvaluationResult = None

# Import DeepEval evaluator (runs independently with Groq)
try:
    from evaluation.deepeval_evaluator import evaluate_with_deepeval
    DEEPEVAL_EVALUATOR_AVAILABLE = True
except ImportError:
    DEEPEVAL_EVALUATOR_AVAILABLE = False
    evaluate_with_deepeval = None

# Import LangSmith integration for trace logging
try:
    from config.langsmith_integration import get_langsmith_integration, LangSmithIntegration
    LANGSMITH_INTEGRATION_AVAILABLE = True
except ImportError:
    LANGSMITH_INTEGRATION_AVAILABLE = False
    get_langsmith_integration = None
    LangSmithIntegration = None

# Import LangGraph event logger
try:
    from run_logging.langgraph_logger import (
        LangGraphEventLogger,
        run_graph_with_event_logging,
        get_langgraph_logger,
    )
    LANGGRAPH_LOGGER_AVAILABLE = True
except ImportError:
    LANGGRAPH_LOGGER_AVAILABLE = False
    LangGraphEventLogger = None
    run_graph_with_event_logging = None
    get_langgraph_logger = None
    logger.warning("LangGraph event logger not available")

# Initialize workflow logger
wf_logger = get_workflow_logger() if get_workflow_logger else None

# Module-level LangGraph event logger (set per-run)
_lg_event_logger: Optional[LangGraphEventLogger] = None

def set_langgraph_event_logger(logger_instance: Optional[LangGraphEventLogger]) -> None:
    """Set the module-level LangGraph event logger for tool event tracking."""
    global _lg_event_logger
    _lg_event_logger = logger_instance

def get_current_event_logger() -> Optional[LangGraphEventLogger]:
    """Get the current LangGraph event logger."""
    return _lg_event_logger

# Configure logging
logger = logging.getLogger(__name__)

# Results directory for saving evaluations
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class CreditWorkflowInput(TypedDict, total=False):
    """Input for the workflow."""
    company_name: str
    run_id: str  # Optional: passed from API for consistent run tracking
    jurisdiction: str  # Optional
    ticker: str  # Optional


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

    # Use run_id from state if provided (from API), otherwise generate one
    run_id = state.get("run_id") or str(uuid.uuid4())

    # Start run in workflow logger with the same run_id
    if wf_logger:
        wf_logger.start_run(company_name, {"source": "langgraph"}, run_id=run_id)

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
                agent_name="llm_parser",
                input_data={"company_name": company_name},
                output_data={"error": "Company name empty"},
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error="Company name cannot be empty",
            )
        return result

    # Use LLM parser if available, otherwise fallback to rule-based
    llm_metrics = {}
    step_number = 1  # parse_input is step 1 in workflow
    if LLM_PARSER_AVAILABLE and llm_parse_company:
        logger.info(f"Using LLM parser for: {company_name}")
        parser = get_llm_parser()
        company_info, llm_metrics = parser.parse_company_llm(
            company_name,
            context={"jurisdiction": state.get("jurisdiction")},
            step_logger=step_logger,
        )

        # Log LLM call for parse_input
        if wf_logger and llm_metrics:
            wf_logger.log_llm_call(
                run_id=run_id,
                company_name=company_name,
                call_type="parse_input",
                model=llm_metrics.get("model", "unknown"),
                prompt=f"Parse company: {company_name}",
                response=json.dumps(company_info, default=str),
                prompt_tokens=llm_metrics.get("prompt_tokens", 0),
                completion_tokens=llm_metrics.get("completion_tokens", 0),
                execution_time_ms=llm_metrics.get("execution_time_ms", 0),
                # Node and task tracking
                node="parse_input",
                agent_name="llm_parser",
                step_number=step_number,
                current_task="company_parsing",
            )

        # Log prompt used for parse_input
        if llm_metrics:
            try:
                from run_logging.sheets_logger import get_sheets_logger
                sheets_logger = get_sheets_logger()
                if sheets_logger and sheets_logger.is_connected():
                    sheets_logger.log_prompt(
                        run_id=run_id,
                        company_name=company_name,
                        prompt_id="company_parser",
                        prompt_name="Company Parser",
                        category="input",
                        system_prompt=llm_metrics.get("system_prompt", ""),
                        user_prompt=llm_metrics.get("prompt_used", ""),
                        variables={"company_name": company_name},
                        node="parse_input",
                        agent_name="llm_parser",
                        step_number=step_number,
                        model=llm_metrics.get("model", ""),
                    )
            except Exception as e:
                logger.warning(f"Failed to log prompt: {e}")
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

    # Log step with LLM metrics (include FULL company_info)
    if wf_logger:
        wf_logger.log_step(
            run_id=run_id,
            company_name=company_name,
            step_name="parse_input",
            agent_name="llm_parser",
            input_data={"company_name": company_name, "jurisdiction": state.get("jurisdiction")},
            output_data={
                "full_company_info": company_info,  # Full parsed company info
                "validation_message": validation_msg,
                "requires_review": requires_review,
                "llm_metrics": llm_metrics,
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
                agent_name="supervisor",
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
            agent_name="supervisor",
        )

    return result


def convert_llm_selection_to_task_plan(selection: Dict[str, Any], company_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert LLM tool selection output to task_plan format.

    The LLM returns:
    {
        "company_analysis": {"is_likely_public": True, "reasoning": "..."},
        "tools_to_use": [
            {"name": "fetch_sec_data", "params": {...}, "reason": "..."},
        ],
        "execution_order_reasoning": "..."
    }

    This function converts to the existing task_plan format:
    [
        {"agent": "api", "action": "fetch_sec_edgar", "params": {...}, "priority": 1, "reason": "..."},
    ]
    """
    task_plan = []
    tools_to_use = selection.get("tools_to_use", [])

    # Tool name to (agent, action) mapping
    tool_action_map = {
        "fetch_sec_data": ("api", "fetch_sec_edgar"),
        "fetch_market_data": ("api", "fetch_finnhub"),
        "fetch_legal_data": ("api", "fetch_court_records"),
        "web_search": ("search", "search_company"),
        # Additional mappings for flexibility
        "sec_edgar": ("api", "fetch_sec_edgar"),
        "finnhub": ("api", "fetch_finnhub"),
        "court_listener": ("api", "fetch_court_records"),
        "search": ("search", "search_company"),
    }

    company_name = company_info.get("company_name", "")
    ticker = company_info.get("ticker", "")
    jurisdiction = company_info.get("jurisdiction", "US")

    for i, tool in enumerate(tools_to_use):
        tool_name = tool.get("name", "")
        params = tool.get("params", {})
        reason = tool.get("reason", "")

        # Normalize tool name
        tool_name_lower = tool_name.lower().replace("-", "_").replace(" ", "_")

        if tool_name_lower in tool_action_map:
            agent, action = tool_action_map[tool_name_lower]

            # Ensure required params are set
            if action == "fetch_sec_edgar":
                if "ticker" not in params and ticker:
                    params["ticker"] = ticker
                if "company_identifier" not in params:
                    params["company_identifier"] = ticker or company_name
            elif action == "fetch_finnhub":
                if "ticker" not in params and ticker:
                    params["ticker"] = ticker
                if "company_name" not in params:
                    params["company_name"] = company_name
            elif action in ["fetch_court_records", "search_company"]:
                if "company_name" not in params:
                    params["company_name"] = company_name
            elif action == "fetch_opencorporates":
                if "company_name" not in params:
                    params["company_name"] = company_name
                if "jurisdiction" not in params:
                    params["jurisdiction"] = jurisdiction

            task_plan.append({
                "agent": agent,
                "action": action,
                "params": params,
                "priority": i + 1,  # Order from LLM
                "reason": reason,  # Keep LLM reasoning
            })
        else:
            logger.warning(f"Unknown tool in LLM selection: {tool_name}")

    return task_plan


def create_plan(state: CreditWorkflowState) -> Dict[str, Any]:
    """Create task plan using LLM-based tool selection."""
    import time
    start_time = time.time()
    step_number = 3  # create_plan is step 3 in workflow
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")
    company_info = state.get("company_info", {})

    # Log node entry for timing
    event_logger = get_current_event_logger()
    if event_logger:
        event_logger.log_node_enter("create_plan", {"company_info": company_info})

    # Prepare context for LLM tool selection
    context = {
        "ticker": company_info.get("ticker"),
        "jurisdiction": company_info.get("jurisdiction"),
        "is_public_company": company_info.get("is_public_company"),
    }

    # Use LLM-based tool selection if available, otherwise fall back to rule-based
    tool_selection = None
    llm_metrics = None

    if TOOL_SUPERVISOR_AVAILABLE:
        try:
            tool_supervisor = ToolSupervisor(model="primary")
            selection_result = tool_supervisor.select_tools(
                company_name=company_name,
                context=context,
                run_id=run_id,
            )

            tool_selection = selection_result.get("selection", {})
            llm_metrics = selection_result.get("llm_metrics", {})

            # Convert LLM output to task_plan format
            task_plan = convert_llm_selection_to_task_plan(tool_selection, company_info)

            logger.info(f"LLM tool selection: {[t.get('action') for t in task_plan]}")

        except Exception as e:
            logger.warning(f"LLM tool selection failed, falling back to rule-based: {e}")
            task_plan = supervisor.create_task_plan(company_info)
            tool_selection = None
    else:
        # Fall back to rule-based tool selection
        task_plan = supervisor.create_task_plan(company_info)

    # Show plan summary for human review
    plan_summary = f"Plan: Fetching data from {len(task_plan)} sources for {company_name}"

    # Build result with LLM selection info if available
    result = {
        "task_plan": task_plan,
        "status": "plan_created",
        "validation_message": plan_summary,
    }

    # Store tool_selection for evaluation phase
    if tool_selection:
        result["tool_selection"] = tool_selection

    execution_time_ms = (time.time() - start_time) * 1000

    # Log step
    if wf_logger:
        wf_logger.log_step(
            run_id=run_id,
            company_name=company_name,
            step_name="create_plan",
            agent_name="tool_supervisor",
            input_data={"company_info": company_info},
            output_data={
                "num_tasks": len(task_plan),
                "full_task_plan": task_plan,
                "tool_selection_reasoning": tool_selection.get("execution_order_reasoning", "") if tool_selection else "",
            },
            execution_time_ms=execution_time_ms,
            success=True,
        )

    # Log LLM call if we used LLM tool selection
    if wf_logger and llm_metrics:
        wf_logger.log_llm_call(
            run_id=run_id,
            company_name=company_name,
            call_type="tool_selection",
            model=llm_metrics.get("model", "llama-3.3-70b-versatile"),
            prompt=f"Tool selection for {company_name}",
            response=json.dumps(tool_selection.get("company_analysis", {})) if tool_selection else "",
            prompt_tokens=llm_metrics.get("prompt_tokens", 0),
            completion_tokens=llm_metrics.get("completion_tokens", 0),
            execution_time_ms=llm_metrics.get("execution_time_ms", 0),
            node="create_plan",
            agent_name="tool_supervisor",
            step_number=step_number,
            current_task="tool_selection",
            temperature=0.1,
        )

    # Log prompt if we used LLM tool selection
    if wf_logger and tool_selection:
        try:
            from config.prompts import get_prompt_text
            system_prompt, user_prompt = get_prompt_text(
                "tool_selection",
                company_name=company_name,
                context=json.dumps(context),
                tool_specs="[available tools]",  # Simplified for logging
            )
            wf_logger.log_prompt(
                run_id=run_id,
                company_name=company_name,
                node="create_plan",
                agent_name="tool_supervisor",
                step_number=step_number,
                prompt_id="tool_selection",
                prompt_name="Tool Selection",
                category="planning",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=llm_metrics.get("model", "llama-3.3-70b-versatile") if llm_metrics else "",
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"Failed to log prompt: {e}")

    # Log full plan to dedicated plans sheet for easy visibility
    try:
        from run_logging.sheets_logger import get_sheets_logger
        sheets_logger = get_sheets_logger()
        if sheets_logger and sheets_logger.is_connected():
            sheets_logger.log_plan(
                run_id=run_id,
                company_name=company_name,
                task_plan=task_plan,
                node="create_plan",
                agent_name="tool_supervisor" if tool_selection else "supervisor",
                status="ok",
            )
    except Exception as e:
        logger.warning(f"Failed to log plan to sheets: {e}")

    # Log full plan to langgraph_events
    if event_logger:
        event_logger.log_node_exit(
            node_name="create_plan",
            output_state={
                "full_plan": task_plan,
                "num_tasks": len(task_plan),
                "company_analysis": tool_selection.get("company_analysis", {}) if tool_selection else {},
                "execution_order_reasoning": tool_selection.get("execution_order_reasoning", "") if tool_selection else "",
            },
        )

    return result


def fetch_api_data(state: CreditWorkflowState) -> Dict[str, Any]:
    """Fetch data from external APIs."""
    import time
    start_time = time.time()
    run_id = state.get("run_id", "unknown")
    company_name = state.get("company_name", "")
    company_info = state["company_info"]
    step_number = 4  # fetch_api_data is step 4 in workflow

    # Get event logger for tool tracking
    event_logger = get_current_event_logger()

    # Log tool starts
    tools_to_fetch = ["sec_edgar", "finnhub", "court_listener"]
    tool_start_times = {}
    for tool_name in tools_to_fetch:
        tool_start_times[tool_name] = time.time()
        if event_logger:
            event_logger.log_tool_event(
                tool_name=f"fetch_{tool_name}",
                status="started",
                node="fetch_api_data",
                input_data=json.dumps({"ticker": company_info.get("ticker"), "company": company_name}),
            )

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

        # Log main step with FULL data
        if wf_logger:
            api_data = api_result.to_dict()
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="fetch_api_data",
                agent_name="api_agent",
                input_data={"ticker": company_info.get("ticker"), "jurisdiction": company_info.get("jurisdiction")},
                output_data={
                    "full_api_data": api_data,  # Full API data from all sources
                    "sources_fetched": list(api_data.keys()),
                    "errors": api_result.errors,
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

                # Calculate tool duration if we tracked the start
                tool_duration = None
                if source_name in tool_start_times:
                    tool_duration = (time.time() - tool_start_times[source_name]) * 1000

                # Log tool completion to LangGraph events
                if event_logger:
                    event_logger.log_tool_event(
                        tool_name=f"fetch_{source_name}",
                        status="completed",
                        node="fetch_api_data",
                        input_data=json.dumps({"ticker": company_info.get("ticker"), "company": company_name}),
                        output_data=json.dumps({"records": records, "success": bool(source_data)}, default=str),
                        duration_ms=tool_duration,
                        error=None if source_data else "No data returned",
                    )

                # Log as data source
                wf_logger.log_data_source(
                    run_id=run_id,
                    company_name=company_name,
                    source_name=source_name,
                    success=bool(source_data),
                    records_found=records,
                    data_summary=source_data if source_data else {},  # Full data
                    execution_time_ms=tool_duration or 0,
                )

                # Log as tool call with hierarchy tracking
                wf_logger.log_tool_call(
                    run_id=run_id,
                    company_name=company_name,
                    tool_name=f"fetch_{source_name}",
                    tool_input={"ticker": company_info.get("ticker"), "company": company_name},
                    tool_output=source_data,
                    execution_time_ms=tool_duration or 0,
                    success=bool(source_data),
                    # Node tracking
                    node="fetch_api_data",
                    agent_name="api_agent",
                    step_number=step_number,
                    # Hierarchy tracking
                    parent_node="fetch_api_data",
                    workflow_phase="data_collection",
                    call_depth=0,
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
                agent_name="api_agent",
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

    # Get event logger for tool tracking
    event_logger = get_current_event_logger()

    # Log tool start
    if event_logger:
        event_logger.log_tool_event(
            tool_name="web_search",
            status="started",
            node="search_web",
            input_data=json.dumps({"company_name": company_name}),
        )

    try:
        search_result = search_agent.search_company(company_name)
        result = {
            "search_data": search_result.to_dict(),
            "errors": state.get("errors", []) + search_result.errors,
            "status": "search_complete",
        }

        search_data = search_result.to_dict()
        tool_duration = (time.time() - start_time) * 1000
        num_results = len(search_data.get("results", [])) if isinstance(search_data, dict) else 0

        # Log tool completion
        if event_logger:
            event_logger.log_tool_event(
                tool_name="web_search",
                status="completed",
                node="search_web",
                input_data=json.dumps({"company_name": company_name}),
                output_data=json.dumps({"num_results": num_results, "success": bool(search_data)}, default=str),
                duration_ms=tool_duration,
            )

        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="search_web",
                input_data={"company_name": company_name},
                output_data={
                    "full_search_data": search_data,  # Full search results
                    "num_results": num_results,
                },
                execution_time_ms=tool_duration,
                success=True,
                agent_name="search_agent",
            )
            # Log as data source
            wf_logger.log_data_source(
                run_id=run_id,
                company_name=company_name,
                source_name="web_search",
                success=bool(search_data),
                records_found=num_results,
                data_summary=search_data if search_data else {},  # Full data
                execution_time_ms=tool_duration,
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
                agent_name="search_agent",
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
    step_number = 6  # synthesize is step 6 in workflow

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

        # Log the primary synthesis step with FULL assessment
        if wf_logger:
            wf_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name="synthesize",
                agent_name="llm_analyst",
                input_data={
                    "company_info": state.get("company_info", {}),
                    "api_data_sources": list(state.get("api_data", {}).keys()),
                },
                output_data={
                    "full_assessment": assessment_dict,  # Full assessment with all details
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

                        # Log each LLM call with token usage and task tracking
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
                                # Node and task tracking
                                node="synthesize",
                                agent_name="llm_analyst",
                                step_number=step_number,
                                current_task="credit_synthesis",
                            )

                            # Log each LLM assessment to assessments sheet
                            eval_status = "good" if llm_result.credit_score_estimate >= 70 else ("average" if llm_result.credit_score_estimate >= 50 else "bad")
                            wf_logger.log_assessment(
                                run_id=run_id,
                                company_name=company_name,
                                risk_level=llm_result.risk_level,
                                credit_score=llm_result.credit_score_estimate,
                                confidence=llm_result.confidence,
                                reasoning=llm_result.reasoning,
                                recommendations=llm_result.recommendations,
                                risk_factors=llm_result.risk_factors,
                                positive_factors=llm_result.positive_factors,
                                node="synthesize",
                                agent_name="llm_analyst",
                                model=model_id,
                                temperature=0.1,
                                step_number=step_number,
                                prompt=LLMAnalystAgent.CREDIT_ANALYSIS_PROMPT[:500] + "...",
                                duration_ms=llm_exec_time,
                                status=eval_status,
                            )

                            # Log prompt for the first synthesis call only (to avoid duplicates)
                            if call_type == "synthesis_primary_1":
                                try:
                                    from run_logging.sheets_logger import get_sheets_logger
                                    sheets_logger = get_sheets_logger()
                                    if sheets_logger and sheets_logger.is_connected():
                                        sheets_logger.log_prompt(
                                            run_id=run_id,
                                            company_name=company_name,
                                            prompt_id="credit_synthesis",
                                            prompt_name="Credit Synthesis",
                                            category="synthesis",
                                            system_prompt="You are an expert credit analyst...",
                                            user_prompt=LLMAnalystAgent.CREDIT_ANALYSIS_PROMPT.format(
                                                company_info=json.dumps(state.get("company_info", {}), default=str)[:500],
                                                financial_data="(summarized)",
                                                legal_data="(summarized)",
                                                market_data="(summarized)",
                                                news_data="(summarized)",
                                            )[:2000],  # Truncate for sheet
                                            variables={
                                                "company_name": company_name,
                                                "api_sources": list(state.get("api_data", {}).keys()),
                                            },
                                            node="synthesize",
                                            agent_name="llm_analyst",
                                            step_number=step_number,
                                            model=model_id,
                                        )
                                except Exception as e:
                                    logger.warning(f"Failed to log synthesis prompt: {e}")

                            # Task 17: Log detailed LLM call
                            wf_logger.log_llm_call_detailed(
                                run_id=run_id,
                                company_name=company_name,
                                llm_provider="groq",
                                agent_name="llm_analyst",
                                model=model_id,
                                prompt=f"Credit analysis for {company_name}",
                                context=f"API data sources: {list(state.get('api_data', {}).keys())}",
                                response=json.dumps({
                                    "risk_level": llm_result.risk_level,
                                    "credit_score": llm_result.credit_score_estimate,
                                    "confidence": llm_result.confidence,
                                }, default=str),
                                reasoning=llm_result.reasoning or "",
                                prompt_tokens=llm_result.prompt_tokens,
                                completion_tokens=llm_result.completion_tokens,
                                response_time_ms=llm_exec_time,
                                input_cost=llm_result.input_cost,
                                output_cost=llm_result.output_cost,
                                total_cost=llm_result.total_cost,
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

        # Note: Individual LLM assessments are logged in the loop above
        # This logs the final consensus assessment
        if wf_logger:
            final_eval_status = "good" if assessment_dict.get('credit_score_estimate', 0) >= 70 else ("average" if assessment_dict.get('credit_score_estimate', 0) >= 50 else "bad")
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
                node="synthesize_final",
                agent_name="consensus",
                model="multi-model",
                step_number=step_number,
                prompt="Final consensus from 6 LLM analyses",
                status=final_eval_status,
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
                agent_name="llm_analyst",
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
                agent_name="db_writer",
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
                agent_name="db_writer",
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
                agent_name="db_writer",
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

        # Get tool selection reasoning from create_plan (if LLM-based selection was used)
        tool_selection_reasoning = state.get("tool_selection", {})

        # Use LLM-as-judge evaluation if available, otherwise fall back to rule-based
        llm_eval_metrics = None
        use_llm_evaluation = LANGCHAIN_GROQ_AVAILABLE if 'LANGCHAIN_GROQ_AVAILABLE' in dir() else False

        # Try to import for LLM evaluation
        try:
            from config.langchain_llm import is_langchain_groq_available
            use_llm_evaluation = is_langchain_groq_available()
        except ImportError:
            use_llm_evaluation = False

        if use_llm_evaluation and tool_selection_reasoning:
            # Use LLM-as-judge for more intelligent evaluation
            tool_eval, llm_eval_metrics = tool_evaluator.evaluate_with_llm(
                company_name=company_name,
                selected_tools=planned_tools,
                tool_selection_reasoning=tool_selection_reasoning,
                actual_data_results=api_data,
            )
            logger.info(f"LLM-as-judge tool selection evaluation: {tool_eval.f1_score:.2f}")
        else:
            # Fall back to rule-based evaluation
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
                agent_name="workflow_evaluator",
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
            # Determine if LLM-as-judge was used
            eval_model = "llm_judge" if llm_eval_metrics else "rule_based"
            eval_agent = "llm_tool_selection_judge" if llm_eval_metrics else "tool_selection_evaluator"

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
                # Node tracking fields
                node="evaluate",
                agent_name=eval_agent,
                step_number=7,  # evaluate is step 7
                model=llm_eval_metrics.get("model", "rule_based") if llm_eval_metrics else "rule_based",
            )

            # Log LLM evaluation call if LLM-as-judge was used
            if llm_eval_metrics:
                wf_logger.log_llm_call(
                    run_id=run_id,
                    company_name=company_name,
                    call_type="tool_selection_evaluation",
                    model=llm_eval_metrics.get("model", "llama-3.3-70b-versatile"),
                    prompt=f"Tool selection evaluation for {company_name}",
                    response=json.dumps({
                        "precision": tool_eval.precision,
                        "recall": tool_eval.recall,
                        "f1_score": tool_eval.f1_score,
                        "is_correct": tool_eval.is_correct,
                    }),
                    prompt_tokens=llm_eval_metrics.get("prompt_tokens", 0),
                    completion_tokens=llm_eval_metrics.get("completion_tokens", 0),
                    execution_time_ms=llm_eval_metrics.get("execution_time_ms", 0),
                    node="evaluate",
                    agent_name="llm_tool_selection_judge",
                    step_number=7,
                    current_task="tool_selection_evaluation",
                    temperature=0.1,
                )

            # Log evaluation with reasoning
            step_number = 7  # evaluate is step 7
            eval_duration_ms = (time.time() - start_time) * 1000
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
                # Node tracking fields
                node="evaluate",
                node_type="agent",
                agent_name="workflow_evaluator",
                step_number=step_number,
                model="rule_based",  # Evaluation is rule-based, not LLM
                duration_ms=eval_duration_ms,
                status="ok",
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

        # ============ CROSS-MODEL EVALUATION ============
        # Run secondary assessment with different model and compare (OUTSIDE wf_logger block)
        try:
            from config.external_config import get_config_manager
            config_manager = get_config_manager()
            eval_config = config_manager.get("evaluation", {})
            consistency_config = eval_config.get("consistency", {})

            logger.info(f"Cross-model config check: cross_model={consistency_config.get('cross_model', False)}")

            if consistency_config.get("cross_model", False):
                logger.info("Running cross-model evaluation...")

                # Get primary assessment
                primary_risk = assessment.get("overall_risk_level", "unknown")
                primary_score = assessment.get("credit_score_estimate", 0)
                primary_confidence = assessment.get("confidence_score", 0)
                primary_model = "llama-3.3-70b-versatile"  # Primary model

                # Run secondary assessment with fast model
                secondary_model_key = consistency_config.get("cross_model_secondary", "fast")
                secondary_model = "llama-3.1-8b-instant"  # Fast model

                # Create a quick secondary assessment using LLM analyst
                try:
                    from agents.llm_analyst import LLMAnalystAgent
                    secondary_analyst = LLMAnalystAgent(model=secondary_model)

                    # Get company_info from state
                    company_info = state.get("company_info", {"name": company_name})
                    search_data = state.get("search_data", {})

                    # Run secondary analysis with same data but different model
                    secondary_analysis = secondary_analyst.analyze_company(
                        company_info=company_info,
                        api_data=api_data,
                        search_data=search_data,
                    )

                    # Extract result from LLMAnalysisResult
                    if secondary_analysis.success:
                        secondary_risk = secondary_analysis.risk_level or "unknown"
                        secondary_score = secondary_analysis.credit_score_estimate or 0
                        secondary_confidence = secondary_analysis.confidence or 0
                    else:
                        raise Exception(f"Secondary analysis failed: {secondary_analysis.error}")

                    # Calculate cross-model agreement
                    risk_agreement = 1.0 if primary_risk.lower() == secondary_risk.lower() else 0.0
                    score_diff = abs(primary_score - secondary_score)
                    score_agreement = max(0, 1.0 - score_diff / 100)
                    cross_model_agreement = risk_agreement * 0.6 + score_agreement * 0.4

                    # Determine best model (the one with higher confidence)
                    if primary_confidence >= secondary_confidence:
                        best_model = primary_model
                        best_reasoning = "Primary model had higher confidence"
                    else:
                        best_model = secondary_model
                        best_reasoning = "Secondary model had higher confidence"

                    # Log to cross_model_eval sheet - use get_sheets_logger directly
                    from run_logging.sheets_logger import get_sheets_logger
                    sheets_logger = get_sheets_logger()
                    if sheets_logger and sheets_logger.is_connected():
                        # Calculate confidence agreement (how close are the confidence scores)
                        confidence_diff = abs(primary_confidence - secondary_confidence)
                        confidence_agreement = max(0, 1.0 - confidence_diff)  # 1.0 = perfect agreement

                        sheets_logger.log_cross_model_eval(
                            eval_id=run_id,  # Full run_id for Google Sheets lookup
                            company_name=company_name,
                            models_compared=[primary_model, secondary_model],
                            num_models=2,
                            # Node tracking fields
                            node="synthesize",
                            node_type="agent",
                            agent_name="cross_model_evaluator",
                            step_number=6,  # synthesize is step 6
                            # Agreement metrics
                            risk_level_agreement=risk_agreement,
                            credit_score_mean=(primary_score + secondary_score) / 2,
                            credit_score_std=abs(primary_score - secondary_score) / 2,
                            credit_score_range=abs(primary_score - secondary_score),
                            confidence_agreement=confidence_agreement,
                            best_model=best_model,
                            best_model_reasoning=best_reasoning,
                            cross_model_agreement=cross_model_agreement,
                            llm_judge_analysis=f"Primary ({primary_model}): {primary_risk}/{primary_score}/{primary_confidence:.2f}, Secondary ({secondary_model}): {secondary_risk}/{secondary_score}/{secondary_confidence:.2f}",
                            model_recommendations=[f"Use {best_model} for this company type"],
                            model_results={
                                primary_model: {"risk_level": primary_risk, "credit_score": primary_score, "confidence": primary_confidence},
                                secondary_model: {"risk_level": secondary_risk, "credit_score": secondary_score, "confidence": secondary_confidence},
                            },
                        )
                        logger.info(f"Cross-model eval logged: agreement={cross_model_agreement:.2f}, confidence_agreement={confidence_agreement:.2f}, best={best_model}")

                except Exception as secondary_error:
                    logger.warning(f"Secondary model analysis failed: {secondary_error}")

        except Exception as cross_model_error:
            logger.warning(f"Cross-model evaluation failed: {cross_model_error}")
        # ============ END CROSS-MODEL EVALUATION ============

        # ============ SAME-MODEL CONSISTENCY (MULTIPLE RUNS) ============
        # Run additional LLM analyses with the same model to calculate consistency
        try:
            from config.external_config import get_config_manager
            config_manager = get_config_manager()
            eval_config = config_manager.get("evaluation", {})
            consistency_config = eval_config.get("consistency", {})

            if consistency_config.get("same_model", True):  # Enabled by default
                num_runs = consistency_config.get("num_runs", 3)

                # Only run if num_runs > 1 (we already have 1 run)
                if num_runs > 1:
                    logger.info(f"Running same-model consistency evaluation ({num_runs - 1} additional runs)...")

                    from agents.llm_analyst import LLMAnalystAgent
                    primary_model = "llama-3.3-70b-versatile"

                    # Store all results including primary
                    all_risk_levels = [assessment.get("overall_risk_level", "unknown")]
                    all_credit_scores = [assessment.get("credit_score_estimate", 0)]
                    all_confidences = [assessment.get("confidence_score", 0)]
                    # Additional fields for similarity metrics
                    all_reasonings = [assessment.get("reasoning", "")]
                    all_risk_factors = [assessment.get("risk_factors", [])]
                    all_recommendations = [assessment.get("recommendations", [])]
                    run_details = [{
                        "run": 1,
                        "risk_level": all_risk_levels[0],
                        "credit_score": all_credit_scores[0],
                        "confidence": all_confidences[0],
                    }]

                    # Run additional analyses
                    for i in range(2, num_runs + 1):
                        try:
                            analyst = LLMAnalystAgent(model=primary_model)
                            company_info = state.get("company_info", {"name": company_name})
                            search_data = state.get("search_data", {})

                            result = analyst.analyze_company(
                                company_info=company_info,
                                api_data=api_data,
                                search_data=search_data,
                            )

                            if result.success:
                                all_risk_levels.append(result.risk_level or "unknown")
                                all_credit_scores.append(result.credit_score_estimate or 0)
                                all_confidences.append(result.confidence or 0)
                                # Collect for similarity metrics
                                all_reasonings.append(result.reasoning or "")
                                all_risk_factors.append(result.risk_factors or [])
                                all_recommendations.append(result.recommendations or [])
                                run_details.append({
                                    "run": i,
                                    "risk_level": result.risk_level,
                                    "credit_score": result.credit_score_estimate,
                                    "confidence": result.confidence,
                                })
                        except Exception as run_error:
                            logger.warning(f"Consistency run {i} failed: {run_error}")

                    # Calculate consistency metrics
                    if len(all_risk_levels) >= 2:
                        import statistics

                        # Risk level consistency (% agreement)
                        most_common_risk = max(set(all_risk_levels), key=all_risk_levels.count)
                        risk_level_consistency = all_risk_levels.count(most_common_risk) / len(all_risk_levels)

                        # Credit score statistics
                        credit_score_mean = statistics.mean(all_credit_scores)
                        credit_score_std = statistics.stdev(all_credit_scores) if len(all_credit_scores) > 1 else 0

                        # Confidence variance
                        confidence_variance = statistics.variance(all_confidences) if len(all_confidences) > 1 else 0

                        # Calculate reasoning similarity (word overlap using Jaccard)
                        def jaccard_similarity_text(texts):
                            """Calculate average Jaccard similarity between pairs of text."""
                            if len(texts) < 2:
                                return 1.0
                            word_sets = [set(t.lower().split()) for t in texts if t]
                            if len(word_sets) < 2:
                                return 1.0
                            similarities = []
                            for i in range(len(word_sets)):
                                for j in range(i + 1, len(word_sets)):
                                    if word_sets[i] or word_sets[j]:
                                        intersection = len(word_sets[i] & word_sets[j])
                                        union = len(word_sets[i] | word_sets[j])
                                        similarities.append(intersection / union if union > 0 else 1.0)
                            return sum(similarities) / len(similarities) if similarities else 1.0

                        def list_overlap(list_of_lists):
                            """Calculate average Jaccard overlap between pairs of lists."""
                            if len(list_of_lists) < 2:
                                return 1.0
                            sets = [set(items) for items in list_of_lists if items]
                            if len(sets) < 2:
                                return 1.0
                            overlaps = []
                            for i in range(len(sets)):
                                for j in range(i + 1, len(sets)):
                                    if sets[i] or sets[j]:
                                        intersection = len(sets[i] & sets[j])
                                        union = len(sets[i] | sets[j])
                                        overlaps.append(intersection / union if union > 0 else 1.0)
                            return sum(overlaps) / len(overlaps) if overlaps else 1.0

                        reasoning_similarity = jaccard_similarity_text(all_reasonings)
                        risk_factors_overlap = list_overlap(all_risk_factors)
                        recommendations_overlap = list_overlap(all_recommendations)

                        # Overall consistency (including new metrics)
                        overall_consistency = (
                            risk_level_consistency * 0.4 +
                            max(0, 1 - credit_score_std / 100) * 0.3 +
                            reasoning_similarity * 0.1 +
                            risk_factors_overlap * 0.1 +
                            recommendations_overlap * 0.1
                        )

                        # Determine grade
                        if overall_consistency >= 0.9:
                            grade = "A"
                        elif overall_consistency >= 0.75:
                            grade = "B"
                        elif overall_consistency >= 0.6:
                            grade = "C"
                        else:
                            grade = "D"

                        # Log to model_consistency sheet
                        from run_logging.sheets_logger import get_sheets_logger
                        sheets_logger = get_sheets_logger()
                        if sheets_logger and sheets_logger.is_connected():
                            sheets_logger.log_model_consistency(
                                eval_id=run_id,  # Full run_id for Google Sheets lookup
                                company_name=company_name,
                                model_name=primary_model,
                                num_runs=len(all_risk_levels),
                                node="evaluate",
                                agent_name="consistency_evaluator",
                                step_number=7,
                                risk_level_consistency=risk_level_consistency,
                                credit_score_mean=credit_score_mean,
                                credit_score_std=credit_score_std,
                                confidence_variance=confidence_variance,
                                reasoning_similarity=reasoning_similarity,
                                risk_factors_overlap=risk_factors_overlap,
                                recommendations_overlap=recommendations_overlap,
                                overall_consistency=overall_consistency,
                                is_consistent=overall_consistency >= 0.75,
                                consistency_grade=grade,
                                llm_judge_analysis=f"Ran {len(all_risk_levels)} analyses. Risk levels: {all_risk_levels}. Scores: {all_credit_scores}. Reasoning similarity: {reasoning_similarity:.2f}. Risk factors overlap: {risk_factors_overlap:.2f}. Recommendations overlap: {recommendations_overlap:.2f}.",
                                run_details=run_details,
                            )
                            logger.info(f"Same-model consistency: {overall_consistency:.2f} (Grade {grade})")

        except Exception as consistency_error:
            logger.warning(f"Same-model consistency evaluation failed: {consistency_error}")
        # ============ END SAME-MODEL CONSISTENCY ============

        # Complete the run in workflow logger
        if wf_logger:
            wf_logger.complete_run(
                run_id=run_id,
                final_result={
                    "risk_level": assessment.get("overall_risk_level"),
                    "credit_score": assessment.get("credit_score_estimate"),
                    "confidence": assessment.get("confidence_score", 0),  # Use confidence_score key
                    "evaluation_score": evaluation["overall_score"],
                    "api_data": state.get("api_data", {}),
                },
                total_metrics={
                    "tool_selection_score": tool_selection_score,
                    "data_quality_score": data_quality_score,
                    "synthesis_score": synthesis_score,
                },
            )

            # Task 17: Log comprehensive run summary
            llm_results = state.get("llm_results", [])
            total_tokens = sum(r.get("total_tokens", 0) for r in llm_results if r.get("success"))
            total_cost = sum(r.get("total_cost", 0) for r in llm_results if r.get("success"))

            wf_logger.log_run_summary(
                run_id=run_id,
                company_name=company_name,
                status="completed",
                risk_level=assessment.get("overall_risk_level", ""),
                credit_score=assessment.get("credit_score_estimate", 0),
                confidence=assessment.get("confidence_score", 0.0),
                reasoning=assessment.get("llm_reasoning", ""),
                tool_selection_score=tool_selection_score,
                data_quality_score=data_quality_score,
                synthesis_score=synthesis_score,
                overall_score=evaluation.get("overall_score", 0.0),
                final_decision="APPROVED" if assessment.get("overall_risk_level") in ["low", "LOW", "moderate", "MODERATE"] else "REVIEW_REQUIRED",
                decision_reasoning=f"Risk level: {assessment.get('overall_risk_level')}, Score: {assessment.get('credit_score_estimate')}",
                errors=state.get("errors", []),
                tools_used=list(state.get("api_data", {}).keys()),
                agents_used=["SupervisorAgent", "APIAgent", "SearchAgent", "LLMAnalystAgent", "EvaluationAgent"],
                duration_ms=(time.time() - start_time) * 1000,
                total_tokens=total_tokens,
                total_cost=total_cost,
                llm_calls_count=len([r for r in llm_results if r.get("success")]),
            )

            # Task 4: Agent efficiency metrics evaluation
            if AGENT_EVALUATOR_AVAILABLE and evaluate_agent_run:
                try:
                    agent_metrics = evaluate_agent_run(
                        run_id=run_id,
                        company_name=company_name,
                        state=state,
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                    # Log agent metrics to workflow logger
                    wf_logger.log_agent_metrics(
                        run_id=run_id,
                        company_name=company_name,
                        intent_correctness=agent_metrics.intent_correctness,
                        plan_quality=agent_metrics.plan_quality,
                        tool_choice_correctness=agent_metrics.tool_choice_correctness,
                        tool_completeness=agent_metrics.tool_completeness,
                        trajectory_match=agent_metrics.trajectory_match,
                        final_answer_quality=agent_metrics.final_answer_quality,
                        step_count=agent_metrics.step_count,
                        tool_calls=agent_metrics.tool_calls,
                        latency_ms=agent_metrics.latency_ms,
                        overall_score=agent_metrics.overall_score(),
                        intent_details=agent_metrics.intent_details,
                        plan_details=agent_metrics.plan_details,
                        tool_details=agent_metrics.tool_details,
                        trajectory_details=agent_metrics.trajectory_details,
                        answer_details=agent_metrics.answer_details,
                    )

                    # Add agent metrics to evaluation result
                    evaluation["agent_metrics"] = {
                        "intent_correctness": agent_metrics.intent_correctness,
                        "plan_quality": agent_metrics.plan_quality,
                        "tool_choice_correctness": agent_metrics.tool_choice_correctness,
                        "tool_completeness": agent_metrics.tool_completeness,
                        "trajectory_match": agent_metrics.trajectory_match,
                        "final_answer_quality": agent_metrics.final_answer_quality,
                        "overall_agent_score": agent_metrics.overall_score(),
                    }

                    logger.info(f"Agent metrics - Intent: {agent_metrics.intent_correctness:.2f}, "
                               f"Plan: {agent_metrics.plan_quality:.2f}, "
                               f"Tool Choice: {agent_metrics.tool_choice_correctness:.2f}, "
                               f"Tool Complete: {agent_metrics.tool_completeness:.2f}, "
                               f"Trajectory: {agent_metrics.trajectory_match:.2f}, "
                               f"Answer: {agent_metrics.final_answer_quality:.2f}, "
                               f"Overall: {agent_metrics.overall_score():.4f}")

                except Exception as agent_eval_error:
                    logger.warning(f"Agent efficiency evaluation failed: {agent_eval_error}")

            # Task 21: LLM-as-a-judge evaluation
            if LLM_JUDGE_AVAILABLE and evaluate_with_llm_judge:
                try:
                    judge_result = evaluate_with_llm_judge(
                        run_id=run_id,
                        company_name=company_name,
                        assessment=assessment,
                        api_data=state.get("api_data", {}),
                        benchmark=None,  # Can be populated with Coalition data if available
                    )

                    # Log LLM judge result to workflow logger
                    wf_logger.log_llm_judge_result(
                        run_id=run_id,
                        company_name=company_name,
                        model_used=judge_result.model_used,
                        accuracy_score=judge_result.accuracy_score,
                        completeness_score=judge_result.completeness_score,
                        consistency_score=judge_result.consistency_score,
                        actionability_score=judge_result.actionability_score,
                        data_utilization_score=judge_result.data_utilization_score,
                        overall_score=judge_result.overall_score,
                        accuracy_reasoning=judge_result.accuracy_reasoning,
                        completeness_reasoning=judge_result.completeness_reasoning,
                        consistency_reasoning=judge_result.consistency_reasoning,
                        actionability_reasoning=judge_result.actionability_reasoning,
                        data_utilization_reasoning=judge_result.data_utilization_reasoning,
                        overall_reasoning=judge_result.overall_reasoning,
                        benchmark_alignment=judge_result.benchmark_alignment,
                        benchmark_comparison=judge_result.benchmark_comparison,
                        suggestions=judge_result.suggestions,
                        tokens_used=judge_result.tokens_used,
                        evaluation_cost=judge_result.evaluation_cost,
                    )

                    # Add LLM judge results to evaluation
                    evaluation["llm_judge"] = {
                        "accuracy_score": judge_result.accuracy_score,
                        "completeness_score": judge_result.completeness_score,
                        "consistency_score": judge_result.consistency_score,
                        "actionability_score": judge_result.actionability_score,
                        "data_utilization_score": judge_result.data_utilization_score,
                        "overall_score": judge_result.overall_score,
                        "suggestions": judge_result.suggestions,
                    }

                    logger.info(f"LLM Judge - Accuracy: {judge_result.accuracy_score:.2f}, "
                               f"Completeness: {judge_result.completeness_score:.2f}, "
                               f"Consistency: {judge_result.consistency_score:.2f}, "
                               f"Actionability: {judge_result.actionability_score:.2f}, "
                               f"Data Util: {judge_result.data_utilization_score:.2f}, "
                               f"Overall: {judge_result.overall_score:.2f}")

                except Exception as judge_error:
                    logger.warning(f"LLM Judge evaluation failed: {judge_error}")

            # Unified evaluation (combines DeepEval + OpenEvals + Built-in)
            if UNIFIED_EVALUATOR_AVAILABLE and evaluate_workflow:
                try:
                    unified_result = evaluate_workflow(
                        run_id=run_id,
                        company_name=company_name,
                        state=state,
                        llm_results=llm_results,
                        llm_consistency_data=llm_consistency,
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                    # Add unified evaluation to result
                    evaluation["unified_metrics"] = {
                        "accuracy": {
                            "faithfulness": unified_result.accuracy.faithfulness,
                            "hallucination": unified_result.accuracy.hallucination,
                            "answer_relevancy": unified_result.accuracy.answer_relevancy,
                            "factual_accuracy": unified_result.accuracy.factual_accuracy,
                            "final_answer_quality": unified_result.accuracy.final_answer_quality,
                            "accuracy_score": unified_result.accuracy.accuracy_score,
                        },
                        "consistency": {
                            "same_model": unified_result.consistency.same_model_consistency,
                            "cross_model": unified_result.consistency.cross_model_consistency,
                            "risk_agreement": unified_result.consistency.risk_level_agreement,
                            "semantic_similarity": unified_result.consistency.semantic_similarity,
                            "consistency_score": unified_result.consistency.consistency_score,
                        },
                        "agent_efficiency": {
                            "intent_correctness": unified_result.agent_efficiency.intent_correctness,
                            "plan_quality": unified_result.agent_efficiency.plan_quality,
                            "tool_choice": unified_result.agent_efficiency.tool_choice_correctness,
                            "tool_completeness": unified_result.agent_efficiency.tool_completeness,
                            "trajectory_match": unified_result.agent_efficiency.trajectory_match,
                            "final_answer": unified_result.agent_efficiency.final_answer_quality,
                            "overall_score": unified_result.agent_efficiency.overall_score,
                        },
                        "overall_quality_score": unified_result.overall_quality_score,
                        "libraries_used": unified_result.libraries_used,
                    }

                    # Log unified metrics to sheets
                    if wf_logger:
                        wf_logger.log_unified_metrics(
                            run_id=run_id,
                            company_name=company_name,
                            # Accuracy
                            faithfulness=unified_result.accuracy.faithfulness,
                            hallucination=unified_result.accuracy.hallucination,
                            answer_relevancy=unified_result.accuracy.answer_relevancy,
                            factual_accuracy=unified_result.accuracy.factual_accuracy,
                            final_answer_quality=unified_result.accuracy.final_answer_quality,
                            accuracy_score=unified_result.accuracy.accuracy_score,
                            # Consistency
                            same_model_consistency=unified_result.consistency.same_model_consistency,
                            cross_model_consistency=unified_result.consistency.cross_model_consistency,
                            risk_level_agreement=unified_result.consistency.risk_level_agreement,
                            semantic_similarity=unified_result.consistency.semantic_similarity,
                            consistency_score=unified_result.consistency.consistency_score,
                            # Agent efficiency
                            intent_correctness=unified_result.agent_efficiency.intent_correctness,
                            plan_quality=unified_result.agent_efficiency.plan_quality,
                            tool_choice_correctness=unified_result.agent_efficiency.tool_choice_correctness,
                            tool_completeness=unified_result.agent_efficiency.tool_completeness,
                            trajectory_match=unified_result.agent_efficiency.trajectory_match,
                            agent_final_answer=unified_result.agent_efficiency.final_answer_quality,
                            agent_efficiency_score=unified_result.agent_efficiency.overall_score,
                            # Overall
                            overall_quality_score=unified_result.overall_quality_score,
                            libraries_used=unified_result.libraries_used,
                            evaluation_time_ms=unified_result.evaluation_time_ms,
                        )

                    logger.info(f"Unified Evaluation - Accuracy: {unified_result.accuracy.accuracy_score:.2f}, "
                               f"Consistency: {unified_result.consistency.consistency_score:.2f}, "
                               f"Agent: {unified_result.agent_efficiency.overall_score:.2f}, "
                               f"Overall: {unified_result.overall_quality_score:.2f}")

                except Exception as unified_error:
                    logger.warning(f"Unified evaluation failed: {unified_error}")

            # Run DeepEval separately (uses Groq, logs to deepeval_metrics sheet)
            if DEEPEVAL_EVALUATOR_AVAILABLE and evaluate_with_deepeval:
                try:
                    deepeval_result = evaluate_with_deepeval(
                        state=state,
                        run_id=run_id,
                        log_to_sheets=True,
                        provider="groq",  # Use free Groq model
                    )

                    # Add DeepEval metrics to evaluation result
                    evaluation["deepeval_metrics"] = {
                        "answer_relevancy": deepeval_result.answer_relevancy,
                        "faithfulness": deepeval_result.faithfulness,
                        "hallucination": deepeval_result.hallucination,
                        "contextual_relevancy": deepeval_result.contextual_relevancy,
                        "bias": deepeval_result.bias,
                        "overall_score": deepeval_result.overall_score,
                    }

                    logger.info(f"DeepEval - Relevancy: {deepeval_result.answer_relevancy:.2f}, "
                               f"Faithfulness: {deepeval_result.faithfulness:.2f}, "
                               f"Hallucination: {deepeval_result.hallucination:.2f}, "
                               f"Overall: {deepeval_result.overall_score:.2f}")

                except Exception as deepeval_error:
                    logger.warning(f"DeepEval evaluation failed: {deepeval_error}")

            # Fetch and log LangSmith traces for this run
            if LANGSMITH_INTEGRATION_AVAILABLE and get_langsmith_integration:
                try:
                    langsmith_integration = get_langsmith_integration()
                    if langsmith_integration.client:
                        # Log current run trace (will fetch from LangSmith API)
                        logged = langsmith_integration.fetch_and_log_traces(limit=5, hours_back=1)
                        if logged > 0:
                            logger.info(f"Logged {logged} LangSmith traces to sheets")
                except Exception as langsmith_error:
                    logger.debug(f"LangSmith trace logging skipped: {langsmith_error}")

            # Log verification - verify all sheet logging worked for this run
            try:
                from run_logging.sheets_logger import get_sheets_logger
                sheets_logger = get_sheets_logger()
                if sheets_logger and sheets_logger.is_connected():
                    # Wait a moment for async logging to complete
                    import time as time_module
                    time_module.sleep(2)  # Give async writers time to finish
                    sheets_logger.log_verification(
                        run_id=run_id,
                        company_name=company_name,
                    )
            except Exception as verify_error:
                logger.debug(f"Log verification skipped: {verify_error}")

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
                agent_name="workflow_evaluator",
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
# ASYNC RUNNER WITH EVENT LOGGING
# =============================================================================

async def run_with_event_logging(
    company_name: str,
    jurisdiction: Optional[str] = None,
    ticker: Optional[str] = None,
    log_to_sheets: bool = True,
    log_to_mongodb: bool = True,
) -> Dict[str, Any]:
    """
    Run the credit intelligence workflow with full LangGraph event logging.

    This function captures all LangGraph events (chain starts, LLM calls,
    tool executions, etc.) and logs them to Google Sheets and MongoDB.

    Args:
        company_name: Company to analyze
        jurisdiction: Optional jurisdiction (e.g., "US", "UK")
        ticker: Optional stock ticker
        log_to_sheets: Whether to log events to Google Sheets
        log_to_mongodb: Whether to log events to MongoDB

    Returns:
        Final workflow state with assessment

    Example:
        import asyncio
        result = asyncio.run(run_with_event_logging("Apple Inc"))
        print(result["assessment"]["overall_risk_level"])
    """
    if not LANGGRAPH_LOGGER_AVAILABLE:
        logger.warning("LangGraph event logger not available, running without event logging")
        # Fallback to sync execution
        initial_state = {
            "company_name": company_name,
            "jurisdiction": jurisdiction,
            "ticker": ticker,
            "company_info": {},
            "task_plan": [],
            "api_data": {},
            "search_data": {},
            "assessment": None,
            "errors": [],
            "status": "started",
            "llm_results": [],
            "human_approved": False,
            "human_feedback": None,
            "validation_message": "",
            "requires_review": False,
        }
        return graph.invoke(initial_state)

    # Generate run_id
    run_id = str(uuid.uuid4())

    # Create initial state
    initial_state = {
        "company_name": company_name,
        "jurisdiction": jurisdiction,
        "ticker": ticker,
        "run_id": run_id,
        "company_info": {},
        "task_plan": [],
        "api_data": {},
        "search_data": {},
        "assessment": None,
        "errors": [],
        "status": "started",
        "llm_results": [],
        "human_approved": False,
        "human_feedback": None,
        "validation_message": "",
        "requires_review": False,
    }

    logger.info(f"Starting workflow with event logging: run_id={run_id}, company={company_name}")

    # Run with event logging
    result = await run_graph_with_event_logging(
        graph=graph,
        input_state=initial_state,
        run_id=run_id,
        company_name=company_name,
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    )

    logger.info(f"Workflow completed with event logging: run_id={run_id}")

    return result


def run_sync_with_logging(
    company_name: str,
    jurisdiction: Optional[str] = None,
    ticker: Optional[str] = None,
    log_to_sheets: bool = True,
    log_to_mongodb: bool = True,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_with_event_logging.

    Use this when you're not in an async context.

    Args:
        company_name: Company to analyze
        jurisdiction: Optional jurisdiction
        ticker: Optional stock ticker
        log_to_sheets: Whether to log to Sheets
        log_to_mongodb: Whether to log to MongoDB

    Returns:
        Final workflow state

    Example:
        result = run_sync_with_logging("Apple Inc")
    """
    import asyncio

    return asyncio.run(run_with_event_logging(
        company_name=company_name,
        jurisdiction=jurisdiction,
        ticker=ticker,
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    ))


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
    parser.add_argument("--run-with-logging", type=str, help="Run workflow with full event logging")
    parser.add_argument("--no-sheets", action="store_true", help="Disable Google Sheets logging")
    parser.add_argument("--no-mongodb", action="store_true", help="Disable MongoDB logging")

    args = parser.parse_args()

    if args.mermaid:
        print(get_mermaid_diagram())
    elif args.ascii:
        print_graph_ascii()
    elif args.save:
        save_graph_image(args.save)
    elif args.run_with_logging:
        # Run with full event logging
        print(f"\nRunning workflow with event logging for: {args.run_with_logging}")
        print(f"  Sheets logging: {'disabled' if args.no_sheets else 'enabled'}")
        print(f"  MongoDB logging: {'disabled' if args.no_mongodb else 'enabled'}")
        print("-" * 60)

        result = run_sync_with_logging(
            company_name=args.run_with_logging,
            log_to_sheets=not args.no_sheets,
            log_to_mongodb=not args.no_mongodb,
        )

        if result.get("assessment"):
            assessment = result["assessment"]
            print(f"\nAssessment for {args.run_with_logging}:")
            print(f"  Risk Level: {assessment.get('overall_risk_level', 'N/A')}")
            print(f"  Credit Score: {assessment.get('credit_score_estimate', 'N/A')}/100")
            print(f"  Confidence: {assessment.get('confidence_score', 0):.2f}")
        else:
            print(f"\nNo assessment generated. Status: {result.get('status')}")
            print(f"Errors: {result.get('errors', [])}")

        print("\nEvent logging complete. Check langgraph_events sheet and MongoDB.")
    elif args.run:
        # Run the workflow (without event logging)
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
