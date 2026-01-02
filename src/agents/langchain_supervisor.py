"""LangChain-Native Supervisor Agent.

Uses LangChain chains (LLMChain/RunnableSequence) for:
- Better LangSmith tracing and observability
- Structured prompt templates
- Automatic output parsing with PydanticOutputParser
- Cleaner chain composition

This is the LangChain-native alternative to ToolSupervisor.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import LangChain components
try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.runnables import RunnableSequence, RunnableLambda
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-core not available")

# Import our LLM factory
try:
    from config.langchain_llm import get_chat_groq, is_langchain_groq_available
    from config.langchain_callbacks import (
        CostTrackerCallback,
        SheetsLoggingCallback,
        create_sheets_logging_callback,
        is_sheets_logging_available,
    )
    from config.cost_tracker import get_cost_tracker
    LLM_AVAILABLE = is_langchain_groq_available()
    SHEETS_LOGGING_AVAILABLE = is_sheets_logging_available()
except ImportError:
    LLM_AVAILABLE = False
    SHEETS_LOGGING_AVAILABLE = False
    logger.warning("LangChain LLM not available")

# Import output schemas
try:
    from config.output_schemas import ToolSelection, CreditAssessment
    from config.output_parsers import result_to_dict
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    logger.warning("Output schemas not available")

# Import tools
from tools import get_tool_executor


# =============================================================================
# Prompt Templates
# =============================================================================

TOOL_SELECTION_TEMPLATE = """You are a credit intelligence agent selecting tools for credit risk assessment.

## Company: {company_name}
## Context: {context}

## Available Tools
{tool_specs}

## Instructions
1. Determine if this is a public or private company
2. Select 1-3 relevant tools from the list above
3. Explain your reasoning

IMPORTANT: You MUST include ALL required fields in your response:
- company_analysis (with is_likely_public and reasoning)
- tools_to_use (list of tools with name, params, reason)
- execution_order_reasoning (why this order)

{format_instructions}"""


SYNTHESIS_TEMPLATE = """You are a senior credit analyst. Analyze the collected data and provide a credit risk assessment.

## Company
Name: {company_name}

## Tool Selection Reasoning
{tool_reasoning}

## Collected Data
{tool_results}

## Your Task
Based on ALL the data collected, provide a comprehensive credit risk assessment.

{format_instructions}"""


# =============================================================================
# LangChain Supervisor
# =============================================================================

class LangChainSupervisor:
    """
    LangChain-native supervisor using chains for tool selection and synthesis.

    Benefits over ToolSupervisor:
    - Automatic LangSmith tracing for all chain steps
    - Structured prompt templates with variables
    - PydanticOutputParser for validated responses
    - Composable chains with RunnableSequence
    """

    def __init__(
        self,
        model: str = "primary",
        temperature: float = 0.1,
        log_to_sheets: bool = True,
        log_to_mongodb: bool = False,
    ):
        """
        Initialize the LangChain supervisor.

        Args:
            model: Model name or alias (primary, fast, balanced)
            temperature: Sampling temperature for LLM
            log_to_sheets: Whether to log LangChain events to Google Sheets
            log_to_mongodb: Whether to log LangChain events to MongoDB
        """
        self.model = model
        self.temperature = temperature
        self.log_to_sheets = log_to_sheets
        self.log_to_mongodb = log_to_mongodb
        self.tool_executor = get_tool_executor()
        self.decision_log: List[Dict[str, Any]] = []

        # Initialize chains
        self._selection_chain = None
        self._synthesis_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """Setup LangChain chains for tool selection and synthesis."""
        if not LANGCHAIN_AVAILABLE or not LLM_AVAILABLE or not SCHEMAS_AVAILABLE:
            logger.warning("Cannot setup chains - dependencies not available")
            return

        # Tool Selection Chain
        selection_parser = PydanticOutputParser(pydantic_object=ToolSelection)
        selection_prompt = PromptTemplate(
            template=TOOL_SELECTION_TEMPLATE,
            input_variables=["company_name", "context", "tool_specs"],
            partial_variables={"format_instructions": selection_parser.get_format_instructions()},
        )

        # Synthesis Chain
        synthesis_parser = PydanticOutputParser(pydantic_object=CreditAssessment)
        synthesis_prompt = PromptTemplate(
            template=SYNTHESIS_TEMPLATE,
            input_variables=["company_name", "tool_reasoning", "tool_results"],
            partial_variables={"format_instructions": synthesis_parser.get_format_instructions()},
        )

        # Get LLM with callbacks
        callbacks = []
        try:
            tracker = get_cost_tracker()
            callbacks.append(CostTrackerCallback(tracker=tracker, call_type="langchain_supervisor"))
        except Exception:
            pass

        llm = get_chat_groq(
            model=self.model,
            temperature=self.temperature,
            callbacks=callbacks,
        )

        if llm:
            # Create chains using LCEL (LangChain Expression Language)
            self._selection_chain = selection_prompt | llm | selection_parser
            self._synthesis_chain = synthesis_prompt | llm | synthesis_parser
            logger.info(f"LangChainSupervisor chains initialized with model: {self.model}")
        else:
            logger.error("Failed to create LLM for chains")

    def is_available(self) -> bool:
        """Check if the supervisor is ready to use."""
        return self._selection_chain is not None and self._synthesis_chain is not None

    def select_tools(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Use LLMChain to select which tools to use.

        Args:
            company_name: Name of the company
            context: Additional context
            run_id: Run ID for tracking

        Returns:
            Dict with tool selection and reasoning
        """
        run_id = run_id or str(uuid.uuid4())

        if not self.is_available():
            return {
                "run_id": run_id,
                "error": "LangChain supervisor not available",
                "selection": {},
            }

        # Prepare inputs
        tool_specs = self.tool_executor.get_tool_specs_text()
        context_str = str(context or {})

        try:
            # Prepare callbacks for logging
            invoke_callbacks = []
            if self.log_to_sheets and SHEETS_LOGGING_AVAILABLE:
                invoke_callbacks.append(create_sheets_logging_callback(
                    run_id=run_id,
                    company_name=company_name,
                    log_to_sheets=True,
                    log_to_mongodb=self.log_to_mongodb,
                ))

            # Invoke the chain with logging callbacks
            invoke_config = {"callbacks": invoke_callbacks} if invoke_callbacks else {}
            result: ToolSelection = self._selection_chain.invoke(
                {
                    "company_name": company_name,
                    "context": context_str,
                    "tool_specs": tool_specs,
                },
                config=invoke_config,
            )

            # Convert to dict
            selection = result_to_dict(result)

            # Log decision
            self.decision_log.append({
                "run_id": run_id,
                "step": "tool_selection",
                "company_name": company_name,
                "selection": selection,
                "timestamp": datetime.utcnow().isoformat(),
            })

            tools_selected = [t.get("name") for t in selection.get("tools_to_use", [])]
            logger.info(f"LangChain tool selection for {company_name}: {tools_selected}")

            return {
                "run_id": run_id,
                "selection": selection,
            }

        except Exception as e:
            logger.warning(f"Tool selection chain failed: {e}, using fallback")
            # Fallback to legacy ToolSupervisor
            try:
                from agents.tool_supervisor import ToolSupervisor
                fallback = ToolSupervisor(model=self.model)
                return fallback.select_tools(company_name, context, run_id)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "run_id": run_id,
                    "error": str(e),
                    "selection": {},
                }

    def execute_selected_tools(
        self,
        company_name: str,
        tool_selection: Dict[str, Any],
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Execute the tools selected by the chain.

        Args:
            company_name: Company name
            tool_selection: Result from select_tools()
            run_id: Run ID for tracking

        Returns:
            Dict with all tool results
        """
        run_id = run_id or tool_selection.get("run_id", str(uuid.uuid4()))

        selection = tool_selection.get("selection", {})
        tools_to_use = selection.get("tools_to_use", [])

        results = {}
        execution_metrics = []

        for tool_call in tools_to_use:
            tool_name = tool_call.get("name")
            params = tool_call.get("params", {})

            # Ensure company identifier is passed
            if "company_identifier" not in params and "company_name" not in params:
                if tool_name == "fetch_sec_data":
                    params["company_identifier"] = company_name
                elif tool_name == "fetch_market_data":
                    params["ticker"] = params.get("ticker", company_name[:4].upper())
                    params["company_name"] = company_name
                else:
                    params["company_name"] = company_name

            # Execute tool
            result = self.tool_executor.execute_tool(tool_name, run_id=run_id, **params)
            results[tool_name] = result.to_dict()

            execution_metrics.append({
                "tool_name": tool_name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
            })

        # Log execution
        self.decision_log.append({
            "run_id": run_id,
            "step": "tool_execution",
            "company_name": company_name,
            "tools_executed": len(tools_to_use),
            "execution_metrics": execution_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return {
            "run_id": run_id,
            "results": results,
            "execution_metrics": execution_metrics,
        }

    def synthesize_assessment(
        self,
        company_name: str,
        tool_selection: Dict[str, Any],
        tool_results: Dict[str, Any],
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Use LLMChain to synthesize final credit assessment.

        Args:
            company_name: Company name
            tool_selection: Tool selection result
            tool_results: Results from tool execution
            run_id: Run ID for tracking

        Returns:
            Dict with final assessment
        """
        run_id = run_id or str(uuid.uuid4())

        if not self.is_available():
            return {
                "run_id": run_id,
                "error": "LangChain supervisor not available",
                "assessment": {},
            }

        # Prepare inputs
        selection = tool_selection.get("selection", {})
        tool_reasoning = str(selection.get("company_analysis", {}))
        tool_results_str = str(tool_results.get("results", {}))[:8000]  # Truncate

        try:
            # Prepare callbacks for logging
            invoke_callbacks = []
            if self.log_to_sheets and SHEETS_LOGGING_AVAILABLE:
                invoke_callbacks.append(create_sheets_logging_callback(
                    run_id=run_id,
                    company_name=company_name,
                    log_to_sheets=True,
                    log_to_mongodb=self.log_to_mongodb,
                ))

            # Invoke the chain with logging callbacks
            invoke_config = {"callbacks": invoke_callbacks} if invoke_callbacks else {}
            result: CreditAssessment = self._synthesis_chain.invoke(
                {
                    "company_name": company_name,
                    "tool_reasoning": tool_reasoning,
                    "tool_results": tool_results_str,
                },
                config=invoke_config,
            )

            # Convert to dict
            assessment = result_to_dict(result)

            # Log synthesis
            self.decision_log.append({
                "run_id": run_id,
                "step": "synthesis",
                "company_name": company_name,
                "assessment": assessment,
                "timestamp": datetime.utcnow().isoformat(),
            })

            logger.info(
                f"LangChain assessment for {company_name}: "
                f"{assessment.get('risk_level')} (score: {assessment.get('credit_score')})"
            )

            return {
                "run_id": run_id,
                "assessment": assessment,
            }

        except Exception as e:
            logger.warning(f"Synthesis chain failed: {e}, using fallback")
            # Fallback to legacy ToolSupervisor
            try:
                from agents.tool_supervisor import ToolSupervisor
                fallback = ToolSupervisor(model=self.model)
                return fallback.synthesize_assessment(
                    company_name, tool_selection, tool_results, run_id
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "run_id": run_id,
                    "error": str(e),
                    "assessment": {},
                }

    def run_full_assessment(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run complete credit assessment workflow using LangChain.

        This is the main entry point that:
        1. Selects appropriate tools (via chain)
        2. Executes selected tools
        3. Synthesizes final assessment (via chain)

        Args:
            company_name: Name of the company
            context: Additional context

        Returns:
            Complete assessment with all metrics
        """
        import time
        run_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting LangChain assessment for: {company_name} (run_id: {run_id})")

        # Step 1: Tool Selection (via chain)
        tool_selection = self.select_tools(company_name, context, run_id)

        if tool_selection.get("error"):
            return {
                "run_id": run_id,
                "error": tool_selection["error"],
                "company_name": company_name,
            }

        # Step 2: Execute Tools
        tool_results = self.execute_selected_tools(company_name, tool_selection, run_id)

        # Step 3: Synthesize Assessment (via chain)
        assessment = self.synthesize_assessment(
            company_name, tool_selection, tool_results, run_id
        )

        total_time = (time.time() - start_time) * 1000

        return {
            "run_id": run_id,
            "company_name": company_name,
            "context": context,
            "tool_selection": tool_selection,
            "tool_results": tool_results,
            "assessment": assessment,
            "total_execution_time_ms": round(total_time, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "supervisor_type": "langchain",
        }

    def get_decision_log(self, run_id: str = None) -> List[Dict[str, Any]]:
        """Get decision log, optionally filtered by run_id."""
        if run_id:
            return [l for l in self.decision_log if l.get("run_id") == run_id]
        return self.decision_log

    def clear_log(self):
        """Clear decision log."""
        self.decision_log = []


# =============================================================================
# Factory and Feature Flag
# =============================================================================

# Feature flag to enable LangChain supervisor
USE_LANGCHAIN_SUPERVISOR = os.getenv("USE_LANGCHAIN_SUPERVISOR", "false").lower() == "true"


def get_supervisor(use_langchain: bool = None, model: str = "primary"):
    """
    Get the appropriate supervisor based on configuration.

    Args:
        use_langchain: Override USE_LANGCHAIN_SUPERVISOR flag
        model: Model to use

    Returns:
        LangChainSupervisor or ToolSupervisor instance
    """
    should_use_langchain = use_langchain if use_langchain is not None else USE_LANGCHAIN_SUPERVISOR

    if should_use_langchain:
        supervisor = LangChainSupervisor(model=model)
        if supervisor.is_available():
            logger.info("Using LangChainSupervisor")
            return supervisor
        logger.warning("LangChainSupervisor not available, falling back to ToolSupervisor")

    # Fallback to legacy supervisor
    from agents.tool_supervisor import ToolSupervisor
    logger.info("Using ToolSupervisor")
    return ToolSupervisor(model=model)


def is_langchain_supervisor_available() -> bool:
    """Check if LangChain supervisor can be used."""
    return LANGCHAIN_AVAILABLE and LLM_AVAILABLE and SCHEMAS_AVAILABLE
