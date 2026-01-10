"""LangChain Agent-Based Tool Selection.

Provides an alternative to the text-based tool selection pattern in tool_supervisor.py.
Uses LangChain's Agent framework with structured tool calling for:
- Automatic callback/trace integration
- Structured argument validation
- Safer tool execution (no hallucinated tool names)

Usage:
    from agents.langchain_tool_agent import AgentToolSelector

    selector = AgentToolSelector()
    result = selector.run_with_agent(company_name="Apple Inc")

Comparison with text-based selection:
    Text-based (tool_supervisor.py):
        - LLM returns JSON with tool names and params
        - Code parses JSON and executes tools manually
        - More control, but needs extra parsing and validation
        - Existing approach, well-tested

    Agent-based (this module):
        - LangChain Agent calls tools as functions
        - Automatic argument validation via schemas
        - Built-in callback support for traces
        - Newer approach, more integrated
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Feature flag for agent-based selection
USE_AGENT_TOOL_SELECTION = os.getenv("USE_AGENT_TOOL_SELECTION", "false").lower() == "true"

# Try to import LangChain Agent components
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AGENT_AVAILABLE = True
except ImportError:
    LANGCHAIN_AGENT_AVAILABLE = False
    logger.info("LangChain Agent not available. Using text-based tool selection.")

# Import LLM factory
try:
    from config.langchain_llm import get_chat_groq, get_llm_for_prompt
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import tool adapters
try:
    from tools.langchain_adapters import get_structured_langchain_tools, is_langchain_tools_available
    TOOLS_AVAILABLE = is_langchain_tools_available()
except ImportError:
    TOOLS_AVAILABLE = False

# Import callbacks
try:
    from config.langchain_callbacks import (
        CostTrackerCallback,
        SheetsLoggingCallback,
        create_cost_callback,
        create_sheets_logging_callback,
    )
    from config.cost_tracker import get_cost_tracker
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False


class AgentToolSelector:
    """
    LangChain Agent-based tool selector for credit intelligence.

    Uses create_tool_calling_agent() with structured tools for automatic
    tool calling, argument validation, and callback integration.

    Key Benefits:
        - Structured tool calling (no JSON parsing needed)
        - Automatic LangSmith/LangFuse tracing via callbacks
        - Built-in argument validation via Pydantic schemas
        - Safer execution (can't call non-existent tools)

    Trade-offs:
        - Less explicit control over tool execution order
        - Agent decides when to stop (may need max_iterations)
        - Requires all tools to be LangChain Tool objects
    """

    def __init__(
        self,
        model: str = "primary",
        temperature: float = 0.1,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize agent-based tool selector.

        Args:
            model: LLM model alias (primary, fast, balanced)
            temperature: Sampling temperature
            max_iterations: Max agent iterations before stopping
            verbose: Enable verbose agent logging
        """
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._agent_executor = None

    def _create_agent(
        self,
        run_id: str,
        company_name: str = "",
    ) -> Optional[Any]:
        """
        Create LangChain Agent with tools and callbacks.

        Args:
            run_id: Run ID for tracking
            company_name: Company being analyzed (for logging)

        Returns:
            AgentExecutor instance or None if not available
        """
        if not all([LANGCHAIN_AGENT_AVAILABLE, LLM_AVAILABLE, TOOLS_AVAILABLE]):
            logger.warning("LangChain Agent dependencies not available")
            return None

        # Get tools
        tools = get_structured_langchain_tools(run_id=run_id)
        if not tools:
            logger.warning("No tools available for agent")
            return None

        # Setup callbacks
        callbacks = []
        if CALLBACKS_AVAILABLE:
            callbacks.append(create_cost_callback(run_id=run_id, call_type="agent_tool_selection"))
            callbacks.append(create_sheets_logging_callback(
                run_id=run_id,
                company_name=company_name,
                log_to_sheets=True,
                log_to_mongodb=True,
            ))

        # Get LLM
        llm = get_llm_for_prompt(
            prompt_id="tool_selection",
            callbacks=callbacks,
        )
        if not llm:
            llm = get_chat_groq(model=self.model, temperature=self.temperature, callbacks=callbacks)

        if not llm:
            logger.error("Failed to create LLM for agent")
            return None

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            callbacks=callbacks,
        )

        return agent_executor

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are a credit intelligence agent analyzing companies for credit risk assessment.

Your task is to gather relevant information about a company using the available tools.

Strategy:
1. First, determine if the company is public (has SEC filings, stock ticker) or private
2. For PUBLIC US companies:
   - Use fetch_sec_data to get SEC EDGAR filings
   - Use fetch_market_data to get stock and financial data
3. For PRIVATE companies:
   - Use web_search to find company information
   - Use fetch_legal_data to check for legal issues
4. For UNKNOWN companies:
   - Use web_search first to determine company type
   - Then call appropriate tools based on findings

Be efficient - only use tools that will provide useful data for credit assessment.
When you have gathered sufficient data, provide a summary of your findings."""

    def run_with_agent(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run credit data gathering using LangChain Agent.

        The agent will automatically select and execute appropriate tools
        based on the company name and context.

        Args:
            company_name: Name of the company to analyze
            context: Additional context (ticker, is_public, etc.)
            run_id: Run ID for tracking

        Returns:
            Dict with:
                - run_id: Run identifier
                - output: Agent's final response
                - tools_called: List of tools that were called
                - intermediate_steps: Agent's execution steps
                - execution_time_ms: Total execution time
        """
        import time
        start_time = time.time()

        run_id = run_id or str(uuid.uuid4())
        context = context or {}

        # Build input for agent
        input_text = f"Analyze {company_name} for credit risk assessment."
        if context.get("ticker"):
            input_text += f" Ticker symbol: {context['ticker']}."
        if context.get("is_public") is True:
            input_text += " This is a public company."
        elif context.get("is_private") is True:
            input_text += " This is a private company."

        # Create agent
        agent_executor = self._create_agent(run_id=run_id, company_name=company_name)

        if not agent_executor:
            logger.warning("Agent not available, returning empty result")
            return {
                "run_id": run_id,
                "output": "Agent not available",
                "tools_called": [],
                "error": "LangChain Agent dependencies not available",
                "execution_time_ms": 0,
            }

        try:
            # Run agent
            result = agent_executor.invoke(
                {"input": input_text},
                config={"run_name": f"credit_analysis_{company_name}"},
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Extract tools called from intermediate steps
            tools_called = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        action = step[0]
                        if hasattr(action, "tool"):
                            tools_called.append(action.tool)

            return {
                "run_id": run_id,
                "company_name": company_name,
                "output": result.get("output", ""),
                "tools_called": tools_called,
                "intermediate_steps": result.get("intermediate_steps", []),
                "execution_time_ms": round(execution_time_ms, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            return {
                "run_id": run_id,
                "company_name": company_name,
                "output": "",
                "tools_called": [],
                "error": str(e),
                "execution_time_ms": round(execution_time_ms, 2),
                "timestamp": datetime.utcnow().isoformat(),
            }


def is_agent_tool_selection_available() -> bool:
    """Check if agent-based tool selection is available."""
    return all([LANGCHAIN_AGENT_AVAILABLE, LLM_AVAILABLE, TOOLS_AVAILABLE])


def is_agent_tool_selection_enabled() -> bool:
    """Check if agent-based tool selection is enabled via feature flag."""
    return USE_AGENT_TOOL_SELECTION and is_agent_tool_selection_available()
