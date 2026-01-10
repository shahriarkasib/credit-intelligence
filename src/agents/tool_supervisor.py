"""Tool-Based Supervisor Agent - LLM selects which tools to use."""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import Groq client (legacy)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed")

# Import LangChain LLM factory (preferred)
try:
    from config.langchain_llm import get_chat_groq, get_llm_for_prompt, get_llm_config_for_prompt, is_langchain_groq_available
    from config.langchain_callbacks import CostTrackerCallback
    from langchain_core.messages import HumanMessage
    LANGCHAIN_GROQ_AVAILABLE = is_langchain_groq_available()
    LLM_FOR_PROMPT_AVAILABLE = True
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False
    LLM_FOR_PROMPT_AVAILABLE = False
    logger.warning("LangChain LLM not available, using legacy Groq client")

# Import cost tracker
try:
    from config.cost_tracker import get_cost_tracker
    COST_TRACKER_AVAILABLE = True
except ImportError:
    COST_TRACKER_AVAILABLE = False

# Import output parsers (Step 3)
try:
    from config.output_parsers import (
        parse_tool_selection,
        parse_credit_assessment,
        get_format_instructions,
        result_to_dict,
        is_parsers_available,
    )
    from config.output_schemas import ToolSelection, CreditAssessment
    OUTPUT_PARSERS_AVAILABLE = is_parsers_available()
except ImportError:
    OUTPUT_PARSERS_AVAILABLE = False
    logger.warning("Output parsers not available, using legacy JSON parsing")

# Import centralized prompts
try:
    from config.prompts import get_prompt_text, get_prompt
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logger.warning("Centralized prompts not available, using inline prompts")

from tools import ToolExecutor, get_tool_executor

# Import MongoDB for decision persistence
try:
    from storage.mongodb import get_db
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.debug("MongoDB not available for decision persistence")


# Available models
MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
    "balanced": "gemma2-9b-it",
}

# Feature flag: Use LangChain ChatGroq instead of raw Groq SDK
USE_LANGCHAIN_LLM = os.getenv("USE_LANGCHAIN_LLM", "true").lower() == "true"


class ToolSupervisor:
    """
    Supervisor agent that uses LLM to decide which tools to call.

    The LLM analyzes the company name and context, then decides:
    1. Which tools are appropriate for this company
    2. In what order to call them
    3. What parameters to pass

    This enables evaluation of:
    - Tool selection accuracy
    - Reasoning quality
    - Decision efficiency
    """

    def __init__(self, model: str = "primary"):
        self.model = MODELS.get(model, model)
        self.tool_executor = get_tool_executor()
        self._client = None  # Legacy Groq client
        self._use_langchain = USE_LANGCHAIN_LLM and LANGCHAIN_GROQ_AVAILABLE

        # Initialize appropriate LLM client
        if self._use_langchain:
            logger.info(f"ToolSupervisor using LangChain ChatGroq: {self.model}")
        elif GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._client = Groq(api_key=api_key)
                logger.info(f"ToolSupervisor using legacy Groq: {self.model}")

        # Logging
        self.decision_log: List[Dict[str, Any]] = []

        # MongoDB connection for persistence
        self._db = None
        if MONGODB_AVAILABLE:
            try:
                self._db = get_db()
            except Exception as e:
                logger.debug(f"MongoDB not connected: {e}")

    def _persist_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Persist a decision to MongoDB for crash recovery and auditing.

        Args:
            decision: Decision log entry to persist

        Returns:
            True if persisted successfully, False otherwise
        """
        if not self._db or not self._db.is_connected():
            return False

        try:
            # Store in a dedicated collection for supervisor decisions
            collection = self._db.db.supervisor_decisions
            collection.insert_one(decision.copy())
            logger.debug(f"Persisted decision for run {decision.get('run_id', 'unknown')}")
            return True
        except Exception as e:
            logger.warning(f"Failed to persist decision: {e}")
            return False

    def _get_tool_selection_prompt(self, company_name: str, context: Dict[str, Any] = None) -> str:
        """Generate prompt for tool selection."""
        tool_specs = self.tool_executor.get_tool_specs_text()

        # Try to use centralized prompts
        if PROMPTS_AVAILABLE:
            try:
                system_prompt, user_prompt = get_prompt_text(
                    "tool_selection",
                    company_name=company_name,
                    context=json.dumps(context or {}),
                    tool_specs=tool_specs,
                )
                # Combine system and user for legacy single-prompt call
                return f"{system_prompt}\n\n{user_prompt}"
            except Exception as e:
                logger.warning(f"Failed to get centralized prompt: {e}")

        # Fallback to inline prompt
        prompt = f"""You are a credit intelligence agent. Your task is to select the appropriate tools to gather information about a company for credit risk assessment.

## Company to Analyze
Name: {company_name}
Additional Context: {json.dumps(context or {})}

## Available Tools
{tool_specs}

## Your Task
Analyze the company name and decide which tools to use. Consider:
1. Is this a public or private company?
2. What data sources would be most relevant?
3. What is the most efficient order to call them?

## Response Format
Respond with a JSON object:
```json
{{
    "company_analysis": {{
        "is_likely_public": true/false,
        "reasoning": "Brief explanation of why you made this determination"
    }},
    "tools_to_use": [
        {{
            "name": "tool_name",
            "params": {{"param_name": "value"}},
            "reason": "Why this tool is needed"
        }}
    ],
    "execution_order_reasoning": "Why this order makes sense"
}}
```

Only include tools that are truly needed. Be efficient.
"""
        return prompt

    def _summarize_tool_results(
        self,
        tool_results: Dict[str, Any],
        max_chars_per_tool: int = 5000,
        max_total_chars: int = 30000,
    ) -> str:
        """
        Summarize large tool results to fit within token budget.

        This prevents token overflow when tool results contain large amounts
        of data (e.g., full SEC filings, extensive court records).

        Args:
            tool_results: Dict of tool_name -> result data
            max_chars_per_tool: Maximum characters per individual tool result
            max_total_chars: Maximum total characters for all results

        Returns:
            JSON string of summarized results
        """
        summarized = {}

        for tool_name, result in tool_results.items():
            result_str = json.dumps(result, default=str)
            result_len = len(result_str)

            if result_len > max_chars_per_tool:
                # Summarize large results
                if isinstance(result, dict):
                    # Keep key structure but truncate values
                    summary = {
                        "_summary": f"Large result truncated ({result_len} chars)",
                        "_keys": list(result.keys())[:20],
                    }
                    # Include small values directly
                    for key, value in result.items():
                        value_str = json.dumps(value, default=str)
                        if len(value_str) < 500:
                            summary[key] = value
                        elif isinstance(value, list) and len(value) > 0:
                            summary[f"{key}_count"] = len(value)
                            summary[f"{key}_sample"] = value[:2] if len(value) > 2 else value
                        elif isinstance(value, dict):
                            summary[f"{key}_keys"] = list(value.keys())[:10]
                    summarized[tool_name] = summary
                elif isinstance(result, list):
                    summarized[tool_name] = {
                        "_summary": f"Large list truncated ({result_len} chars)",
                        "_count": len(result),
                        "_sample": result[:3] if len(result) > 3 else result,
                    }
                else:
                    summarized[tool_name] = {
                        "_summary": f"Large result truncated ({result_len} chars)",
                        "_preview": str(result)[:500],
                    }
            else:
                summarized[tool_name] = result

        # Check total size and further truncate if needed
        total_str = json.dumps(summarized, default=str, indent=2)
        if len(total_str) > max_total_chars:
            logger.warning(f"Tool results still too large ({len(total_str)} chars), further truncating")
            # Remove detailed data, keep only summaries
            for tool_name in summarized:
                if isinstance(summarized[tool_name], dict):
                    if "_summary" not in summarized[tool_name]:
                        summarized[tool_name] = {
                            "_summary": f"Result available ({len(json.dumps(summarized[tool_name], default=str))} chars)",
                            "_keys": list(summarized[tool_name].keys())[:10] if isinstance(summarized[tool_name], dict) else None,
                        }
            total_str = json.dumps(summarized, default=str, indent=2)

        logger.debug(f"Summarized tool results: {len(total_str)} chars from {len(json.dumps(tool_results, default=str))} chars")
        return total_str

    def _get_synthesis_prompt(
        self,
        company_name: str,
        tool_results: Dict[str, Any],
        tool_selection: Dict[str, Any]
    ) -> str:
        """Generate prompt for final credit synthesis."""
        tool_reasoning = json.dumps(tool_selection.get('company_analysis', {}), indent=2)
        # Use summarized results to prevent token overflow
        tool_results_str = self._summarize_tool_results(tool_results)

        # Try to use centralized prompts
        if PROMPTS_AVAILABLE:
            try:
                system_prompt, user_prompt = get_prompt_text(
                    "credit_synthesis",
                    company_name=company_name,
                    tool_reasoning=tool_reasoning,
                    tool_results=tool_results_str,
                )
                # Combine system and user for legacy single-prompt call
                return f"{system_prompt}\n\n{user_prompt}"
            except Exception as e:
                logger.warning(f"Failed to get centralized prompt: {e}")

        # Fallback to inline prompt
        prompt = f"""You are a senior credit analyst. Analyze the collected data and provide a credit risk assessment.

## Company
Name: {company_name}

## Tool Selection Reasoning
{tool_reasoning}

## Collected Data
{tool_results_str}

## Your Task
Based on ALL the data collected, provide a comprehensive credit risk assessment.

## Response Format
Respond with a JSON object:
```json
{{
    "risk_level": "low" | "medium" | "high" | "critical",
    "credit_score": 0-100,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "risk_factors": ["list", "of", "risk", "factors"],
    "positive_factors": ["list", "of", "positive", "factors"],
    "recommendations": ["list", "of", "recommendations"],
    "data_quality_assessment": {{
        "data_completeness": 0.0-1.0,
        "sources_used": ["list of sources"],
        "missing_data": ["what data was not available"]
    }}
}}
```
"""
        return prompt

    def _call_llm(self, prompt: str, temperature: float = 0.1, call_type: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        Call LLM and return response with metrics.

        Uses LangChain ChatGroq with callbacks when available, falls back to legacy Groq SDK.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0-1)
            call_type: Type of call for cost tracking (e.g., "tool_selection", "synthesis")

        Returns:
            Tuple of (response_text, metrics_dict)
        """
        start_time = time.time()

        # Use LangChain ChatGroq with callbacks (preferred)
        if self._use_langchain:
            return self._call_llm_langchain(prompt, temperature, call_type, start_time)

        # Fallback to legacy Groq SDK
        return self._call_llm_legacy(prompt, temperature, start_time)

    def _call_llm_langchain(
        self,
        prompt: str,
        temperature: float,
        call_type: str,
        start_time: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM using LangChain ChatGroq with automatic token tracking."""
        # Setup callbacks for automatic token/cost tracking
        callbacks = []
        if COST_TRACKER_AVAILABLE:
            tracker = get_cost_tracker()
            callbacks.append(CostTrackerCallback(tracker=tracker, call_type=call_type))

        # Get ChatGroq instance
        llm = get_chat_groq(
            model=self.model,
            temperature=temperature,
            callbacks=callbacks,
        )

        if not llm:
            raise RuntimeError("Failed to create ChatGroq instance")

        # Make the call
        response = llm.invoke([HumanMessage(content=prompt)])
        execution_time = (time.time() - start_time) * 1000

        # Debug: Log the raw response
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"LLM raw response ({call_type}): {response_content[:500]}...")

        # Extract metrics from response metadata
        usage_metadata = getattr(response, 'usage_metadata', {}) or {}
        response_metadata = getattr(response, 'response_metadata', {}) or {}

        # Try to get token usage from various sources
        prompt_tokens = usage_metadata.get('input_tokens', 0)
        completion_tokens = usage_metadata.get('output_tokens', 0)
        total_tokens = usage_metadata.get('total_tokens', prompt_tokens + completion_tokens)

        # Fallback: check response_metadata
        if not total_tokens and response_metadata:
            token_usage = response_metadata.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', prompt_tokens + completion_tokens)

        metrics = {
            "model": self.model,
            "execution_time_ms": round(execution_time, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "llm_backend": "langchain_chatgroq",
        }

        logger.debug(f"LangChain LLM call: {self.model}, {total_tokens} tokens, {execution_time:.0f}ms")

        return response.content, metrics

    def _call_llm_legacy(
        self,
        prompt: str,
        temperature: float,
        start_time: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM using legacy Groq SDK (fallback)."""
        if not self._client:
            raise RuntimeError("Groq client not available")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000,
        )

        execution_time = (time.time() - start_time) * 1000

        # Extract metrics
        usage = response.usage
        metrics = {
            "model": self.model,
            "execution_time_ms": round(execution_time, 2),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "llm_backend": "groq_sdk",
        }

        return response.choices[0].message.content, metrics

    def _call_llm_for_prompt(
        self,
        prompt_id: str,
        prompt: str,
        call_type: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call LLM using the configuration specified for a prompt.

        Uses get_llm_for_prompt to get the LLM configured for this specific prompt.

        Args:
            prompt_id: The prompt identifier (e.g., "tool_selection", "credit_synthesis")
            prompt: The formatted prompt text
            call_type: Type of call for cost tracking

        Returns:
            Tuple of (response_text, metrics_dict)
        """
        if not LLM_FOR_PROMPT_AVAILABLE:
            # Fall back to the standard _call_llm method
            logger.debug(f"get_llm_for_prompt not available, using default LLM for {prompt_id}")
            return self._call_llm(prompt, call_type=call_type)

        start_time = time.time()

        # Get resolved config for logging
        resolved_config = get_llm_config_for_prompt(prompt_id)

        # Setup callbacks
        callbacks = []
        if COST_TRACKER_AVAILABLE:
            tracker = get_cost_tracker()
            callbacks.append(CostTrackerCallback(tracker=tracker, call_type=call_type))

        # Get LLM configured for this prompt
        llm = get_llm_for_prompt(prompt_id, callbacks=callbacks)

        if not llm:
            # Fallback to default
            logger.warning(f"Failed to get LLM for prompt '{prompt_id}', falling back to default")
            llm = get_chat_groq(model=self.model, temperature=0.1, callbacks=callbacks)

        if not llm:
            raise RuntimeError(f"Failed to create LLM instance for prompt '{prompt_id}'")

        response = llm.invoke([HumanMessage(content=prompt)])
        execution_time = (time.time() - start_time) * 1000

        # Extract metrics
        usage_metadata = getattr(response, 'usage_metadata', {}) or {}
        prompt_tokens = usage_metadata.get('input_tokens', 0)
        completion_tokens = usage_metadata.get('output_tokens', 0)
        total_tokens = usage_metadata.get('total_tokens', prompt_tokens + completion_tokens)

        metrics = {
            "model": resolved_config.get("model_id", self.model),
            "provider": resolved_config.get("provider", "groq"),
            "prompt_id": prompt_id,
            "execution_time_ms": round(execution_time, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "temperature": resolved_config.get("temperature", 0.1),
            "llm_backend": "per_prompt_config",
        }

        logger.debug(f"LLM call for prompt '{prompt_id}': {metrics['model']}, {total_tokens} tokens, {execution_time:.0f}ms")

        return response.content, metrics

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"error": str(e), "raw_response": response}

    def select_tools(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to select which tools to use for a company.

        Args:
            company_name: Name of the company
            context: Additional context (ticker, jurisdiction, etc.)
            run_id: Run ID for tracking

        Returns:
            Dict with tool selection and reasoning
        """
        run_id = run_id or str(uuid.uuid4())

        prompt = self._get_tool_selection_prompt(company_name, context)
        # Use per-prompt LLM configuration
        response, llm_metrics = self._call_llm_for_prompt(
            prompt_id="tool_selection",
            prompt=prompt,
            call_type="tool_selection",
        )

        # Parse with new OutputParser (with legacy fallback)
        if OUTPUT_PARSERS_AVAILABLE:
            parsed = parse_tool_selection(response, legacy_parser=self._parse_json_response)
            selection = result_to_dict(parsed)
        else:
            selection = self._parse_json_response(response)

        # Log the decision
        decision_log = {
            "run_id": run_id,
            "step": "tool_selection",
            "company_name": company_name,
            "context": context,
            "selection": selection,
            "llm_metrics": llm_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.decision_log.append(decision_log)

        # Persist to MongoDB for crash recovery
        self._persist_decision(decision_log)

        logger.info(f"Tool selection for {company_name}: {[t.get('name') for t in selection.get('tools_to_use', [])]}")

        return {
            "run_id": run_id,
            "selection": selection,
            "llm_metrics": llm_metrics,
        }

    def execute_selected_tools(
        self,
        company_name: str,
        tool_selection: Dict[str, Any],
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Execute the tools selected by the LLM.

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
                "reason": tool_call.get("reason", ""),
            })

        # Log execution
        execution_log = {
            "run_id": run_id,
            "step": "tool_execution",
            "company_name": company_name,
            "tools_executed": len(tools_to_use),
            "execution_metrics": execution_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.decision_log.append(execution_log)

        # Persist to MongoDB for crash recovery
        self._persist_decision(execution_log)

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
        Synthesize final credit assessment from tool results.

        Args:
            company_name: Company name
            tool_selection: Tool selection result
            tool_results: Results from tool execution
            run_id: Run ID for tracking

        Returns:
            Dict with final assessment
        """
        run_id = run_id or str(uuid.uuid4())

        prompt = self._get_synthesis_prompt(
            company_name,
            tool_results.get("results", {}),
            tool_selection.get("selection", {}),
        )

        # Use per-prompt LLM configuration
        response, llm_metrics = self._call_llm_for_prompt(
            prompt_id="credit_synthesis",
            prompt=prompt,
            call_type="synthesis",
        )

        # Parse with new OutputParser (with legacy fallback)
        if OUTPUT_PARSERS_AVAILABLE:
            parsed = parse_credit_assessment(response, legacy_parser=self._parse_json_response)
            assessment = result_to_dict(parsed)
        else:
            assessment = self._parse_json_response(response)

        # Log synthesis
        synthesis_log = {
            "run_id": run_id,
            "step": "synthesis",
            "company_name": company_name,
            "assessment": assessment,
            "llm_metrics": llm_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.decision_log.append(synthesis_log)

        logger.info(f"Assessment for {company_name}: {assessment.get('risk_level')} (score: {assessment.get('credit_score')})")

        return {
            "run_id": run_id,
            "assessment": assessment,
            "llm_metrics": llm_metrics,
        }

    def run_full_assessment(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run complete credit assessment workflow.

        This is the main entry point that:
        1. Selects appropriate tools
        2. Executes selected tools
        3. Synthesizes final assessment

        Args:
            company_name: Name of the company
            context: Additional context

        Returns:
            Complete assessment with all metrics
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting full assessment for: {company_name} (run_id: {run_id})")

        # Step 1: Tool Selection
        tool_selection = self.select_tools(company_name, context, run_id)

        # Step 2: Execute Tools
        tool_results = self.execute_selected_tools(company_name, tool_selection, run_id)

        # Step 3: Synthesize Assessment
        assessment = self.synthesize_assessment(
            company_name, tool_selection, tool_results, run_id
        )

        total_time = (time.time() - start_time) * 1000

        # Compile complete result
        result = {
            "run_id": run_id,
            "company_name": company_name,
            "context": context,
            "tool_selection": tool_selection,
            "tool_results": tool_results,
            "assessment": assessment,
            "total_execution_time_ms": round(total_time, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    def get_decision_log(self, run_id: str = None) -> List[Dict[str, Any]]:
        """Get decision log, optionally filtered by run_id."""
        if run_id:
            return [l for l in self.decision_log if l.get("run_id") == run_id]
        return self.decision_log

    def clear_log(self):
        """Clear decision log."""
        self.decision_log = []
