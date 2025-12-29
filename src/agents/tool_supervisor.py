"""Tool-Based Supervisor Agent - LLM selects which tools to use."""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed")

from tools import ToolExecutor, get_tool_executor


# Available models
MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
    "balanced": "gemma2-9b-it",
}


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
        self._client = None

        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._client = Groq(api_key=api_key)
                logger.info(f"ToolSupervisor initialized with model: {self.model}")

        # Logging
        self.decision_log: List[Dict[str, Any]] = []

    def _get_tool_selection_prompt(self, company_name: str, context: Dict[str, Any] = None) -> str:
        """Generate prompt for tool selection."""
        tool_specs = self.tool_executor.get_tool_specs_text()

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

    def _get_synthesis_prompt(
        self,
        company_name: str,
        tool_results: Dict[str, Any],
        tool_selection: Dict[str, Any]
    ) -> str:
        """Generate prompt for final credit synthesis."""
        prompt = f"""You are a senior credit analyst. Analyze the collected data and provide a credit risk assessment.

## Company
Name: {company_name}

## Tool Selection Reasoning
{json.dumps(tool_selection.get('company_analysis', {}), indent=2)}

## Collected Data
{json.dumps(tool_results, indent=2, default=str)}

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

    def _call_llm(self, prompt: str, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """
        Call LLM and return response with metrics.

        Returns:
            Tuple of (response_text, metrics_dict)
        """
        if not self._client:
            raise RuntimeError("Groq client not available")

        start_time = time.time()

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
        }

        return response.choices[0].message.content, metrics

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
        response, llm_metrics = self._call_llm(prompt)

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

        response, llm_metrics = self._call_llm(prompt)
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
