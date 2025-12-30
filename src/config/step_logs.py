"""Step Logs - All prompts and step configurations for the Credit Intelligence workflow.

This file contains:
1. All LLM prompts used in each workflow step
2. Step metadata and descriptions
3. Logging utilities for tracking prompt usage
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# STEP DEFINITIONS
# =============================================================================

WORKFLOW_STEPS = {
    "parse_input": {
        "name": "Parse Input",
        "description": "Parse and enrich company input using LLM",
        "order": 1,
        "uses_llm": True,
    },
    "validate_company": {
        "name": "Validate Company",
        "description": "Validate company information and check for issues",
        "order": 2,
        "uses_llm": False,
    },
    "create_plan": {
        "name": "Create Plan",
        "description": "Create task plan for data collection",
        "order": 3,
        "uses_llm": False,
    },
    "fetch_api_data": {
        "name": "Fetch API Data",
        "description": "Fetch data from SEC, Finnhub, CourtListener APIs",
        "order": 4,
        "uses_llm": False,
    },
    "search_web": {
        "name": "Search Web",
        "description": "Search web for news and additional information",
        "order": 5,
        "uses_llm": False,
    },
    "synthesize": {
        "name": "Synthesize Assessment",
        "description": "Use LLM to analyze data and generate credit assessment",
        "order": 6,
        "uses_llm": True,
    },
    "save_to_database": {
        "name": "Save to Database",
        "description": "Save results to MongoDB and Google Sheets",
        "order": 7,
        "uses_llm": False,
    },
    "evaluate": {
        "name": "Evaluate",
        "description": "Evaluate workflow performance and consistency",
        "order": 8,
        "uses_llm": False,
    },
}


# =============================================================================
# LLM PROMPTS
# =============================================================================

PROMPTS = {
    # -------------------------------------------------------------------------
    # PARSE INPUT PROMPT (NEW - LLM-based company parsing)
    # -------------------------------------------------------------------------
    "parse_input": {
        "system": """You are a financial data specialist. Your task is to analyze company names and identify key information about them.

Given a company name, determine:
1. Whether it's likely a public or private company
2. The stock ticker symbol (if public)
3. The jurisdiction/country
4. The industry sector
5. Any known subsidiaries or parent companies

Be accurate and conservative - if unsure, indicate uncertainty.""",

        "user": """Analyze this company and provide structured information:

Company Name: {company_name}
Additional Context: {context}

Respond in JSON format:
```json
{{
    "company_name": "Official company name",
    "normalized_name": "lowercase normalized name",
    "is_public_company": true/false,
    "confidence": 0.0-1.0,
    "ticker": "TICKER or null",
    "exchange": "NYSE/NASDAQ/etc or null",
    "jurisdiction": "US/UK/DE/etc",
    "industry": "Technology/Finance/Healthcare/etc",
    "parent_company": "Parent company name or null",
    "reasoning": "Brief explanation of your determination"
}}
```""",
    },

    # -------------------------------------------------------------------------
    # TOOL SELECTION PROMPT
    # -------------------------------------------------------------------------
    "tool_selection": {
        "system": """You are a credit intelligence agent. Your task is to select appropriate data collection tools based on company characteristics.""",

        "user": """Select the appropriate tools to gather information about a company for credit risk assessment.

## Company to Analyze
Name: {company_name}
Additional Context: {context}

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

Only include tools that are truly needed. Be efficient.""",
    },

    # -------------------------------------------------------------------------
    # SYNTHESIS PROMPT (Credit Assessment)
    # -------------------------------------------------------------------------
    "synthesis": {
        "system": """You are a senior credit analyst with expertise in corporate credit risk assessment.
Analyze all available data comprehensively and provide accurate, well-reasoned assessments.""",

        "user": """Analyze the collected data and provide a comprehensive credit risk assessment.

## Company
Name: {company_name}

## Tool Selection Reasoning
{tool_selection_reasoning}

## Collected Data
{collected_data}

## Your Task
Based on ALL the data collected, provide a comprehensive credit risk assessment.

Consider:
- Financial health (revenue, cash flow, debt levels)
- Legal history (lawsuits, bankruptcies)
- Market position (stock performance, industry standing)
- News sentiment and recent developments
- Data quality and gaps

## Response Format
Respond with a JSON object:
```json
{{
    "risk_level": "low" | "medium" | "high" | "critical",
    "credit_score": 0-100,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment (2-3 paragraphs)",
    "risk_factors": ["list", "of", "identified", "risk", "factors"],
    "positive_factors": ["list", "of", "positive", "factors"],
    "recommendations": ["list", "of", "actionable", "recommendations"],
    "data_quality_assessment": {{
        "data_completeness": 0.0-1.0,
        "sources_used": ["list of sources"],
        "missing_data": ["what data was not available"],
        "confidence_impact": "How missing data affects confidence"
    }}
}}
```""",
    },

    # -------------------------------------------------------------------------
    # LLM ANALYST PROMPT (for multi-LLM consistency)
    # -------------------------------------------------------------------------
    "llm_analyst": {
        "system": """You are an expert credit analyst. Analyze company data and provide credit risk assessments.
Be consistent, thorough, and base your assessment on the evidence provided.""",

        "user": """Analyze this company for credit risk:

Company: {company_name}
Company Info: {company_info}

Financial Data:
{financial_data}

Legal Data:
{legal_data}

Market Data:
{market_data}

News/Search Data:
{news_data}

Provide your assessment in this JSON format:
```json
{{
    "risk_level": "low|medium|high|critical",
    "credit_score": 0-100,
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed analysis",
    "risk_factors": ["factor1", "factor2"],
    "positive_factors": ["factor1", "factor2"],
    "recommendations": ["rec1", "rec2"]
}}
```""",
    },
}


# =============================================================================
# STEP LOG DATACLASS
# =============================================================================

@dataclass
class StepLog:
    """Log entry for a workflow step."""
    step_name: str
    run_id: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # running, completed, failed

    # LLM details (if applicable)
    prompt_used: Optional[str] = None
    prompt_template: Optional[str] = None
    prompt_variables: Optional[Dict[str, Any]] = None
    llm_response: Optional[str] = None
    llm_model: Optional[str] = None
    tokens_used: Optional[int] = None

    # Input/Output
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def complete(self, output_data: Dict[str, Any] = None, error: str = None):
        """Mark step as completed."""
        self.completed_at = datetime.utcnow().isoformat()
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_ms = (end - start).total_seconds() * 1000
        self.status = "failed" if error else "completed"
        self.output_data = output_data
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "prompt_used": self.prompt_used,
            "prompt_template": self.prompt_template,
            "prompt_variables": self.prompt_variables,
            "llm_response": self.llm_response,
            "llm_model": self.llm_model,
            "tokens_used": self.tokens_used,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
        }


# =============================================================================
# STEP LOGGER CLASS
# =============================================================================

class StepLogger:
    """Logger for tracking workflow steps with prompts."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.logs: List[StepLog] = []
        self.current_step: Optional[StepLog] = None

    def start_step(
        self,
        step_name: str,
        input_data: Dict[str, Any] = None,
        prompt_template: str = None,
        prompt_variables: Dict[str, Any] = None,
    ) -> StepLog:
        """Start logging a new step."""
        log = StepLog(
            step_name=step_name,
            run_id=self.run_id,
            input_data=input_data,
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
        )

        # Build actual prompt if template provided
        if prompt_template and prompt_variables:
            try:
                log.prompt_used = prompt_template.format(**prompt_variables)
            except KeyError as e:
                logger.warning(f"Could not format prompt: {e}")

        self.current_step = log
        self.logs.append(log)

        logger.info(f"[{self.run_id[:8]}] Started step: {step_name}")
        return log

    def log_llm_call(
        self,
        model: str,
        response: str,
        tokens: int = None,
    ):
        """Log LLM call details for current step."""
        if self.current_step:
            self.current_step.llm_model = model
            self.current_step.llm_response = response
            self.current_step.tokens_used = tokens

    def complete_step(
        self,
        output_data: Dict[str, Any] = None,
        error: str = None,
    ):
        """Complete the current step."""
        if self.current_step:
            self.current_step.complete(output_data, error)
            logger.info(
                f"[{self.run_id[:8]}] Completed step: {self.current_step.step_name} "
                f"({self.current_step.duration_ms:.0f}ms)"
            )
            self.current_step = None

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Get all step logs as dictionaries."""
        return [log.to_dict() for log in self.logs]

    def get_prompts_used(self) -> List[Dict[str, str]]:
        """Get all prompts used in this run."""
        return [
            {
                "step": log.step_name,
                "prompt": log.prompt_used,
                "model": log.llm_model,
            }
            for log in self.logs
            if log.prompt_used
        ]

    def export_to_json(self, filepath: str):
        """Export all logs to JSON file."""
        import json
        data = {
            "run_id": self.run_id,
            "steps": self.get_all_logs(),
            "prompts": self.get_prompts_used(),
            "exported_at": datetime.utcnow().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported logs to: {filepath}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_prompt(step_name: str, prompt_type: str = "user") -> Optional[str]:
    """Get a prompt template by step name and type."""
    step_prompts = PROMPTS.get(step_name, {})
    return step_prompts.get(prompt_type)


def format_prompt(step_name: str, prompt_type: str = "user", **variables) -> str:
    """Format a prompt with variables."""
    template = get_prompt(step_name, prompt_type)
    if not template:
        raise ValueError(f"No {prompt_type} prompt found for step: {step_name}")
    return template.format(**variables)


def get_step_info(step_name: str) -> Dict[str, Any]:
    """Get step metadata."""
    return WORKFLOW_STEPS.get(step_name, {})


def list_all_prompts() -> Dict[str, Dict[str, str]]:
    """List all available prompts."""
    return PROMPTS.copy()
