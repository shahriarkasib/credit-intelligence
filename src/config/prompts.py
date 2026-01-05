"""
Centralized Prompt Management for Credit Intelligence.

All prompts are stored here and can be:
- Retrieved via API
- Modified at runtime
- Reset to defaults
- Tested with sample data
- Loaded from external YAML config (if available)
"""

import json
import os
import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to load from external config
try:
    from config.external_config import (
        get_all_prompts_from_config,
        get_prompt_config as get_external_prompt,
        get_prompt_text_from_config,
        on_config_change,
    )
    EXTERNAL_CONFIG_AVAILABLE = True
    logger.info("External config available for prompts")
except ImportError:
    EXTERNAL_CONFIG_AVAILABLE = False
    logger.info("Using inline prompts (external config not available)")

# Default prompts - these are the system defaults
DEFAULT_PROMPTS: Dict[str, Dict[str, Any]] = {
    "company_parser": {
        "id": "company_parser",
        "name": "Company Parser",
        "description": "Parses company name to identify type, ticker, and industry",
        "category": "input",
        "variables": ["company_name"],
        "system_prompt": """You are a financial data specialist. Your task is to analyze company names and identify key information about them.

Given a company name, determine:
1. Whether it's likely a public or private company
2. The probable stock ticker (if public)
3. The industry sector
4. Any parent company or subsidiaries
5. The primary jurisdiction/country

Respond in JSON format with the following structure:
{
    "is_public_company": true/false,
    "ticker": "TICKER" or null,
    "industry": "Industry name",
    "sector": "Sector name",
    "jurisdiction": "Country/Region",
    "parent_company": "Parent name" or null,
    "confidence": 0.0-1.0
}""",
        "user_template": "Analyze this company: {company_name}",
    },

    "tool_selection": {
        "id": "tool_selection",
        "name": "Tool Selection",
        "description": "Selects appropriate data collection tools based on company type",
        "category": "planning",
        "variables": ["company_name", "context", "tool_specs"],
        "system_prompt": """You are a credit intelligence agent selecting tools for credit risk assessment.

Your task is to analyze the company and select the most appropriate data collection tools.

Consider:
- Is this a public or private company?
- What data sources are most relevant?
- What is the optimal order of tool execution?

Select tools wisely - don't select tools that won't return useful data.""",
        "user_template": """## Company: {company_name}

## Context
{context}

## Available Tools
{tool_specs}

## Instructions
Analyze the company and select the appropriate tools. Return your response as JSON with:
- company_analysis: Your analysis of the company (is_likely_public, reasoning)
- tools_to_use: List of tools with name, params, and reason
- execution_order_reasoning: Why you chose this order""",
    },

    "credit_synthesis": {
        "id": "credit_synthesis",
        "name": "Credit Synthesis",
        "description": "Synthesizes collected data into a final credit assessment",
        "category": "synthesis",
        "variables": ["company_name", "tool_reasoning", "tool_results"],
        "system_prompt": """You are a senior credit analyst. Analyze the collected data and provide a credit risk assessment.

Your assessment should be:
- Data-driven and evidence-based
- Balanced, considering both positive and negative factors
- Clear about confidence levels and data gaps
- Actionable with specific recommendations

CRITICAL: You MUST respond with a valid JSON object only. No markdown, no explanations, no code blocks - just the raw JSON.""",
        "user_template": """## Company
{company_name}

## Analysis Context
{tool_reasoning}

## Collected Data
{tool_results}

## Required JSON Response
You MUST respond with ONLY a JSON object in this exact format (no markdown, no code blocks):

{{
    "risk_level": "low" | "medium" | "high" | "critical",
    "credit_score": <number 0-100>,
    "confidence": <number 0.0-1.0>,
    "reasoning": "<detailed explanation of your assessment>",
    "risk_factors": ["<risk 1>", "<risk 2>", ...],
    "positive_factors": ["<positive 1>", "<positive 2>", ...],
    "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
    "data_quality_assessment": {{
        "completeness": <number 0.0-1.0>,
        "reliability": <number 0.0-1.0>,
        "gaps": ["<data gap 1>", "<data gap 2>", ...]
    }}
}}

IMPORTANT: Return ONLY the JSON object. Do not include any text before or after the JSON.""",
    },

    "credit_analysis": {
        "id": "credit_analysis",
        "name": "Credit Analysis",
        "description": "Full credit risk analysis from raw company data",
        "category": "analysis",
        "variables": ["company_name", "company_data"],
        "system_prompt": """You are an expert credit analyst. Analyze the following company data and provide a credit risk assessment.

Be consistent, thorough, and base your assessment on the evidence provided.
If data is limited, clearly state your confidence level and what additional information would be helpful.""",
        "user_template": """## Company Information
Company: {company_name}

## Available Data
{company_data}

## Required Output
Provide a structured credit assessment with:
1. Overall Risk Level (LOW/MODERATE/HIGH/CRITICAL)
2. Credit Score Estimate (300-850)
3. Confidence Score (0-100%)
4. Key Risk Factors
5. Positive Indicators
6. Recommendations
7. Data Gaps and Limitations""",
    },

    "validation": {
        "id": "validation",
        "name": "Assessment Validation",
        "description": "Reviews and validates rule-based assessments",
        "category": "validation",
        "variables": ["assessment", "company_data"],
        "system_prompt": """You are a credit analyst reviewer. A rule-based system produced an assessment that you need to validate.

Your task is to:
1. Review the assessment against the raw data
2. Identify any inconsistencies or errors
3. Suggest corrections if needed
4. Provide a confidence score for the assessment""",
        "user_template": """## Rule-Based Assessment
{assessment}

## Original Company Data
{company_data}

## Validation Tasks
1. Is the risk level appropriate given the data?
2. Is the credit score estimate reasonable?
3. Are all major risk factors captured?
4. Are there any data points that contradict the assessment?
5. What is your confidence in this assessment (0-100%)?

Provide your validation as JSON with:
- is_valid: true/false
- corrections: list of suggested changes
- confidence: your confidence score
- reasoning: explanation of your review""",
    },
}

# Runtime storage for modified prompts
_custom_prompts: Dict[str, Dict[str, Any]] = {}
_prompts_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'custom_prompts.json')


def _load_custom_prompts() -> None:
    """Load custom prompts from file if exists."""
    global _custom_prompts
    try:
        if os.path.exists(_prompts_file):
            with open(_prompts_file, 'r') as f:
                _custom_prompts = json.load(f)
    except Exception:
        _custom_prompts = {}


def _save_custom_prompts() -> None:
    """Save custom prompts to file."""
    try:
        os.makedirs(os.path.dirname(_prompts_file), exist_ok=True)
        with open(_prompts_file, 'w') as f:
            json.dump(_custom_prompts, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save custom prompts: {e}")


# Load on module import
_load_custom_prompts()


def get_all_prompts() -> Dict[str, Dict[str, Any]]:
    """
    Get all prompts (external config + custom overrides + defaults).

    Priority: custom overrides > external config > defaults

    Returns:
        Dict with prompt_id -> prompt config
    """
    result = {}

    # Start with defaults
    for prompt_id, default_config in DEFAULT_PROMPTS.items():
        result[prompt_id] = {**default_config, "is_custom": False, "source": "default"}

    # Override with external config if available
    if EXTERNAL_CONFIG_AVAILABLE:
        try:
            external_prompts = get_all_prompts_from_config()
            for prompt_id, prompt_config in external_prompts.items():
                if prompt_id in result:
                    # Merge with existing
                    result[prompt_id].update({
                        "system_prompt": prompt_config.system_prompt,
                        "user_template": prompt_config.user_template,
                        "name": prompt_config.name,
                        "description": prompt_config.description,
                        "category": prompt_config.category,
                        "variables": prompt_config.variables,
                        "source": "external_config",
                    })
                else:
                    # Add new prompt from external config
                    result[prompt_id] = {
                        "id": prompt_id,
                        "name": prompt_config.name,
                        "description": prompt_config.description,
                        "category": prompt_config.category,
                        "variables": prompt_config.variables,
                        "system_prompt": prompt_config.system_prompt,
                        "user_template": prompt_config.user_template,
                        "is_custom": False,
                        "source": "external_config",
                    }
        except Exception as e:
            logger.warning(f"Failed to load external prompts: {e}")

    # Override with custom prompts (runtime modifications)
    for prompt_id, custom_config in _custom_prompts.items():
        if prompt_id in result:
            result[prompt_id].update(custom_config)
            result[prompt_id]["is_custom"] = True
            result[prompt_id]["source"] = "custom"
        else:
            result[prompt_id] = {**custom_config, "is_custom": True, "source": "custom"}

    return result


def get_prompt(prompt_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific prompt by ID.

    Args:
        prompt_id: The prompt identifier

    Returns:
        Prompt config dict or None if not found
    """
    if prompt_id in _custom_prompts:
        return {**DEFAULT_PROMPTS.get(prompt_id, {}), **_custom_prompts[prompt_id], "is_custom": True}
    elif prompt_id in DEFAULT_PROMPTS:
        return {**DEFAULT_PROMPTS[prompt_id], "is_custom": False}
    return None


def get_prompt_text(prompt_id: str, **variables) -> tuple[str, str]:
    """
    Get formatted prompt text ready for LLM.

    Args:
        prompt_id: The prompt identifier
        **variables: Variables to substitute in template

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    prompt = get_prompt(prompt_id)
    if not prompt:
        raise ValueError(f"Unknown prompt: {prompt_id}")

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_template"].format(**variables)

    return system_prompt, user_prompt


def update_prompt(prompt_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a prompt with custom values.

    Args:
        prompt_id: The prompt identifier
        updates: Dict with fields to update (system_prompt, user_template, etc.)

    Returns:
        Updated prompt config
    """
    if prompt_id not in DEFAULT_PROMPTS:
        raise ValueError(f"Unknown prompt: {prompt_id}")

    # Store custom values
    if prompt_id not in _custom_prompts:
        _custom_prompts[prompt_id] = {}

    _custom_prompts[prompt_id].update(updates)
    _custom_prompts[prompt_id]["updated_at"] = datetime.utcnow().isoformat()

    _save_custom_prompts()

    return get_prompt(prompt_id)


def reset_prompt(prompt_id: str) -> Dict[str, Any]:
    """
    Reset a prompt to its default values.

    Args:
        prompt_id: The prompt identifier

    Returns:
        Default prompt config
    """
    if prompt_id in _custom_prompts:
        del _custom_prompts[prompt_id]
        _save_custom_prompts()

    return get_prompt(prompt_id)


def reset_all_prompts() -> Dict[str, Dict[str, Any]]:
    """
    Reset all prompts to defaults.

    Returns:
        All default prompts
    """
    global _custom_prompts
    _custom_prompts = {}
    _save_custom_prompts()

    return get_all_prompts()


def get_prompt_categories() -> Dict[str, list]:
    """
    Get prompts organized by category.

    Returns:
        Dict with category -> list of prompt configs
    """
    prompts = get_all_prompts()
    categories = {}

    for prompt_id, config in prompts.items():
        category = config.get("category", "other")
        if category not in categories:
            categories[category] = []
        categories[category].append(config)

    return categories
