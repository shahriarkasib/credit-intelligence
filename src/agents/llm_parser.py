"""LLM-Based Company Parser - Uses LLM to parse and enrich company input.

Replaces the rule-based lookup with intelligent LLM-powered parsing that can:
- Identify public vs private companies
- Determine stock tickers
- Classify industry and jurisdiction
- Handle variations in company names

Now supports LangChain ChatGroq with automatic token tracking.
"""

import os
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Import LangChain ChatGroq (preferred)
try:
    from config.langchain_llm import get_chat_groq, is_langchain_groq_available
    from config.langchain_callbacks import CostTrackerCallback
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_GROQ_AVAILABLE = is_langchain_groq_available()
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False
    logger.warning("LangChain Groq not available, using legacy Groq client")

# Import cost tracker
try:
    from config.cost_tracker import get_cost_tracker, calculate_cost_for_tokens
    COST_TRACKER_AVAILABLE = True
except ImportError:
    COST_TRACKER_AVAILABLE = False

# Try to import Groq (legacy fallback)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed - LLM parsing not available")

# Feature flag: Use LangChain ChatGroq instead of raw Groq SDK
USE_LANGCHAIN_LLM = os.getenv("USE_LANGCHAIN_LLM", "true").lower() == "true"

# Import output parsers (Step 3)
try:
    from config.output_parsers import (
        parse_company,
        result_to_dict,
        is_parsers_available,
    )
    from config.output_schemas import ParsedCompany
    OUTPUT_PARSERS_AVAILABLE = is_parsers_available()
except ImportError:
    OUTPUT_PARSERS_AVAILABLE = False
    logger.warning("Output parsers not available, using legacy JSON parsing")

# Import prompts
try:
    from config.step_logs import PROMPTS, StepLogger
except ImportError:
    PROMPTS = {}
    StepLogger = None


# Fallback known tickers (used if LLM unavailable or for validation)
KNOWN_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "jpmorgan": "JPM",
    "walmart": "WMT",
    "disney": "DIS",
    "coca-cola": "KO",
    "visa": "V",
    "mastercard": "MA",
    "intel": "INTC",
    "amd": "AMD",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
}


class LLMCompanyParser:
    """
    LLM-powered company input parser.

    Uses Groq LLM to analyze company names and determine:
    - Whether it's a public or private company
    - Stock ticker symbol (if public)
    - Jurisdiction/country
    - Industry classification
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self._client = None  # Legacy Groq client
        self._use_langchain = USE_LANGCHAIN_LLM and LANGCHAIN_GROQ_AVAILABLE

        if self._use_langchain:
            logger.info(f"LLMCompanyParser using LangChain ChatGroq: {model}")
        elif GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._client = Groq(api_key=api_key)
                logger.info(f"LLMCompanyParser using legacy Groq: {model}")
            else:
                logger.warning("GROQ_API_KEY not set - using fallback parsing")

    def is_available(self) -> bool:
        """Check if LLM parsing is available."""
        return self._use_langchain or self._client is not None

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        call_type: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call LLM with system and user prompts, return response with metrics.

        Uses LangChain ChatGroq when available, falls back to legacy Groq SDK.
        """
        start_time = time.time()

        if self._use_langchain:
            return self._call_llm_langchain(
                system_prompt, user_prompt, temperature, max_tokens, call_type, start_time
            )

        return self._call_llm_legacy(
            system_prompt, user_prompt, temperature, max_tokens, start_time
        )

    def _call_llm_langchain(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        call_type: str,
        start_time: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM using LangChain ChatGroq."""
        callbacks = []
        if COST_TRACKER_AVAILABLE:
            tracker = get_cost_tracker()
            callbacks.append(CostTrackerCallback(tracker=tracker, call_type=call_type))

        llm = get_chat_groq(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        if not llm:
            raise RuntimeError("Failed to create ChatGroq instance")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        execution_time = (time.time() - start_time) * 1000

        # Extract token usage
        usage_metadata = getattr(response, 'usage_metadata', {}) or {}
        prompt_tokens = usage_metadata.get('input_tokens', 0)
        completion_tokens = usage_metadata.get('output_tokens', 0)
        total_tokens = usage_metadata.get('total_tokens', prompt_tokens + completion_tokens)

        metrics = {
            "model": self.model,
            "execution_time_ms": round(execution_time, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_used": user_prompt,
            "system_prompt": system_prompt,
            "llm_backend": "langchain_chatgroq",
        }

        return response.content, metrics

    def _call_llm_legacy(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        start_time: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM using legacy Groq SDK."""
        if not self._client:
            raise RuntimeError("Groq client not available")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        execution_time = (time.time() - start_time) * 1000

        usage = response.usage
        metrics = {
            "model": self.model,
            "execution_time_ms": round(execution_time, 2),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "prompt_used": user_prompt,
            "system_prompt": system_prompt,
            "llm_backend": "groq_sdk",
        }

        return response.choices[0].message.content, metrics

    def _get_system_prompt(self) -> str:
        """Get system prompt for company parsing."""
        if PROMPTS and "parse_input" in PROMPTS:
            return PROMPTS["parse_input"]["system"]

        return """You are a financial data specialist. Your task is to analyze company names and identify key information about them.

Given a company name, determine:
1. Whether it's likely a PUBLIC or PRIVATE company
2. The stock ticker symbol (if public)
3. The jurisdiction/country
4. The industry sector

PUBLIC COMPANY INDICATORS:
- Has a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- Listed on stock exchanges (NYSE, NASDAQ, LSE, etc.)
- Well-known publicly traded corporations
- Files with SEC (10-K, 10-Q reports)
- Examples: Apple Inc, Microsoft, Tesla, Amazon, JPMorgan Chase

PRIVATE COMPANY INDICATORS:
- No stock ticker or exchange listing
- Contains "LLC", "Private", "Holdings", "Family", "Partners" in name
- Small/medium businesses without public filings
- Subsidiaries not independently traded
- Unknown or fictional company names
- Examples: "Acme Private Holdings", "Johnson Family Enterprises", "Private Tech Solutions"

IMPORTANT: If you cannot find evidence of a stock ticker or public trading, classify as PRIVATE (is_public_company: false).
Be accurate and conservative - when in doubt, classify as private."""

    def _get_user_prompt(self, company_name: str, context: Dict[str, Any] = None) -> str:
        """Get user prompt for company parsing."""
        context_str = json.dumps(context or {})

        if PROMPTS and "parse_input" in PROMPTS:
            template = PROMPTS["parse_input"]["user"]
            try:
                return template.format(company_name=company_name, context=context_str)
            except KeyError:
                # Fallback if template has issues
                pass

        return f"""Analyze this company and provide structured information:

Company Name: {company_name}
Additional Context: {context_str}

IMPORTANT CLASSIFICATION RULES:
1. Set is_public_company=true ONLY if the company has a verifiable stock ticker (e.g., AAPL, MSFT)
2. Set is_public_company=false for:
   - Companies with no known stock ticker
   - Private companies, LLCs, partnerships
   - Unknown or fictional company names
   - Companies with "Private", "Holdings", "LLC", "Family" in name
3. If unsure about public status, default to is_public_company=false

Respond in JSON format:
```json
{{
    "company_name": "Official company name",
    "normalized_name": "lowercase normalized name",
    "is_public_company": true if has stock ticker else false,
    "confidence": 0.0-1.0,
    "ticker": "TICKER symbol if public else null",
    "exchange": "NYSE/NASDAQ/etc if public else null",
    "jurisdiction": "US/UK/DE/etc",
    "industry": "Technology/Finance/Healthcare/etc",
    "parent_company": "Parent company name or null",
    "reasoning": "Explain why company is public or private"
}}
```"""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
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

    def parse_company_llm(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
        step_logger: Optional["StepLogger"] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse company name using LLM.

        Args:
            company_name: Name of the company
            context: Additional context (jurisdiction, ticker hint, etc.)
            step_logger: Optional step logger for tracking prompts

        Returns:
            Tuple of (company_info dict, llm_metrics dict)
        """
        if not self.is_available():
            logger.info("LLM not available, using fallback parsing")
            return self._fallback_parse(company_name, context), {}

        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(company_name, context)

        # Log prompt if logger provided
        if step_logger:
            step_logger.start_step(
                "parse_input",
                input_data={"company_name": company_name, "context": context},
                prompt_template=user_prompt,
                prompt_variables={"company_name": company_name, "context": context},
            )

        try:
            # Call LLM using new unified helper
            response_text, llm_metrics = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=500,
                call_type="parse_company",
            )

            # Log LLM call
            if step_logger:
                step_logger.log_llm_call(
                    model=self.model,
                    response=response_text,
                    tokens=llm_metrics.get("total_tokens", 0),
                )

            # Parse response with new OutputParser (with fallback)
            if OUTPUT_PARSERS_AVAILABLE:
                parsed_result = parse_company(response_text, legacy_parser=self._parse_json_response)
                parsed = result_to_dict(parsed_result)
            else:
                parsed = self._parse_json_response(response_text)

            if "error" in parsed:
                logger.warning(f"Failed to parse LLM response, using fallback")
                return self._fallback_parse(company_name, context), llm_metrics

            # Validate and enrich result
            company_info = self._validate_and_enrich(parsed, company_name, context)

            if step_logger:
                step_logger.complete_step(output_data=company_info)

            logger.info(
                f"LLM parsed '{company_name}': "
                f"public={company_info.get('is_public_company')}, "
                f"ticker={company_info.get('ticker')}"
            )

            return company_info, llm_metrics

        except Exception as e:
            logger.error(f"LLM parsing error: {e}")
            if step_logger:
                step_logger.complete_step(error=str(e))
            return self._fallback_parse(company_name, context), {"error": str(e)}

    def _validate_and_enrich(
        self,
        parsed: Dict[str, Any],
        original_name: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Validate LLM response and enrich with defaults."""
        context = context or {}

        # Validate ticker against known list (if provided)
        ticker = parsed.get("ticker")
        if ticker:
            # Cross-check with known tickers
            normalized = original_name.lower().strip()
            known_ticker = None
            for name, t in KNOWN_TICKERS.items():
                if name in normalized:
                    known_ticker = t
                    break

            # If LLM ticker differs from known, prefer known
            if known_ticker and ticker != known_ticker:
                logger.info(f"Correcting ticker from {ticker} to {known_ticker}")
                ticker = known_ticker

        return {
            "company_name": parsed.get("company_name", original_name),
            "normalized_name": parsed.get("normalized_name", original_name.lower().strip()),
            "is_public_company": parsed.get("is_public_company", False),
            "confidence": parsed.get("confidence", 0.5),
            "ticker": ticker,
            "exchange": parsed.get("exchange"),
            "jurisdiction": parsed.get("jurisdiction", context.get("jurisdiction", "US")),
            "industry": parsed.get("industry"),
            "parent_company": parsed.get("parent_company"),
            "reasoning": parsed.get("reasoning", ""),
            "parsed_by": "llm",
        }

    def _fallback_parse(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Fallback rule-based parsing when LLM unavailable."""
        context = context or {}
        normalized_name = company_name.lower().strip()

        # Look up ticker
        ticker = None
        for known_name, known_ticker in KNOWN_TICKERS.items():
            if known_name in normalized_name or normalized_name in known_name:
                ticker = known_ticker
                break

        return {
            "company_name": company_name,
            "normalized_name": normalized_name,
            "is_public_company": ticker is not None,
            "confidence": 0.8 if ticker else 0.3,
            "ticker": ticker,
            "exchange": None,
            "jurisdiction": context.get("jurisdiction") or ("US" if ticker else None),
            "industry": None,
            "parent_company": None,
            "reasoning": "Parsed using rule-based fallback (LLM not available)",
            "parsed_by": "fallback",
        }


# Singleton instance
_parser: Optional[LLMCompanyParser] = None


def get_llm_parser() -> LLMCompanyParser:
    """Get the global LLMCompanyParser instance."""
    global _parser
    if _parser is None:
        _parser = LLMCompanyParser()
    return _parser


def parse_company_input(
    company_name: str,
    jurisdiction: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Parse company input - main entry point.

    Args:
        company_name: Name of the company
        jurisdiction: Optional jurisdiction hint
        use_llm: Whether to use LLM parsing (default True)

    Returns:
        Parsed company information
    """
    parser = get_llm_parser()
    context = {"jurisdiction": jurisdiction} if jurisdiction else None

    if use_llm and parser.is_available():
        company_info, _ = parser.parse_company_llm(company_name, context)
        return company_info
    else:
        return parser._fallback_parse(company_name, context)
