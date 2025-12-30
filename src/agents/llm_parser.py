"""LLM-Based Company Parser - Uses LLM to parse and enrich company input.

Replaces the rule-based lookup with intelligent LLM-powered parsing that can:
- Identify public vs private companies
- Determine stock tickers
- Classify industry and jurisdiction
- Handle variations in company names
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq not installed - LLM parsing not available")

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
        self._client = None

        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._client = Groq(api_key=api_key)
                logger.info(f"LLMCompanyParser initialized with model: {model}")
            else:
                logger.warning("GROQ_API_KEY not set - using fallback parsing")

    def is_available(self) -> bool:
        """Check if LLM parsing is available."""
        return self._client is not None

    def _get_system_prompt(self) -> str:
        """Get system prompt for company parsing."""
        if PROMPTS and "parse_input" in PROMPTS:
            return PROMPTS["parse_input"]["system"]

        return """You are a financial data specialist. Your task is to analyze company names and identify key information about them.

Given a company name, determine:
1. Whether it's likely a public or private company
2. The stock ticker symbol (if public)
3. The jurisdiction/country
4. The industry sector

Be accurate and conservative - if unsure, indicate uncertainty."""

    def _get_user_prompt(self, company_name: str, context: Dict[str, Any] = None) -> str:
        """Get user prompt for company parsing."""
        context_str = json.dumps(context or {})

        if PROMPTS and "parse_input" in PROMPTS:
            template = PROMPTS["parse_input"]["user"]
            return template.format(company_name=company_name, context=context_str)

        return f"""Analyze this company and provide structured information:

Company Name: {company_name}
Additional Context: {context_str}

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
        if not self._client:
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
            import time
            start_time = time.time()

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract response and metrics
            response_text = response.choices[0].message.content
            usage = response.usage

            llm_metrics = {
                "model": self.model,
                "execution_time_ms": round(execution_time, 2),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "prompt_used": user_prompt,
                "system_prompt": system_prompt,
            }

            # Log LLM call
            if step_logger:
                step_logger.log_llm_call(
                    model=self.model,
                    response=response_text,
                    tokens=usage.total_tokens if usage else 0,
                )

            # Parse response
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
