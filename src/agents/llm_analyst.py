"""LLM-Powered Analyst Agent for Credit Intelligence.

Uses Groq (free) for AI-powered credit analysis and reasoning.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis."""
    success: bool = False
    risk_level: str = "unknown"  # low, medium, high, critical
    confidence: float = 0.0
    reasoning: str = ""
    key_findings: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    credit_score_estimate: int = 50
    raw_response: str = ""
    model_used: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_findings": self.key_findings,
            "risk_factors": self.risk_factors,
            "positive_factors": self.positive_factors,
            "recommendations": self.recommendations,
            "credit_score_estimate": self.credit_score_estimate,
            "model_used": self.model_used,
            "error": self.error,
        }


class LLMAnalystAgent:
    """
    LLM-powered analyst agent for credit intelligence.

    Uses Groq's free API for AI-powered analysis:
    - Analyzes complex patterns in financial data
    - Provides nuanced risk assessments
    - Generates human-readable explanations
    - Validates rule-based assessments
    """

    # Groq models (all free)
    MODELS = {
        "primary": "llama-3.3-70b-versatile",  # Best quality
        "fast": "llama-3.1-8b-instant",  # Fastest
        "balanced": "mixtral-8x7b-32768",  # Good balance
    }

    CREDIT_ANALYSIS_PROMPT = """You are an expert credit analyst. Analyze the following company data and provide a credit risk assessment.

## Company Information
{company_info}

## Financial Data
{financial_data}

## Legal/Court Data
{legal_data}

## Market Data
{market_data}

## News/Sentiment Data
{news_data}

## Task
Based on the above data, provide a comprehensive credit risk assessment. Your response MUST be in the following JSON format:

{{
    "risk_level": "<low|medium|high|critical>",
    "credit_score": <0-100>,
    "confidence": <0.0-1.0>,
    "reasoning": "<2-3 sentence summary of your analysis>",
    "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
    "risk_factors": ["<risk 1>", "<risk 2>"],
    "positive_factors": ["<positive 1>", "<positive 2>"],
    "recommendations": ["<recommendation 1>", "<recommendation 2>"]
}}

Important guidelines:
- risk_level: "low" (score 75-100), "medium" (50-74), "high" (25-49), "critical" (0-24)
- If sanctions or bankruptcy data exists, weight heavily in assessment
- Consider data completeness in your confidence score
- Be specific and actionable in recommendations

Respond ONLY with the JSON object, no additional text."""

    VALIDATION_PROMPT = """You are a credit analyst reviewer. A rule-based system produced the following assessment:

## Rule-Based Assessment
Risk Level: {rule_risk_level}
Credit Score: {rule_credit_score}
Ability to Pay Score: {ability_score}
Willingness to Pay Score: {willingness_score}
Fraud Risk Score: {fraud_score}

## Raw Company Data
{company_data}

## Task
Review the rule-based assessment and provide your validation. Respond in JSON:

{{
    "agrees_with_assessment": <true|false>,
    "adjusted_risk_level": "<low|medium|high|critical>",
    "adjusted_score": <0-100>,
    "disagreement_reasons": ["<reason 1>", "<reason 2>"],
    "additional_considerations": ["<consideration 1>", "<consideration 2>"],
    "confidence": <0.0-1.0>
}}

Respond ONLY with the JSON object."""

    def __init__(self, api_key: Optional[str] = None, model: str = "primary"):
        """
        Initialize the LLM Analyst Agent.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model to use ("primary", "fast", "balanced")
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_id = self.MODELS.get(model, self.MODELS["primary"])
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"LLM Analyst initialized with model: {self.model_id}")
            except ImportError:
                logger.warning("groq package not installed. Run: pip install groq")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("No GROQ_API_KEY found. LLM analysis will be unavailable.")

    def is_available(self) -> bool:
        """Check if LLM analysis is available."""
        return self.client is not None

    def analyze_company(
        self,
        company_info: Dict[str, Any],
        api_data: Dict[str, Any],
        search_data: Dict[str, Any],
    ) -> LLMAnalysisResult:
        """
        Perform AI-powered credit analysis.

        Args:
            company_info: Parsed company information
            api_data: Data from API Agent
            search_data: Data from Search Agent

        Returns:
            LLMAnalysisResult with AI-powered assessment
        """
        if not self.is_available():
            return LLMAnalysisResult(
                success=False,
                error="Groq API not available. Set GROQ_API_KEY environment variable.",
            )

        # Prepare data summaries for the prompt
        financial_summary = self._summarize_financial_data(api_data)
        legal_summary = self._summarize_legal_data(api_data)
        market_summary = self._summarize_market_data(api_data)
        news_summary = self._summarize_news_data(search_data)

        # Build the prompt
        prompt = self.CREDIT_ANALYSIS_PROMPT.format(
            company_info=json.dumps(company_info, indent=2, default=str),
            financial_data=financial_summary,
            legal_data=legal_summary,
            market_data=market_summary,
            news_data=news_summary,
        )

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1024,
            )

            raw_response = response.choices[0].message.content

            # Parse JSON response
            result = self._parse_llm_response(raw_response)
            result.model_used = self.model_id
            result.raw_response = raw_response

            return result

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return LLMAnalysisResult(
                success=False,
                error=str(e),
                model_used=self.model_id,
            )

    def validate_assessment(
        self,
        rule_based_assessment: Dict[str, Any],
        api_data: Dict[str, Any],
        search_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a rule-based assessment using LLM reasoning.

        Args:
            rule_based_assessment: Assessment from rule-based supervisor
            api_data: Raw API data
            search_data: Raw search data

        Returns:
            Validation result with agreement/disagreement analysis
        """
        if not self.is_available():
            return {"error": "Groq API not available"}

        # Combine all data for context
        company_data = {
            "api_data": api_data,
            "search_data": search_data,
        }

        prompt = self.VALIDATION_PROMPT.format(
            rule_risk_level=rule_based_assessment.get("overall_risk_level", "unknown"),
            rule_credit_score=rule_based_assessment.get("credit_score_estimate", "N/A"),
            ability_score=rule_based_assessment.get("ability_to_pay", {}).get("score", "N/A"),
            willingness_score=rule_based_assessment.get("willingness_to_pay", {}).get("score", "N/A"),
            fraud_score=rule_based_assessment.get("fraud_risk", {}).get("score", "N/A"),
            company_data=json.dumps(company_data, indent=2, default=str)[:8000],  # Truncate
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )

            raw_response = response.choices[0].message.content

            # Try to parse JSON
            try:
                # Find JSON in response
                start_idx = raw_response.find("{")
                end_idx = raw_response.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = raw_response[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            return {"raw_response": raw_response, "parse_error": True}

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"error": str(e)}

    def explain_risk_factor(
        self,
        risk_factor: str,
        company_info: Dict[str, Any],
        context_data: Dict[str, Any],
    ) -> str:
        """
        Generate a detailed explanation for a specific risk factor.

        Args:
            risk_factor: The risk factor to explain
            company_info: Company information
            context_data: Supporting data

        Returns:
            Detailed explanation string
        """
        if not self.is_available():
            return f"Risk factor: {risk_factor} (LLM explanation unavailable)"

        prompt = f"""Explain the following credit risk factor for {company_info.get('company_name', 'this company')}:

Risk Factor: {risk_factor}

Context:
{json.dumps(context_data, indent=2, default=str)[:2000]}

Provide a 2-3 sentence explanation of:
1. Why this is a concern
2. Potential impact on creditworthiness
3. What to monitor going forward

Keep your response concise and professional."""

        try:
            response = self.client.chat.completions.create(
                model=self.MODELS["fast"],  # Use fast model for explanations
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return f"Risk factor: {risk_factor}"

    def _summarize_financial_data(self, api_data: Dict[str, Any]) -> str:
        """Summarize financial data for the prompt."""
        summary_parts = []

        # SEC EDGAR data
        sec_data = api_data.get("sec_edgar", {})
        if sec_data and not sec_data.get("error"):
            financials = sec_data.get("financials", {})
            if financials:
                summary_parts.append("SEC EDGAR Financials:")
                for metric, data in financials.items():
                    if isinstance(data, dict) and "value" in data:
                        value = data["value"]
                        if isinstance(value, (int, float)) and value > 1000000:
                            value = f"${value/1000000:.1f}M"
                        summary_parts.append(f"  - {metric}: {value}")
            else:
                summary_parts.append("SEC EDGAR: No detailed financials available")
        else:
            summary_parts.append("SEC EDGAR: Data not available (may be private company)")

        # Finnhub data
        finnhub = api_data.get("finnhub", {})
        if finnhub and not finnhub.get("error"):
            profile = finnhub.get("profile", {})
            if profile:
                summary_parts.append("\nMarket Profile (Finnhub):")
                if profile.get("market_cap"):
                    summary_parts.append(f"  - Market Cap: ${profile['market_cap']:.0f}M")
                if profile.get("exchange"):
                    summary_parts.append(f"  - Exchange: {profile['exchange']}")
                if profile.get("industry"):
                    summary_parts.append(f"  - Industry: {profile['industry']}")

            quote = finnhub.get("quote", {})
            if quote and quote.get("current_price"):
                summary_parts.append(f"  - Stock Price: ${quote['current_price']:.2f}")
                if quote.get("percent_change"):
                    summary_parts.append(f"  - Day Change: {quote['percent_change']:.2f}%")

        return "\n".join(summary_parts) if summary_parts else "No financial data available"

    def _summarize_legal_data(self, api_data: Dict[str, Any]) -> str:
        """Summarize legal/court data for the prompt."""
        summary_parts = []

        # CourtListener data
        court_data = api_data.get("court_listener", {})
        if court_data and not court_data.get("error"):
            risk = court_data.get("risk_indicators", {})
            summary_parts.append("Legal/Court Data:")
            summary_parts.append(f"  - Risk Level: {risk.get('risk_level', 'unknown')}")
            summary_parts.append(f"  - Bankruptcy on record: {risk.get('has_bankruptcy', False)}")
            summary_parts.append(f"  - Civil cases: {risk.get('civil_case_count', 0)}")

            # Recent cases
            bankruptcy = court_data.get("bankruptcy_cases", [])
            if bankruptcy:
                summary_parts.append(f"  - Recent bankruptcy cases: {len(bankruptcy)}")
        else:
            summary_parts.append("Legal Data: Not available")

        # Sanctions data
        sanctions = api_data.get("opensanctions", {})
        if sanctions and not sanctions.get("error"):
            summary_parts.append("\nSanctions Check:")
            summary_parts.append(f"  - Is Sanctioned: {sanctions.get('is_sanctioned', False)}")
            risk = sanctions.get("overall_risk", {})
            summary_parts.append(f"  - Sanctions Risk: {risk.get('level', 'unknown')}")

        return "\n".join(summary_parts) if summary_parts else "No legal data available"

    def _summarize_market_data(self, api_data: Dict[str, Any]) -> str:
        """Summarize market data for the prompt."""
        summary_parts = []

        finnhub = api_data.get("finnhub", {})
        if finnhub and not finnhub.get("error"):
            metrics = finnhub.get("metrics", {})
            if metrics:
                summary_parts.append("Financial Metrics:")
                if metrics.get("pe_ratio"):
                    summary_parts.append(f"  - P/E Ratio: {metrics['pe_ratio']:.2f}")
                if metrics.get("current_ratio"):
                    summary_parts.append(f"  - Current Ratio: {metrics['current_ratio']:.2f}")
                if metrics.get("debt_equity"):
                    summary_parts.append(f"  - Debt/Equity: {metrics['debt_equity']:.2f}")
                if metrics.get("52_week_high"):
                    summary_parts.append(f"  - 52W High: ${metrics['52_week_high']:.2f}")
                if metrics.get("52_week_low"):
                    summary_parts.append(f"  - 52W Low: ${metrics['52_week_low']:.2f}")

        # OpenCorporates data
        oc_data = api_data.get("opencorporates", {})
        if oc_data and not oc_data.get("error"):
            summary_parts.append("\nCompany Registry:")
            summary_parts.append(f"  - Status: {oc_data.get('current_status', 'unknown')}")
            if oc_data.get("incorporation_date"):
                summary_parts.append(f"  - Incorporated: {oc_data['incorporation_date']}")
            officers = oc_data.get("officers", [])
            summary_parts.append(f"  - Officers on record: {len(officers)}")

        return "\n".join(summary_parts) if summary_parts else "No market data available"

    def _summarize_news_data(self, search_data: Dict[str, Any]) -> str:
        """Summarize news/sentiment data for the prompt."""
        summary_parts = []

        sentiment = search_data.get("sentiment", {})
        if sentiment:
            summary_parts.append("News Sentiment:")
            summary_parts.append(f"  - Overall: {sentiment.get('sentiment', 'unknown')}")
            summary_parts.append(f"  - Score: {sentiment.get('score', 0)}")

        articles = search_data.get("news_articles", [])
        if articles:
            summary_parts.append(f"\nRecent News ({len(articles)} articles found):")
            for article in articles[:3]:
                title = article.get("title", "")[:80]
                summary_parts.append(f"  - {title}")

        findings = search_data.get("key_findings", [])
        if findings:
            summary_parts.append("\nKey Findings:")
            for finding in findings[:5]:
                summary_parts.append(f"  - {finding[:100]}")

        return "\n".join(summary_parts) if summary_parts else "No news data available"

    def _parse_llm_response(self, raw_response: str) -> LLMAnalysisResult:
        """Parse LLM JSON response into structured result."""
        result = LLMAnalysisResult()

        try:
            # Find JSON in response
            start_idx = raw_response.find("{")
            end_idx = raw_response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = raw_response[start_idx:end_idx]
                data = json.loads(json_str)

                result.success = True
                result.risk_level = data.get("risk_level", "unknown").lower()
                result.credit_score_estimate = int(data.get("credit_score", 50))
                result.confidence = float(data.get("confidence", 0.5))
                result.reasoning = data.get("reasoning", "")
                result.key_findings = data.get("key_findings", [])
                result.risk_factors = data.get("risk_factors", [])
                result.positive_factors = data.get("positive_factors", [])
                result.recommendations = data.get("recommendations", [])
            else:
                result.error = "No JSON found in response"
                result.reasoning = raw_response[:500]

        except json.JSONDecodeError as e:
            result.error = f"JSON parse error: {e}"
            result.reasoning = raw_response[:500]
        except Exception as e:
            result.error = f"Parse error: {e}"

        return result


# Convenience function
def analyze_with_llm(
    company_info: Dict[str, Any],
    api_data: Dict[str, Any],
    search_data: Dict[str, Any],
    api_key: Optional[str] = None,
) -> LLMAnalysisResult:
    """
    Convenience function to run LLM analysis.

    Args:
        company_info: Parsed company information
        api_data: Data from API Agent
        search_data: Data from Search Agent
        api_key: Optional Groq API key

    Returns:
        LLMAnalysisResult with AI-powered assessment
    """
    analyst = LLMAnalystAgent(api_key=api_key)
    return analyst.analyze_company(company_info, api_data, search_data)
