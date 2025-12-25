"""Supervisor Agent - Orchestrates the credit intelligence workflow.

Enhanced with LLM-powered reasoning for AI-augmented credit analysis.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LLM analyst (optional)
try:
    from .llm_analyst import LLMAnalystAgent, LLMAnalysisResult
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.info("LLM analyst not available")

# Common US company tickers for demo
KNOWN_TICKERS = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "meta": "META",
    "meta platforms": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "jpmorgan": "JPM",
    "jpmorgan chase": "JPM",
    "walmart": "WMT",
    "walt disney": "DIS",
    "disney": "DIS",
    "coca-cola": "KO",
    "coca cola": "KO",
    "johnson & johnson": "JNJ",
    "visa": "V",
    "mastercard": "MA",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
}


@dataclass
class CreditAssessment:
    """Final credit assessment report."""
    company: str
    assessment_date: str

    # Overall scores
    overall_risk_level: str = "unknown"  # low, medium, high, critical
    credit_score_estimate: int = 0  # 0-100

    # Component assessments
    ability_to_pay: Dict[str, Any] = field(default_factory=dict)
    willingness_to_pay: Dict[str, Any] = field(default_factory=dict)
    fraud_risk: Dict[str, Any] = field(default_factory=dict)

    # Data summaries
    financial_summary: Dict[str, Any] = field(default_factory=dict)
    legal_summary: Dict[str, Any] = field(default_factory=dict)
    market_summary: Dict[str, Any] = field(default_factory=dict)
    news_summary: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)

    # Metadata
    data_sources_used: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0

    # LLM Analysis (new)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    analysis_method: str = "rule_based"  # "rule_based", "llm", "hybrid"
    confidence_score: float = 0.0
    llm_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": self.company,
            "assessment_date": self.assessment_date,
            "overall_risk_level": self.overall_risk_level,
            "credit_score_estimate": self.credit_score_estimate,
            "ability_to_pay": self.ability_to_pay,
            "willingness_to_pay": self.willingness_to_pay,
            "fraud_risk": self.fraud_risk,
            "financial_summary": self.financial_summary,
            "legal_summary": self.legal_summary,
            "market_summary": self.market_summary,
            "news_summary": self.news_summary,
            "recommendations": self.recommendations,
            "risk_factors": self.risk_factors,
            "positive_factors": self.positive_factors,
            "data_sources_used": self.data_sources_used,
            "data_quality_score": self.data_quality_score,
            "llm_analysis": self.llm_analysis,
            "analysis_method": self.analysis_method,
            "confidence_score": self.confidence_score,
            "llm_reasoning": self.llm_reasoning,
        }


class SupervisorAgent:
    """
    Supervisor Agent that orchestrates the credit intelligence workflow.

    Responsible for:
    - Receiving and parsing applicant details
    - Deciding which sub-agents to call
    - Synthesizing results into final credit assessment

    Enhanced with LLM-powered reasoning for hybrid analysis.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Analysis mode: "rule_based", "llm", or "hybrid"
        # Default to "llm" - AI makes the final decision
        self.analysis_mode = self.config.get("analysis_mode", "llm")

        # Initialize LLM analyst if available
        self.llm_analyst = None
        if LLM_AVAILABLE and self.analysis_mode in ["llm", "hybrid"]:
            groq_key = self.config.get("groq_api_key") or os.getenv("GROQ_API_KEY")
            if groq_key:
                self.llm_analyst = LLMAnalystAgent(api_key=groq_key)
                logger.info(f"LLM Analyst initialized (mode: {self.analysis_mode})")
            else:
                logger.info("No GROQ_API_KEY found, falling back to rule-based analysis")

    def parse_company_input(self, company_name: str, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse and enrich company input with known information.

        Args:
            company_name: Name of the company
            jurisdiction: Optional jurisdiction code

        Returns:
            Enriched company information
        """
        normalized_name = company_name.lower().strip()
        ticker = KNOWN_TICKERS.get(normalized_name)

        # Try partial match if exact match fails
        if not ticker:
            for known_name, known_ticker in KNOWN_TICKERS.items():
                if known_name in normalized_name or normalized_name in known_name:
                    ticker = known_ticker
                    break

        return {
            "company_name": company_name,
            "normalized_name": normalized_name,
            "ticker": ticker,
            "jurisdiction": jurisdiction or "US" if ticker else jurisdiction,
            "is_public_company": ticker is not None,
        }

    def create_task_plan(self, company_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a task plan based on company information.

        Args:
            company_info: Parsed company information

        Returns:
            List of tasks to execute
        """
        tasks = []

        # Always search for general information
        tasks.append({
            "agent": "search",
            "action": "search_company",
            "params": {"company_name": company_info["company_name"]},
            "priority": 1,
        })

        # If public company, fetch financial data
        if company_info.get("is_public_company") and company_info.get("ticker"):
            tasks.append({
                "agent": "api",
                "action": "fetch_sec_edgar",
                "params": {"ticker": company_info["ticker"]},
                "priority": 1,
            })
            tasks.append({
                "agent": "api",
                "action": "fetch_finnhub",
                "params": {"ticker": company_info["ticker"]},
                "priority": 1,
            })

        # Always fetch registry and compliance data
        tasks.append({
            "agent": "api",
            "action": "fetch_opencorporates",
            "params": {
                "company_name": company_info["company_name"],
                "jurisdiction": company_info.get("jurisdiction"),
            },
            "priority": 2,
        })

        tasks.append({
            "agent": "api",
            "action": "fetch_court_records",
            "params": {"company_name": company_info["company_name"]},
            "priority": 2,
        })

        tasks.append({
            "agent": "api",
            "action": "fetch_sanctions",
            "params": {"company_name": company_info["company_name"]},
            "priority": 1,  # High priority - fraud check
        })

        return sorted(tasks, key=lambda x: x["priority"])

    def synthesize_assessment(
        self,
        company_info: Dict[str, Any],
        api_data: Dict[str, Any],
        search_data: Dict[str, Any],
    ) -> CreditAssessment:
        """
        Synthesize all collected data into a final credit assessment.

        Supports three modes:
        - "rule_based": Traditional scoring with hardcoded thresholds
        - "llm": AI-powered analysis using Groq
        - "hybrid": Combines both, with confidence-weighted scoring

        Args:
            company_info: Parsed company information
            api_data: Data from API Agent
            search_data: Data from Search Agent

        Returns:
            CreditAssessment with full analysis
        """
        assessment = CreditAssessment(
            company=company_info["company_name"],
            assessment_date=datetime.utcnow().isoformat(),
        )

        # ----- STEP 1: Rule-Based Assessment (always run) -----
        assessment.ability_to_pay = self._assess_ability_to_pay(api_data)
        assessment.willingness_to_pay = self._assess_willingness_to_pay(api_data)
        assessment.fraud_risk = self._assess_fraud_risk(api_data, company_info)

        # Create summaries
        assessment.financial_summary = self._create_financial_summary(api_data)
        assessment.legal_summary = self._create_legal_summary(api_data)
        assessment.market_summary = self._create_market_summary(api_data)
        assessment.news_summary = self._create_news_summary(search_data)

        # Calculate rule-based risk scores
        rule_scores = self._calculate_risk_scores(assessment)
        rule_risk_level = rule_scores["overall_risk"]
        rule_credit_score = rule_scores["credit_score"]

        # Rule-based findings
        rule_risk_factors = self._identify_risk_factors(api_data, search_data)
        rule_positive_factors = self._identify_positive_factors(api_data, search_data)

        # ----- STEP 2: LLM Analysis (if enabled) -----
        llm_result = None
        if self.llm_analyst and self.llm_analyst.is_available():
            logger.info("Running LLM analysis...")
            try:
                llm_result = self.llm_analyst.analyze_company(
                    company_info=company_info,
                    api_data=api_data,
                    search_data=search_data,
                )
                if llm_result.success:
                    assessment.llm_analysis = llm_result.to_dict()
                    logger.info(f"LLM analysis complete: {llm_result.risk_level}")
                else:
                    logger.warning(f"LLM analysis failed: {llm_result.error}")
            except Exception as e:
                logger.error(f"LLM analysis error: {e}")

        # ----- STEP 3: Combine Results Based on Mode -----
        if self.analysis_mode == "llm" and llm_result and llm_result.success:
            # LLM-only mode
            assessment.overall_risk_level = llm_result.risk_level
            assessment.credit_score_estimate = llm_result.credit_score_estimate
            assessment.risk_factors = llm_result.risk_factors
            assessment.positive_factors = llm_result.positive_factors
            assessment.recommendations = llm_result.recommendations
            assessment.llm_reasoning = llm_result.reasoning
            assessment.confidence_score = llm_result.confidence
            assessment.analysis_method = "llm"

        elif self.analysis_mode == "hybrid" and llm_result and llm_result.success:
            # Hybrid mode: combine rule-based and LLM results
            hybrid_result = self._combine_hybrid_assessment(
                rule_risk_level=rule_risk_level,
                rule_credit_score=rule_credit_score,
                rule_risk_factors=rule_risk_factors,
                rule_positive_factors=rule_positive_factors,
                llm_result=llm_result,
            )
            assessment.overall_risk_level = hybrid_result["risk_level"]
            assessment.credit_score_estimate = hybrid_result["credit_score"]
            assessment.risk_factors = hybrid_result["risk_factors"]
            assessment.positive_factors = hybrid_result["positive_factors"]
            assessment.recommendations = hybrid_result["recommendations"]
            assessment.llm_reasoning = llm_result.reasoning
            assessment.confidence_score = hybrid_result["confidence"]
            assessment.analysis_method = "hybrid"

        else:
            # Rule-based fallback
            assessment.overall_risk_level = rule_risk_level
            assessment.credit_score_estimate = rule_credit_score
            assessment.risk_factors = rule_risk_factors
            assessment.positive_factors = rule_positive_factors
            assessment.recommendations = self._generate_recommendations(assessment)
            assessment.confidence_score = 0.7  # Default confidence for rule-based
            assessment.analysis_method = "rule_based"

        # Track data sources
        assessment.data_sources_used = self._get_data_sources_used(api_data, search_data)
        if llm_result and llm_result.success:
            assessment.data_sources_used.append(f"LLM ({llm_result.model_used})")
        assessment.data_quality_score = self._calculate_data_quality(api_data, search_data)

        return assessment

    def _combine_hybrid_assessment(
        self,
        rule_risk_level: str,
        rule_credit_score: int,
        rule_risk_factors: List[str],
        rule_positive_factors: List[str],
        llm_result: "LLMAnalysisResult",
    ) -> Dict[str, Any]:
        """
        Combine rule-based and LLM assessments into a hybrid result.

        Uses agreement-weighted scoring:
        - If both agree: high confidence, use agreed score
        - If disagree: moderate confidence, weighted average
        """
        RISK_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        RISK_NAMES = {0: "low", 1: "medium", 2: "high", 3: "critical"}

        rule_level_num = RISK_LEVELS.get(rule_risk_level, 1)
        llm_level_num = RISK_LEVELS.get(llm_result.risk_level, 1)

        # Check agreement
        levels_agree = abs(rule_level_num - llm_level_num) <= 1
        scores_agree = abs(rule_credit_score - llm_result.credit_score_estimate) <= 15

        if levels_agree and scores_agree:
            # Strong agreement - high confidence
            confidence = min(0.95, llm_result.confidence + 0.15)
            # Average the scores
            final_score = int((rule_credit_score + llm_result.credit_score_estimate) / 2)
            # Use the more conservative risk level
            final_level_num = max(rule_level_num, llm_level_num)
        else:
            # Disagreement - moderate confidence, investigate further
            confidence = min(0.7, llm_result.confidence)
            # Weight LLM slightly higher (it has more context)
            final_score = int(rule_credit_score * 0.4 + llm_result.credit_score_estimate * 0.6)
            # Use the more conservative (higher risk) assessment
            final_level_num = max(rule_level_num, llm_level_num)

        final_risk_level = RISK_NAMES.get(final_level_num, "medium")

        # Combine risk factors (deduplicate)
        combined_risks = list(set(rule_risk_factors + llm_result.risk_factors))
        combined_positives = list(set(rule_positive_factors + llm_result.positive_factors))

        # Use LLM recommendations (more context-aware)
        recommendations = llm_result.recommendations if llm_result.recommendations else self._generate_recommendations_from_score(final_score, final_risk_level)

        # Add disagreement note if applicable
        if not (levels_agree and scores_agree):
            recommendations.insert(0, f"Note: Rule-based ({rule_risk_level}) and AI ({llm_result.risk_level}) assessments differ - manual review recommended")

        return {
            "risk_level": final_risk_level,
            "credit_score": final_score,
            "confidence": confidence,
            "risk_factors": combined_risks[:10],
            "positive_factors": combined_positives[:10],
            "recommendations": recommendations[:6],
            "agreement": levels_agree and scores_agree,
        }

    def _generate_recommendations_from_score(self, score: int, risk_level: str) -> List[str]:
        """Generate recommendations based on score and risk level."""
        if risk_level == "critical":
            return ["DO NOT PROCEED - Critical risk factors identified", "Refer to compliance team"]
        elif risk_level == "high":
            return ["Proceed with extreme caution", "Require additional documentation", "Consider reduced credit limits"]
        elif risk_level == "medium":
            return ["Standard due diligence recommended", "Consider periodic reviews"]
        else:
            return ["Standard credit terms may apply", "May qualify for preferred terms" if score > 80 else "Monitor periodically"]

    def _assess_ability_to_pay(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess financial capacity to pay."""
        result = {
            "score": 50,  # Default middle score
            "level": "medium",
            "factors": [],
        }

        # Check SEC EDGAR financials
        sec_data = api_data.get("sec_edgar", {})
        financials = sec_data.get("financials", {})

        if financials:
            # Revenue check
            revenue = financials.get("revenue", {}).get("value", 0)
            if revenue and revenue > 1_000_000_000:  # > $1B
                result["score"] += 20
                result["factors"].append("Strong revenue (>$1B)")
            elif revenue and revenue > 100_000_000:  # > $100M
                result["score"] += 10
                result["factors"].append("Solid revenue (>$100M)")

            # Cash flow check
            cash_flow = financials.get("operating_cash_flow", {}).get("value", 0)
            if cash_flow and cash_flow > 0:
                result["score"] += 10
                result["factors"].append("Positive operating cash flow")
            elif cash_flow and cash_flow < 0:
                result["score"] -= 10
                result["factors"].append("Negative operating cash flow")

            # Debt ratio check
            debt_ratio = financials.get("debt_to_assets_ratio", {}).get("value", 0)
            if debt_ratio and debt_ratio < 0.3:
                result["score"] += 10
                result["factors"].append("Low debt ratio (<30%)")
            elif debt_ratio and debt_ratio > 0.7:
                result["score"] -= 15
                result["factors"].append("High debt ratio (>70%)")

        # Check Finnhub market data
        finnhub_data = api_data.get("finnhub", {})
        profile = finnhub_data.get("profile", {})
        metrics = finnhub_data.get("metrics", {})

        if profile:
            market_cap = profile.get("market_cap")
            if market_cap and market_cap > 10_000:  # > $10B (in millions)
                result["score"] += 10
                result["factors"].append("Large market cap company")

        if metrics:
            current_ratio = metrics.get("current_ratio")
            if current_ratio and current_ratio > 1.5:
                result["score"] += 5
                result["factors"].append("Good liquidity (current ratio >1.5)")
            elif current_ratio and current_ratio < 1.0:
                result["score"] -= 10
                result["factors"].append("Poor liquidity (current ratio <1)")

        # Normalize score
        result["score"] = max(0, min(100, result["score"]))

        # Set level
        if result["score"] >= 70:
            result["level"] = "high"
        elif result["score"] >= 40:
            result["level"] = "medium"
        else:
            result["level"] = "low"

        return result

    def _assess_willingness_to_pay(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess willingness to pay based on legal history."""
        result = {
            "score": 70,  # Default good score
            "level": "medium",
            "factors": [],
        }

        court_data = api_data.get("court_listener", {})
        risk_indicators = court_data.get("risk_indicators", {})

        if risk_indicators:
            risk_level = risk_indicators.get("risk_level", "low")
            has_bankruptcy = risk_indicators.get("has_bankruptcy", False)
            civil_count = risk_indicators.get("civil_case_count", 0)

            if has_bankruptcy:
                result["score"] -= 40
                result["factors"].append("Bankruptcy filing on record")

            if civil_count > 5:
                result["score"] -= 15
                result["factors"].append(f"Multiple civil cases ({civil_count})")
            elif civil_count > 0:
                result["score"] -= 5
                result["factors"].append(f"Some civil litigation ({civil_count} cases)")

            if risk_level == "high":
                result["score"] -= 10
            elif risk_level == "low" and civil_count == 0:
                result["score"] += 10
                result["factors"].append("Clean legal record")

        # Normalize score
        result["score"] = max(0, min(100, result["score"]))

        # Set level
        if result["score"] >= 70:
            result["level"] = "high"
        elif result["score"] >= 40:
            result["level"] = "medium"
        else:
            result["level"] = "low"

        return result

    def _assess_fraud_risk(self, api_data: Dict[str, Any], company_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess fraud/legitimacy risk."""
        result = {
            "score": 80,  # Default low risk (high score = good)
            "level": "low",
            "factors": [],
        }

        # Check OpenSanctions
        sanctions_data = api_data.get("opensanctions", {})
        is_sanctioned = sanctions_data.get("is_sanctioned", False)
        overall_risk = sanctions_data.get("overall_risk", {})

        if is_sanctioned:
            result["score"] = 0
            result["level"] = "critical"
            result["factors"].append("SANCTIONED ENTITY - DO NOT PROCEED")
            return result

        sanctions_level = overall_risk.get("level", "low")
        if sanctions_level == "high":
            result["score"] -= 30
            result["factors"].append("High sanctions risk")
        elif sanctions_level == "medium":
            result["score"] -= 15
            result["factors"].append("Medium sanctions risk")

        # Check OpenCorporates company status
        oc_data = api_data.get("opencorporates", {})
        if oc_data:
            status = oc_data.get("current_status", "").lower()
            if status and "active" in status:
                result["factors"].append("Company is active")
            elif status and ("dissolved" in status or "inactive" in status):
                result["score"] -= 40
                result["factors"].append(f"Company status: {status}")

            # Check for officers/directors
            officers = oc_data.get("officers", [])
            if officers:
                result["factors"].append(f"Has {len(officers)} registered officers")
            else:
                result["score"] -= 10
                result["factors"].append("No officers on record")

            # Check incorporation date
            inc_date = oc_data.get("incorporation_date")
            if inc_date:
                try:
                    from datetime import datetime
                    inc_year = int(inc_date[:4])
                    years_old = datetime.now().year - inc_year
                    if years_old < 2:
                        result["score"] -= 10
                        result["factors"].append(f"Recently incorporated ({years_old} years)")
                    elif years_old > 10:
                        result["score"] += 5
                        result["factors"].append(f"Established company ({years_old} years)")
                except (ValueError, TypeError):
                    pass

        # Normalize score
        result["score"] = max(0, min(100, result["score"]))

        # Set level
        if result["score"] >= 80:
            result["level"] = "low"
        elif result["score"] >= 50:
            result["level"] = "medium"
        elif result["score"] >= 20:
            result["level"] = "high"
        else:
            result["level"] = "critical"

        return result

    def _create_financial_summary(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create financial summary from SEC and Finnhub data."""
        summary = {}

        sec_data = api_data.get("sec_edgar", {})
        if sec_data:
            summary["company_name"] = sec_data.get("company_name")
            summary["industry"] = sec_data.get("sic_description")
            financials = sec_data.get("financials", {})
            if financials:
                for metric, data in financials.items():
                    if isinstance(data, dict) and "value" in data:
                        summary[metric] = data["value"]

        finnhub_data = api_data.get("finnhub", {})
        if finnhub_data:
            profile = finnhub_data.get("profile", {})
            if profile:
                summary["market_cap"] = profile.get("market_cap")
                summary["exchange"] = profile.get("exchange")
                summary["industry"] = profile.get("industry")

            quote = finnhub_data.get("quote", {})
            if quote:
                summary["stock_price"] = quote.get("current_price")
                summary["price_change"] = quote.get("percent_change")

        return summary

    def _create_legal_summary(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create legal summary from CourtListener data."""
        court_data = api_data.get("court_listener", {})
        return {
            "total_cases": court_data.get("total_dockets", 0),
            "bankruptcy_cases": len(court_data.get("bankruptcy_cases", [])),
            "civil_cases": len(court_data.get("civil_cases", [])),
            "risk_level": court_data.get("risk_indicators", {}).get("risk_level", "unknown"),
        }

    def _create_market_summary(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create market summary from Finnhub data."""
        finnhub_data = api_data.get("finnhub", {})
        summary = {}

        profile = finnhub_data.get("profile", {})
        quote = finnhub_data.get("quote", {})
        metrics = finnhub_data.get("metrics", {})

        if profile:
            summary["company"] = profile.get("name")
            summary["market_cap"] = profile.get("market_cap")
            summary["industry"] = profile.get("industry")

        if quote:
            summary["current_price"] = quote.get("current_price")
            summary["day_change_pct"] = quote.get("percent_change")
            summary["52_week_high"] = metrics.get("52_week_high") if metrics else None
            summary["52_week_low"] = metrics.get("52_week_low") if metrics else None

        if metrics:
            summary["pe_ratio"] = metrics.get("pe_ratio")
            summary["debt_equity"] = metrics.get("debt_equity")

        return summary

    def _create_news_summary(self, search_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create news summary from search data."""
        return {
            "article_count": len(search_data.get("news_articles", [])),
            "sentiment": search_data.get("sentiment", {}).get("sentiment", "unknown"),
            "sentiment_score": search_data.get("sentiment", {}).get("score", 0),
            "key_findings": search_data.get("key_findings", [])[:5],
        }

    def _calculate_risk_scores(self, assessment: CreditAssessment) -> Dict[str, Any]:
        """Calculate overall risk level and credit score."""
        # Weight the component scores
        ability_score = assessment.ability_to_pay.get("score", 50)
        willingness_score = assessment.willingness_to_pay.get("score", 50)
        fraud_score = assessment.fraud_risk.get("score", 50)

        # Fraud is critical - if fraud risk is high, everything is high risk
        if assessment.fraud_risk.get("level") == "critical":
            return {
                "overall_risk": "critical",
                "credit_score": 0,
            }

        # Weighted average (fraud has highest weight)
        credit_score = int(
            ability_score * 0.35 +
            willingness_score * 0.25 +
            fraud_score * 0.40
        )

        # Determine risk level
        if credit_score >= 75:
            risk_level = "low"
        elif credit_score >= 50:
            risk_level = "medium"
        elif credit_score >= 25:
            risk_level = "high"
        else:
            risk_level = "critical"

        return {
            "overall_risk": risk_level,
            "credit_score": credit_score,
        }

    def _identify_risk_factors(self, api_data: Dict[str, Any], search_data: Dict[str, Any]) -> List[str]:
        """Identify risk factors from all data."""
        risks = []

        # Sanctions risk
        sanctions = api_data.get("opensanctions", {})
        if sanctions.get("is_sanctioned"):
            risks.append("CRITICAL: Entity is on sanctions list")
        elif sanctions.get("overall_risk", {}).get("level") in ["high", "medium"]:
            risks.append(f"Sanctions risk level: {sanctions.get('overall_risk', {}).get('level')}")

        # Legal risk
        court = api_data.get("court_listener", {})
        if court.get("risk_indicators", {}).get("has_bankruptcy"):
            risks.append("Bankruptcy filing on record")
        if court.get("risk_indicators", {}).get("civil_case_count", 0) > 5:
            risks.append("Significant litigation history")

        # Financial risk
        sec = api_data.get("sec_edgar", {})
        financials = sec.get("financials", {})
        if financials:
            cash_flow = financials.get("operating_cash_flow", {}).get("value", 0)
            if cash_flow and cash_flow < 0:
                risks.append("Negative operating cash flow")
            debt_ratio = financials.get("debt_to_assets_ratio", {}).get("value", 0)
            if debt_ratio and debt_ratio > 0.7:
                risks.append("High debt ratio (>70%)")

        # News sentiment risk
        sentiment = search_data.get("sentiment", {})
        if sentiment.get("sentiment") == "negative":
            risks.append("Negative news sentiment")

        # Search findings risk
        findings = search_data.get("key_findings", [])
        for finding in findings:
            if any(kw in finding.lower() for kw in ["bankruptcy", "lawsuit", "investigation", "layoff"]):
                risks.append(finding)

        return list(set(risks))[:10]

    def _identify_positive_factors(self, api_data: Dict[str, Any], search_data: Dict[str, Any]) -> List[str]:
        """Identify positive factors from all data."""
        positives = []

        # Financial positives
        sec = api_data.get("sec_edgar", {})
        financials = sec.get("financials", {})
        if financials:
            revenue = financials.get("revenue", {}).get("value", 0)
            if revenue and revenue > 1_000_000_000:
                positives.append("Strong revenue (>$1B)")
            cash_flow = financials.get("operating_cash_flow", {}).get("value", 0)
            if cash_flow and cash_flow > 0:
                positives.append("Positive operating cash flow")
            debt_ratio = financials.get("debt_to_assets_ratio", {}).get("value", 0)
            if debt_ratio and debt_ratio < 0.3:
                positives.append("Low debt ratio (<30%)")

        # Market positives
        finnhub = api_data.get("finnhub", {})
        profile = finnhub.get("profile", {})
        if profile:
            market_cap = profile.get("market_cap")
            if market_cap and market_cap > 10000:
                positives.append("Large market cap company")

        # Registry positives
        oc = api_data.get("opencorporates", {})
        if oc:
            status = oc.get("current_status", "").lower()
            if "active" in status:
                positives.append("Active company status")
            officers = oc.get("officers", [])
            if officers and len(officers) >= 3:
                positives.append("Well-established leadership team")

        # Sanctions clean
        sanctions = api_data.get("opensanctions", {})
        if not sanctions.get("is_sanctioned") and sanctions.get("overall_risk", {}).get("level") == "low":
            positives.append("Clear sanctions check")

        # Legal clean
        court = api_data.get("court_listener", {})
        if court.get("risk_indicators", {}).get("risk_level") == "low":
            positives.append("Clean legal record")

        # News positives
        sentiment = search_data.get("sentiment", {})
        if sentiment.get("sentiment") == "positive":
            positives.append("Positive news sentiment")

        return list(set(positives))[:10]

    def _generate_recommendations(self, assessment: CreditAssessment) -> List[str]:
        """Generate recommendations based on assessment."""
        recommendations = []

        risk_level = assessment.overall_risk_level
        credit_score = assessment.credit_score_estimate

        if risk_level == "critical":
            recommendations.append("DO NOT PROCEED - Critical risk factors identified")
            recommendations.append("Refer to compliance/legal team immediately")
            return recommendations

        if risk_level == "high":
            recommendations.append("Proceed with extreme caution")
            recommendations.append("Require additional documentation and guarantees")
            recommendations.append("Consider reduced credit limits")

        elif risk_level == "medium":
            recommendations.append("Standard due diligence recommended")
            recommendations.append("Consider periodic reviews")
            if credit_score < 60:
                recommendations.append("May require additional collateral")

        else:  # low risk
            recommendations.append("Standard credit terms may apply")
            if credit_score > 80:
                recommendations.append("May qualify for preferred credit terms")

        # Specific recommendations based on data gaps
        if not assessment.financial_summary:
            recommendations.append("Request financial statements for private company")

        if assessment.willingness_to_pay.get("score", 100) < 50:
            recommendations.append("Verify payment history with references")

        return recommendations

    def _get_data_sources_used(self, api_data: Dict[str, Any], search_data: Dict[str, Any]) -> List[str]:
        """Get list of data sources that returned data."""
        sources = []

        if api_data.get("sec_edgar") and not api_data["sec_edgar"].get("error"):
            sources.append("SEC EDGAR")
        if api_data.get("finnhub") and not api_data["finnhub"].get("error"):
            sources.append("Finnhub")
        if api_data.get("opencorporates") and not api_data["opencorporates"].get("error"):
            sources.append("OpenCorporates")
        if api_data.get("court_listener") and not api_data["court_listener"].get("error"):
            sources.append("CourtListener")
        if api_data.get("opensanctions") and not api_data["opensanctions"].get("error"):
            sources.append("OpenSanctions")
        if search_data.get("web_results") or search_data.get("news_articles"):
            sources.append("Web Search")

        return sources

    def _calculate_data_quality(self, api_data: Dict[str, Any], search_data: Dict[str, Any]) -> float:
        """Calculate data quality score based on data completeness."""
        total_sources = 6
        available_sources = len(self._get_data_sources_used(api_data, search_data))

        # Base score on source availability
        base_score = available_sources / total_sources

        # Bonus for having financial data
        if api_data.get("sec_edgar", {}).get("financials"):
            base_score += 0.1

        # Bonus for having market data
        if api_data.get("finnhub", {}).get("profile"):
            base_score += 0.05

        return min(1.0, base_score)
