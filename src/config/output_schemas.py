"""Pydantic Output Schemas for LLM Responses.

Provides validated, type-safe schemas for:
- Tool selection decisions
- Credit risk assessments
- Company parsing results

These schemas enable PydanticOutputParser for reliable JSON parsing.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Tool Selection Schemas (for ToolSupervisor)
# =============================================================================

class CompanyAnalysis(BaseModel):
    """Analysis of whether a company is public or private."""
    is_likely_public: bool = Field(
        description="Whether the company is likely publicly traded"
    )
    reasoning: str = Field(
        description="Brief explanation of the determination"
    )


class ToolCall(BaseModel):
    """A single tool call decision."""
    name: str = Field(
        description="Name of the tool to call (e.g., fetch_sec_data, fetch_market_data)"
    )
    params: dict = Field(
        default_factory=dict,
        description="Parameters to pass to the tool"
    )
    reason: str = Field(
        description="Why this tool is needed for the analysis"
    )


class ToolSelection(BaseModel):
    """Complete tool selection response from LLM."""
    company_analysis: CompanyAnalysis = Field(
        description="Analysis of the company type"
    )
    tools_to_use: List[ToolCall] = Field(
        description="List of tools to execute in order"
    )
    execution_order_reasoning: str = Field(
        description="Why this order of tool execution makes sense"
    )


# =============================================================================
# Credit Assessment Schemas (for LLMAnalyst and Synthesis)
# =============================================================================

class DataQualityAssessment(BaseModel):
    """Assessment of data quality and completeness."""
    data_completeness: float = Field(
        ge=0.0, le=1.0,
        description="Score from 0-1 indicating how complete the data is"
    )
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of data sources that provided information"
    )
    missing_data: List[str] = Field(
        default_factory=list,
        description="List of data that was not available"
    )


class CreditAssessment(BaseModel):
    """Complete credit risk assessment response."""
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="Overall risk level: low (75-100), medium (50-74), high (25-49), critical (0-24)"
    )
    credit_score: int = Field(
        ge=0, le=100,
        description="Credit score from 0-100"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the assessment from 0-1"
    )
    reasoning: str = Field(
        description="2-3 sentence summary of the analysis"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from the analysis"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
    positive_factors: List[str] = Field(
        default_factory=list,
        description="Positive factors supporting creditworthiness"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    data_quality_assessment: Optional[DataQualityAssessment] = Field(
        default=None,
        description="Assessment of the data quality"
    )

    @field_validator('risk_level', mode='before')
    @classmethod
    def normalize_risk_level(cls, v):
        """Normalize risk level to lowercase."""
        if isinstance(v, str):
            return v.lower().strip()
        return v


# =============================================================================
# Company Parsing Schemas (for LLMParser)
# =============================================================================

class ParsedCompany(BaseModel):
    """Parsed company information from LLM."""
    company_name: str = Field(
        description="Official company name"
    )
    normalized_name: str = Field(
        description="Lowercase normalized name for matching"
    )
    is_public_company: bool = Field(
        description="Whether the company is publicly traded"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the parsing from 0-1"
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Stock ticker symbol if public"
    )
    exchange: Optional[str] = Field(
        default=None,
        description="Stock exchange (NYSE, NASDAQ, etc.)"
    )
    jurisdiction: str = Field(
        default="US",
        description="Country/jurisdiction code"
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry classification"
    )
    parent_company: Optional[str] = Field(
        default=None,
        description="Parent company name if subsidiary"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the determination"
    )


# =============================================================================
# Validation Schemas (for assessment validation)
# =============================================================================

class ValidationResult(BaseModel):
    """Result of validating a rule-based assessment."""
    agrees_with_assessment: bool = Field(
        description="Whether LLM agrees with the rule-based assessment"
    )
    adjusted_risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="LLM's adjusted risk level"
    )
    adjusted_score: int = Field(
        ge=0, le=100,
        description="LLM's adjusted credit score"
    )
    disagreement_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for disagreement if any"
    )
    additional_considerations: List[str] = Field(
        default_factory=list,
        description="Additional factors to consider"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the validation"
    )

    @field_validator('adjusted_risk_level', mode='before')
    @classmethod
    def normalize_risk_level(cls, v):
        """Normalize risk level to lowercase."""
        if isinstance(v, str):
            return v.lower().strip()
        return v


# =============================================================================
# Export all schemas
# =============================================================================

__all__ = [
    # Tool Selection
    "CompanyAnalysis",
    "ToolCall",
    "ToolSelection",
    # Credit Assessment
    "DataQualityAssessment",
    "CreditAssessment",
    # Company Parsing
    "ParsedCompany",
    # Validation
    "ValidationResult",
]
