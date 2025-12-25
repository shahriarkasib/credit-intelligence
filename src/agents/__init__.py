"""Credit Intelligence Agents - LangGraph-based agentic workflow.

Enhanced with LLM-powered reasoning for AI-augmented credit analysis.
"""

from .supervisor import SupervisorAgent, CreditAssessment
from .search_agent import SearchAgent
from .api_agent import APIAgent
from .workflow import CreditIntelligenceWorkflow, run_credit_analysis

# Optional LLM analyst (requires groq)
try:
    from .llm_analyst import LLMAnalystAgent, LLMAnalysisResult, analyze_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMAnalystAgent = None
    LLMAnalysisResult = None
    analyze_with_llm = None

__all__ = [
    "SupervisorAgent",
    "CreditAssessment",
    "SearchAgent",
    "APIAgent",
    "CreditIntelligenceWorkflow",
    "run_credit_analysis",
    "LLMAnalystAgent",
    "LLMAnalysisResult",
    "analyze_with_llm",
    "LLM_AVAILABLE",
]
