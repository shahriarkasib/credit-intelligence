"""Credit Intelligence Agents - LangGraph-based agentic workflow.

Enhanced with LLM-powered reasoning for AI-augmented credit analysis.
"""

from .supervisor import SupervisorAgent, CreditAssessment
from .search_agent import SearchAgent
from .api_agent import APIAgent
from .workflow import CreditIntelligenceWorkflow, run_credit_analysis
from .tool_supervisor import ToolSupervisor

# Optional LLM analyst (requires groq)
try:
    from .llm_analyst import LLMAnalystAgent, LLMAnalysisResult, analyze_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMAnalystAgent = None
    LLMAnalysisResult = None
    analyze_with_llm = None

# LangChain-native supervisor (Step 4)
try:
    from .langchain_supervisor import (
        LangChainSupervisor,
        get_supervisor,
        is_langchain_supervisor_available,
        USE_LANGCHAIN_SUPERVISOR,
    )
    LANGCHAIN_SUPERVISOR_AVAILABLE = is_langchain_supervisor_available()
except ImportError:
    LANGCHAIN_SUPERVISOR_AVAILABLE = False
    LangChainSupervisor = None
    get_supervisor = None
    is_langchain_supervisor_available = None
    USE_LANGCHAIN_SUPERVISOR = False

__all__ = [
    "SupervisorAgent",
    "CreditAssessment",
    "SearchAgent",
    "APIAgent",
    "CreditIntelligenceWorkflow",
    "run_credit_analysis",
    "ToolSupervisor",
    "LLMAnalystAgent",
    "LLMAnalysisResult",
    "analyze_with_llm",
    "LLM_AVAILABLE",
    # LangChain supervisor
    "LangChainSupervisor",
    "get_supervisor",
    "is_langchain_supervisor_available",
    "LANGCHAIN_SUPERVISOR_AVAILABLE",
    "USE_LANGCHAIN_SUPERVISOR",
]
