"""Consistency-as-Correctness Evaluation Framework."""

from .execution_wrapper import ExecutionWrapper, ModelConfig
from .consistency_scorer import ConsistencyScorer
from .correctness_scorer import CorrectnessScorer
from .analyzer import CorrelationAnalyzer
from .tool_selection_evaluator import ToolSelectionEvaluator, ToolSelectionResult
from .workflow_evaluator import WorkflowEvaluator, WorkflowEvaluationResult
from .langsmith_evaluator import (
    CreditEvaluationResult,
    CreditIntelligenceEvaluator,
    get_credit_evaluator,
)

# Task 4: Agent Efficiency Evaluator
from .agent_efficiency_evaluator import (
    AgentEfficiencyEvaluator,
    AgentEfficiencyMetrics,
    evaluate_agent_run,
    get_agent_evaluator,
)

# Task 21: LLM-as-a-Judge Evaluator
from .llm_judge_evaluator import (
    LLMJudgeEvaluator,
    LLMJudgeResult,
    evaluate_with_llm_judge,
    get_llm_judge,
)

# Task 3: Evaluation Brain (orchestration layer)
from .evaluation_brain import (
    EvaluationBrain,
    ComprehensiveEvaluationResult,
    evaluate_comprehensive,
    get_evaluation_brain,
)

__all__ = [
    "ExecutionWrapper",
    "ModelConfig",
    "ConsistencyScorer",
    "CorrectnessScorer",
    "CorrelationAnalyzer",
    "ToolSelectionEvaluator",
    "ToolSelectionResult",
    "WorkflowEvaluator",
    "WorkflowEvaluationResult",
    # LangSmith Evaluation
    "CreditEvaluationResult",
    "CreditIntelligenceEvaluator",
    "get_credit_evaluator",
    # Task 4: Agent Efficiency
    "AgentEfficiencyEvaluator",
    "AgentEfficiencyMetrics",
    "evaluate_agent_run",
    "get_agent_evaluator",
    # Task 21: LLM Judge
    "LLMJudgeEvaluator",
    "LLMJudgeResult",
    "evaluate_with_llm_judge",
    "get_llm_judge",
    # Task 3: Evaluation Brain
    "EvaluationBrain",
    "ComprehensiveEvaluationResult",
    "evaluate_comprehensive",
    "get_evaluation_brain",
]
