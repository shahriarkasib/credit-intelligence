"""Credit Intelligence Workflow - LangGraph-based orchestration."""

import logging
import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from dataclasses import asdict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .supervisor import SupervisorAgent, CreditAssessment
from .search_agent import SearchAgent
from .api_agent import APIAgent

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State for the credit intelligence workflow."""
    # Input
    company_name: str
    jurisdiction: Optional[str]
    ticker: Optional[str]

    # Parsed company info
    company_info: Dict[str, Any]

    # Task plan
    task_plan: List[Dict[str, Any]]

    # Agent results
    api_data: Dict[str, Any]
    search_data: Dict[str, Any]

    # Final output
    assessment: Optional[Dict[str, Any]]

    # Metadata
    errors: List[str]
    status: str


class CreditIntelligenceWorkflow:
    """
    LangGraph-based workflow for credit intelligence.

    Orchestrates:
    1. Supervisor: Parse input, create task plan
    2. API Agent: Fetch structured data
    3. Search Agent: Gather web information
    4. Supervisor: Synthesize final assessment
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.supervisor = SupervisorAgent(config)
        self.api_agent = APIAgent(config)
        self.search_agent = SearchAgent(config)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("parse_input", self._parse_input_node)
        workflow.add_node("create_plan", self._create_plan_node)
        workflow.add_node("fetch_api_data", self._fetch_api_data_node)
        workflow.add_node("search_web", self._search_web_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Add edges
        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "create_plan")
        workflow.add_edge("create_plan", "fetch_api_data")
        workflow.add_edge("fetch_api_data", "search_web")
        workflow.add_edge("search_web", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _parse_input_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Parse and enrich company input."""
        logger.info(f"Parsing input for company: {state['company_name']}")

        company_info = self.supervisor.parse_company_input(
            state["company_name"],
            state.get("jurisdiction"),
        )

        return {
            "company_info": company_info,
            "ticker": company_info.get("ticker"),
            "status": "input_parsed",
        }

    def _create_plan_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Create task plan based on company info."""
        logger.info("Creating task plan")

        task_plan = self.supervisor.create_task_plan(state["company_info"])

        return {
            "task_plan": task_plan,
            "status": "plan_created",
        }

    def _fetch_api_data_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Fetch data from external APIs."""
        logger.info("Fetching API data")

        company_info = state["company_info"]

        try:
            api_result = self.api_agent.fetch_all_data(
                company_name=company_info["company_name"],
                ticker=company_info.get("ticker"),
                jurisdiction=company_info.get("jurisdiction"),
                parallel=True,
            )

            return {
                "api_data": api_result.to_dict(),
                "errors": state.get("errors", []) + api_result.errors,
                "status": "api_data_fetched",
            }

        except Exception as e:
            logger.error(f"API data fetch error: {e}")
            return {
                "api_data": {},
                "errors": state.get("errors", []) + [f"API fetch error: {str(e)}"],
                "status": "api_data_error",
            }

    def _search_web_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Search for web information."""
        logger.info("Searching web for company information")

        company_name = state["company_info"]["company_name"]

        try:
            search_result = self.search_agent.search_company(company_name)

            return {
                "search_data": search_result.to_dict(),
                "errors": state.get("errors", []) + search_result.errors,
                "status": "search_complete",
            }

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "search_data": {},
                "errors": state.get("errors", []) + [f"Search error: {str(e)}"],
                "status": "search_error",
            }

    def _synthesize_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Synthesize final credit assessment."""
        logger.info("Synthesizing credit assessment")

        try:
            assessment = self.supervisor.synthesize_assessment(
                company_info=state["company_info"],
                api_data=state.get("api_data", {}),
                search_data=state.get("search_data", {}),
            )

            return {
                "assessment": assessment.to_dict(),
                "status": "complete",
            }

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                "assessment": None,
                "errors": state.get("errors", []) + [f"Synthesis error: {str(e)}"],
                "status": "synthesis_error",
            }

    def run(
        self,
        company_name: str,
        jurisdiction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the credit intelligence workflow.

        Args:
            company_name: Name of the company to analyze
            jurisdiction: Optional jurisdiction code

        Returns:
            Complete workflow result including assessment
        """
        initial_state: WorkflowState = {
            "company_name": company_name,
            "jurisdiction": jurisdiction,
            "ticker": None,
            "company_info": {},
            "task_plan": [],
            "api_data": {},
            "search_data": {},
            "assessment": None,
            "errors": [],
            "status": "started",
        }

        logger.info(f"Starting credit intelligence workflow for: {company_name}")

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            return dict(final_state)

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {
                **initial_state,
                "errors": [str(e)],
                "status": "error",
            }

    def get_assessment_report(self, result: Dict[str, Any]) -> str:
        """Generate a human-readable assessment report."""
        assessment = result.get("assessment", {})

        if not assessment:
            return f"Error: Could not generate assessment for {result.get('company_name')}"

        # Get analysis method info
        analysis_method = assessment.get("analysis_method", "rule_based")
        confidence = assessment.get("confidence_score", 0)
        method_display = {
            "rule_based": "Rule-Based",
            "llm": "AI-Powered (LLM)",
            "hybrid": "Hybrid (Rules + AI)",
        }.get(analysis_method, analysis_method)

        lines = [
            "=" * 60,
            f"CREDIT INTELLIGENCE REPORT",
            f"Company: {assessment.get('company', 'Unknown')}",
            f"Date: {assessment.get('assessment_date', 'Unknown')}",
            f"Analysis: {method_display} | Confidence: {confidence:.0%}",
            "=" * 60,
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"Risk Level: {assessment.get('overall_risk_level', 'Unknown').upper()}",
            f"Credit Score: {assessment.get('credit_score_estimate', 'N/A')}/100",
            "",
            "COMPONENT SCORES",
            "-" * 40,
        ]

        # Ability to Pay
        ability = assessment.get("ability_to_pay", {})
        lines.append(f"Ability to Pay: {ability.get('level', 'Unknown')} ({ability.get('score', 'N/A')}/100)")
        for factor in ability.get("factors", [])[:3]:
            lines.append(f"  - {factor}")

        # Willingness to Pay
        willingness = assessment.get("willingness_to_pay", {})
        lines.append(f"Willingness to Pay: {willingness.get('level', 'Unknown')} ({willingness.get('score', 'N/A')}/100)")
        for factor in willingness.get("factors", [])[:3]:
            lines.append(f"  - {factor}")

        # Fraud Risk
        fraud = assessment.get("fraud_risk", {})
        lines.append(f"Fraud Risk: {fraud.get('level', 'Unknown')} ({fraud.get('score', 'N/A')}/100)")
        for factor in fraud.get("factors", [])[:3]:
            lines.append(f"  - {factor}")

        # Risk Factors
        lines.extend(["", "RISK FACTORS", "-" * 40])
        risk_factors = assessment.get("risk_factors", [])
        if risk_factors:
            for rf in risk_factors[:5]:
                lines.append(f"  ! {rf}")
        else:
            lines.append("  No significant risk factors identified")

        # Positive Factors
        lines.extend(["", "POSITIVE FACTORS", "-" * 40])
        positive_factors = assessment.get("positive_factors", [])
        if positive_factors:
            for pf in positive_factors[:5]:
                lines.append(f"  + {pf}")
        else:
            lines.append("  No significant positive factors identified")

        # Recommendations
        lines.extend(["", "RECOMMENDATIONS", "-" * 40])
        recommendations = assessment.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")

        # AI Analysis (if available)
        llm_reasoning = assessment.get("llm_reasoning", "")
        if llm_reasoning:
            lines.extend(["", "AI ANALYSIS", "-" * 40])
            # Word wrap the reasoning
            words = llm_reasoning.split()
            current_line = "  "
            for word in words:
                if len(current_line) + len(word) + 1 > 58:
                    lines.append(current_line)
                    current_line = "  " + word
                else:
                    current_line += " " + word if current_line != "  " else word
            if current_line.strip():
                lines.append(current_line)

        # Data Sources
        lines.extend(["", "DATA SOURCES", "-" * 40])
        sources = assessment.get("data_sources_used", [])
        lines.append(f"  Sources used: {', '.join(sources) if sources else 'None'}")
        lines.append(f"  Data quality score: {assessment.get('data_quality_score', 0):.0%}")

        # Errors
        errors = result.get("errors", [])
        if errors:
            lines.extend(["", "WARNINGS/ERRORS", "-" * 40])
            for err in errors[:5]:
                lines.append(f"  * {err}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def run_credit_analysis(
    company_name: str,
    jurisdiction: Optional[str] = None,
    print_report: bool = True,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run credit analysis.

    Args:
        company_name: Name of the company to analyze
        jurisdiction: Optional jurisdiction code
        print_report: Whether to print the report
        config: Optional configuration

    Returns:
        Complete workflow result
    """
    workflow = CreditIntelligenceWorkflow(config)
    result = workflow.run(company_name, jurisdiction)

    if print_report:
        report = workflow.get_assessment_report(result)
        print(report)

    return result


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Intelligence Analysis")
    parser.add_argument("--company", required=True, help="Company name to analyze")
    parser.add_argument("--jurisdiction", help="Jurisdiction code (e.g., US, GB, DE)")
    parser.add_argument("--output", help="Output file for JSON result")
    parser.add_argument("--quiet", action="store_true", help="Suppress report output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run analysis
    result = run_credit_analysis(
        company_name=args.company,
        jurisdiction=args.jurisdiction,
        print_report=not args.quiet,
    )

    # Save JSON output if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nJSON result saved to: {args.output}")
