"""LangGraph entry point for LangGraph Studio visualization.

This module exports the compiled graph for use with:
1. LangGraph Studio (desktop app)
2. LangSmith (cloud tracing)
3. Programmatic visualization
"""

import os
import sys
from typing import Any, Dict, List, Optional, TypedDict

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

# Import agents
from agents.supervisor import SupervisorAgent
from agents.api_agent import APIAgent
from agents.search_agent import SearchAgent

# Import MongoDB storage
try:
    from storage.mongodb import CreditIntelligenceDB
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


# =============================================================================
# STATE DEFINITION
# =============================================================================

class CreditWorkflowInput(TypedDict):
    """Simple input - just the company name."""
    company_name: str


class CreditWorkflowOutput(TypedDict):
    """Output from the workflow."""
    company_name: str
    risk_level: str
    credit_score: int
    confidence: float
    reasoning: str
    recommendations: List[str]


class CreditWorkflowState(TypedDict):
    """State for the credit intelligence workflow."""
    # Input (required)
    company_name: str

    # Optional inputs with defaults
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

    # Human-in-the-loop fields
    human_approved: bool
    human_feedback: Optional[str]
    validation_message: str
    requires_review: bool


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

# Initialize agents
supervisor = SupervisorAgent({"analysis_mode": "llm"})
api_agent = APIAgent()
search_agent = SearchAgent()

# Initialize MongoDB (optional)
db = CreditIntelligenceDB() if MONGODB_AVAILABLE else None


def parse_input(state: CreditWorkflowState) -> Dict[str, Any]:
    """Parse and enrich company input."""
    company_name = state["company_name"].strip()

    # Validate input
    if not company_name:
        return {
            "company_info": {},
            "status": "error",
            "errors": ["Company name cannot be empty"],
            "validation_message": "ERROR: Please provide a valid company name.",
            "requires_review": False,
        }

    company_info = supervisor.parse_company_input(
        company_name,
        state.get("jurisdiction"),
    )

    # Check if company was recognized
    is_known = company_info.get("is_public_company", False)
    ticker = company_info.get("ticker")

    if is_known and ticker:
        validation_msg = f"Found: {company_name} (Ticker: {ticker}) - Public company, full data available."
        requires_review = False
    else:
        validation_msg = f"Note: '{company_name}' not found in known companies database. Will search as private company with limited data sources."
        requires_review = True

    return {
        "company_info": company_info,
        "ticker": ticker,
        "status": "input_parsed",
        "validation_message": validation_msg,
        "requires_review": requires_review,
    }


def validate_company(state: CreditWorkflowState) -> Dict[str, Any]:
    """
    Human-in-the-loop checkpoint: Validate company before proceeding.
    This node pauses for human approval if the company is unknown.
    """
    # This is where human reviews the validation_message
    # If human_approved is set (by LangGraph Studio), we continue
    # Otherwise, workflow waits here

    validation_msg = state.get("validation_message", "")
    requires_review = state.get("requires_review", False)
    company_info = state.get("company_info", {})

    if not company_info:
        return {
            "status": "validation_failed",
            "errors": state.get("errors", []) + ["Company validation failed"],
        }

    return {
        "status": "validated",
        "human_approved": True,  # Will be overwritten by human input in Studio
    }


def create_plan(state: CreditWorkflowState) -> Dict[str, Any]:
    """Create task plan based on company info."""
    task_plan = supervisor.create_task_plan(state["company_info"])

    # Show plan summary for human review
    plan_summary = f"Plan: Fetching data from {len(task_plan)} sources for {state['company_name']}"

    return {
        "task_plan": task_plan,
        "status": "plan_created",
        "validation_message": plan_summary,
    }


def fetch_api_data(state: CreditWorkflowState) -> Dict[str, Any]:
    """Fetch data from external APIs."""
    company_info = state["company_info"]

    try:
        api_result = api_agent.fetch_all_data(
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
        return {
            "api_data": {},
            "errors": state.get("errors", []) + [f"API fetch error: {str(e)}"],
            "status": "api_data_error",
        }


def search_web(state: CreditWorkflowState) -> Dict[str, Any]:
    """Search for web information."""
    company_name = state["company_info"]["company_name"]

    try:
        search_result = search_agent.search_company(company_name)
        return {
            "search_data": search_result.to_dict(),
            "errors": state.get("errors", []) + search_result.errors,
            "status": "search_complete",
        }
    except Exception as e:
        return {
            "search_data": {},
            "errors": state.get("errors", []) + [f"Search error: {str(e)}"],
            "status": "search_error",
        }


def synthesize(state: CreditWorkflowState) -> Dict[str, Any]:
    """Synthesize final credit assessment with LLM analysis."""
    try:
        assessment = supervisor.synthesize_assessment(
            company_info=state["company_info"],
            api_data=state.get("api_data", {}),
            search_data=state.get("search_data", {}),
        )
        assessment_dict = assessment.to_dict()

        # Return both full assessment and simplified output
        return {
            "assessment": assessment_dict,
            "status": "complete",
            # Output fields for CreditWorkflowOutput
            "company_name": state["company_name"],
            "risk_level": assessment_dict.get("overall_risk_level", "unknown"),
            "credit_score": assessment_dict.get("credit_score_estimate", 0),
            "confidence": assessment_dict.get("confidence_score", 0.0),
            "reasoning": assessment_dict.get("llm_reasoning", ""),
            "recommendations": assessment_dict.get("recommendations", []),
        }
    except Exception as e:
        return {
            "assessment": None,
            "errors": state.get("errors", []) + [f"Synthesis error: {str(e)}"],
            "status": "synthesis_error",
            "company_name": state.get("company_name", ""),
            "risk_level": "error",
            "credit_score": 0,
            "confidence": 0.0,
            "reasoning": str(e),
            "recommendations": [],
        }


def save_to_database(state: CreditWorkflowState) -> Dict[str, Any]:
    """Save assessment and raw data to MongoDB."""
    if not db or not db.is_connected():
        return {"status": "complete_no_db"}

    try:
        company_name = state.get("company_name", "")
        assessment = state.get("assessment", {})
        api_data = state.get("api_data", {})
        search_data = state.get("search_data", {})

        # Save assessment
        if assessment:
            db.save_assessment(assessment)

        # Save raw data for auditing
        db.save_all_raw_data(company_name, api_data, search_data)

        # Save/update company profile
        company_info = state.get("company_info", {})
        if company_info:
            db.save_company(company_info)

        return {"status": "complete_saved"}

    except Exception as e:
        return {
            "status": "complete_db_error",
            "errors": state.get("errors", []) + [f"DB save error: {str(e)}"],
        }


# =============================================================================
# BUILD GRAPH
# =============================================================================

def should_continue_after_validation(state: CreditWorkflowState) -> str:
    """Decide whether to continue or stop based on validation."""
    if state.get("status") == "error":
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow with human-in-the-loop."""
    workflow = StateGraph(
        CreditWorkflowState,
        input=CreditWorkflowInput,
        output=CreditWorkflowOutput,
    )

    # Add nodes with descriptive names
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("validate_company", validate_company)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("fetch_api_data", fetch_api_data)
    workflow.add_node("search_web", search_web)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("save_to_database", save_to_database)

    # Add edges with human-in-the-loop checkpoints
    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "validate_company")

    # Conditional: stop if validation failed
    workflow.add_conditional_edges(
        "validate_company",
        should_continue_after_validation,
        {
            "continue": "create_plan",
            "end": END,
        }
    )

    workflow.add_edge("create_plan", "fetch_api_data")
    workflow.add_edge("fetch_api_data", "search_web")
    workflow.add_edge("search_web", "synthesize")
    workflow.add_edge("synthesize", "save_to_database")
    workflow.add_edge("save_to_database", END)

    return workflow


def build_graph_with_hitl() -> StateGraph:
    """Build graph with Human-in-the-Loop interrupts at key points."""
    return build_graph()


# Compiled graph for LangGraph Studio
# interrupt_before pauses workflow for human approval at validate_company only
# (to confirm company is correct before fetching data)
graph = build_graph().compile(
    interrupt_before=["validate_company"]
)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def get_mermaid_diagram() -> str:
    """Generate a Mermaid diagram of the workflow."""
    return """
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4F46E5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#3730A3', 'lineColor': '#6366F1', 'secondaryColor': '#818CF8', 'tertiaryColor': '#C7D2FE'}}}%%

graph TD
    subgraph CREDIT_INTELLIGENCE_WORKFLOW["Credit Intelligence Workflow"]
        START((Start)) --> PARSE[/"📝 Parse Input<br/>(Supervisor)"/]
        PARSE --> PLAN["📋 Create Plan<br/>(Supervisor)"]
        PLAN --> API["🔌 Fetch API Data<br/>(API Agent)"]

        subgraph DATA_SOURCES["Data Sources"]
            API --> SEC["SEC EDGAR<br/>Financials"]
            API --> FH["Finnhub<br/>Market Data"]
            API --> CL["CourtListener<br/>Legal Records"]
            API --> OC["OpenCorporates<br/>Registry"]
            API --> OS["OpenSanctions<br/>Compliance"]
        end

        SEC --> SEARCH
        FH --> SEARCH
        CL --> SEARCH
        OC --> SEARCH
        OS --> SEARCH

        SEARCH["🔍 Search Web<br/>(Search Agent)"]
        SEARCH --> SYNTH

        subgraph SYNTHESIS["AI-Powered Synthesis"]
            SYNTH["🧠 Synthesize<br/>(Supervisor + LLM)"]
            SYNTH --> RULES["Rule-Based<br/>Scoring"]
            SYNTH --> LLM["LLM Analysis<br/>(Groq)"]
            RULES --> HYBRID["Hybrid<br/>Assessment"]
            LLM --> HYBRID
        end

        HYBRID --> RESULT[/"📊 Credit Assessment"/]
        RESULT --> END_NODE((End))
    end

    style START fill:#10B981,stroke:#059669,color:#fff
    style END_NODE fill:#EF4444,stroke:#DC2626,color:#fff
    style PARSE fill:#6366F1,stroke:#4F46E5,color:#fff
    style PLAN fill:#6366F1,stroke:#4F46E5,color:#fff
    style API fill:#F59E0B,stroke:#D97706,color:#fff
    style SEARCH fill:#8B5CF6,stroke:#7C3AED,color:#fff
    style SYNTH fill:#EC4899,stroke:#DB2777,color:#fff
    style RESULT fill:#10B981,stroke:#059669,color:#fff
    style HYBRID fill:#14B8A6,stroke:#0D9488,color:#fff
```
"""


def print_graph_ascii():
    """Print an ASCII representation of the workflow."""
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CREDIT INTELLIGENCE WORKFLOW                              ║
║                         (LangGraph Visualization)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║    ┌─────────────┐                                                            ║
║    │   START     │                                                            ║
║    └──────┬──────┘                                                            ║
║           │                                                                   ║
║           ▼                                                                   ║
║    ┌─────────────────┐                                                        ║
║    │  PARSE INPUT    │  ◄── Supervisor: Parse company name, lookup ticker     ║
║    │  (Supervisor)   │                                                        ║
║    └────────┬────────┘                                                        ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐                                                        ║
║    │  CREATE PLAN    │  ◄── Supervisor: Decide which data sources to query    ║
║    │  (Supervisor)   │                                                        ║
║    └────────┬────────┘                                                        ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐      ┌─────────────────────────────────────────┐       ║
║    │  FETCH API DATA │      │  Data Sources:                          │       ║
║    │  (API Agent)    │ ───► │  • SEC EDGAR (Financials)               │       ║
║    │                 │      │  • Finnhub (Market Data)                │       ║
║    │                 │      │  • CourtListener (Legal)                │       ║
║    │                 │      │  • OpenCorporates (Registry)            │       ║
║    │                 │      │  • OpenSanctions (Compliance)           │       ║
║    └────────┬────────┘      └─────────────────────────────────────────┘       ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐      ┌─────────────────────────────────────────┐       ║
║    │  SEARCH WEB     │      │  Web Intelligence:                      │       ║
║    │  (Search Agent) │ ───► │  • Company information search           │       ║
║    │                 │      │  • News articles & sentiment            │       ║
║    └────────┬────────┘      └─────────────────────────────────────────┘       ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐      ┌─────────────────────────────────────────┐       ║
║    │  SYNTHESIZE     │      │  Hybrid Analysis:                       │       ║
║    │  (Supervisor    │ ───► │  • Rule-based scoring (35% Ability,     │       ║
║    │   + LLM)        │      │    25% Willingness, 40% Fraud)          │       ║
║    │                 │      │  • LLM Analysis (Groq LLaMA 70B)        │       ║
║    │                 │      │  • Confidence-weighted combination      │       ║
║    └────────┬────────┘      └─────────────────────────────────────────┘       ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────────┐                                                        ║
║    │  ASSESSMENT     │  ◄── Final credit report with risk level & score      ║
║    │  (Output)       │                                                        ║
║    └────────┬────────┘                                                        ║
║             │                                                                 ║
║             ▼                                                                 ║
║    ┌─────────────┐                                                            ║
║    │    END      │                                                            ║
║    └─────────────┘                                                            ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)


def save_graph_image(output_path: str = "workflow_graph.png"):
    """
    Save the graph as a PNG image (requires graphviz).

    Args:
        output_path: Path to save the image
    """
    try:
        # Try to use LangGraph's built-in visualization
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("Try installing: pip install pygraphviz")
        return None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Intelligence Graph Visualization")
    parser.add_argument("--mermaid", action="store_true", help="Print Mermaid diagram")
    parser.add_argument("--ascii", action="store_true", help="Print ASCII diagram")
    parser.add_argument("--save", type=str, help="Save graph as PNG image")
    parser.add_argument("--run", type=str, help="Run workflow for a company")

    args = parser.parse_args()

    if args.mermaid:
        print(get_mermaid_diagram())
    elif args.ascii:
        print_graph_ascii()
    elif args.save:
        save_graph_image(args.save)
    elif args.run:
        # Run the workflow
        initial_state = {
            "company_name": args.run,
            "jurisdiction": None,
            "ticker": None,
            "company_info": {},
            "task_plan": [],
            "api_data": {},
            "search_data": {},
            "assessment": None,
            "errors": [],
            "status": "started",
        }
        result = graph.invoke(initial_state)
        print(f"\nAssessment for {args.run}:")
        print(f"  Risk Level: {result['assessment'].get('overall_risk_level', 'N/A')}")
        print(f"  Credit Score: {result['assessment'].get('credit_score_estimate', 'N/A')}/100")
    else:
        # Default: print ASCII diagram
        print_graph_ascii()
