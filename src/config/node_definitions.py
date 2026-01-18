"""
Static node definitions for consistent logging across all sheets.

This file defines the hierarchy:
- node: The LangGraph node name
- node_type: Type of node (llm, agent, tool, router, storage)
- agent_name: The agent executing this node
- master_agent: Always "supervisor"

All logging calls should use these constants to ensure consistency.
"""

# Master agent (always the same)
MASTER_AGENT = "supervisor"

# Node type definitions
class NodeType:
    LLM = "llm"              # Makes LLM API calls
    AGENT = "agent"          # Makes decisions, orchestrates
    TOOL = "tool"            # Fetches external data
    ROUTER = "router"        # Conditional routing
    STORAGE = "storage"      # Database operations
    EVALUATOR = "evaluator"  # Evaluation sub-agents
    ORCHESTRATOR = "orchestrator"  # Master orchestrator


# Static node definitions
# Format: node_name -> (node_type, agent_name)
NODE_DEFINITIONS = {
    # Workflow nodes
    "parse_input": (NodeType.LLM, "llm_parser"),
    "validate_company": (NodeType.AGENT, "supervisor"),
    "create_execution_plan": (NodeType.AGENT, "supervisor"),  # Execution plan after validation
    "create_plan": (NodeType.AGENT, "tool_supervisor"),
    "fetch_api_data": (NodeType.TOOL, "api_agent"),
    "search_web": (NodeType.TOOL, "search_agent"),
    "search_web_enhanced": (NodeType.TOOL, "search_agent"),
    "synthesize": (NodeType.LLM, "llm_analyst"),
    "save_to_database": (NodeType.STORAGE, "db_writer"),
    "evaluate": (NodeType.AGENT, "workflow_evaluator"),

    # Router nodes (conditional edges)
    "should_continue_after_validation": (NodeType.ROUTER, "validation_router"),
    "route_after_api_data": (NodeType.ROUTER, "api_data_router"),
    "route_after_search_by_company_type": (NodeType.ROUTER, "company_type_router"),
    "route_after_save_by_company_type": (NodeType.ROUTER, "save_router"),

    # Orchestrator
    "workflow_orchestrator": (NodeType.ORCHESTRATOR, "supervisor"),

    # Evaluation sub-agents (called by workflow_evaluator)
    "eval_coalition": (NodeType.EVALUATOR, "coalition_evaluator"),
    "eval_llm_judge": (NodeType.EVALUATOR, "llm_judge"),
    "eval_agent_metrics": (NodeType.EVALUATOR, "agent_metrics_evaluator"),
    "eval_consistency": (NodeType.EVALUATOR, "consistency_checker"),
}


def get_node_info(node: str) -> dict:
    """
    Get node information for logging.

    Args:
        node: The node name

    Returns:
        dict with keys: node, node_type, agent_name, master_agent
    """
    if node in NODE_DEFINITIONS:
        node_type, agent_name = NODE_DEFINITIONS[node]
    else:
        # Default for unknown nodes
        node_type = NodeType.AGENT
        agent_name = node

    return {
        "node": node,
        "node_type": node_type,
        "agent_name": agent_name,
        "master_agent": MASTER_AGENT,
    }


def get_agent_name(node: str) -> str:
    """Get the agent name for a node."""
    if node in NODE_DEFINITIONS:
        return NODE_DEFINITIONS[node][1]
    return node


def get_node_type(node: str) -> str:
    """Get the node type for a node."""
    if node in NODE_DEFINITIONS:
        return NODE_DEFINITIONS[node][0]
    return NodeType.AGENT


# Convenience exports
NODES = {
    # Workflow nodes
    "PARSE_INPUT": "parse_input",
    "VALIDATE_COMPANY": "validate_company",
    "CREATE_PLAN": "create_plan",
    "FETCH_API_DATA": "fetch_api_data",
    "SEARCH_WEB": "search_web",
    "SEARCH_WEB_ENHANCED": "search_web_enhanced",
    "SYNTHESIZE": "synthesize",
    "SAVE_TO_DATABASE": "save_to_database",
    "EVALUATE": "evaluate",
    # Router nodes
    "SHOULD_CONTINUE_AFTER_VALIDATION": "should_continue_after_validation",
    "ROUTE_AFTER_API_DATA": "route_after_api_data",
    "ROUTE_AFTER_SEARCH_BY_COMPANY_TYPE": "route_after_search_by_company_type",
    "ROUTE_AFTER_SAVE_BY_COMPANY_TYPE": "route_after_save_by_company_type",
    # Orchestrator
    "WORKFLOW_ORCHESTRATOR": "workflow_orchestrator",
    # Evaluator nodes
    "EVAL_COALITION": "eval_coalition",
    "EVAL_LLM_JUDGE": "eval_llm_judge",
    "EVAL_AGENT_METRICS": "eval_agent_metrics",
    "EVAL_CONSISTENCY": "eval_consistency",
}

AGENTS = {
    # Workflow agents
    "LLM_PARSER": "llm_parser",
    "SUPERVISOR": "supervisor",
    "TOOL_SUPERVISOR": "tool_supervisor",
    "API_AGENT": "api_agent",
    "SEARCH_AGENT": "search_agent",
    "LLM_ANALYST": "llm_analyst",
    "DB_WRITER": "db_writer",
    "WORKFLOW_EVALUATOR": "workflow_evaluator",
    "SUPERVISOR": "supervisor",
    # Router agents
    "VALIDATION_ROUTER": "validation_router",
    "API_DATA_ROUTER": "api_data_router",
    "COMPANY_TYPE_ROUTER": "company_type_router",
    "SAVE_ROUTER": "save_router",
    # Evaluator agents
    "COALITION_EVALUATOR": "coalition_evaluator",
    "LLM_JUDGE": "llm_judge",
    "AGENT_METRICS_EVALUATOR": "agent_metrics_evaluator",
    "CONSISTENCY_CHECKER": "consistency_checker",
}
