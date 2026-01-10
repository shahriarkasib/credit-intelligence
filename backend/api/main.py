"""
Credit Intelligence API - FastAPI Backend with WebSocket Streaming

This provides a LangGraph Studio-like experience via API:
- Real-time workflow execution with step-by-step updates
- WebSocket streaming for live progress
- REST endpoints for simple queries
"""

import os
import sys
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import prompt management
try:
    from config.prompts import (
        get_all_prompts,
        get_prompt,
        update_prompt,
        reset_prompt,
        reset_all_prompts,
        get_prompt_categories,
        get_prompt_text,
    )
    PROMPTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Prompts module not available: {e}")
    PROMPTS_AVAILABLE = False

# Import LLM factory for running prompts
try:
    from config.langchain_llm import (
        get_chat_groq,
        get_llm_for_prompt,
        get_llm_config_for_prompt,
        is_langchain_groq_available,
    )
    LLM_AVAILABLE = is_langchain_groq_available()
    LLM_FOR_PROMPT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain LLM not available: {e}")
    LLM_AVAILABLE = False
    LLM_FOR_PROMPT_AVAILABLE = False
    get_chat_groq = None
    get_llm_for_prompt = None
    get_llm_config_for_prompt = None

# Import storage modules for logs
try:
    from storage.mongodb import CreditIntelligenceDB
    MONGODB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MongoDB module not available: {e}")
    MONGODB_AVAILABLE = False
    CreditIntelligenceDB = None

try:
    from run_logging.sheets_logger import SheetsLogger, get_sheets_logger
    SHEETS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Sheets logger not available: {e}")
    SHEETS_AVAILABLE = False
    SheetsLogger = None
    get_sheets_logger = None

# Import LangGraph event logger for capturing all events
try:
    from run_logging.langgraph_logger import LangGraphEventLogger, get_langgraph_logger
    LANGGRAPH_LOGGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangGraph logger not available: {e}")
    LANGGRAPH_LOGGER_AVAILABLE = False
    LangGraphEventLogger = None
    get_langgraph_logger = None

# Import set_langgraph_event_logger for manual tool event logging
try:
    from agents.graph import set_langgraph_event_logger
    SET_LOGGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"set_langgraph_event_logger not available: {e}")
    SET_LOGGER_AVAILABLE = False
    set_langgraph_event_logger = None

# Import external configuration manager
try:
    from config.external_config import (
        get_config_manager,
        get_sanitized_config,
        get_credential_status,
        get_credential_categories,
        update_env_file,
        reload_config,
    )
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"External config manager not available: {e}")
    CONFIG_MANAGER_AVAILABLE = False

# Import API error callback for broadcasting errors to frontend
try:
    from config.langchain_callbacks import set_api_error_async_callback
    API_ERROR_CALLBACK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"API error callback not available: {e}")
    API_ERROR_CALLBACK_AVAILABLE = False
    set_api_error_async_callback = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CompanyRequest(BaseModel):
    """Request to analyze a company."""
    company_name: str
    jurisdiction: Optional[str] = None
    ticker: Optional[str] = None


class PromptUpdateRequest(BaseModel):
    """Request to update a prompt."""
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None  # provider, model, temperature, max_tokens


class PromptTestRequest(BaseModel):
    """Request to test a prompt with sample data."""
    prompt_id: str
    variables: Dict[str, str] = {}


class PromptLLMConfigRequest(BaseModel):
    """LLM configuration for a prompt request."""
    provider: Optional[str] = None  # groq, openai, anthropic
    model: Optional[str] = None     # primary, fast, balanced, or specific model ID
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class PromptRunRequest(BaseModel):
    """Request to run a prompt with LLM."""
    prompt_id: str
    variables: Dict[str, str] = {}
    model: Optional[str] = None       # Optional model override (legacy)
    provider: Optional[str] = None    # Optional provider override
    temperature: Optional[float] = None  # Optional temperature override
    max_tokens: Optional[int] = None  # Optional max_tokens override


# Configuration management models
class LLMProviderUpdate(BaseModel):
    """Update for a single LLM provider."""
    enabled: Optional[bool] = None
    default_model: Optional[str] = None


class LLMConfigUpdate(BaseModel):
    """Update LLM configuration."""
    default_provider: Optional[str] = None
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None
    providers: Optional[Dict[str, LLMProviderUpdate]] = None


class DataSourceUpdate(BaseModel):
    """Update for a single data source."""
    enabled: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


class CredentialUpdate(BaseModel):
    """Update a single credential."""
    value: str


class RuntimeConfigUpdate(BaseModel):
    """Update runtime settings."""
    hot_reload: Optional[bool] = None
    watch_interval_seconds: Optional[int] = None
    cache_enabled: Optional[bool] = None
    cache_ttl_seconds: Optional[int] = None


class WorkflowStep(BaseModel):
    """A single step in the workflow."""
    step_id: str  # Node name (e.g., parse_input, synthesize)
    name: str  # Display name (e.g., "Parsing Input")
    agent_name: str  # Canonical agent name (e.g., llm_parser, llm_analyst)
    status: str  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    output_summary: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None  # Store full output data
    error: Optional[str] = None


class WorkflowStatus(BaseModel):
    """Current status of a workflow run."""
    run_id: str
    company_name: str
    status: str  # pending, running, completed, failed
    current_step: Optional[str] = None
    steps: List[WorkflowStep] = []
    result: Optional[Dict[str, Any]] = None
    started_at: str
    completed_at: Optional[str] = None


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
        logger.info(f"WebSocket connected for run: {run_id}")

    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
        logger.info(f"WebSocket disconnected for run: {run_id}")

    async def broadcast(self, run_id: str, message: dict):
        """Send message to all connections for a run."""
        if run_id in self.active_connections:
            connections = self.active_connections[run_id]
            logger.info(f"Broadcasting to {len(connections)} connections for run {run_id[:8]}: {message.get('type')}")
            for connection in connections:
                try:
                    await connection.send_json(message)
                    logger.debug(f"Message sent successfully: {message.get('type')}")
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
        else:
            logger.warning(f"No connections for run {run_id[:8]} to broadcast {message.get('type')}")


manager = ConnectionManager()

# Global reference for broadcasting API errors from other modules
_current_run_id: Optional[str] = None

async def broadcast_api_error(error_type: str, error_message: str, details: Dict[str, Any] = None):
    """
    Broadcast an API error to the frontend.

    Args:
        error_type: Type of error (rate_limit, api_error, quota_exceeded, etc.)
        error_message: Human-readable error message
        details: Additional error details
    """
    global _current_run_id
    if _current_run_id:
        await manager.broadcast(_current_run_id, {
            "type": "api_error",
            "data": {
                "error_type": error_type,
                "message": error_message,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        logger.warning(f"API error broadcast: {error_type} - {error_message}")

def set_current_run_id(run_id: Optional[str]):
    """Set the current run ID for error broadcasting."""
    global _current_run_id
    _current_run_id = run_id

def get_current_run_id() -> Optional[str]:
    """Get the current run ID."""
    return _current_run_id


# =============================================================================
# WORKFLOW RUNNER
# =============================================================================

# Store active runs
active_runs: Dict[str, WorkflowStatus] = {}


async def run_workflow_with_streaming(
    run_id: str,
    company_name: str,
    jurisdiction: Optional[str] = None,
    ticker: Optional[str] = None,
):
    """
    Run the credit intelligence workflow with real-time streaming.

    Broadcasts updates via WebSocket as each step completes.
    """
    import time

    # Initialize workflow status
    workflow_status = WorkflowStatus(
        run_id=run_id,
        company_name=company_name,
        status="running",
        started_at=datetime.utcnow().isoformat(),
        steps=[]
    )
    active_runs[run_id] = workflow_status

    # Set current run ID for API error broadcasting
    set_current_run_id(run_id)

    # Set up API error callback to broadcast errors to frontend
    if API_ERROR_CALLBACK_AVAILABLE and set_api_error_async_callback:
        set_api_error_async_callback(broadcast_api_error)

    # Define workflow steps (matching the LangGraph nodes)
    # Format: (node_name, display_name, agent_name)
    step_definitions = [
        ("parse_input", "Parsing Input", "llm_parser"),
        ("validate_company", "Validating Company", "supervisor"),
        ("create_plan", "Creating Plan", "tool_supervisor"),
        ("fetch_api_data", "Fetching API Data", "api_agent"),
        ("search_web", "Searching Web", "search_agent"),
        ("synthesize", "Synthesizing Assessment", "llm_analyst"),
        ("save_to_database", "Saving to Database", "db_writer"),
        ("evaluate", "Evaluating Results", "workflow_evaluator"),
    ]

    # Initialize all steps as pending
    for step_id, step_name, agent_name in step_definitions:
        workflow_status.steps.append(WorkflowStep(
            step_id=step_id,
            name=step_name,
            agent_name=agent_name,
            status="pending"
        ))

    # Broadcast initial state
    await manager.broadcast(run_id, {
        "type": "workflow_started",
        "data": workflow_status.model_dump()
    })

    try:
        # Import the graph
        from agents.graph import graph

        # Step name mapping (LangGraph node names to display names)
        step_name_map = {
            "parse_input": 0,
            "validate_company": 1,
            "create_plan": 2,
            "fetch_api_data": 3,
            "search_web": 4,
            "synthesize": 5,
            "save_to_database": 6,
            "evaluate": 7,
        }

        async def update_step(step_idx: int, status: str, output_summary: str = None, error: str = None, output_data: dict = None):
            """Update a step's status and broadcast."""
            if step_idx < len(workflow_status.steps):
                step = workflow_status.steps[step_idx]
                step.status = status

                if status == "running":
                    step.started_at = datetime.utcnow().isoformat()
                    workflow_status.current_step = step.name
                elif status in ("completed", "failed"):
                    step.completed_at = datetime.utcnow().isoformat()
                    if step.started_at:
                        start = datetime.fromisoformat(step.started_at)
                        end = datetime.fromisoformat(step.completed_at)
                        step.duration_ms = (end - start).total_seconds() * 1000
                    step.output_summary = output_summary
                    step.error = error
                    # IMPORTANT: Store output_data in the step for persistence
                    if output_data:
                        step.output_data = output_data

                await manager.broadcast(run_id, {
                    "type": "step_update",
                    "data": {
                        "step": step.model_dump(),
                        "step_index": step_idx,
                        "output_data": output_data
                    }
                })

        def extract_step_output(node_name: str, state: dict, node_output: dict = None) -> tuple:
            """Extract meaningful output data from state after a node runs.

            Args:
                node_name: Name of the node that just ran
                state: Accumulated state
                node_output: Direct output from the node (preferred for extraction)
            """
            output_summary = ""
            output_data = {}

            # Use node_output if available, otherwise fall back to state
            source = node_output if node_output else state
            logger.info(f"Extracting output for {node_name}: source keys = {list(source.keys()) if source else 'None'}")

            if node_name == "parse_input":
                company_info = source.get("company_info", state.get("company_info", {}))
                company_name = state.get("company_name", source.get("company_name", "Unknown"))
                output_summary = f"Company: {company_name}"
                if company_info:
                    output_summary += f" | Public: {company_info.get('is_public_company', 'Unknown')}"
                    if company_info.get('ticker'):
                        output_summary += f" | Ticker: {company_info.get('ticker')}"
                output_data = {
                    "company_name": company_name,
                    "company_info": company_info,
                    "run_id": str(source.get("run_id", "")),  # Full run_id
                    "validation_message": source.get("validation_message", ""),
                    "requires_review": source.get("requires_review", False),
                }

            elif node_name == "validate_company":
                output_summary = f"Validated: {state.get('company_name')}"
                output_data = {
                    "status": source.get("status", state.get("status", "")),
                    "human_approved": source.get("human_approved", True),
                    "errors": source.get("errors", state.get("errors", [])),
                }

            elif node_name == "create_plan":
                task_plan = source.get("task_plan", state.get("task_plan", []))
                output_summary = f"Created {len(task_plan)} tasks"
                output_data = {
                    "num_tasks": len(task_plan),
                    "task_plan": task_plan,  # Full task plan
                    "validation_message": source.get("validation_message", ""),
                }

            elif node_name == "fetch_api_data":
                api_data = source.get("api_data", state.get("api_data", {}))
                sources_list = list(api_data.keys()) if api_data else []
                output_summary = f"Fetched from {len(sources_list)} sources: {', '.join(sources_list)}"
                # Summarize each source
                output_data = {"sources_fetched": sources_list, "source_details": {}}
                for src_name, data in api_data.items():
                    if isinstance(data, dict):
                        # Extract key metrics
                        records = 0
                        if "filings" in data:
                            records = len(data.get("filings", []))
                        elif "data" in data:
                            records = len(data.get("data", [])) if isinstance(data.get("data"), list) else 1
                        else:
                            records = 1 if data else 0

                        output_data["source_details"][src_name] = {
                            "success": True,
                            "records": records,
                            "preview": str(data)[:300] + "..." if len(str(data)) > 300 else str(data)
                        }
                    else:
                        output_data["source_details"][src_name] = {"data": str(data)[:300]}

            elif node_name == "search_web":
                search_data = source.get("search_data", state.get("search_data", {}))
                output_summary = "Web search completed"
                if search_data:
                    num_results = len(search_data.get("results", []))
                    num_news = len(search_data.get("news", []))
                    output_summary = f"Found {num_results} web results, {num_news} news articles"
                output_data = {
                    "num_results": len(search_data.get("results", [])),
                    "num_news": len(search_data.get("news", [])),
                    "top_results": search_data.get("results", [])[:3],
                    "top_news": search_data.get("news", [])[:3],
                    "status": source.get("status", ""),
                }

            elif node_name == "synthesize":
                assessment = source.get("assessment", state.get("assessment", {}))
                llm_consistency = assessment.get("llm_consistency", {}) if isinstance(assessment, dict) else {}

                if isinstance(assessment, dict):
                    risk = assessment.get("overall_risk_level", source.get("risk_level", "unknown"))
                    score = assessment.get("credit_score_estimate", source.get("credit_score", 0))
                    confidence = assessment.get("confidence_score", source.get("confidence", 0))

                    if isinstance(confidence, (int, float)):
                        output_summary = f"Risk: {risk} | Score: {score} | Confidence: {confidence:.0%}"
                    else:
                        output_summary = f"Risk: {risk} | Score: {score}"

                    reasoning = assessment.get("llm_reasoning", source.get("reasoning", ""))
                    output_data = {
                        "risk_level": risk,
                        "credit_score": score,
                        "confidence": confidence,
                        "reasoning": reasoning if reasoning else "",
                        "risk_factors": assessment.get("risk_factors", []),
                        "positive_factors": assessment.get("positive_factors", []),
                        "recommendations": assessment.get("recommendations", source.get("recommendations", [])),
                        "llm_consistency": {
                            "num_calls": llm_consistency.get("num_llm_calls", 0),
                            "same_model": llm_consistency.get("same_model_consistency", 0),
                            "cross_model": llm_consistency.get("cross_model_consistency", 0),
                            "risk_levels": llm_consistency.get("risk_levels", []),
                            "credit_scores": llm_consistency.get("credit_scores", [])
                        } if llm_consistency else {}
                    }
                else:
                    output_summary = "Assessment generated"
                    output_data = {
                        "risk_level": source.get("risk_level", "unknown"),
                        "credit_score": source.get("credit_score", 0),
                        "assessment_raw": str(assessment) if assessment else "No assessment"
                    }

            elif node_name == "save_to_database":
                output_summary = "Saved to MongoDB and Google Sheets"
                output_data = {
                    "status": source.get("status", "complete"),
                    "mongodb": True,
                    "google_sheets": True,
                    "company": state.get("company_name", "")
                }

            elif node_name == "evaluate":
                evaluation = source.get("evaluation", state.get("evaluation", {}))
                if isinstance(evaluation, dict):
                    overall = evaluation.get("overall_score", source.get("evaluation_score", 0))
                    if isinstance(overall, (int, float)):
                        output_summary = f"Evaluation: {overall:.0%}"
                    else:
                        output_summary = "Evaluation complete"

                    output_data = {
                        "overall_score": evaluation.get("overall_score", 0),
                        "tool_selection": evaluation.get("tool_selection", {}),
                        "data_quality": evaluation.get("data_quality", {}),
                        "synthesis": evaluation.get("synthesis", {}),
                        "tool_selection_score": source.get("tool_selection_score", 0),
                        "data_quality_score": source.get("data_quality_score", 0),
                        "synthesis_score": source.get("synthesis_score", 0),
                    }
                else:
                    output_summary = "Evaluation complete"
                    output_data = {
                        "evaluated": True,
                        "tool_selection_score": source.get("tool_selection_score", 0),
                        "data_quality_score": source.get("data_quality_score", 0),
                        "synthesis_score": source.get("synthesis_score", 0),
                    }

            logger.info(f"Extracted output for {node_name}: summary='{output_summary}', data_keys={list(output_data.keys())}")
            return output_summary, output_data

        # Use graph.stream() to get real-time node outputs
        import concurrent.futures
        import queue

        # Queue for streaming results from thread
        result_queue = queue.Queue()
        final_result = [None]  # Use list for mutability

        # Initialize LangGraph event logger for capturing ALL events
        event_logger = None
        if LANGGRAPH_LOGGER_AVAILABLE and get_langgraph_logger:
            event_logger = get_langgraph_logger(
                run_id=run_id,
                company_name=company_name,
                log_to_sheets=True,
                log_to_mongodb=True,
            )
            logger.info(f"LangGraph event logger initialized for run {run_id}")

            # Set the event logger globally so graph nodes can log tool events
            if SET_LOGGER_AVAILABLE and set_langgraph_event_logger:
                set_langgraph_event_logger(event_logger)
                logger.info("Event logger set globally for tool event logging")

        def run_graph_streaming():
            """Run graph with streaming and put results in queue."""
            try:
                for event in graph.stream({
                    'company_name': company_name,
                    'jurisdiction': jurisdiction,
                    'ticker': ticker,
                    'run_id': run_id,  # Pass API run_id to graph
                }):
                    # event is a dict with node_name -> output
                    result_queue.put(("step", event))

                # Signal completion
                result_queue.put(("done", None))
            except Exception as e:
                result_queue.put(("error", str(e)))

        async def run_graph_with_all_events():
            """Run graph with astream_events to capture ALL events (LLM, tool, chain)."""
            try:
                if event_logger:
                    event_logger.log_graph_start({
                        'company_name': company_name,
                        'jurisdiction': jurisdiction,
                        'ticker': ticker,
                    })

                async for event in graph.astream_events(
                    {
                        'company_name': company_name,
                        'jurisdiction': jurisdiction,
                        'ticker': ticker,
                        'run_id': run_id,  # Pass API run_id to graph
                    },
                    version="v2"
                ):
                    # Log ALL events to sheets/mongodb
                    if event_logger:
                        event_logger.log_event(event)

                    # Also put node outputs in queue for UI updates
                    event_type = event.get("event", "")
                    if event_type == "on_chain_end":
                        node_name = event.get("name", "")
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and node_name:
                            result_queue.put(("step", {node_name: output}))

                # Flush remaining events
                if event_logger:
                    event_logger.log_graph_end({})
                    event_logger.flush()

                result_queue.put(("done", None))
            except Exception as e:
                if event_logger:
                    event_logger.log_graph_end({}, error=str(e))
                    event_logger.flush()
                result_queue.put(("error", str(e)))

        # Start graph - use async version with full event logging
        import threading

        # Use astream_events for full event capture (LLM, tool, chain events)
        if LANGGRAPH_LOGGER_AVAILABLE and event_logger:
            # Run async version in background task
            asyncio.create_task(run_graph_with_all_events())
            graph_thread = None  # No separate thread needed
        else:
            # Fallback to thread-based streaming
            graph_thread = threading.Thread(target=run_graph_streaming)
            graph_thread.start()

        # Process streaming results
        current_state = {}
        completed_steps = set()

        while True:
            try:
                # Wait for next event with timeout
                # Use run_in_executor for blocking queue.get to not block the event loop
                loop = asyncio.get_event_loop()
                try:
                    msg_type, data = await asyncio.wait_for(
                        loop.run_in_executor(None, result_queue.get),
                        timeout=120
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for graph output")
                    break

                if msg_type == "done":
                    # Mark any remaining steps as completed
                    for idx in range(len(step_definitions)):
                        if idx not in completed_steps:
                            await update_step(idx, "completed", "Completed")
                    break

                elif msg_type == "error":
                    raise Exception(data)

                elif msg_type == "step":
                    # data is dict: {node_name: node_output}
                    for node_name, node_output in data.items():
                        logger.info(f"Received step event: {node_name}, output_keys={list(node_output.keys()) if isinstance(node_output, dict) else 'not dict'}")

                        # Update current state with node output
                        if isinstance(node_output, dict):
                            current_state.update(node_output)

                        # Find step index
                        step_idx = step_name_map.get(node_name)
                        if step_idx is not None:
                            # Mark previous steps as completed if not already
                            for prev_idx in range(step_idx):
                                if prev_idx not in completed_steps:
                                    prev_name = step_definitions[prev_idx][0]
                                    prev_summary, prev_data = extract_step_output(prev_name, current_state, None)
                                    await update_step(prev_idx, "completed", prev_summary or "Completed", None, prev_data)
                                    completed_steps.add(prev_idx)

                            # Mark this step as running then completed
                            if step_idx not in completed_steps:
                                await update_step(step_idx, "running")
                                await asyncio.sleep(0.1)  # Brief pause to show running state

                                # Extract output - pass node_output directly for more accurate data
                                output_summary, output_data = extract_step_output(node_name, current_state, node_output)
                                logger.info(f"Sending step {node_name} with output_data: {list(output_data.keys()) if output_data else 'None'}")
                                await update_step(step_idx, "completed", output_summary, None, output_data)
                                completed_steps.add(step_idx)

            except Exception as e:
                logger.error(f"Error processing graph event: {e}")
                break

        # Wait for thread to finish (if using threaded mode)
        if graph_thread:
            graph_thread.join(timeout=5)

        # Get final result from state
        result = current_state

        # Update final status
        workflow_status.status = "completed"
        workflow_status.completed_at = datetime.utcnow().isoformat()
        workflow_status.result = {
            "risk_level": result.get("risk_level", "unknown"),
            "credit_score": result.get("credit_score", 0),
            "confidence": result.get("confidence", 0),
            "reasoning": result.get("reasoning", ""),
            "recommendations": result.get("recommendations", []),
            "evaluation_score": result.get("evaluation_score", 0),
        }

        # Broadcast completion
        await manager.broadcast(run_id, {
            "type": "workflow_completed",
            "data": workflow_status.model_dump()
        })

        logger.info(f"Workflow completed for {company_name}: {workflow_status.result}")

        # Clear the global event logger
        if SET_LOGGER_AVAILABLE and set_langgraph_event_logger:
            set_langgraph_event_logger(None)

        # Clear current run ID
        set_current_run_id(None)

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        workflow_status.status = "failed"
        workflow_status.completed_at = datetime.utcnow().isoformat()

        # Mark current step as failed
        for step in workflow_status.steps:
            if step.status == "running":
                step.status = "failed"
                step.error = str(e)
                break

        await manager.broadcast(run_id, {
            "type": "workflow_failed",
            "data": {
                "error": str(e),
                "status": workflow_status.model_dump()
            }
        })

        # Clear the global event logger on error too
        if SET_LOGGER_AVAILABLE and set_langgraph_event_logger:
            set_langgraph_event_logger(None)

        # Clear current run ID on error
        set_current_run_id(None)

    return workflow_status


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Credit Intelligence API...")
    yield
    logger.info("Shutting down Credit Intelligence API...")


app = FastAPI(
    title="Credit Intelligence API",
    description="LangGraph-powered credit risk assessment with real-time streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Serve frontend or health check."""
    # Try to serve frontend index.html first
    frontend_index = Path(__file__).parent.parent.parent / "frontend" / "out" / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    # Fallback to API health check
    return {
        "status": "healthy",
        "service": "Credit Intelligence API",
        "version": "1.0.0",
        "note": "Frontend not built. Run 'cd frontend && npm run build' to enable UI."
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "active_runs": len(active_runs),
        "active_connections": sum(len(conns) for conns in manager.active_connections.values())
    }


@app.post("/analyze", response_model=Dict[str, str])
async def start_analysis(request: CompanyRequest):
    """
    Start a credit analysis workflow.

    Returns a run_id that can be used to:
    - Connect via WebSocket for real-time updates
    - Poll the /status/{run_id} endpoint
    """
    run_id = str(uuid.uuid4())

    async def delayed_workflow_start():
        """Start workflow after a brief delay to allow WebSocket connection."""
        await asyncio.sleep(0.5)  # Give frontend time to connect WebSocket
        await run_workflow_with_streaming(
            run_id=run_id,
            company_name=request.company_name,
            jurisdiction=request.jurisdiction,
            ticker=request.ticker,
        )

    # Start workflow in background with delay
    asyncio.create_task(delayed_workflow_start())

    return {
        "run_id": run_id,
        "status": "started",
        "websocket_url": f"/ws/{run_id}",
        "status_url": f"/status/{run_id}"
    }


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """Get the current status of a workflow run."""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    return active_runs[run_id].model_dump()


@app.get("/runs")
async def list_runs():
    """List all active and recent runs."""
    return {
        "runs": [
            {
                "run_id": run_id,
                "company_name": status.company_name,
                "status": status.status,
                "started_at": status.started_at
            }
            for run_id, status in active_runs.items()
        ]
    }


# =============================================================================
# PROMPT MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/prompts")
async def list_prompts():
    """Get all available prompts."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    prompts = get_all_prompts()
    return {
        "prompts": list(prompts.values()),
        "count": len(prompts)
    }


@app.get("/api/prompts/categories")
async def list_prompt_categories():
    """Get prompts organized by category."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    return get_prompt_categories()


@app.get("/api/prompts/{prompt_id}")
async def get_prompt_by_id(prompt_id: str):
    """Get a specific prompt by ID."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    prompt = get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    return prompt


@app.put("/api/prompts/{prompt_id}")
async def update_prompt_by_id(prompt_id: str, request: PromptUpdateRequest):
    """Update a prompt with custom values."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    try:
        updates = request.model_dump(exclude_none=True)
        updated = update_prompt(prompt_id, updates)
        return {
            "status": "updated",
            "prompt": updated
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/prompts/{prompt_id}/reset")
async def reset_prompt_by_id(prompt_id: str):
    """Reset a prompt to its default values."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    try:
        prompt = reset_prompt(prompt_id)
        return {
            "status": "reset",
            "prompt": prompt
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/prompts/reset-all")
async def reset_all_prompts_endpoint():
    """Reset all prompts to their default values."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    prompts = reset_all_prompts()
    return {
        "status": "all_reset",
        "count": len(prompts)
    }


@app.post("/api/prompts/test")
async def test_prompt(request: PromptTestRequest):
    """
    Test a prompt with sample variables.

    Returns the formatted prompt without executing it.
    """
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    try:
        system_prompt, user_prompt = get_prompt_text(request.prompt_id, **request.variables)
        return {
            "prompt_id": request.prompt_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "variables_used": request.variables
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing variable: {e}")


@app.post("/api/prompts/run")
async def run_prompt(request: PromptRunRequest):
    """
    Run a prompt with the LLM and return the response.

    This allows testing individual prompts with real LLM execution.
    Supports per-prompt LLM configuration with runtime overrides.

    The LLM is selected based on:
    1. Runtime overrides (provider, model, temperature, max_tokens in request)
    2. Prompt's llm_config (defined in prompts.py or settings.yaml)
    3. Global defaults
    """
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompts module not available")

    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM not available. Check API keys.")

    import time
    start_time = time.time()

    try:
        # Get formatted prompt
        system_prompt, user_prompt = get_prompt_text(request.prompt_id, **request.variables)

        # Get resolved LLM config for this prompt
        resolved_config = {}
        if LLM_FOR_PROMPT_AVAILABLE and get_llm_config_for_prompt:
            resolved_config = get_llm_config_for_prompt(request.prompt_id)

        # Create LLM instance using per-prompt configuration
        if LLM_FOR_PROMPT_AVAILABLE and get_llm_for_prompt:
            llm = get_llm_for_prompt(
                request.prompt_id,
                override_provider=request.provider,
                override_model=request.model,
                override_temperature=request.temperature,
                override_max_tokens=request.max_tokens,
            )
        else:
            # Fallback to legacy get_chat_groq
            model = request.model or "primary"
            llm = get_chat_groq(model=model)

        if not llm:
            raise HTTPException(status_code=503, detail="Failed to create LLM instance. Check API keys.")

        # Create messages for the LLM
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        # Run the LLM
        response = llm.invoke(messages)

        execution_time_ms = (time.time() - start_time) * 1000

        # Extract usage metadata
        usage_metadata = getattr(response, 'usage_metadata', {}) or {}

        return {
            "prompt_id": request.prompt_id,
            "provider": request.provider or resolved_config.get("provider", "groq"),
            "model": request.model or resolved_config.get("model_alias", "primary"),
            "model_id": resolved_config.get("model_id", "unknown"),
            "temperature": request.temperature if request.temperature is not None else resolved_config.get("temperature", 0.1),
            "max_tokens": request.max_tokens if request.max_tokens is not None else resolved_config.get("max_tokens", 2000),
            "config_source": resolved_config.get("source", "defaults"),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "variables_used": request.variables,
            "response": response.content,
            "execution_time_ms": round(execution_time_ms, 2),
            "usage": {
                "input_tokens": usage_metadata.get('input_tokens', 0),
                "output_tokens": usage_metadata.get('output_tokens', 0),
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing variable: {e}")
    except Exception as e:
        logger.error(f"Error running prompt: {e}")
        raise HTTPException(status_code=500, detail=f"LLM execution failed: {str(e)}")


# =============================================================================
# LOGS API ENDPOINTS
# =============================================================================

# Lazy-loaded database instances
_db_instance = None
_sheets_instance = None


def get_db():
    """Get or create MongoDB instance."""
    global _db_instance
    if _db_instance is None and MONGODB_AVAILABLE:
        _db_instance = CreditIntelligenceDB()
    return _db_instance


def get_sheets():
    """Get or create SheetsLogger instance."""
    global _sheets_instance
    if _sheets_instance is None and SHEETS_AVAILABLE:
        _sheets_instance = SheetsLogger()
    return _sheets_instance


@app.get("/logs/evaluations")
async def get_evaluations(limit: int = 50):
    """
    Get recent evaluations from MongoDB.

    Returns evaluation results with scores and reasoning.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"evaluations": [], "source": "none", "message": "MongoDB not connected"}

    evaluations = db.get_evaluations(limit=limit)

    # Convert ObjectId to string for JSON serialization
    for e in evaluations:
        if "_id" in e:
            e["_id"] = str(e["_id"])

    return {
        "evaluations": evaluations,
        "count": len(evaluations),
        "source": "mongodb"
    }


@app.get("/logs/evaluations/{run_id}")
async def get_evaluation_by_run(run_id: str):
    """
    Get evaluation for a specific run.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"evaluation": None, "source": "none", "message": "MongoDB not connected"}

    # Find evaluation by run_id
    evaluation = db.db.evaluations.find_one({"run_id": run_id})

    if evaluation:
        evaluation["_id"] = str(evaluation["_id"])

    return {
        "evaluation": evaluation,
        "source": "mongodb"
    }


@app.get("/logs/traces/{run_id}")
async def get_traces_by_run(run_id: str, limit: int = 100):
    """
    Get LangGraph events/traces for a specific run.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"traces": [], "source": "none", "message": "MongoDB not connected"}

    events = db.get_langgraph_events(run_id=run_id, limit=limit)

    # Convert ObjectId to string
    for e in events:
        if "_id" in e:
            e["_id"] = str(e["_id"])

    # Also get run summary
    summary = db.get_langgraph_run_summary(run_id)

    return {
        "traces": events,
        "summary": summary,
        "count": len(events),
        "source": "mongodb"
    }


@app.get("/logs/assessments")
async def get_assessments(limit: int = 50):
    """
    Get recent credit assessments.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"assessments": [], "source": "none", "message": "MongoDB not connected"}

    assessments = db.get_all_assessments(limit=limit)

    # Convert ObjectId to string
    for a in assessments:
        if "_id" in a:
            a["_id"] = str(a["_id"])

    return {
        "assessments": assessments,
        "count": len(assessments),
        "source": "mongodb"
    }


@app.get("/logs/assessments/{company_name}")
async def get_assessment_by_company(company_name: str, limit: int = 10):
    """
    Get assessment history for a specific company.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"assessments": [], "source": "none", "message": "MongoDB not connected"}

    assessments = db.get_assessment_history(company_name, limit=limit)

    for a in assessments:
        if "_id" in a:
            a["_id"] = str(a["_id"])

    return {
        "assessments": assessments,
        "count": len(assessments),
        "company_name": company_name,
        "source": "mongodb"
    }


@app.get("/logs/stats")
async def get_log_stats():
    """
    Get database statistics and log counts.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"connected": False, "source": "none"}

    stats = db.get_stats()
    risk_distribution = db.get_risk_distribution()

    return {
        **stats,
        "risk_distribution": risk_distribution,
        "source": "mongodb"
    }


@app.get("/logs/runs/history")
async def get_run_history(limit: int = 50):
    """
    Get recent runs with summaries from MongoDB.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"runs": [], "source": "none", "message": "MongoDB not connected"}

    # Try run_summaries first (new Task 17 format), fallback to assessments
    run_summaries = db.get_run_summaries(limit=limit)

    if run_summaries:
        runs = []
        for r in run_summaries:
            runs.append({
                "run_id": r.get("run_id", str(r.get("_id", ""))),
                "company_name": r.get("company_name", ""),
                "status": r.get("status", "completed"),
                "risk_level": r.get("risk_level", ""),
                "credit_score": r.get("credit_score", 0),
                "confidence": r.get("confidence", 0),
                "overall_score": r.get("overall_score", 0),
                "final_decision": r.get("final_decision", ""),
                "duration_ms": r.get("duration_ms", 0),
                "total_tokens": r.get("total_tokens", 0),
                "total_cost": r.get("total_cost", 0),
                "timestamp": r.get("saved_at", r.get("completed_at", "")),
            })
        return {"runs": runs, "count": len(runs), "source": "run_summaries"}

    # Fallback to assessments
    assessments = db.get_all_assessments(limit=limit)
    runs = []
    for a in assessments:
        runs.append({
            "run_id": a.get("run_id", str(a.get("_id", ""))),
            "company_name": a.get("company_name", a.get("company", "")),
            "risk_level": a.get("overall_risk_level", a.get("risk_level", "")),
            "credit_score": a.get("credit_score_estimate", a.get("credit_score", 0)),
            "confidence": a.get("confidence_score", a.get("confidence", 0)),
            "timestamp": a.get("saved_at", a.get("timestamp", "")),
        })

    return {
        "runs": runs,
        "count": len(runs),
        "source": "assessments"
    }


@app.get("/logs/llm-calls")
async def get_llm_calls(run_id: str = None, company_name: str = None, limit: int = 100):
    """
    Get LLM call logs from MongoDB.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"llm_calls": [], "source": "none", "message": "MongoDB not connected"}

    calls = db.get_llm_calls(run_id=run_id, company_name=company_name, limit=limit)

    for c in calls:
        if "_id" in c:
            c["_id"] = str(c["_id"])

    return {
        "llm_calls": calls,
        "count": len(calls),
        "source": "mongodb"
    }


@app.get("/logs/llm-calls/{run_id}/summary")
async def get_llm_calls_summary(run_id: str):
    """
    Get summary of LLM calls for a specific run.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"summary": {}, "source": "none", "message": "MongoDB not connected"}

    summary = db.get_llm_calls_summary(run_id)

    return {
        "summary": summary,
        "source": "mongodb"
    }


@app.get("/logs/run-summaries")
async def get_run_summaries(company_name: str = None, status: str = None, limit: int = 50):
    """
    Get run summaries from MongoDB.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"summaries": [], "source": "none", "message": "MongoDB not connected"}

    summaries = db.get_run_summaries(company_name=company_name, status=status, limit=limit)

    for s in summaries:
        if "_id" in s:
            s["_id"] = str(s["_id"])

    return {
        "summaries": summaries,
        "count": len(summaries),
        "source": "mongodb"
    }


@app.get("/logs/run-summaries/{run_id}")
async def get_run_summary_by_id(run_id: str):
    """
    Get a specific run summary.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"summary": None, "source": "none", "message": "MongoDB not connected"}

    summary = db.get_run_summary(run_id)

    if summary and "_id" in summary:
        summary["_id"] = str(summary["_id"])

    return {
        "summary": summary,
        "source": "mongodb"
    }


@app.get("/logs/statistics")
async def get_run_statistics():
    """
    Get aggregate statistics across all runs.
    """
    db = get_db()
    if not db or not db.is_connected():
        return {"statistics": {}, "source": "none", "message": "MongoDB not connected"}

    stats = db.get_run_statistics()

    return {
        "statistics": stats,
        "source": "mongodb"
    }


# =============================================================================
# CONFIGURATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/config")
async def get_config():
    """
    Get full configuration (with secrets masked).

    Returns sanitized configuration safe for display in UI.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    config = get_sanitized_config()
    manager = get_config_manager()

    return {
        "config": config,
        "config_path": str(manager.config_path),
        "last_modified": datetime.fromtimestamp(manager._last_modified).isoformat() if manager._last_modified else None,
        "hot_reload_enabled": manager._watching,
    }


@app.get("/config/llm")
async def get_llm_config():
    """
    Get LLM provider configuration.

    Returns providers, models, and default settings.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()
    llm_config = manager.get_llm_config()

    # Mask API keys in response
    providers = {}
    for name, config in llm_config.get("providers", {}).items():
        providers[name] = {
            "enabled": config.get("enabled", False),
            "default_model": config.get("default_model", "primary"),
            "models": config.get("models", {}),
            "has_api_key": bool(config.get("api_key")),
        }

    return {
        "default_provider": llm_config.get("default_provider", "groq"),
        "default_temperature": llm_config.get("default_temperature", 0.1),
        "default_max_tokens": llm_config.get("default_max_tokens", 2000),
        "providers": providers,
    }


@app.put("/config/llm")
async def update_llm_config(request: LLMConfigUpdate):
    """
    Update LLM configuration.

    Updates are saved to settings.yaml and hot-reloaded.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()
    updated = []

    # Update default provider
    if request.default_provider is not None:
        manager.update_and_save("llm.default_provider", request.default_provider)
        updated.append("default_provider")

    # Update temperature
    if request.default_temperature is not None:
        manager.update_and_save("llm.default_temperature", request.default_temperature)
        updated.append("default_temperature")

    # Update max tokens
    if request.default_max_tokens is not None:
        manager.update_and_save("llm.default_max_tokens", request.default_max_tokens)
        updated.append("default_max_tokens")

    # Update provider settings
    if request.providers:
        for provider_name, provider_update in request.providers.items():
            if provider_update.enabled is not None:
                manager.update_and_save(f"llm.providers.{provider_name}.enabled", provider_update.enabled)
                updated.append(f"providers.{provider_name}.enabled")
            if provider_update.default_model is not None:
                manager.update_and_save(f"llm.providers.{provider_name}.default_model", provider_update.default_model)
                updated.append(f"providers.{provider_name}.default_model")

    return {
        "success": True,
        "updated": updated,
        "message": f"Updated {len(updated)} settings"
    }


@app.get("/config/data-sources")
async def get_data_sources_config():
    """
    Get data sources configuration.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()
    data_sources = manager.get_data_sources_config()

    return {
        "data_sources": data_sources
    }


@app.put("/config/data-sources/{source_id}")
async def update_data_source(source_id: str, request: DataSourceUpdate):
    """
    Update a data source configuration.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()
    updated = []

    if request.enabled is not None:
        manager.update_and_save(f"data_sources.{source_id}.enabled", request.enabled)
        updated.append("enabled")

    if request.settings:
        for key, value in request.settings.items():
            manager.update_and_save(f"data_sources.{source_id}.{key}", value)
            updated.append(key)

    return {
        "success": True,
        "source_id": source_id,
        "updated": updated
    }


@app.get("/config/credentials")
async def get_credentials_config():
    """
    Get credentials status (masked values, never actual secrets).

    Returns which credentials are set and their masked values.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    credentials = get_credential_status()
    categories = get_credential_categories()

    return {
        "credentials": credentials,
        "categories": categories
    }


@app.put("/config/credentials/{credential_id}")
async def update_credential(credential_id: str, request: CredentialUpdate):
    """
    Update a credential (writes to .env file).

    The value is written to the .env file and the environment is updated.
    The actual value is NEVER returned in the response.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    # Map credential ID to env var name
    credential_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google_ai": "GOOGLE_API_KEY",
        "finnhub": "FINNHUB_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "courtlistener": "COURTLISTENER_API_KEY",
        "opencorporates": "OPENCORPORATES_API_KEY",
        "parallel_ai": "PARALLEL_API_KEY",
        "sec_edgar": "SEC_EDGAR_USER_AGENT",
        "mongodb": "MONGODB_URI",
        "google_sheets_creds": "GOOGLE_CREDENTIALS_PATH",
        "google_sheets_id": "GOOGLE_SPREADSHEET_ID",
        "langchain": "LANGCHAIN_API_KEY",
        "langfuse_public": "LANGFUSE_PUBLIC_KEY",
        "langfuse_secret": "LANGFUSE_SECRET_KEY",
    }

    if credential_id not in credential_map:
        raise HTTPException(status_code=400, detail=f"Unknown credential: {credential_id}")

    env_key = credential_map[credential_id]
    success = update_env_file(env_key, request.value)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update credential")

    # Get updated status (masked)
    updated_status = get_credential_status().get(credential_id, {})

    return {
        "success": True,
        "credential_id": credential_id,
        "env_key": env_key,
        "is_set": updated_status.get("is_set", False),
        "masked": updated_status.get("masked"),
    }


@app.get("/config/runtime")
async def get_runtime_config():
    """
    Get runtime configuration settings.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()
    runtime = manager.get_runtime_config()

    return {
        "runtime": runtime,
        "hot_reload_active": manager._watching,
        "config_path": str(manager.config_path),
    }


@app.put("/config/runtime")
async def update_runtime_config(request: RuntimeConfigUpdate):
    """
    Update runtime configuration settings.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()
    updated = []

    if request.hot_reload is not None:
        manager.update_and_save("runtime.hot_reload", request.hot_reload)
        updated.append("hot_reload")

        # Apply change immediately
        if request.hot_reload and not manager._watching:
            manager.start_watching()
        elif not request.hot_reload and manager._watching:
            manager.stop_watching()

    if request.watch_interval_seconds is not None:
        manager.update_and_save("runtime.watch_interval_seconds", request.watch_interval_seconds)
        updated.append("watch_interval_seconds")

    if request.cache_enabled is not None:
        manager.update_and_save("runtime.cache.enabled", request.cache_enabled)
        updated.append("cache_enabled")

    if request.cache_ttl_seconds is not None:
        manager.update_and_save("runtime.cache.ttl_seconds", request.cache_ttl_seconds)
        updated.append("cache_ttl_seconds")

    return {
        "success": True,
        "updated": updated
    }


@app.post("/config/reload")
async def force_reload_config():
    """
    Force reload configuration from file.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    success = reload_config()

    return {
        "success": success,
        "message": "Configuration reloaded" if success else "No changes detected or reload failed"
    }


@app.get("/config/status")
async def get_config_status():
    """
    Get configuration system status.
    """
    if not CONFIG_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration manager not available")

    manager = get_config_manager()

    return {
        "config_path": str(manager.config_path),
        "config_exists": manager.config_path.exists(),
        "last_modified": datetime.fromtimestamp(manager._last_modified).isoformat() if manager._last_modified else None,
        "hot_reload_enabled": manager._watching,
        "callbacks_registered": len(manager._callbacks),
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time workflow updates.

    Connect to receive:
    - workflow_started: Initial workflow state
    - step_update: Each step completion
    - workflow_completed: Final results
    - workflow_failed: Error information
    """
    await manager.connect(websocket, run_id)

    try:
        # Send current state if run exists
        if run_id in active_runs:
            await websocket.send_json({
                "type": "current_state",
                "data": active_runs[run_id].model_dump()
            })

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for messages from client (ping/pong)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, run_id)


# =============================================================================
# STATIC FRONTEND SERVING
# =============================================================================

# Serve static frontend files (Next.js export)
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "out"

if FRONTEND_DIR.exists():
    logger.info(f"Serving frontend from: {FRONTEND_DIR}")

    # Mount static assets (_next folder)
    next_static = FRONTEND_DIR / "_next"
    if next_static.exists():
        app.mount("/_next", StaticFiles(directory=str(next_static)), name="next-static")

    # Serve other static files
    @app.get("/favicon.ico")
    async def favicon():
        favicon_path = FRONTEND_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(str(favicon_path))
        raise HTTPException(status_code=404)

    # Catch-all route for frontend pages (must be last!)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend static files."""
        # Try exact path first
        file_path = FRONTEND_DIR / full_path

        # If path ends with / or has no extension, try index.html
        if file_path.is_dir() or not full_path or '.' not in full_path.split('/')[-1]:
            index_path = FRONTEND_DIR / full_path / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            # Fallback to root index.html for SPA routing
            root_index = FRONTEND_DIR / "index.html"
            if root_index.exists():
                return FileResponse(str(root_index))

        # Serve file directly if it exists
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        # 404 for non-existent files
        raise HTTPException(status_code=404, detail="Not found")
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_DIR}")
    logger.warning("Run 'npm run build' in frontend/ to build static files")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
