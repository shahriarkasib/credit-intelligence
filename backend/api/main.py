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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CompanyRequest(BaseModel):
    """Request to analyze a company."""
    company_name: str
    jurisdiction: Optional[str] = None
    ticker: Optional[str] = None


class WorkflowStep(BaseModel):
    """A single step in the workflow."""
    step_id: str
    name: str
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
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")


manager = ConnectionManager()


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

    # Define workflow steps (matching the LangGraph nodes)
    step_definitions = [
        ("parse_input", "Parsing Input"),
        ("validate_company", "Validating Company"),
        ("create_plan", "Creating Plan"),
        ("fetch_api_data", "Fetching API Data"),
        ("search_web", "Searching Web"),
        ("synthesize", "Synthesizing Assessment"),
        ("save_to_database", "Saving to Database"),
        ("evaluate", "Evaluating Results"),
    ]

    # Initialize all steps as pending
    for step_id, step_name in step_definitions:
        workflow_status.steps.append(WorkflowStep(
            step_id=step_id,
            name=step_name,
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
                    "run_id": str(source.get("run_id", ""))[:8] + "...",
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
                    "task_plan": task_plan[:5],  # First 5 tasks
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
                        "reasoning": reasoning[:500] if reasoning else "",
                        "risk_factors": assessment.get("risk_factors", [])[:5],
                        "positive_factors": assessment.get("positive_factors", [])[:5],
                        "recommendations": assessment.get("recommendations", source.get("recommendations", []))[:5],
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
                        "assessment_raw": str(assessment)[:500] if assessment else "No assessment"
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

        def run_graph_streaming():
            """Run graph with streaming and put results in queue."""
            try:
                for event in graph.stream({
                    'company_name': company_name,
                    'jurisdiction': jurisdiction,
                    'ticker': ticker,
                }):
                    # event is a dict with node_name -> output
                    result_queue.put(("step", event))

                # Signal completion
                result_queue.put(("done", None))
            except Exception as e:
                result_queue.put(("error", str(e)))

        # Start graph in thread
        import threading
        graph_thread = threading.Thread(target=run_graph_streaming)
        graph_thread.start()

        # Process streaming results
        current_state = {}
        completed_steps = set()

        while True:
            try:
                # Wait for next event with timeout
                msg_type, data = result_queue.get(timeout=120)

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

            except queue.Empty:
                logger.warning("Timeout waiting for graph output")
                break

        # Wait for thread to finish
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Credit Intelligence API",
        "version": "1.0.0"
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

    # Start workflow in background
    asyncio.create_task(
        run_workflow_with_streaming(
            run_id=run_id,
            company_name=request.company_name,
            jurisdiction=request.jurisdiction,
            ticker=request.ticker,
        )
    )

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
