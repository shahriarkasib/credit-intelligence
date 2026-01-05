"""
LangGraph Event Logger - Captures and logs all LangGraph events.

Logs events to:
- Google Sheets (langgraph_events sheet)
- MongoDB (langgraph_events collection)

Captures:
- on_chain_start / on_chain_end / on_chain_stream
- on_chat_model_start / on_chat_model_stream / on_chat_model_end
- on_tool_start / on_tool_end
- on_custom_event
- Graph node enter/exit
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Import SheetsLogger
try:
    from run_logging.sheets_logger import SheetsLogger, get_sheets_logger
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    SheetsLogger = None  # Define as None for type hints
    get_sheets_logger = None
    logger.warning("SheetsLogger not available for LangGraph logging")

# Import MongoDB
try:
    from storage.mongodb import CreditIntelligenceDB, get_db
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    CreditIntelligenceDB = None  # Define as None for type hints
    get_db = None
    logger.warning("MongoDB not available for LangGraph logging")


@dataclass
class LangGraphEvent:
    """Represents a single LangGraph event."""
    run_id: str
    company_name: str
    event_type: str  # on_chain_start, on_chat_model_stream, etc.
    event_name: str  # Name of the chain/node/model
    parent_ids: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Current graph node (e.g., parse_input, validate_company, etc.)
    node: str = ""
    # Node type: "agent", "tool", "llm", or "chain"
    node_type: str = ""

    # Event data
    input_data: Optional[str] = None
    output_data: Optional[str] = None

    # Timing
    duration_ms: Optional[float] = None

    # For LLM events
    model: Optional[str] = None
    tokens: Optional[int] = None

    # For streaming
    chunk_content: Optional[str] = None
    chunk_index: Optional[int] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "started"  # started, streaming, completed, error
    error: Optional[str] = None


class LangGraphEventLogger:
    """
    Logger for LangGraph events.

    Captures events from astream_events() and logs them to
    Google Sheets and MongoDB.

    Usage:
        from run_logging.langgraph_logger import LangGraphEventLogger

        logger = LangGraphEventLogger(
            run_id="abc123",
            company_name="Apple Inc",
            log_to_sheets=True,
            log_to_mongodb=True,
        )

        async for event in graph.astream_events(input_state, version="v2"):
            logger.log_event(event)

        logger.flush()  # Ensure all events are written
    """

    def __init__(
        self,
        run_id: str,
        company_name: str,
        log_to_sheets: bool = True,
        log_to_mongodb: bool = True,
    ):
        self.run_id = run_id
        self.company_name = company_name
        self.log_to_sheets = log_to_sheets and SHEETS_AVAILABLE
        self.log_to_mongodb = log_to_mongodb and MONGODB_AVAILABLE

        # Event buffer for batch logging
        self._event_buffer: List[LangGraphEvent] = []
        self._buffer_size = 50  # Flush every N events (increased to reduce API calls)

        # Track event timing
        self._event_starts: Dict[str, float] = {}

        # Track current graph node (for node column in langgraph_events_2)
        self._current_node: str = ""
        self._node_stack: List[str] = []  # Stack for nested nodes

        # Initialize loggers
        self._sheets_logger: Optional[Any] = None  # SheetsLogger when available
        self._mongodb: Optional[Any] = None  # CreditIntelligenceDB when available

        # Streaming token buffer
        self._token_buffer: Dict[str, List[str]] = {}

        logger.info(f"LangGraphEventLogger initialized: run_id={run_id}, sheets={self.log_to_sheets}, mongodb={self.log_to_mongodb}")

    def _get_sheets_logger(self) -> Optional[Any]:
        """Get or create SheetsLogger instance."""
        if not self.log_to_sheets:
            return None
        if self._sheets_logger is None:
            try:
                self._sheets_logger = get_sheets_logger()
            except Exception as e:
                logger.warning(f"Failed to get SheetsLogger: {e}")
        return self._sheets_logger

    def _get_mongodb(self) -> Optional[Any]:
        """Get or create MongoDB client."""
        if not self.log_to_mongodb:
            return None
        if self._mongodb is None:
            try:
                self._mongodb = get_db()
            except Exception as e:
                logger.warning(f"Failed to get MongoDB: {e}")
        return self._mongodb

    def log_event(self, event: Dict[str, Any]) -> None:
        """
        Log a single LangGraph event.

        Args:
            event: Event dict from astream_events()
        """
        try:
            event_type = event.get("event", "unknown")
            event_name = event.get("name", "unknown")
            run_id = str(event.get("run_id", self.run_id))
            parent_ids = [str(pid) for pid in event.get("parent_ids", [])]
            tags = event.get("tags", [])
            metadata = event.get("metadata", {})
            data = event.get("data", {})

            # Create event object
            lg_event = LangGraphEvent(
                run_id=self.run_id,
                company_name=self.company_name,
                event_type=event_type,
                event_name=event_name,
                parent_ids=parent_ids,
                tags=tags,
                metadata=metadata,
            )

            # SKIP streaming events - they flood the logs (one per token)
            if event_type in ("on_chain_stream", "on_chat_model_stream", "on_llm_stream"):
                return  # Skip streaming events entirely

            # Known workflow nodes (graph nodes we want to track)
            GRAPH_NODES = {
                "parse_input", "validate_company", "create_plan",
                "fetch_api_data", "search_web", "synthesize", "save_to_database",
                "evaluate", "evaluate_assessment", "should_continue_after_validation",
                "human_review",
            }

            # Known workflow nodes to log (skip internal LangChain chains)
            WORKFLOW_NODES = GRAPH_NODES | {
                "LangGraph", "credit_intelligence_workflow",
                # Common LangGraph/LangChain patterns
                "RunnableSequence", "RunnableLambda", "ChatGroq", "PromptTemplate",
            }

            # Check if this is a graph node (for node tracking)
            def is_graph_node(name: str) -> bool:
                return name in GRAPH_NODES

            # Also log if event_name contains any workflow node name (handles prefixes/suffixes)
            def is_workflow_event(name: str) -> bool:
                if name in WORKFLOW_NODES:
                    return True
                # Check if any workflow node is in the name
                for node in ["parse_input", "validate", "create_plan", "fetch", "search",
                             "synthesize", "save", "evaluate"]:
                    if node in name.lower():
                        return True
                return False

            # Track node transitions for the "node" column
            if event_type == "on_chain_start" and is_graph_node(event_name):
                self._node_stack.append(event_name)
                self._current_node = event_name

            # Set the current node on the event BEFORE popping (so chain_end gets the node)
            lg_event.node = self._current_node

            # Determine node_type based on event_type
            if event_type in ("on_tool_start", "on_tool_end"):
                lg_event.node_type = "tool"
            elif event_type in ("on_chat_model_start", "on_chat_model_end", "on_llm_start", "on_llm_end"):
                lg_event.node_type = "llm"
            elif event_type in ("on_chain_start", "on_chain_end"):
                # Check if this is a known agent node or just a chain
                if is_graph_node(event_name):
                    lg_event.node_type = "agent"
                else:
                    lg_event.node_type = "chain"
            elif event_type in ("on_retriever_start", "on_retriever_end"):
                lg_event.node_type = "tool"
            else:
                lg_event.node_type = "other"

            # Pop node stack AFTER setting the event's node
            if event_type == "on_chain_end" and is_graph_node(event_name):
                if self._node_stack and self._node_stack[-1] == event_name:
                    self._node_stack.pop()
                    self._current_node = self._node_stack[-1] if self._node_stack else ""

            # Handle different event types (both LangChain naming conventions)
            if event_type == "on_chain_start":
                # Only log known workflow nodes, skip internal chains
                if not is_workflow_event(event_name):
                    return
                self._handle_chain_start(lg_event, data)
            elif event_type == "on_chain_end":
                if not is_workflow_event(event_name):
                    return
                self._handle_chain_end(lg_event, data, run_id)
            # LLM events - handle both naming conventions
            elif event_type in ("on_chat_model_start", "on_llm_start"):
                self._handle_llm_start(lg_event, data)
            elif event_type in ("on_chat_model_end", "on_llm_end"):
                self._handle_llm_end(lg_event, data, run_id)
            # Tool events
            elif event_type == "on_tool_start":
                self._handle_tool_start(lg_event, data)
            elif event_type == "on_tool_end":
                self._handle_tool_end(lg_event, data, run_id)
            # Retriever events
            elif event_type == "on_retriever_start":
                self._handle_tool_start(lg_event, data)
            elif event_type == "on_retriever_end":
                self._handle_tool_end(lg_event, data, run_id)
            # Parser events
            elif event_type == "on_parser_start":
                self._handle_chain_start(lg_event, data)
            elif event_type == "on_parser_end":
                self._handle_chain_end(lg_event, data, run_id)
            # Custom events
            elif event_type == "on_custom_event":
                self._handle_custom_event(lg_event, data)
            else:
                # Skip unknown event types to avoid clutter
                return

            # Add to buffer
            self._event_buffer.append(lg_event)

            # Flush if buffer is full
            if len(self._event_buffer) >= self._buffer_size:
                self.flush()

        except Exception as e:
            logger.error(f"Error logging LangGraph event: {e}")

    def _handle_chain_start(self, event: LangGraphEvent, data: Dict) -> None:
        """Handle chain start event."""
        event.status = "started"
        event.input_data = self._truncate(str(data.get("input", {})))
        self._event_starts[event.event_name] = time.time()

    def _handle_chain_end(self, event: LangGraphEvent, data: Dict, run_id: str) -> None:
        """Handle chain end event."""
        event.status = "completed"

        # Extract meaningful output summary
        output = data.get("output", {})
        if isinstance(output, dict):
            # Extract key fields for summary
            summary_parts = []
            if "status" in output:
                summary_parts.append(f"status={output['status']}")
            if "risk_level" in output:
                summary_parts.append(f"risk={output['risk_level']}")
            if "credit_score" in output:
                summary_parts.append(f"score={output['credit_score']}")
            if "company_info" in output and isinstance(output["company_info"], dict):
                info = output["company_info"]
                if "is_public_company" in info:
                    summary_parts.append(f"public={info['is_public_company']}")
            if summary_parts:
                event.output_data = ", ".join(summary_parts)
            else:
                event.output_data = self._truncate(str(output))
        else:
            event.output_data = self._truncate(str(output))

        # Calculate duration
        start_time = self._event_starts.pop(event.event_name, None)
        if start_time:
            event.duration_ms = (time.time() - start_time) * 1000

    def _handle_chain_stream(self, event: LangGraphEvent, data: Dict) -> None:
        """Handle chain stream event."""
        event.status = "streaming"
        chunk = data.get("chunk", "")
        event.chunk_content = self._truncate(str(chunk))

    def _handle_llm_start(self, event: LangGraphEvent, data: Dict) -> None:
        """Handle LLM start event."""
        event.status = "started"

        # Model name is in event.metadata.ls_model_name (from LangSmith metadata)
        # This is passed to the handler via the event object's metadata field
        model = event.metadata.get("ls_model_name") if event.metadata else None

        # Fallback: try data dict
        if not model:
            invocation_params = data.get("invocation_params", {})
            model = invocation_params.get("model") or invocation_params.get("model_name")

        if not model:
            kwargs = data.get("kwargs", {})
            model = kwargs.get("model") or kwargs.get("model_name")

        event.model = model or "unknown"

        messages = data.get("messages", [])
        if messages:
            event.input_data = self._truncate(str(messages))
        self._event_starts[f"llm_{event.event_name}"] = time.time()
        self._token_buffer[event.event_name] = []

    def _handle_llm_stream(self, event: LangGraphEvent, data: Dict, run_id: str) -> None:
        """Handle LLM stream event (token by token)."""
        event.status = "streaming"
        chunk = data.get("chunk", {})

        # Extract token content
        content = ""
        if hasattr(chunk, "content"):
            content = chunk.content
        elif isinstance(chunk, dict):
            content = chunk.get("content", "")

        event.chunk_content = content

        # Buffer tokens
        if event.event_name in self._token_buffer:
            self._token_buffer[event.event_name].append(content)
            event.chunk_index = len(self._token_buffer[event.event_name])

    def _handle_llm_end(self, event: LangGraphEvent, data: Dict, run_id: str) -> None:
        """Handle LLM end event."""
        event.status = "completed"

        # Model name from metadata
        event.model = event.metadata.get("ls_model_name") if event.metadata else None

        # Extract output - it's usually an AIMessage with .content
        output = data.get("output")
        if output:
            # AIMessage has .content attribute
            if hasattr(output, "content"):
                event.output_data = self._truncate(str(output.content))
            # Or it could be a dict
            elif isinstance(output, dict):
                event.output_data = self._truncate(str(output.get("content", output)))
            # Or just stringify
            else:
                event.output_data = self._truncate(str(output))

        # Token usage from response metadata
        if hasattr(output, "response_metadata"):
            usage = output.response_metadata.get("token_usage", {})
            event.tokens = usage.get("total_tokens", 0)

        # Calculate duration
        start_time = self._event_starts.pop(f"llm_{event.event_name}", None)
        if start_time:
            event.duration_ms = (time.time() - start_time) * 1000

    def _handle_tool_start(self, event: LangGraphEvent, data: Dict) -> None:
        """Handle tool start event."""
        event.status = "started"
        event.input_data = self._truncate(str(data.get("input", "")))
        self._event_starts[f"tool_{event.event_name}"] = time.time()

    def _handle_tool_end(self, event: LangGraphEvent, data: Dict, run_id: str) -> None:
        """Handle tool end event."""
        event.status = "completed"
        event.output_data = self._truncate(str(data.get("output", "")))

        # Calculate duration
        start_time = self._event_starts.pop(f"tool_{event.event_name}", None)
        if start_time:
            event.duration_ms = (time.time() - start_time) * 1000

    def _handle_custom_event(self, event: LangGraphEvent, data: Dict) -> None:
        """Handle custom event."""
        event.status = "custom"
        event.input_data = self._truncate(json.dumps(data, default=str))

    def _truncate(self, text: str, max_length: int = 2000) -> str:
        """Truncate text to max length."""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def flush(self) -> None:
        """Flush event buffer to storage."""
        if not self._event_buffer:
            logger.debug("[LangGraphLogger] flush called but buffer is empty")
            return

        events_to_write = self._event_buffer.copy()
        self._event_buffer.clear()
        logger.info(f"[LangGraphLogger] Flushing {len(events_to_write)} events (sheets={self.log_to_sheets}, mongodb={self.log_to_mongodb})")

        # Write to Google Sheets
        if self.log_to_sheets:
            self._write_to_sheets(events_to_write)

        # Write to MongoDB
        if self.log_to_mongodb:
            self._write_to_mongodb(events_to_write)

    def _write_to_sheets(self, events: List[LangGraphEvent]) -> None:
        """Write events to Google Sheets using batch append."""
        sheets_logger = self._get_sheets_logger()
        if not sheets_logger:
            logger.warning("[LangGraphLogger] No sheets_logger available")
            return
        if not sheets_logger.is_connected():
            logger.warning("[LangGraphLogger] sheets_logger not connected")
            return

        try:
            # Prepare rows for langgraph_events (original format)
            rows_v1 = []
            # Prepare rows for langgraph_events_3 (with node and node_type columns)
            rows_v3 = []

            for event in events:
                # Original format (langgraph_events)
                row_v1 = [
                    event.run_id,
                    event.company_name,
                    event.event_type,
                    event.event_name,
                    event.status,
                    event.duration_ms if event.duration_ms is not None else "",
                    event.model or "",
                    event.tokens if event.tokens is not None else "",
                    (event.input_data[:500] if event.input_data else ""),
                    (event.output_data[:500] if event.output_data else ""),
                    event.error or "",
                    event.timestamp,
                ]
                rows_v1.append(row_v1)

                # New format with node and node_type columns (langgraph_events_3)
                row_v3 = [
                    event.run_id,
                    event.company_name,
                    event.node or "",  # node column
                    event.node_type or "",  # node_type column (agent/tool/llm/chain)
                    event.event_type,
                    event.event_name,
                    event.status,
                    event.duration_ms if event.duration_ms is not None else "",
                    event.model or "",
                    event.tokens if event.tokens is not None else "",
                    (event.input_data[:500] if event.input_data else ""),
                    (event.output_data[:500] if event.output_data else ""),
                    event.error or "",
                    event.timestamp,
                ]
                rows_v3.append(row_v3)

            # Write to langgraph_events (original)
            if rows_v1:
                sheet = sheets_logger._get_sheet("langgraph_events")
                if sheet:
                    sheet.append_rows(rows_v1)
                    logger.debug(f"Batch wrote {len(rows_v1)} events to langgraph_events")

            # Write to langgraph_events_3 (with node and node_type columns)
            if rows_v3:
                sheet3 = sheets_logger._get_sheet("langgraph_events_3")
                if sheet3:
                    sheet3.append_rows(rows_v3)
                    logger.debug(f"Batch wrote {len(rows_v3)} events to langgraph_events_3")
        except Exception as e:
            logger.warning(f"Failed to write LangGraph events to Sheets: {e}")

    def _write_to_mongodb(self, events: List[LangGraphEvent]) -> None:
        """Write events to MongoDB."""
        mongodb = self._get_mongodb()
        if not mongodb or not mongodb.is_connected():
            return

        try:
            # Convert events to dicts and insert
            event_dicts = []
            for event in events:
                event_dict = asdict(event)
                event_dict["logged_at"] = datetime.utcnow()
                event_dicts.append(event_dict)

            if event_dicts:
                mongodb.db.langgraph_events.insert_many(event_dicts)
                logger.debug(f"Wrote {len(event_dicts)} LangGraph events to MongoDB")
        except Exception as e:
            logger.warning(f"Failed to write LangGraph events to MongoDB: {e}")

    def log_node_enter(self, node_name: str, input_state: Dict[str, Any]) -> None:
        """Log when entering a graph node."""
        event = LangGraphEvent(
            run_id=self.run_id,
            company_name=self.company_name,
            event_type="node_enter",
            event_name=node_name,
            status="started",
            input_data=self._truncate(json.dumps(input_state, default=str)),
        )
        self._event_starts[f"node_{node_name}"] = time.time()
        self._event_buffer.append(event)

    def log_node_exit(self, node_name: str, output_state: Dict[str, Any], error: Optional[str] = None) -> None:
        """Log when exiting a graph node."""
        duration_ms = None
        start_time = self._event_starts.pop(f"node_{node_name}", None)
        if start_time:
            duration_ms = (time.time() - start_time) * 1000

        event = LangGraphEvent(
            run_id=self.run_id,
            company_name=self.company_name,
            event_type="node_exit",
            event_name=node_name,
            status="completed" if not error else "error",
            output_data=self._truncate(json.dumps(output_state, default=str)),
            duration_ms=duration_ms,
            error=error,
        )
        self._event_buffer.append(event)

        # Always flush on node exit
        self.flush()

    def log_graph_start(self, input_state: Dict[str, Any]) -> None:
        """Log graph execution start."""
        logger.info(f"[LangGraphLogger] graph_start for {self.company_name}, run_id={self.run_id}")
        event = LangGraphEvent(
            run_id=self.run_id,
            company_name=self.company_name,
            event_type="graph_start",
            event_name="LangGraph",
            status="started",
            input_data=self._truncate(json.dumps(input_state, default=str)),
        )
        self._event_starts["graph"] = time.time()
        self._event_buffer.append(event)
        self.flush()
        logger.info(f"[LangGraphLogger] graph_start flushed, sheets={self.log_to_sheets}")

    def log_graph_end(self, output_state: Dict[str, Any], error: Optional[str] = None) -> None:
        """Log graph execution end."""
        duration_ms = None
        start_time = self._event_starts.pop("graph", None)
        if start_time:
            duration_ms = (time.time() - start_time) * 1000

        event = LangGraphEvent(
            run_id=self.run_id,
            company_name=self.company_name,
            event_type="graph_end",
            event_name="LangGraph",
            status="completed" if not error else "error",
            output_data=self._truncate(json.dumps(output_state, default=str)),
            duration_ms=duration_ms,
            error=error,
        )
        self._event_buffer.append(event)
        self.flush()


async def run_graph_with_event_logging(
    graph,
    input_state: Dict[str, Any],
    run_id: str,
    company_name: str,
    log_to_sheets: bool = True,
    log_to_mongodb: bool = True,
) -> Dict[str, Any]:
    """
    Run a LangGraph with full event logging.

    Uses astream_events() to capture all events and logs them
    to Google Sheets and MongoDB.

    Args:
        graph: Compiled LangGraph
        input_state: Initial state
        run_id: Run ID for tracking
        company_name: Company being analyzed
        log_to_sheets: Whether to log to Sheets
        log_to_mongodb: Whether to log to MongoDB

    Returns:
        Final state from graph execution
    """
    event_logger = LangGraphEventLogger(
        run_id=run_id,
        company_name=company_name,
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    )

    event_logger.log_graph_start(input_state)

    final_state = None
    error = None

    try:
        async for event in graph.astream_events(input_state, version="v2"):
            event_logger.log_event(event)

            # Capture final state from last chain_end event
            if event.get("event") == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final_state = output

        event_logger.log_graph_end(final_state or {})

    except Exception as e:
        error = str(e)
        logger.error(f"Graph execution error: {e}")
        event_logger.log_graph_end({}, error=error)
        raise

    finally:
        event_logger.flush()

    return final_state or {}


def get_langgraph_logger(
    run_id: str,
    company_name: str,
    log_to_sheets: bool = True,
    log_to_mongodb: bool = True,
) -> LangGraphEventLogger:
    """
    Factory function to create a LangGraphEventLogger.

    Args:
        run_id: Run ID for tracking
        company_name: Company being analyzed
        log_to_sheets: Whether to log to Sheets
        log_to_mongodb: Whether to log to MongoDB

    Returns:
        LangGraphEventLogger instance
    """
    return LangGraphEventLogger(
        run_id=run_id,
        company_name=company_name,
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    )
