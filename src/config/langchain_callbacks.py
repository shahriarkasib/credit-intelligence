"""LangChain Callbacks for metrics integration.

Provides:
- CostTrackerCallback - integrates with existing CostTracker
- MetricsCollectorCallback - integrates with existing MetricsCollector
- Automatic token counting from LLM responses
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

logger = logging.getLogger(__name__)

# Try to import LangChain callback base
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    CALLBACKS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to older import path
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain_core.outputs import LLMResult
        CALLBACKS_AVAILABLE = True
    except ImportError:
        CALLBACKS_AVAILABLE = False
        BaseCallbackHandler = object
        LLMResult = None
        logger.warning("langchain callbacks not available")

# Import our cost tracker
try:
    from config.cost_tracker import CostTracker, TokenUsage, get_cost_tracker
    COST_TRACKER_AVAILABLE = True
except ImportError:
    COST_TRACKER_AVAILABLE = False
    logger.warning("CostTracker not available")


class CostTrackerCallback(BaseCallbackHandler):
    """
    LangChain callback that integrates with CostTracker.

    Automatically captures token usage from LLM responses and
    adds them to the CostTracker for cost calculation.

    Usage:
        from config.langchain_callbacks import CostTrackerCallback
        from config.cost_tracker import get_cost_tracker

        tracker = get_cost_tracker(run_id="abc123")
        callback = CostTrackerCallback(tracker, call_type="synthesize")

        llm = ChatGroq(callbacks=[callback])
        llm.invoke("Hello")

        # Tokens and cost automatically tracked in tracker
    """

    def __init__(
        self,
        tracker: Optional[Any] = None,
        call_type: str = "",
        provider: str = "groq",
    ):
        """
        Initialize callback.

        Args:
            tracker: CostTracker instance (or uses global if None)
            call_type: Type of call for categorization (e.g., "synthesize", "parse_input")
            provider: LLM provider for cost calculation
        """
        super().__init__()
        self.tracker = tracker
        self.call_type = call_type
        self.provider = provider
        self._start_time: Optional[float] = None
        self._model: str = ""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        self._start_time = time.time()
        # Try to extract model from serialized data
        self._model = serialized.get("kwargs", {}).get("model", "")
        if not self._model:
            self._model = serialized.get("id", ["", ""])[-1] if serialized.get("id") else ""
        logger.debug(f"LLM call started: model={self._model}, call_type={self.call_type}")

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes. Extract token usage and add to tracker."""
        if not COST_TRACKER_AVAILABLE:
            return

        # Get tracker (use provided or global)
        tracker = self.tracker or get_cost_tracker()
        if not tracker:
            return

        try:
            # Extract token usage from LLMResult
            if hasattr(response, 'llm_output') and response.llm_output:
                llm_output = response.llm_output
                token_usage = llm_output.get('token_usage', {})

                if token_usage:
                    tokens = TokenUsage(
                        prompt_tokens=token_usage.get('prompt_tokens', 0),
                        completion_tokens=token_usage.get('completion_tokens', 0),
                        total_tokens=token_usage.get('total_tokens', 0),
                    )

                    # Get model from response if not captured earlier
                    model = self._model or llm_output.get('model_name', 'unknown')

                    # Add to tracker
                    cost = tracker.add_call(
                        model=model,
                        tokens=tokens,
                        provider=self.provider,
                        call_type=self.call_type,
                    )

                    elapsed = time.time() - self._start_time if self._start_time else 0
                    logger.debug(
                        f"LLM call completed: {model} - "
                        f"{tokens.total_tokens} tokens - "
                        f"${cost.total_cost:.6f} - "
                        f"{elapsed*1000:.0f}ms"
                    )
            else:
                logger.debug("No token usage in LLM response")

        except Exception as e:
            logger.warning(f"Failed to extract token usage: {e}")

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.error(f"LLM call failed after {elapsed*1000:.0f}ms: {error}")


class MetricsCollectorCallback(BaseCallbackHandler):
    """
    LangChain callback that integrates with MetricsCollector.

    Tracks timing and step information for workflow logging.

    Usage:
        from config.langchain_callbacks import MetricsCollectorCallback
        from run_logging import MetricsCollector

        metrics = MetricsCollector()
        callback = MetricsCollectorCallback(metrics, step_name="synthesize")

        llm = ChatGroq(callbacks=[callback])
        llm.invoke("Hello")
    """

    def __init__(
        self,
        collector: Optional[Any] = None,
        step_name: str = "llm_call",
    ):
        """
        Initialize callback.

        Args:
            collector: MetricsCollector instance
            step_name: Name of the step for logging
        """
        super().__init__()
        self.collector = collector
        self.step_name = step_name
        self._start_time: Optional[float] = None

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        self._start_time = time.time()
        if self.collector and hasattr(self.collector, 'start_step'):
            self.collector.start_step(self.step_name)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes."""
        if self.collector and hasattr(self.collector, 'end_step'):
            # Extract token info for metrics
            tokens = 0
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                tokens = token_usage.get('total_tokens', 0)

            self.collector.end_step(self.step_name, tokens=tokens, success=True)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        if self.collector and hasattr(self.collector, 'end_step'):
            self.collector.end_step(self.step_name, tokens=0, success=False)


def create_cost_callback(
    run_id: str = "",
    call_type: str = "",
    provider: str = "groq",
) -> CostTrackerCallback:
    """
    Convenience function to create a CostTrackerCallback.

    Args:
        run_id: Run ID for the tracker
        call_type: Type of LLM call
        provider: LLM provider

    Returns:
        CostTrackerCallback configured with global tracker
    """
    tracker = get_cost_tracker(run_id=run_id) if COST_TRACKER_AVAILABLE else None
    return CostTrackerCallback(tracker=tracker, call_type=call_type, provider=provider)


def is_callbacks_available() -> bool:
    """Check if LangChain callbacks are available."""
    return CALLBACKS_AVAILABLE


# =============================================================================
# Sheets & MongoDB Logging Callback
# =============================================================================

# Import SheetsLogger
try:
    from run_logging.sheets_logger import SheetsLogger, get_sheets_logger
    SHEETS_LOGGER_AVAILABLE = True
except ImportError:
    SHEETS_LOGGER_AVAILABLE = False
    logger.warning("SheetsLogger not available")

# Import MongoDB
try:
    from storage.mongodb import CreditIntelligenceDB, get_db
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("MongoDB not available for logging")


class SheetsLoggingCallback(BaseCallbackHandler):
    """
    Comprehensive LangChain callback that logs all events to Google Sheets and MongoDB.

    Captures:
    - LLM Events: on_llm_start, on_llm_end, on_llm_error
    - Chain Events: on_chain_start, on_chain_end, on_chain_error
    - Tool Events: on_tool_start, on_tool_end, on_tool_error

    Usage:
        from config.langchain_callbacks import SheetsLoggingCallback

        callback = SheetsLoggingCallback(
            run_id="abc123",
            company_name="Apple Inc",
            log_to_sheets=True,
            log_to_mongodb=True,
        )

        llm = ChatGroq(callbacks=[callback])
        llm.invoke("Hello")

        # All events automatically logged to Sheets and MongoDB
    """

    def __init__(
        self,
        run_id: str = "",
        company_name: str = "",
        log_to_sheets: bool = True,
        log_to_mongodb: bool = False,
        sheets_logger: Optional[Any] = None,
        mongodb_client: Optional[Any] = None,
    ):
        """
        Initialize the logging callback.

        Args:
            run_id: Run ID for tracking
            company_name: Company being analyzed
            log_to_sheets: Whether to log to Google Sheets
            log_to_mongodb: Whether to log to MongoDB
            sheets_logger: Optional SheetsLogger instance
            mongodb_client: Optional MongoDB client instance
        """
        super().__init__()
        self.run_id = run_id
        self.company_name = company_name
        self.log_to_sheets = log_to_sheets and SHEETS_LOGGER_AVAILABLE
        self.log_to_mongodb = log_to_mongodb and MONGODB_AVAILABLE

        # Initialize loggers
        self._sheets_logger = sheets_logger
        self._mongodb = mongodb_client

        # Event tracking
        self._llm_starts: Dict[str, Dict[str, Any]] = {}
        self._chain_starts: Dict[str, Dict[str, Any]] = {}
        self._tool_starts: Dict[str, Dict[str, Any]] = {}

    def _get_sheets_logger(self) -> Optional[Any]:
        """Get or create SheetsLogger instance."""
        if not self.log_to_sheets:
            return None
        if self._sheets_logger is None and SHEETS_LOGGER_AVAILABLE:
            try:
                self._sheets_logger = get_sheets_logger()
            except Exception as e:
                logger.warning(f"Failed to get SheetsLogger: {e}")
        return self._sheets_logger

    def _get_mongodb(self) -> Optional[Any]:
        """Get or create MongoDB client."""
        if not self.log_to_mongodb:
            return None
        if self._mongodb is None and MONGODB_AVAILABLE:
            try:
                self._mongodb = get_db()
            except Exception as e:
                logger.warning(f"Failed to get MongoDB: {e}")
        return self._mongodb

    # ==================== LLM EVENTS ====================

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        run_id_str = str(run_id)
        model = serialized.get("kwargs", {}).get("model", "")
        if not model:
            model = serialized.get("id", ["", ""])[-1] if serialized.get("id") else "unknown"

        self._llm_starts[run_id_str] = {
            "start_time": time.time(),
            "model": model,
            "prompt": prompts[0] if prompts else "",
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

        logger.debug(f"LLM started: run_id={run_id_str}, model={model}")

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes."""
        run_id_str = str(run_id)
        start_info = self._llm_starts.pop(run_id_str, {})
        start_time = start_info.get("start_time", time.time())
        execution_time_ms = (time.time() - start_time) * 1000

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)

        # Extract response text
        response_text = ""
        if hasattr(response, 'generations') and response.generations:
            if response.generations[0]:
                gen = response.generations[0][0]
                response_text = gen.text if hasattr(gen, 'text') else str(gen)

        model = start_info.get("model", "unknown")
        prompt = start_info.get("prompt", "")

        # Log to Google Sheets
        sheets_logger = self._get_sheets_logger()
        if sheets_logger:
            try:
                sheets_logger.log_llm_call(
                    run_id=self.run_id or run_id_str,
                    company_name=self.company_name,
                    call_type="langchain_llm",
                    model=model,
                    prompt=prompt[:1000],  # Truncate
                    response=response_text[:2000],  # Truncate
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    execution_time_ms=round(execution_time_ms, 2),
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM call to Sheets: {e}")

        # Log to MongoDB
        mongodb = self._get_mongodb()
        if mongodb and mongodb.is_connected():
            try:
                mongodb.db.langchain_events.insert_one({
                    "event_type": "llm_end",
                    "run_id": self.run_id or run_id_str,
                    "llm_run_id": run_id_str,
                    "company_name": self.company_name,
                    "model": model,
                    "prompt": prompt[:2000],
                    "response": response_text[:5000],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "execution_time_ms": round(execution_time_ms, 2),
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Failed to log LLM call to MongoDB: {e}")

        logger.debug(f"LLM completed: {model}, {total_tokens} tokens, {execution_time_ms:.0f}ms")

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        run_id_str = str(run_id)
        start_info = self._llm_starts.pop(run_id_str, {})
        start_time = start_info.get("start_time", time.time())
        execution_time_ms = (time.time() - start_time) * 1000

        error_msg = str(error)

        # Log to Google Sheets
        sheets_logger = self._get_sheets_logger()
        if sheets_logger:
            try:
                sheets_logger.log_llm_call(
                    run_id=self.run_id or run_id_str,
                    company_name=self.company_name,
                    call_type="langchain_llm_error",
                    model=start_info.get("model", "unknown"),
                    prompt=start_info.get("prompt", "")[:1000],
                    response=f"ERROR: {error_msg}",
                    prompt_tokens=0,
                    completion_tokens=0,
                    execution_time_ms=round(execution_time_ms, 2),
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM error to Sheets: {e}")

        # Log to MongoDB
        mongodb = self._get_mongodb()
        if mongodb and mongodb.is_connected():
            try:
                mongodb.db.langchain_events.insert_one({
                    "event_type": "llm_error",
                    "run_id": self.run_id or run_id_str,
                    "llm_run_id": run_id_str,
                    "company_name": self.company_name,
                    "model": start_info.get("model", "unknown"),
                    "error": error_msg,
                    "execution_time_ms": round(execution_time_ms, 2),
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Failed to log LLM error to MongoDB: {e}")

        logger.error(f"LLM error: {error_msg}")

    # ==================== CHAIN EVENTS ====================

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""
        run_id_str = str(run_id)
        # Handle None serialized (can happen with some chain types)
        if serialized is None:
            chain_name = "unknown_chain"
        else:
            chain_id = serialized.get("id", ["unknown"])
            chain_name = serialized.get("name", chain_id[-1] if chain_id else "unknown")

        self._chain_starts[run_id_str] = {
            "start_time": time.time(),
            "chain_name": chain_name,
            "inputs": inputs,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

        logger.debug(f"Chain started: {chain_name}, run_id={run_id_str}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain finishes."""
        run_id_str = str(run_id)
        start_info = self._chain_starts.pop(run_id_str, {})
        start_time = start_info.get("start_time", time.time())
        execution_time_ms = (time.time() - start_time) * 1000

        chain_name = start_info.get("chain_name", "unknown")
        inputs = start_info.get("inputs", {})

        # Log to Google Sheets (using langsmith_traces sheet)
        sheets_logger = self._get_sheets_logger()
        if sheets_logger:
            try:
                sheets_logger.log_langsmith_trace(
                    run_id=self.run_id or run_id_str,
                    company_name=self.company_name,
                    step_name=chain_name,
                    run_type="chain",
                    status="success",
                    latency_ms=round(execution_time_ms, 2),
                    error=None,
                    input_preview=str(inputs)[:500],
                    output_preview=str(outputs)[:500],
                )
            except Exception as e:
                logger.warning(f"Failed to log chain to Sheets: {e}")

        # Log to MongoDB
        mongodb = self._get_mongodb()
        if mongodb and mongodb.is_connected():
            try:
                mongodb.db.langchain_events.insert_one({
                    "event_type": "chain_end",
                    "run_id": self.run_id or run_id_str,
                    "chain_run_id": run_id_str,
                    "company_name": self.company_name,
                    "chain_name": chain_name,
                    "inputs": str(inputs)[:2000],
                    "outputs": str(outputs)[:2000],
                    "execution_time_ms": round(execution_time_ms, 2),
                    "status": "success",
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Failed to log chain to MongoDB: {e}")

        logger.debug(f"Chain completed: {chain_name}, {execution_time_ms:.0f}ms")

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        run_id_str = str(run_id)
        start_info = self._chain_starts.pop(run_id_str, {})
        start_time = start_info.get("start_time", time.time())
        execution_time_ms = (time.time() - start_time) * 1000

        chain_name = start_info.get("chain_name", "unknown")
        error_msg = str(error)

        # Log to Google Sheets
        sheets_logger = self._get_sheets_logger()
        if sheets_logger:
            try:
                sheets_logger.log_langsmith_trace(
                    run_id=self.run_id or run_id_str,
                    company_name=self.company_name,
                    step_name=chain_name,
                    run_type="chain",
                    status="error",
                    latency_ms=round(execution_time_ms, 2),
                    error=error_msg[:500],
                    input_preview=str(start_info.get("inputs", {}))[:500],
                    output_preview="",
                )
            except Exception as e:
                logger.warning(f"Failed to log chain error to Sheets: {e}")

        # Log to MongoDB
        mongodb = self._get_mongodb()
        if mongodb and mongodb.is_connected():
            try:
                mongodb.db.langchain_events.insert_one({
                    "event_type": "chain_error",
                    "run_id": self.run_id or run_id_str,
                    "chain_run_id": run_id_str,
                    "company_name": self.company_name,
                    "chain_name": chain_name,
                    "error": error_msg,
                    "execution_time_ms": round(execution_time_ms, 2),
                    "status": "error",
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Failed to log chain error to MongoDB: {e}")

        logger.error(f"Chain error: {chain_name} - {error_msg}")

    # ==================== TOOL EVENTS ====================

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        run_id_str = str(run_id)
        tool_name = serialized.get("name", "unknown")

        self._tool_starts[run_id_str] = {
            "start_time": time.time(),
            "tool_name": tool_name,
            "input": input_str,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

        logger.debug(f"Tool started: {tool_name}, run_id={run_id_str}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool finishes."""
        run_id_str = str(run_id)
        start_info = self._tool_starts.pop(run_id_str, {})
        start_time = start_info.get("start_time", time.time())
        execution_time_ms = (time.time() - start_time) * 1000

        tool_name = start_info.get("tool_name", "unknown")
        tool_input = start_info.get("input", "")

        # Log to Google Sheets
        sheets_logger = self._get_sheets_logger()
        if sheets_logger:
            try:
                sheets_logger.log_tool_call(
                    run_id=self.run_id or run_id_str,
                    company_name=self.company_name,
                    tool_name=tool_name,
                    tool_input=tool_input[:500],
                    tool_output=str(output)[:1000],
                    execution_time_ms=round(execution_time_ms, 2),
                    success=True,
                    error=None,
                )
            except Exception as e:
                logger.warning(f"Failed to log tool call to Sheets: {e}")

        # Log to MongoDB
        mongodb = self._get_mongodb()
        if mongodb and mongodb.is_connected():
            try:
                mongodb.db.langchain_events.insert_one({
                    "event_type": "tool_end",
                    "run_id": self.run_id or run_id_str,
                    "tool_run_id": run_id_str,
                    "company_name": self.company_name,
                    "tool_name": tool_name,
                    "input": tool_input[:2000],
                    "output": str(output)[:5000],
                    "execution_time_ms": round(execution_time_ms, 2),
                    "status": "success",
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Failed to log tool call to MongoDB: {e}")

        logger.debug(f"Tool completed: {tool_name}, {execution_time_ms:.0f}ms")

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        run_id_str = str(run_id)
        start_info = self._tool_starts.pop(run_id_str, {})
        start_time = start_info.get("start_time", time.time())
        execution_time_ms = (time.time() - start_time) * 1000

        tool_name = start_info.get("tool_name", "unknown")
        error_msg = str(error)

        # Log to Google Sheets
        sheets_logger = self._get_sheets_logger()
        if sheets_logger:
            try:
                sheets_logger.log_tool_call(
                    run_id=self.run_id or run_id_str,
                    company_name=self.company_name,
                    tool_name=tool_name,
                    tool_input=start_info.get("input", "")[:500],
                    tool_output="",
                    execution_time_ms=round(execution_time_ms, 2),
                    success=False,
                    error=error_msg[:500],
                )
            except Exception as e:
                logger.warning(f"Failed to log tool error to Sheets: {e}")

        # Log to MongoDB
        mongodb = self._get_mongodb()
        if mongodb and mongodb.is_connected():
            try:
                mongodb.db.langchain_events.insert_one({
                    "event_type": "tool_error",
                    "run_id": self.run_id or run_id_str,
                    "tool_run_id": run_id_str,
                    "company_name": self.company_name,
                    "tool_name": tool_name,
                    "error": error_msg,
                    "execution_time_ms": round(execution_time_ms, 2),
                    "status": "error",
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Failed to log tool error to MongoDB: {e}")

        logger.error(f"Tool error: {tool_name} - {error_msg}")


def create_sheets_logging_callback(
    run_id: str = "",
    company_name: str = "",
    log_to_sheets: bool = True,
    log_to_mongodb: bool = False,
) -> SheetsLoggingCallback:
    """
    Convenience function to create a SheetsLoggingCallback.

    Args:
        run_id: Run ID for tracking
        company_name: Company being analyzed
        log_to_sheets: Whether to log to Google Sheets
        log_to_mongodb: Whether to log to MongoDB

    Returns:
        SheetsLoggingCallback configured for logging
    """
    return SheetsLoggingCallback(
        run_id=run_id,
        company_name=company_name,
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    )


def is_sheets_logging_available() -> bool:
    """Check if Sheets logging is available."""
    return CALLBACKS_AVAILABLE and SHEETS_LOGGER_AVAILABLE
