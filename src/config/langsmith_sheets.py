"""LangSmith to Google Sheets Integration - Fetches traces and logs to sheets."""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import LangSmith client
try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_SDK_AVAILABLE = True
except ImportError:
    LANGSMITH_SDK_AVAILABLE = False
    LangSmithClient = None
    logger.warning("langsmith SDK not installed")

# Import sheets logger
try:
    from run_logging.sheets_logger import get_sheets_logger
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    logger.warning("Sheets logger not available")


class LangSmithSheetsLogger:
    """
    Fetches traces from LangSmith and logs them to Google Sheets.

    Uses LangSmith SDK to:
    - List recent runs/traces
    - Get run details (tokens, latency, etc.)
    - Export to Google Sheets for analysis
    """

    def __init__(self):
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "credit-intelligence")
        self.sheets_logger = get_sheets_logger() if SHEETS_AVAILABLE else None
        self.client = None

        if LANGSMITH_SDK_AVAILABLE and self.api_key:
            try:
                self.client = LangSmithClient(api_key=self.api_key)
                logger.info("LangSmith client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith client: {e}")
        else:
            logger.warning("LangSmith SDK not available or LANGCHAIN_API_KEY not set")

    def is_available(self) -> bool:
        """Check if LangSmith client is available."""
        return self.client is not None

    def fetch_recent_runs(
        self,
        limit: int = 50,
        hours_ago: int = 24,
    ) -> List[Any]:
        """
        Fetch recent runs from LangSmith.

        Args:
            limit: Maximum number of runs to fetch
            hours_ago: Fetch runs from the last N hours

        Returns:
            List of run objects
        """
        if not self.is_available():
            return []

        try:
            # Use SDK to list runs
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit,
            ))
            logger.info(f"Fetched {len(runs)} runs from LangSmith")
            return runs

        except Exception as e:
            logger.error(f"Failed to fetch runs from LangSmith: {e}")
            return []

    def fetch_run_details(self, run_id: str) -> Optional[Any]:
        """
        Fetch detailed information about a specific run.

        Args:
            run_id: The LangSmith run ID

        Returns:
            Run object
        """
        if not self.is_available():
            return None

        try:
            return self.client.read_run(run_id)
        except Exception as e:
            logger.error(f"Failed to fetch run details: {e}")
            return None

    def log_traces_to_sheets(
        self,
        workflow_run_id: str,
        company_name: str,
        limit: int = 20,
    ) -> int:
        """
        Fetch recent LangSmith traces and log them to Google Sheets.

        Args:
            workflow_run_id: The workflow run ID (for linking)
            company_name: Company name being analyzed
            limit: Number of traces to fetch

        Returns:
            Number of traces logged
        """
        if not self.sheets_logger or not self.sheets_logger.is_connected():
            logger.warning("Sheets logger not connected")
            return 0

        runs = self.fetch_recent_runs(limit=limit, hours_ago=1)
        logged_count = 0

        for run in runs:
            try:
                # Extract run info from SDK Run object
                run_id = str(run.id) if hasattr(run, 'id') else ""
                name = run.name if hasattr(run, 'name') else "unknown"
                run_type = run.run_type if hasattr(run, 'run_type') else "chain"
                status = run.status if hasattr(run, 'status') else "unknown"

                # Token usage
                total_tokens = run.total_tokens or 0 if hasattr(run, 'total_tokens') else 0
                prompt_tokens = run.prompt_tokens or 0 if hasattr(run, 'prompt_tokens') else 0
                completion_tokens = run.completion_tokens or 0 if hasattr(run, 'completion_tokens') else 0

                # Latency
                latency_ms = 0
                if hasattr(run, 'start_time') and hasattr(run, 'end_time') and run.start_time and run.end_time:
                    latency_ms = (run.end_time - run.start_time).total_seconds() * 1000

                # Model info
                model = ""
                if hasattr(run, 'extra') and run.extra:
                    extra = run.extra
                    if isinstance(extra, dict):
                        model = extra.get("invocation_params", {}).get("model", "")
                        if not model:
                            model = extra.get("metadata", {}).get("ls_model_name", "")

                # Error
                error = run.error if hasattr(run, 'error') else ""

                # Input/Output previews
                inputs = run.inputs if hasattr(run, 'inputs') else {}
                outputs = run.outputs if hasattr(run, 'outputs') else {}
                input_preview = str(inputs)[:500] if inputs else ""
                output_preview = str(outputs)[:500] if outputs else ""

                # Log to sheets
                self.sheets_logger.log_langsmith_trace(
                    run_id=workflow_run_id,
                    company_name=company_name,
                    step_name=name,
                    run_type=run_type,
                    status=status,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    model=model,
                    error=error or "",
                    input_preview=input_preview,
                    output_preview=output_preview,
                )
                logged_count += 1

            except Exception as e:
                logger.error(f"Failed to log trace: {e}")

        logger.info(f"Logged {logged_count} LangSmith traces to Google Sheets")
        return logged_count

    def sync_all_traces(self, hours_ago: int = 24, limit: int = 100) -> int:
        """
        Sync all recent traces to Google Sheets.

        Args:
            hours_ago: Fetch traces from the last N hours
            limit: Maximum traces to sync

        Returns:
            Number of traces synced
        """
        if not self.sheets_logger or not self.sheets_logger.is_connected():
            logger.warning("Sheets logger not connected")
            return 0

        runs = self.fetch_recent_runs(limit=limit, hours_ago=hours_ago)
        synced_count = 0

        for run in runs:
            try:
                # Extract run info from SDK Run object
                run_id = str(run.id) if hasattr(run, 'id') else ""
                name = run.name if hasattr(run, 'name') else "unknown"
                run_type = run.run_type if hasattr(run, 'run_type') else "chain"
                status = run.status if hasattr(run, 'status') else "unknown"

                # Extract company name from inputs if available
                inputs = run.inputs if hasattr(run, 'inputs') else {}
                company_name = ""
                if isinstance(inputs, dict):
                    company_name = inputs.get("company_name", "")
                    if not company_name:
                        # Try to find company_name in nested structure
                        for key, value in inputs.items():
                            if isinstance(value, dict) and "company_name" in value:
                                company_name = value["company_name"]
                                break
                            elif key == "input" and isinstance(value, str):
                                company_name = value[:50]

                # Token usage
                total_tokens = run.total_tokens or 0 if hasattr(run, 'total_tokens') else 0
                prompt_tokens = run.prompt_tokens or 0 if hasattr(run, 'prompt_tokens') else 0
                completion_tokens = run.completion_tokens or 0 if hasattr(run, 'completion_tokens') else 0

                # Latency
                latency_ms = 0
                if hasattr(run, 'start_time') and hasattr(run, 'end_time') and run.start_time and run.end_time:
                    latency_ms = (run.end_time - run.start_time).total_seconds() * 1000

                # Model info
                model = ""
                if hasattr(run, 'extra') and run.extra:
                    extra = run.extra
                    if isinstance(extra, dict):
                        model = extra.get("invocation_params", {}).get("model", "")
                        if not model:
                            model = extra.get("metadata", {}).get("ls_model_name", "")

                # Error
                error = run.error if hasattr(run, 'error') else ""

                # Input/Output previews
                outputs = run.outputs if hasattr(run, 'outputs') else {}
                input_preview = str(inputs)[:500] if inputs else ""
                output_preview = str(outputs)[:500] if outputs else ""

                # Log to sheets
                self.sheets_logger.log_langsmith_trace(
                    run_id=run_id[:8] if run_id else "",
                    company_name=company_name or "Unknown",
                    step_name=name,
                    run_type=run_type,
                    status=status,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    model=model,
                    error=error or "",
                    input_preview=input_preview,
                    output_preview=output_preview,
                )
                synced_count += 1

            except Exception as e:
                logger.error(f"Failed to sync trace: {e}")

        logger.info(f"Synced {synced_count} LangSmith traces to Google Sheets")
        return synced_count


# Singleton instance
_langsmith_sheets: Optional[LangSmithSheetsLogger] = None


def get_langsmith_sheets_logger() -> LangSmithSheetsLogger:
    """Get the global LangSmithSheetsLogger instance."""
    global _langsmith_sheets
    if _langsmith_sheets is None:
        _langsmith_sheets = LangSmithSheetsLogger()
    return _langsmith_sheets


def sync_langsmith_to_sheets(hours_ago: int = 24, limit: int = 100) -> int:
    """
    Convenience function to sync LangSmith traces to Google Sheets.

    Args:
        hours_ago: Fetch traces from the last N hours
        limit: Maximum traces to sync

    Returns:
        Number of traces synced
    """
    logger = get_langsmith_sheets_logger()
    return logger.sync_all_traces(hours_ago=hours_ago, limit=limit)
