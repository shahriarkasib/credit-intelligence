"""Google Sheets Logger - Logs workflow runs to Google Sheets."""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Thread pool for async logging (non-blocking)
_sheets_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sheets_logger")

# Try to import gspread
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logger.warning("gspread not installed. Google Sheets logging disabled.")


class SheetsLogger:
    """
    Logger that saves workflow data to Google Sheets.

    Creates/uses the following sheets:
    - runs: Run summaries (run_id, company, status, timestamps, scores)
    - tool_calls: Tool execution logs
    - assessments: Credit assessments
    - evaluations: Evaluation results
    """

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        spreadsheet_id: Optional[str] = None,
        spreadsheet_name: str = "Credit Intelligence Logs"
    ):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SPREADSHEET_ID")
        self.spreadsheet_name = spreadsheet_name
        self.client = None
        self.spreadsheet = None
        self._sheets_cache: Dict[str, Any] = {}

        if GSPREAD_AVAILABLE and self.credentials_path:
            self._connect()

    def _connect(self):
        """Connect to Google Sheets."""
        try:
            # Resolve credentials path relative to project root
            creds_path = Path(self.credentials_path)
            if not creds_path.is_absolute():
                # Try relative to project root (parent of src)
                project_root = Path(__file__).parent.parent.parent
                creds_path = project_root / self.credentials_path

            if not creds_path.exists():
                logger.warning(f"Google credentials file not found: {creds_path}")
                return

            creds = Credentials.from_service_account_file(
                str(creds_path),
                scopes=self.SCOPES
            )
            self.client = gspread.authorize(creds)

            # Open or create spreadsheet
            if self.spreadsheet_id:
                self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            else:
                # Try to open by name, or create new
                try:
                    self.spreadsheet = self.client.open(self.spreadsheet_name)
                except gspread.SpreadsheetNotFound:
                    self.spreadsheet = self.client.create(self.spreadsheet_name)
                    logger.info(f"Created new spreadsheet: {self.spreadsheet_name}")

            # Initialize sheets
            self._init_sheets()
            logger.info(f"SheetsLogger connected to: {self.spreadsheet.title}")

        except Exception as e:
            logger.warning(f"Google Sheets connection failed: {e}")

    def _init_sheets(self):
        """Initialize required sheets with headers."""
        sheet_configs = {
            "runs": [
                "run_id", "company_name", "status", "started_at", "completed_at",
                "risk_level", "credit_score", "confidence", "total_time_ms",
                "tools_used", "evaluation_score"
            ],
            "tool_calls": [
                "run_id", "company_name", "tool_name", "success",
                "execution_time_ms", "timestamp", "tool_input", "tool_output", "error"
            ],
            "assessments": [
                "run_id", "company_name", "risk_level", "credit_score",
                "confidence", "reasoning", "recommendations", "timestamp"
            ],
            "evaluations": [
                "run_id", "company_name", "tool_selection_score", "data_quality_score",
                "synthesis_score", "overall_score", "timestamp"
            ],
            "tool_selections": [
                "run_id", "company_name", "selected_tools", "expected_tools",
                "precision", "recall", "f1_score", "timestamp"
            ],
            # NEW: Detailed step-by-step logs
            "step_logs": [
                "run_id", "company_name", "step_name", "step_number",
                "input_summary", "output_summary", "execution_time_ms",
                "success", "error", "timestamp"
            ],
            # NEW: LLM call logs
            "llm_calls": [
                "run_id", "company_name", "call_type", "model",
                "prompt_summary", "response_summary", "prompt_tokens",
                "completion_tokens", "total_tokens", "input_cost", "output_cost",
                "total_cost", "execution_time_ms", "timestamp"
            ],
            # NEW: Consistency scores (includes model name for per-model tracking)
            "consistency_scores": [
                "run_id", "company_name", "model_name", "evaluation_type", "num_runs",
                "risk_level_consistency", "score_consistency", "score_std",
                "overall_consistency", "risk_levels", "credit_scores", "timestamp"
            ],
            # NEW: Data source results
            "data_sources": [
                "run_id", "company_name", "source_name", "success",
                "records_found", "data_summary", "execution_time_ms", "timestamp"
            ],
            # NEW: LangSmith traces
            "langsmith_traces": [
                "run_id", "company_name", "step_name", "run_type", "status",
                "total_tokens", "prompt_tokens", "completion_tokens",
                "latency_ms", "model", "error", "input_preview",
                "output_preview", "timestamp"
            ]
        }

        existing_sheets = [ws.title for ws in self.spreadsheet.worksheets()]

        for sheet_name, headers in sheet_configs.items():
            if sheet_name not in existing_sheets:
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name, rows=1000, cols=len(headers)
                )
                worksheet.append_row(headers)
                logger.info(f"Created sheet: {sheet_name}")
            self._sheets_cache[sheet_name] = self.spreadsheet.worksheet(sheet_name)

    def is_connected(self) -> bool:
        return self.spreadsheet is not None

    def _get_sheet(self, name: str):
        """Get worksheet by name."""
        if name not in self._sheets_cache:
            try:
                self._sheets_cache[name] = self.spreadsheet.worksheet(name)
            except Exception:
                return None
        return self._sheets_cache.get(name)

    def _safe_str(self, value: Any, max_length: int = 50000) -> str:
        """Convert value to safe string for sheets (max 50k chars per cell)."""
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            s = json.dumps(value, default=str, indent=2)  # Pretty print for readability
        else:
            s = str(value)
        return s[:max_length] if len(s) > max_length else s

    # ==================== LOGGING METHODS ====================

    def log_run(
        self,
        run_id: str,
        company_name: str,
        status: str,
        risk_level: str = "",
        credit_score: int = None,
        confidence: float = None,
        total_time_ms: float = 0,
        tools_used: List[str] = None,
        evaluation_score: float = None,
    ):
        """Log a run summary (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            status,
            datetime.utcnow().isoformat(),  # started_at (approximate)
            datetime.utcnow().isoformat(),  # completed_at
            risk_level or "",
            credit_score if credit_score is not None else "",
            confidence if confidence is not None else "",
            total_time_ms,
            ", ".join(tools_used) if tools_used else "",
            evaluation_score if evaluation_score is not None else "",
        ]

        def _write():
            try:
                sheet = self._get_sheet("runs")
                sheet.append_row(row)
                logger.debug(f"Logged run to sheets: {run_id}")
            except Exception as e:
                logger.error(f"Failed to log run to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_tool_call(
        self,
        run_id: str,
        company_name: str,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        execution_time_ms: float = 0,
        success: bool = True,
        error: str = None,
    ):
        """Log a tool call (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            tool_name,
            "Yes" if success else "No",
            execution_time_ms,
            datetime.utcnow().isoformat(),
            self._safe_str(tool_input),
            self._safe_str(tool_output),
            error or "",
        ]

        def _write():
            try:
                sheet = self._get_sheet("tool_calls")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log tool call to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_assessment(
        self,
        run_id: str,
        company_name: str,
        risk_level: str,
        credit_score: int,
        confidence: float,
        reasoning: str = "",
        recommendations: List[str] = None,
    ):
        """Log a credit assessment (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            risk_level,
            credit_score,
            confidence,
            self._safe_str(reasoning),
            self._safe_str(recommendations),
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("assessments")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log assessment to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_evaluation(
        self,
        run_id: str,
        company_name: str,
        tool_selection_score: float,
        data_quality_score: float,
        synthesis_score: float,
        overall_score: float,
    ):
        """Log evaluation results (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            tool_selection_score,
            data_quality_score,
            synthesis_score,
            overall_score,
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("evaluations")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log evaluation to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_tool_selection(
        self,
        run_id: str,
        company_name: str,
        selected_tools: List[str],
        expected_tools: List[str] = None,
        precision: float = 0,
        recall: float = 0,
        f1_score: float = 0,
    ):
        """Log tool selection decision (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            ", ".join(selected_tools) if selected_tools else "",
            ", ".join(expected_tools) if expected_tools else "",
            precision,
            recall,
            f1_score,
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("tool_selections")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log tool selection to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_step(
        self,
        run_id: str,
        company_name: str,
        step_name: str,
        step_number: int,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: float,
        success: bool = True,
        error: str = "",
    ):
        """Log a workflow step execution (non-blocking)."""
        if not self.is_connected():
            return

        # Prepare data synchronously
        row = [
            run_id,
            company_name,
            step_name,
            step_number,
            self._safe_str(input_data),  # Full data (up to 50k)
            self._safe_str(output_data),  # Full data (up to 50k)
            execution_time_ms,
            "Yes" if success else "No",
            error or "",  # Full error message
            datetime.utcnow().isoformat(),
        ]

        # Write asynchronously (non-blocking)
        def _write():
            try:
                sheet = self._get_sheet("step_logs")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log step to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_llm_call(
        self,
        run_id: str,
        company_name: str,
        call_type: str,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        execution_time_ms: float,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        total_cost: float = 0.0,
    ):
        """Log an LLM API call with cost tracking (non-blocking)."""
        if not self.is_connected():
            return

        # Calculate cost if not provided
        if total_cost == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            try:
                from config.cost_tracker import calculate_cost_for_tokens
                cost = calculate_cost_for_tokens(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider="groq",
                )
                input_cost = cost["input_cost"]
                output_cost = cost["output_cost"]
                total_cost = cost["total_cost"]
            except Exception:
                pass  # Cost calculation optional

        row = [
            run_id,
            company_name,
            call_type,
            model,
            self._safe_str(prompt),  # Full prompt (up to 50k)
            self._safe_str(response),  # Full response (up to 50k)
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            round(input_cost, 6),
            round(output_cost, 6),
            round(total_cost, 6),
            execution_time_ms,
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("llm_calls")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log LLM call to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_consistency_score(
        self,
        run_id: str,
        company_name: str,
        model_name: str,  # e.g., "llama-3.3-70b-versatile" or "overall"
        evaluation_type: str,  # "same_model" or "cross_model"
        num_runs: int,
        risk_level_consistency: float,
        score_consistency: float,
        score_std: float,  # Standard deviation of scores
        overall_consistency: float,
        risk_levels: List[str],
        credit_scores: List[int],
    ):
        """Log consistency evaluation scores (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            model_name,
            evaluation_type,
            num_runs,
            risk_level_consistency,
            score_consistency,
            score_std,
            overall_consistency,
            ", ".join(risk_levels),
            ", ".join(str(s) for s in credit_scores),
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("consistency_scores")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log consistency score to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_data_source(
        self,
        run_id: str,
        company_name: str,
        source_name: str,
        success: bool,
        records_found: int,
        data_summary: str,
        execution_time_ms: float,
    ):
        """Log data source fetch result (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            source_name,
            "Yes" if success else "No",
            records_found,
            self._safe_str(data_summary),  # Full data (up to 50k)
            execution_time_ms,
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("data_sources")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log data source to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_langsmith_trace(
        self,
        run_id: str,
        company_name: str,
        step_name: str,
        run_type: str,
        status: str,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0,
        model: str = "",
        error: str = "",
        input_preview: str = "",
        output_preview: str = "",
    ):
        """Log a LangSmith trace (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            step_name,
            run_type,
            status,
            total_tokens,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            model,
            error or "",
            self._safe_str(input_preview, max_length=1000),
            self._safe_str(output_preview, max_length=1000),
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("langsmith_traces")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log LangSmith trace to sheets: {e}")

        _sheets_executor.submit(_write)

    def get_spreadsheet_url(self) -> Optional[str]:
        """Get the URL of the spreadsheet."""
        if self.spreadsheet:
            return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet.id}"
        return None


# Singleton instance
_sheets_logger: Optional[SheetsLogger] = None


def get_sheets_logger(force_new: bool = False) -> SheetsLogger:
    """Get the global SheetsLogger instance."""
    global _sheets_logger
    if _sheets_logger is None or force_new:
        _sheets_logger = SheetsLogger()
    return _sheets_logger
