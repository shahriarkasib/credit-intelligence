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

        # Check for credentials (file or env var)
        has_credentials = (
            self.credentials_path or
            os.getenv("GOOGLE_SHEETS_CREDENTIALS")
        )
        if GSPREAD_AVAILABLE and has_credentials:
            self._connect()

    def _connect(self):
        """Connect to Google Sheets.

        Supports two credential methods:
        1. File-based: GOOGLE_CREDENTIALS_PATH points to a JSON file
        2. Environment variable: GOOGLE_SHEETS_CREDENTIALS contains base64-encoded JSON
           (for Heroku and other cloud deployments)
        """
        try:
            import base64

            creds = None

            # Method 1: Environment variable (base64-encoded JSON) - preferred for cloud
            env_credentials = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
            if env_credentials:
                try:
                    # Decode base64 credentials
                    creds_json = base64.b64decode(env_credentials).decode('utf-8')
                    creds_dict = json.loads(creds_json)
                    creds = Credentials.from_service_account_info(
                        creds_dict,
                        scopes=self.SCOPES
                    )
                    logger.info("Using Google credentials from environment variable")
                except Exception as e:
                    logger.warning(f"Failed to load credentials from env var: {e}")

            # Method 2: File-based credentials (for local development)
            if creds is None and self.credentials_path:
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
                logger.info("Using Google credentials from file")

            if creds is None:
                logger.warning("No Google credentials available")
                return

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
                "run_id", "company_name", "tool_selection_score", "tool_reasoning",
                "data_quality_score", "data_reasoning", "synthesis_score",
                "synthesis_reasoning", "overall_score", "timestamp"
            ],
            "tool_selections": [
                "run_id", "company_name", "selected_tools", "expected_tools",
                "correct_tools", "missing_tools", "extra_tools",
                "precision", "recall", "f1_score", "reasoning", "timestamp"
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
                "latency_ms", "error", "input_preview", "output_preview", "timestamp"
            ],
            # NEW: LangGraph events (from astream_events)
            "langgraph_events": [
                "run_id", "company_name", "event_type", "event_name", "status",
                "duration_ms", "model", "tokens", "input_preview", "output_preview",
                "error", "timestamp"
            ],
            # Task 17: Detailed LLM call logs (full Task 17 spec)
            "llm_calls_detailed": [
                "run_id", "company_name", "llm_provider", "agent_name", "model",
                "prompt", "context", "response", "reasoning", "error",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_cost", "output_cost", "total_cost",
                "response_time_ms", "timestamp"
            ],
            # Task 17: Comprehensive run summaries
            "run_summaries": [
                "run_id", "company_name", "status",
                "risk_level", "credit_score", "confidence", "reasoning",
                "tool_selection_score", "data_quality_score", "synthesis_score", "overall_score",
                "final_decision", "decision_reasoning",
                "errors", "warnings", "tools_used", "agents_used",
                "started_at", "completed_at", "duration_ms",
                "total_tokens", "total_cost", "llm_calls_count", "timestamp"
            ],
            # Task 4: Agent efficiency metrics
            "agent_metrics": [
                "run_id", "company_name", "timestamp",
                # Core agent metrics (0-1 scores)
                "intent_correctness", "plan_quality", "tool_choice_correctness",
                "tool_completeness", "trajectory_match", "final_answer_quality",
                # Execution metrics
                "step_count", "tool_calls", "latency_ms",
                # Overall score
                "overall_score",
                # Details (JSON)
                "intent_details", "plan_details", "tool_details",
                "trajectory_details", "answer_details"
            ],
            # Task 21: LLM-as-a-judge evaluations
            "llm_judge_results": [
                "run_id", "company_name", "timestamp", "model_used",
                # Dimension scores (0-1)
                "accuracy_score", "completeness_score", "consistency_score",
                "actionability_score", "data_utilization_score", "overall_score",
                # Reasoning
                "accuracy_reasoning", "completeness_reasoning", "consistency_reasoning",
                "actionability_reasoning", "data_utilization_reasoning", "overall_reasoning",
                # Benchmark comparison
                "benchmark_alignment", "benchmark_comparison",
                # Suggestions
                "suggestions",
                # Metadata
                "tokens_used", "evaluation_cost"
            ],
            # Task 21 Enhanced: Model consistency evaluation
            "model_consistency": [
                "eval_id", "company_name", "model_name", "num_runs", "timestamp",
                # Core consistency metrics
                "risk_level_consistency", "credit_score_mean", "credit_score_std",
                "confidence_variance", "reasoning_similarity",
                "risk_factors_overlap", "recommendations_overlap",
                # Overall
                "overall_consistency", "is_consistent", "consistency_grade",
                # LLM Judge analysis
                "llm_judge_analysis", "llm_judge_concerns",
                # Run details
                "run_details"
            ],
            # Task 21 Enhanced: Cross-model comparison
            "cross_model_eval": [
                "eval_id", "company_name", "models_compared", "num_models", "timestamp",
                # Agreement metrics
                "risk_level_agreement", "credit_score_mean", "credit_score_std",
                "credit_score_range", "confidence_agreement",
                # Best model
                "best_model", "best_model_reasoning",
                # Overall
                "cross_model_agreement",
                # LLM analysis
                "llm_judge_analysis", "model_recommendations",
                # Per-model results
                "model_results", "pairwise_comparisons"
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
        tool_reasoning: str = "",
        data_reasoning: str = "",
        synthesis_reasoning: str = "",
    ):
        """Log evaluation results with reasoning (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            tool_selection_score,
            self._safe_str(tool_reasoning, max_length=500),
            data_quality_score,
            self._safe_str(data_reasoning, max_length=500),
            synthesis_score,
            self._safe_str(synthesis_reasoning, max_length=500),
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
        correct_tools: List[str] = None,
        missing_tools: List[str] = None,
        extra_tools: List[str] = None,
        reasoning: str = "",
    ):
        """Log tool selection decision with reasoning (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            ", ".join(selected_tools) if selected_tools else "",
            ", ".join(expected_tools) if expected_tools else "",
            ", ".join(correct_tools) if correct_tools else "",
            ", ".join(missing_tools) if missing_tools else "",
            ", ".join(extra_tools) if extra_tools else "",
            precision,
            recall,
            f1_score,
            self._safe_str(reasoning, max_length=500),
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
        latency_ms: float = 0,
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
            latency_ms,
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

    def log_langgraph_event(
        self,
        run_id: str,
        company_name: str,
        event_type: str,
        event_name: str,
        status: str,
        duration_ms: float = None,
        model: str = "",
        tokens: int = None,
        input_preview: str = "",
        output_preview: str = "",
        error: str = "",
    ):
        """Log a LangGraph event from astream_events (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            event_type,
            event_name,
            status,
            duration_ms if duration_ms is not None else "",
            model or "",
            tokens if tokens is not None else "",
            self._safe_str(input_preview, max_length=1000),
            self._safe_str(output_preview, max_length=1000),
            error or "",
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("langgraph_events")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log LangGraph event to sheets: {e}")

        _sheets_executor.submit(_write)

    # ==================== TASK 17: DETAILED LOGGING ====================

    def log_llm_call_detailed(
        self,
        run_id: str,
        company_name: str,
        llm_provider: str,
        agent_name: str,
        model: str,
        prompt: str,
        context: str = "",
        response: str = "",
        reasoning: str = "",
        error: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_time_ms: float = 0,
        input_cost: float = 0,
        output_cost: float = 0,
        total_cost: float = 0,
    ):
        """
        Log a detailed LLM call (Task 17 compliant).

        Logs to llm_calls_detailed sheet with all fields:
        - llm_provider, run_id, agent_name
        - prompt, context, response, reasoning
        - error, tokens, response_time_ms, costs

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            llm_provider,
            agent_name,
            model,
            self._safe_str(prompt, max_length=10000),
            self._safe_str(context, max_length=5000),
            self._safe_str(response, max_length=10000),
            self._safe_str(reasoning, max_length=2000),
            error or "",
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            round(input_cost, 6),
            round(output_cost, 6),
            round(total_cost, 6),
            response_time_ms,
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("llm_calls_detailed")
                sheet.append_row(row)
                logger.debug(f"Logged detailed LLM call for run: {run_id}")
            except Exception as e:
                logger.error(f"Failed to log detailed LLM call to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_run_summary(
        self,
        run_id: str,
        company_name: str,
        status: str = "completed",
        # Assessment
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        reasoning: str = "",
        # Eval metrics
        tool_selection_score: float = 0.0,
        data_quality_score: float = 0.0,
        synthesis_score: float = 0.0,
        overall_score: float = 0.0,
        # Decision
        final_decision: str = "",
        decision_reasoning: str = "",
        # Execution details
        errors: List[str] = None,
        warnings: List[str] = None,
        tools_used: List[str] = None,
        agents_used: List[str] = None,
        # Timing
        started_at: str = "",
        completed_at: str = "",
        duration_ms: float = 0.0,
        # Costs
        total_tokens: int = 0,
        total_cost: float = 0.0,
        llm_calls_count: int = 0,
    ):
        """
        Log a comprehensive run summary (Task 17 compliant).

        Logs to run_summaries sheet with all fields:
        - company_name, run_id, status
        - risk_level, credit_score, confidence, reasoning
        - ALL eval metrics (tool_selection, data_quality, synthesis, overall)
        - final_decision (Good/Not Good), decision_reasoning
        - errors, warnings, tools_used, agents_used
        - timing (started_at, completed_at, duration_ms)
        - costs (total_tokens, total_cost, llm_calls_count)

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            status,
            risk_level,
            credit_score,
            round(confidence, 4),
            self._safe_str(reasoning, max_length=5000),
            round(tool_selection_score, 4),
            round(data_quality_score, 4),
            round(synthesis_score, 4),
            round(overall_score, 4),
            final_decision,
            self._safe_str(decision_reasoning, max_length=2000),
            ", ".join(errors) if errors else "",
            ", ".join(warnings) if warnings else "",
            ", ".join(tools_used) if tools_used else "",
            ", ".join(agents_used) if agents_used else "",
            started_at,
            completed_at,
            duration_ms,
            total_tokens,
            round(total_cost, 6),
            llm_calls_count,
            datetime.utcnow().isoformat(),
        ]

        def _write():
            try:
                sheet = self._get_sheet("run_summaries")
                sheet.append_row(row)
                logger.info(f"Logged run summary for: {company_name} (run: {run_id})")
            except Exception as e:
                logger.error(f"Failed to log run summary to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_agent_metrics(
        self,
        run_id: str,
        company_name: str,
        # Core agent metrics (0-1 scores)
        intent_correctness: float = 0.0,
        plan_quality: float = 0.0,
        tool_choice_correctness: float = 0.0,
        tool_completeness: float = 0.0,
        trajectory_match: float = 0.0,
        final_answer_quality: float = 0.0,
        # Execution metrics
        step_count: int = 0,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
        # Overall score
        overall_score: float = 0.0,
        # Details (dicts)
        intent_details: Dict[str, Any] = None,
        plan_details: Dict[str, Any] = None,
        tool_details: Dict[str, Any] = None,
        trajectory_details: Dict[str, Any] = None,
        answer_details: Dict[str, Any] = None,
    ):
        """
        Log agent efficiency metrics (Task 4 compliant).

        Logs standard agentic metrics to agent_metrics sheet:
        - intent_correctness: Did the agent understand the task?
        - plan_quality: How good was the execution plan?
        - tool_choice_correctness: Did agent choose correct tools? (precision)
        - tool_completeness: Did agent use all needed tools? (recall)
        - trajectory_match: Did agent follow expected execution path?
        - final_answer_quality: Is the final output correct and complete?
        - step_count, tool_calls, latency_ms: Execution metrics

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            datetime.utcnow().isoformat(),
            # Core metrics
            round(intent_correctness, 4),
            round(plan_quality, 4),
            round(tool_choice_correctness, 4),
            round(tool_completeness, 4),
            round(trajectory_match, 4),
            round(final_answer_quality, 4),
            # Execution metrics
            step_count,
            tool_calls,
            round(latency_ms, 2),
            # Overall
            round(overall_score, 4),
            # Details as JSON
            self._safe_str(intent_details or {}, max_length=5000),
            self._safe_str(plan_details or {}, max_length=5000),
            self._safe_str(tool_details or {}, max_length=5000),
            self._safe_str(trajectory_details or {}, max_length=5000),
            self._safe_str(answer_details or {}, max_length=5000),
        ]

        def _write():
            try:
                sheet = self._get_sheet("agent_metrics")
                sheet.append_row(row)
                logger.info(f"Logged agent metrics for: {company_name} (run: {run_id})")
            except Exception as e:
                logger.error(f"Failed to log agent metrics to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_llm_judge_result(
        self,
        run_id: str,
        company_name: str,
        model_used: str,
        # Dimension scores (0-1)
        accuracy_score: float = 0.0,
        completeness_score: float = 0.0,
        consistency_score: float = 0.0,
        actionability_score: float = 0.0,
        data_utilization_score: float = 0.0,
        overall_score: float = 0.0,
        # Reasoning
        accuracy_reasoning: str = "",
        completeness_reasoning: str = "",
        consistency_reasoning: str = "",
        actionability_reasoning: str = "",
        data_utilization_reasoning: str = "",
        overall_reasoning: str = "",
        # Benchmark comparison
        benchmark_alignment: float = 0.0,
        benchmark_comparison: str = "",
        # Suggestions
        suggestions: List[str] = None,
        # Metadata
        tokens_used: int = 0,
        evaluation_cost: float = 0.0,
    ):
        """
        Log LLM-as-a-judge evaluation result (Task 21 compliant).

        Logs comprehensive LLM judge evaluation to llm_judge_results sheet:
        - Dimension scores: accuracy, completeness, consistency, actionability, data_utilization
        - Reasoning for each dimension
        - Benchmark comparison (if provided)
        - Suggestions for improvement
        - Cost and token metadata

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            datetime.utcnow().isoformat(),
            model_used,
            # Dimension scores
            round(accuracy_score, 4),
            round(completeness_score, 4),
            round(consistency_score, 4),
            round(actionability_score, 4),
            round(data_utilization_score, 4),
            round(overall_score, 4),
            # Reasoning (truncated)
            self._safe_str(accuracy_reasoning, max_length=2000),
            self._safe_str(completeness_reasoning, max_length=2000),
            self._safe_str(consistency_reasoning, max_length=2000),
            self._safe_str(actionability_reasoning, max_length=2000),
            self._safe_str(data_utilization_reasoning, max_length=2000),
            self._safe_str(overall_reasoning, max_length=5000),
            # Benchmark
            round(benchmark_alignment, 4) if benchmark_alignment else 0,
            self._safe_str(benchmark_comparison, max_length=5000),
            # Suggestions
            self._safe_str(suggestions or [], max_length=5000),
            # Metadata
            tokens_used,
            round(evaluation_cost, 6),
        ]

        def _write():
            try:
                sheet = self._get_sheet("llm_judge_results")
                sheet.append_row(row)
                logger.info(f"Logged LLM judge result for: {company_name} (run: {run_id})")
            except Exception as e:
                logger.error(f"Failed to log LLM judge result to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_model_consistency(
        self,
        eval_id: str,
        company_name: str,
        model_name: str,
        num_runs: int,
        # Core consistency metrics
        risk_level_consistency: float = 0.0,
        credit_score_mean: float = 0.0,
        credit_score_std: float = 0.0,
        confidence_variance: float = 0.0,
        reasoning_similarity: float = 0.0,
        risk_factors_overlap: float = 0.0,
        recommendations_overlap: float = 0.0,
        # Overall
        overall_consistency: float = 0.0,
        is_consistent: bool = False,
        consistency_grade: str = "",
        # LLM Judge analysis
        llm_judge_analysis: str = "",
        llm_judge_concerns: List[str] = None,
        # Run details
        run_details: List[Dict[str, Any]] = None,
    ):
        """
        Log model consistency evaluation result.

        Measures how consistent a model is when run multiple times
        on the same company.

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            eval_id,
            company_name,
            model_name,
            num_runs,
            datetime.utcnow().isoformat(),
            # Core metrics
            round(risk_level_consistency, 4),
            round(credit_score_mean, 2),
            round(credit_score_std, 2),
            round(confidence_variance, 4),
            round(reasoning_similarity, 4),
            round(risk_factors_overlap, 4),
            round(recommendations_overlap, 4),
            # Overall
            round(overall_consistency, 4),
            "Yes" if is_consistent else "No",
            consistency_grade,
            # LLM analysis
            self._safe_str(llm_judge_analysis, max_length=5000),
            self._safe_str(llm_judge_concerns or [], max_length=2000),
            # Run details
            self._safe_str(run_details or [], max_length=10000),
        ]

        def _write():
            try:
                sheet = self._get_sheet("model_consistency")
                sheet.append_row(row)
                logger.info(f"Logged model consistency for: {company_name} ({model_name})")
            except Exception as e:
                logger.error(f"Failed to log model consistency to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_cross_model_eval(
        self,
        eval_id: str,
        company_name: str,
        models_compared: List[str],
        num_models: int,
        # Agreement metrics
        risk_level_agreement: float = 0.0,
        credit_score_mean: float = 0.0,
        credit_score_std: float = 0.0,
        credit_score_range: float = 0.0,
        confidence_agreement: float = 0.0,
        # Best model
        best_model: str = "",
        best_model_reasoning: str = "",
        # Overall
        cross_model_agreement: float = 0.0,
        # LLM analysis
        llm_judge_analysis: str = "",
        model_recommendations: List[str] = None,
        # Per-model results
        model_results: Dict[str, Dict[str, Any]] = None,
        pairwise_comparisons: List[Dict[str, Any]] = None,
    ):
        """
        Log cross-model evaluation result.

        Compares assessments from different models for the same company.

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            eval_id,
            company_name,
            ", ".join(models_compared) if models_compared else "",
            num_models,
            datetime.utcnow().isoformat(),
            # Agreement metrics
            round(risk_level_agreement, 4),
            round(credit_score_mean, 2),
            round(credit_score_std, 2),
            round(credit_score_range, 2),
            round(confidence_agreement, 4),
            # Best model
            best_model,
            self._safe_str(best_model_reasoning, max_length=2000),
            # Overall
            round(cross_model_agreement, 4),
            # LLM analysis
            self._safe_str(llm_judge_analysis, max_length=5000),
            self._safe_str(model_recommendations or [], max_length=2000),
            # Per-model results
            self._safe_str(model_results or {}, max_length=10000),
            self._safe_str(pairwise_comparisons or [], max_length=5000),
        ]

        def _write():
            try:
                sheet = self._get_sheet("cross_model_eval")
                sheet.append_row(row)
                logger.info(f"Logged cross-model eval for: {company_name} ({len(models_compared)} models)")
            except Exception as e:
                logger.error(f"Failed to log cross-model eval to sheets: {e}")

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
