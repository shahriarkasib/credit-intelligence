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
        # generated_by values:
        # - "Us": Data generated by our custom code
        # - "FW": Data generated by Framework (LangGraph/LangChain/LangSmith)
        # - "Mixed": Row logged by us, but some column values come from Framework
        #
        # Common columns for full flow traceability:
        # - node: Current graph node (parse_input, validate_company, synthesize, etc.)
        # - node_type: Type of node (agent, tool, llm, chain)
        # - agent_name: Name of the agent/component executing
        # - step_number: Sequential step number in workflow
        # - model: LLM model used (if applicable)
        # - temperature: LLM temperature (if applicable)
        # - status: ok/fail/error
        sheet_configs = {
            # Sheet 1: Run summaries
            "runs": [
                "run_id", "company_name", "node", "agent_name", "model", "temperature",
                "status", "started_at", "completed_at",
                "risk_level", "credit_score", "confidence", "total_time_ms",
                "total_steps", "total_llm_calls", "tools_used", "evaluation_score",
                "timestamp", "generated_by"
            ],
            # Sheet 2: Tool execution logs (with hierarchy tracking)
            "tool_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "tool_name", "tool_input", "tool_output",
                # Hierarchy columns for traceability
                "parent_node", "workflow_phase", "call_depth", "parent_tool_id",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 3: Credit assessments
            "assessments": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model", "temperature", "prompt",
                "risk_level", "credit_score", "confidence", "reasoning", "recommendations",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 4: Evaluation results
            "evaluations": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
                "tool_selection_score", "tool_reasoning",
                "data_quality_score", "data_reasoning",
                "synthesis_score", "synthesis_reasoning", "overall_score",
                "eval_status",  # good/average/bad based on overall_score
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 5: Tool selection decisions
            "tool_selections": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
                "selected_tools", "expected_tools", "correct_tools", "missing_tools", "extra_tools",
                "precision", "recall", "f1_score", "reasoning",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 6: LLM call logs - ALL LLM calls with full details
            # Includes: parse_input, tool_selection, credit_analysis, evaluations
            "llm_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "call_type", "model", "temperature",
                # Full prompt and response for auditability
                "prompt", "response", "reasoning",
                # Context and task tracking
                "context", "current_task",
                # Token usage and costs
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_cost", "output_cost", "total_cost",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 8: Consistency scores (includes model name for per-model tracking)
            "consistency_scores": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model_name", "evaluation_type", "num_runs",
                "risk_level_consistency", "score_consistency", "score_std",
                "overall_consistency", "eval_status",  # good/average/bad
                "risk_levels", "credit_scores",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 9: Data source results
            "data_sources": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "source_name", "records_found", "data_summary",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 10: LangGraph events (FW: events from astream_events)
            "langgraph_events": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "event_type", "event_name", "model", "temperature", "tokens",
                "input_preview", "output_preview",
                "duration_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 10: Plans - Full task plans created for each run
            "plans": [
                "run_id", "company_name", "node", "agent_name",
                "num_tasks", "plan_summary",
                # Full plan as JSON with all task details
                "full_plan",
                # Individual task columns for easy viewing
                "task_1", "task_2", "task_3", "task_4", "task_5",
                "task_6", "task_7", "task_8", "task_9", "task_10",
                "created_at", "status", "generated_by"
            ],
            # Sheet 21: Prompts - All prompts used in runs
            "prompts": [
                "run_id", "company_name", "node", "agent_name", "step_number",
                "prompt_id", "prompt_name", "category",
                # Full prompt text
                "system_prompt", "user_prompt",
                # Variables used
                "variables_json",
                # Model and execution info
                "model", "temperature",
                "timestamp", "generated_by"
            ],
            # Sheet 12: Log Tests - Simple verification of sheet logging per run
            "log_tests": [
                "run_id", "company_name",
                # Per-sheet verification (has_data: yes/no, count)
                "runs", "langgraph_events", "llm_calls", "tool_calls",
                "assessments", "evaluations", "tool_selections",
                "consistency_scores", "data_sources", "plans", "prompts",
                # Summary
                "total_sheets_logged", "verification_status",
                "timestamp", "generated_by"
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

    def _get_eval_status(self, score: float) -> str:
        """
        Get evaluation status based on score.

        Args:
            score: Score between 0 and 1

        Returns:
            'good' if score >= 0.8
            'average' if score >= 0.6
            'bad' if score < 0.6
        """
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "average"
        else:
            return "bad"

    # ==================== LOGGING METHODS ====================

    def log_run(
        self,
        run_id: str,
        company_name: str,
        status: str,
        # New common fields
        node: str = "",
        agent_name: str = "",
        model: str = "",
        temperature: float = None,
        # Original fields
        risk_level: str = "",
        credit_score: int = None,
        confidence: float = None,
        total_time_ms: float = 0,
        total_steps: int = 0,
        total_llm_calls: int = 0,
        tools_used: List[str] = None,
        evaluation_score: float = None,
        started_at: str = "",
        completed_at: str = "",
    ):
        """Log a run summary (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "",
            agent_name or "",
            model or "",
            temperature if temperature is not None else 0.1,  # Default temperature
            status,
            started_at or datetime.utcnow().isoformat(),
            completed_at or datetime.utcnow().isoformat(),
            risk_level or "",
            credit_score if credit_score is not None else "",
            confidence if confidence is not None else "",
            total_time_ms,
            total_steps,
            total_llm_calls,
            ", ".join(tools_used) if tools_used else "",
            evaluation_score if evaluation_score is not None else "",
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We generate run summaries
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
        # New common fields
        node: str = "",
        node_type: str = "tool",
        agent_name: str = "",
        step_number: int = 0,
        # Original fields
        tool_input: Any = None,
        tool_output: Any = None,
        execution_time_ms: float = 0,
        success: bool = True,
        error: str = None,
        # Hierarchy fields for traceability
        parent_node: str = "",
        workflow_phase: str = "",
        call_depth: int = 0,
        parent_tool_id: str = "",
    ):
        """Log a tool call with hierarchy tracking (non-blocking)."""
        if not self.is_connected():
            return

        status = "ok" if success else "fail"
        row = [
            run_id,
            company_name,
            node or "",
            node_type or "tool",
            agent_name or "",
            step_number,
            tool_name,
            self._safe_str(tool_input),
            self._safe_str(tool_output),
            # Hierarchy columns
            parent_node or node or "",  # Default to current node if no parent
            workflow_phase or "",  # e.g., "data_collection", "synthesis", "evaluation"
            call_depth,  # 0 = top-level, 1 = nested call, etc.
            parent_tool_id or "",  # ID of parent tool call if nested
            execution_time_ms,
            status,
            error or "",
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We log tool calls
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
        # New common fields
        node: str = "synthesize",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        model: str = "",
        temperature: float = None,
        prompt: str = "",
        duration_ms: float = 0,
        status: str = "ok",
        # Original fields
        reasoning: str = "",
        recommendations: List[str] = None,
    ):
        """Log a credit assessment (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "synthesize",
            node_type or "agent",
            agent_name or "",
            step_number,
            model or "",
            temperature if temperature is not None else 0.1,  # Default temperature
            self._safe_str(prompt, max_length=5000),
            risk_level,
            credit_score,
            confidence,
            self._safe_str(reasoning),
            self._safe_str(recommendations),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We generate assessments in synthesize node
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
        # New common fields
        node: str = "evaluate",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        model: str = "",
        duration_ms: float = 0,
        status: str = "ok",
        # Original fields
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
            node or "evaluate",
            node_type or "agent",
            agent_name or "",
            step_number,
            model or "",
            tool_selection_score,
            self._safe_str(tool_reasoning, max_length=500),
            data_quality_score,
            self._safe_str(data_reasoning, max_length=500),
            synthesis_score,
            self._safe_str(synthesis_reasoning, max_length=500),
            overall_score,
            self._get_eval_status(overall_score),  # eval_status
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We calculate evaluation scores
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
        # New common fields
        node: str = "create_plan",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        model: str = "",
        duration_ms: float = 0,
        status: str = "ok",
        # Original fields
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

        # Deduplicate tools while preserving order
        def dedupe(tools: List[str]) -> List[str]:
            if not tools:
                return []
            seen = set()
            return [t for t in tools if not (t in seen or seen.add(t))]

        selected_tools = dedupe(selected_tools)
        expected_tools = dedupe(expected_tools) if expected_tools else None
        correct_tools = dedupe(correct_tools) if correct_tools else None
        missing_tools = dedupe(missing_tools) if missing_tools else None
        extra_tools = dedupe(extra_tools) if extra_tools else None

        row = [
            run_id,
            company_name,
            node or "create_plan",
            node_type or "agent",
            agent_name or "",
            step_number,
            model or "",
            ", ".join(selected_tools) if selected_tools else "",
            ", ".join(expected_tools) if expected_tools else "",
            ", ".join(correct_tools) if correct_tools else "",
            ", ".join(missing_tools) if missing_tools else "",
            ", ".join(extra_tools) if extra_tools else "",
            precision,
            recall,
            f1_score,
            self._safe_str(reasoning, max_length=500),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We evaluate tool selection
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
        step_number: int = 0,
        input_data: Dict[str, Any] = None,
        output_data: Dict[str, Any] = None,
        execution_time_ms: float = 0,
        # New common fields
        node: str = "",
        node_type: str = "agent",
        agent_name: str = "",
        model: str = "",
        temperature: float = None,
        status: str = "ok",
        # Original fields
        success: bool = True,
        error: str = "",
    ):
        """
        Log a workflow step execution.

        NOTE: Step logging is now handled by langgraph_events sheet.
        This method is kept for backwards compatibility but does not write to sheets.
        """
        # Step logging is now done via langgraph_events - no separate step_logs sheet
        pass

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
        # New common fields
        node: str = "",
        node_type: str = "llm",
        agent_name: str = "",
        step_number: int = 0,
        temperature: float = None,
        context: str = "",
        status: str = "ok",
        error: str = "",
        # Task tracking
        current_task: str = "",  # Which task triggered this LLM call
        reasoning: str = "",  # LLM reasoning/explanation
        # Original fields
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        total_cost: float = 0.0,
    ):
        """Log an LLM API call with cost and task tracking (non-blocking)."""
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

        # Row matches llm_calls sheet columns:
        # run_id, company_name, node, node_type, agent_name, step_number,
        # call_type, model, temperature,
        # prompt, response, reasoning,
        # context, current_task,
        # prompt_tokens, completion_tokens, total_tokens,
        # input_cost, output_cost, total_cost,
        # execution_time_ms, status, error,
        # timestamp, generated_by
        row = [
            run_id,
            company_name,
            node or "",
            node_type or "llm",
            agent_name or "",
            step_number,
            call_type,
            model,
            temperature if temperature is not None else 0.1,
            self._safe_str(prompt),  # Full prompt
            self._safe_str(response),  # Full response
            self._safe_str(reasoning, max_length=10000),  # Reasoning
            self._safe_str(context, max_length=5000),
            current_task or "",
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            round(input_cost, 6),
            round(output_cost, 6),
            round(total_cost, 6),
            execution_time_ms,
            status,
            error or "",
            datetime.utcnow().isoformat(),
            "Mixed",
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
        # New common fields
        node: str = "evaluate",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        duration_ms: float = 0,
        status: str = "ok",
    ):
        """Log consistency evaluation scores (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "evaluate",
            node_type or "agent",
            agent_name or "",
            step_number,
            model_name,
            evaluation_type,
            num_runs,
            risk_level_consistency,
            score_consistency,
            score_std,
            overall_consistency,
            self._get_eval_status(overall_consistency),  # eval_status
            ", ".join(risk_levels),
            ", ".join(str(s) for s in credit_scores),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We calculate consistency scores
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
        # New common fields
        node: str = "fetch_api_data",
        node_type: str = "tool",
        agent_name: str = "",
        step_number: int = 0,
        error: str = "",
    ):
        """Log data source fetch result (non-blocking)."""
        if not self.is_connected():
            return

        status = "ok" if success else "fail"
        row = [
            run_id,
            company_name,
            node or "fetch_api_data",
            node_type or "tool",
            agent_name or "",
            step_number,
            source_name,
            records_found,
            self._safe_str(data_summary),  # Full data (up to 50k)
            execution_time_ms,
            status,
            error or "",
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We log data source fetches
        ]

        def _write():
            try:
                sheet = self._get_sheet("data_sources")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log data source to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_langgraph_event(
        self,
        run_id: str,
        company_name: str,
        event_type: str,
        event_name: str,
        status: str,
        node: str = "",
        node_type: str = "",
        # New common fields
        agent_name: str = "",
        step_number: int = 0,
        temperature: float = None,
        # Original fields
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
            node or "",
            node_type or "",
            agent_name or "",
            step_number,
            event_type,
            event_name,
            model or "",
            temperature if temperature is not None else 0.1,  # Default temperature
            tokens if tokens is not None else "",
            self._safe_str(input_preview, max_length=10000),
            self._safe_str(output_preview, max_length=10000),
            duration_ms if duration_ms is not None else "",
            status,
            error or "",
            datetime.utcnow().isoformat(),  # timestamp
            "FW",  # generated_by: Events from LangGraph astream_events
        ]

        def _write():
            try:
                sheet = self._get_sheet("langgraph_events")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log LangGraph event to sheets: {e}")

        _sheets_executor.submit(_write)

    # ==================== TASK 17: DETAILED LOGGING ====================

    def log_llm_call_detailed(self, *args, **kwargs):
        """DEPRECATED: Use log_llm_call instead. This sheet has been removed."""
        pass

    def log_run_summary(self, *args, **kwargs):
        """DEPRECATED: Use runs sheet via log_run instead. run_summaries sheet has been removed."""
        pass

    def log_agent_metrics(self, *args, **kwargs):
        """DEPRECATED: Agent metrics now logged via evaluations sheet. agent_metrics sheet has been removed."""
        pass

    def log_llm_judge_result(self, *args, **kwargs):
        """DEPRECATED: LLM judge results now logged via evaluations sheet. llm_judge_results sheet has been removed."""
        pass

    def log_model_consistency(self, *args, **kwargs):
        """DEPRECATED: Model consistency now logged via consistency_scores sheet. model_consistency sheet has been removed."""
        pass

    def log_cross_model_eval(self, *args, **kwargs):
        """DEPRECATED: Cross-model evaluation now logged via consistency_scores sheet. cross_model_eval sheet has been removed."""
        pass

    def log_deepeval_metrics(self, *args, **kwargs):
        """DEPRECATED: DeepEval metrics not currently in use. deepeval_metrics sheet has been removed."""
        pass

    def log_plan(
        self,
        run_id: str,
        company_name: str,
        task_plan: List[Dict[str, Any]],
        # Common fields
        node: str = "create_plan",
        agent_name: str = "",
        status: str = "ok",
    ):
        """
        Log a full task plan to the plans sheet (non-blocking).

        This creates a dedicated record of each plan created during workflow execution,
        making it easy to see all plans at a glance.

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            task_plan: List of task dictionaries with full details
            node: Current node (default: create_plan)
            agent_name: Agent that created the plan
            status: Plan status (ok, error)
        """
        if not self.is_connected():
            return

        num_tasks = len(task_plan)
        plan_summary = f"Created {num_tasks} tasks for {company_name}"

        # Extract individual tasks for easy viewing (up to 10)
        task_columns = []
        for i in range(10):
            if i < len(task_plan):
                task = task_plan[i]
                # Format task as readable string
                task_str = json.dumps(task, default=str) if isinstance(task, dict) else str(task)
                task_columns.append(self._safe_str(task_str, max_length=5000))
            else:
                task_columns.append("")

        row = [
            run_id,
            company_name,
            node or "create_plan",
            agent_name or "",
            num_tasks,
            plan_summary,
            self._safe_str(task_plan),  # Full plan as JSON
            *task_columns,  # task_1 through task_10
            datetime.utcnow().isoformat(),  # created_at
            status,
            "Us",  # generated_by
        ]

        def _write():
            try:
                sheet = self._get_sheet("plans")
                sheet.append_row(row)
                logger.info(f"Logged plan for: {company_name} (run: {run_id}, {num_tasks} tasks)")
            except Exception as e:
                logger.error(f"Failed to log plan to sheets: {e}")

        _sheets_executor.submit(_write)

    def log_prompt(
        self,
        run_id: str,
        company_name: str,
        prompt_id: str,
        prompt_name: str,
        category: str,
        system_prompt: str,
        user_prompt: str,
        variables: Dict[str, Any] = None,
        # Common fields
        node: str = "",
        agent_name: str = "",
        step_number: int = 0,
        model: str = "",
        temperature: float = None,
    ):
        """
        Log a prompt used during a run (non-blocking).

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            prompt_id: ID of the prompt (e.g., "company_parser")
            prompt_name: Human-readable name
            category: Prompt category (input, planning, synthesis, etc.)
            system_prompt: Full system prompt text
            user_prompt: Full user prompt text (after variable substitution)
            variables: Variables used in the prompt
            node: Current workflow node
            agent_name: Agent using the prompt
            step_number: Step number in workflow
            model: LLM model being used
            temperature: LLM temperature setting
        """
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "",
            agent_name or "",
            step_number,
            prompt_id,
            prompt_name,
            category,
            self._safe_str(system_prompt),  # Full system prompt
            self._safe_str(user_prompt),  # Full user prompt
            json.dumps(variables or {}, default=str),  # Variables as JSON
            model or "",
            temperature if temperature is not None else "",
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by
        ]

        def _write():
            try:
                sheet = self._get_sheet("prompts")
                sheet.append_row(row)
                logger.info(f"Logged prompt {prompt_id} for: {company_name} (run: {run_id})")
            except Exception as e:
                logger.error(f"Failed to log prompt to sheets: {e}")

        _sheets_executor.submit(_write)

    def get_spreadsheet_url(self) -> Optional[str]:
        """Get the URL of the spreadsheet."""
        if self.spreadsheet:
            return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet.id}"
        return None

    def log_verification(
        self,
        run_id: str,
        company_name: str,
        expected_sheets: List[str] = None,
    ):
        """
        Log verification of all sheet logging for a run.

        Counts rows in each sheet for this run_id and logs summary to log_tests sheet.
        Simple format: shows "count" or "0" for each sheet.

        Args:
            run_id: The run ID to verify
            company_name: Company name for the run
            expected_sheets: Optional list of sheets that should have data
        """
        if not self.is_connected():
            return

        # Core sheets to check (matches log_tests columns)
        core_sheets = [
            "runs", "langgraph_events", "llm_calls", "tool_calls",
            "assessments", "evaluations", "tool_selections",
            "consistency_scores", "data_sources", "plans", "prompts"
        ]

        def _verify_and_log():
            try:
                sheet_counts = {}
                sheets_with_data = 0

                for sheet_name in core_sheets:
                    try:
                        sheet = self._get_sheet(sheet_name)
                        if sheet:
                            all_values = sheet.get_all_values()
                            count = sum(1 for row in all_values[1:] if row and row[0] == run_id)
                            sheet_counts[sheet_name] = count
                            if count > 0:
                                sheets_with_data += 1
                        else:
                            sheet_counts[sheet_name] = 0
                    except Exception as e:
                        logger.warning(f"Failed to check sheet {sheet_name}: {e}")
                        sheet_counts[sheet_name] = 0

                # Determine status
                status = "pass" if sheets_with_data >= 5 else ("partial" if sheets_with_data > 0 else "fail")

                # Build row - simple format showing count for each sheet
                row = [
                    run_id,
                    company_name,
                    # Per-sheet: show count (or 0)
                    sheet_counts.get("runs", 0),
                    sheet_counts.get("langgraph_events", 0),
                    sheet_counts.get("llm_calls", 0),
                    sheet_counts.get("tool_calls", 0),
                    sheet_counts.get("assessments", 0),
                    sheet_counts.get("evaluations", 0),
                    sheet_counts.get("tool_selections", 0),
                    sheet_counts.get("consistency_scores", 0),
                    sheet_counts.get("data_sources", 0),
                    sheet_counts.get("plans", 0),
                    sheet_counts.get("prompts", 0),
                    # Summary
                    sheets_with_data,
                    status,
                    datetime.utcnow().isoformat(),
                    "Us",
                ]

                sheet = self._get_sheet("log_tests")
                sheet.append_row(row)
                logger.info(f"Log verification for run {run_id}: {status} ({sheets_with_data}/11 sheets)")

            except Exception as e:
                logger.error(f"Failed to log verification: {e}")

        _sheets_executor.submit(_verify_and_log)

    def verify_run_logging(self, run_id: str) -> Dict[str, Any]:
        """
        Verify logging for a specific run_id and return results.

        This is a synchronous version for debugging/testing.

        Args:
            run_id: The run ID to verify

        Returns:
            Dict with verification results per sheet
        """
        if not self.is_connected():
            return {"error": "Not connected to Google Sheets"}

        # Only include sheets that exist (redundant sheets have been deleted)
        all_sheets = [
            "runs", "langgraph_events", "llm_calls", "tool_calls",
            "assessments", "evaluations", "tool_selections",
            "consistency_scores", "data_sources", "plans", "prompts", "log_tests"
        ]

        results = {
            "run_id": run_id,
            "sheets": {},
            "total_rows": 0,
            "sheets_with_data": 0,
        }

        for sheet_name in all_sheets:
            try:
                sheet = self._get_sheet(sheet_name)
                if sheet:
                    all_values = sheet.get_all_values()
                    count = sum(1 for row in all_values[1:] if row and row[0] == run_id)
                    results["sheets"][sheet_name] = count
                    results["total_rows"] += count
                    if count > 0:
                        results["sheets_with_data"] += 1
                else:
                    results["sheets"][sheet_name] = 0
            except Exception as e:
                results["sheets"][sheet_name] = f"Error: {e}"

        return results


# Singleton instance
_sheets_logger: Optional[SheetsLogger] = None


def get_sheets_logger(force_new: bool = False) -> SheetsLogger:
    """Get the global SheetsLogger instance."""
    global _sheets_logger
    if _sheets_logger is None or force_new:
        _sheets_logger = SheetsLogger()
    return _sheets_logger
