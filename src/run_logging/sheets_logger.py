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
            os.getenv("GOOGLE_CREDENTIALS_JSON") or
            os.getenv("GOOGLE_SHEETS_CREDENTIALS")
        )
        if GSPREAD_AVAILABLE and has_credentials:
            self._connect()

    def _connect(self):
        """Connect to Google Sheets.

        Supports three credential methods:
        1. GOOGLE_CREDENTIALS_JSON: Plain JSON string (for Heroku - easiest)
        2. GOOGLE_SHEETS_CREDENTIALS: Base64-encoded JSON (legacy cloud method)
        3. GOOGLE_CREDENTIALS_PATH: File path to JSON file (local development)
        """
        try:
            import base64

            creds = None

            # Method 1: Plain JSON environment variable (easiest for Heroku)
            plain_json_creds = os.getenv("GOOGLE_CREDENTIALS_JSON")
            if plain_json_creds:
                try:
                    creds_dict = json.loads(plain_json_creds)
                    creds = Credentials.from_service_account_info(
                        creds_dict,
                        scopes=self.SCOPES
                    )
                    logger.info("Using Google credentials from GOOGLE_CREDENTIALS_JSON env var")
                except Exception as e:
                    logger.warning(f"Failed to load credentials from GOOGLE_CREDENTIALS_JSON: {e}")

            # Method 2: Base64-encoded environment variable (legacy)
            if creds is None:
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
                        logger.info("Using Google credentials from GOOGLE_SHEETS_CREDENTIALS env var")
                    except Exception as e:
                        logger.warning(f"Failed to load credentials from GOOGLE_SHEETS_CREDENTIALS: {e}")

            # Method 3: File-based credentials (for local development)
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
                "run_id", "company_name", "node", "agent_name", "master_agent", "model", "temperature",
                "status", "started_at", "completed_at",
                "risk_level", "credit_score", "confidence", "total_time_ms",
                "total_steps", "total_llm_calls", "tools_used", "evaluation_score",
                "timestamp", "generated_by"
            ],
            # Sheet 2: Tool execution logs (with hierarchy tracking)
            "tool_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "tool_name", "tool_input", "tool_output",
                # Hierarchy columns for traceability
                "parent_node", "workflow_phase", "call_depth", "parent_tool_id",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 3: Credit assessments
            "assessments": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "model", "temperature", "prompt",
                "risk_level", "credit_score", "confidence", "reasoning", "recommendations",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 4: Evaluation results
            "evaluations": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number", "model",
                "tool_selection_score", "tool_reasoning",
                "data_quality_score", "data_reasoning",
                "synthesis_score", "synthesis_reasoning", "overall_score",
                "eval_status",  # good/average/bad based on overall_score
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 5: Tool selection decisions
            "tool_selections": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number", "model",
                "selected_tools", "expected_tools", "correct_tools", "missing_tools", "extra_tools",
                "precision", "recall", "f1_score", "reasoning",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 6: LLM call logs - ALL LLM calls with full details
            # Includes: parse_input, tool_selection, credit_analysis, evaluations
            "llm_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
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
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "model_name", "evaluation_type", "num_runs",
                "risk_level_consistency", "score_consistency", "score_std",
                "overall_consistency", "eval_status",  # good/average/bad
                "risk_levels", "credit_scores",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 9: Data source results
            "data_sources": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "source_name", "records_found", "data_summary",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 10: LangGraph events (FW: events from astream_events)
            "langgraph_events": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "event_type", "event_name", "model", "temperature", "tokens",
                "input_preview", "output_preview",
                "duration_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 10: Plans - Full task plans created for each run
            "plans": [
                "run_id", "company_name", "node", "agent_name", "master_agent",
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
                "run_id", "company_name", "node", "agent_name", "master_agent", "step_number",
                "prompt_id", "prompt_name", "category",
                # Full prompt text
                "system_prompt", "user_prompt",
                # Variables used
                "variables_json",
                # Model and execution info
                "model", "temperature",
                "timestamp", "generated_by"
            ],
            # Sheet 12: Cross-model evaluation results
            "cross_model_eval": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "models_compared", "num_models",
                "risk_level_agreement", "credit_score_mean", "credit_score_std", "credit_score_range",
                "confidence_agreement", "best_model", "best_model_reasoning",
                "cross_model_agreement", "eval_status",
                "llm_judge_analysis", "model_recommendations",
                "model_results", "pairwise_comparisons",
                "duration_ms", "status", "timestamp", "generated_by"
            ],
            # Sheet 13: LLM-as-judge evaluation results
            "llm_judge_results": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                "model_used", "temperature",
                "accuracy_score", "completeness_score", "consistency_score",
                "actionability_score", "data_utilization_score", "overall_score", "eval_status",
                "accuracy_reasoning", "completeness_reasoning", "consistency_reasoning",
                "actionability_reasoning", "data_utilization_reasoning", "overall_reasoning",
                "benchmark_alignment", "benchmark_comparison", "suggestions",
                "tokens_used", "evaluation_cost", "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 14: Agent efficiency metrics
            "agent_metrics": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number", "model",
                "intent_correctness", "plan_quality", "tool_choice_correctness",
                "tool_completeness", "trajectory_match", "final_answer_quality",
                "step_count", "tool_calls", "latency_ms",
                "overall_score", "eval_status",
                "intent_details", "plan_details", "tool_details", "trajectory_details", "answer_details",
                "status", "timestamp", "generated_by"
            ],
            # Sheet 15: Coalition evaluation results
            "coalition": [
                "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
                # Overall correctness
                "is_correct", "correctness_score", "confidence", "correctness_category",
                # Component scores
                "efficiency_score", "quality_score", "tool_score", "consistency_score",
                # Coalition details
                "agreement_score", "num_evaluators",
                # Individual evaluator votes (JSON)
                "votes_json",
                # Metadata
                "evaluation_time_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 16: Log Tests - Simple verification of sheet logging per run
            "log_tests": [
                "run_id", "company_name",
                # Per-sheet verification (count for each sheet)
                "runs", "langgraph_events", "llm_calls", "tool_calls",
                "assessments", "evaluations", "tool_selections",
                "consistency_scores", "data_sources", "plans", "prompts",
                "cross_model_eval", "llm_judge_results", "agent_metrics", "coalition",
                # Summary
                "total_sheets_logged", "verification_status",
                "timestamp", "generated_by"
            ],
            # Sheet 17: State Dumps - Full workflow state snapshots
            "state_dumps": [
                "run_id", "company_name", "node", "master_agent", "step_number",
                # Company info (JSON)
                "company_info_json",
                # Plan (JSON)
                "plan_json", "plan_size_bytes", "plan_tasks_count",
                # API data (JSON summary)
                "api_data_summary", "api_data_size_bytes", "api_sources_count",
                # Search data (JSON summary)
                "search_data_summary", "search_data_size_bytes",
                # Assessment
                "risk_level", "credit_score", "confidence", "assessment_json",
                # Evaluation scores
                "coalition_score", "agent_metrics_score", "evaluation_json",
                # Errors
                "errors_json", "error_count",
                # Metadata
                "total_state_size_bytes", "duration_ms", "status",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
        master_agent: str = "supervisor",
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
        # run_id, company_name, node, node_type, agent_name, master_agent, step_number,
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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

    def log_agent_metrics(
        self,
        run_id: str,
        company_name: str,
        node: str = "",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        model: str = "",
        status: str = "ok",
        intent_correctness: float = 0.0,
        plan_quality: float = 0.0,
        tool_choice_correctness: float = 0.0,
        tool_completeness: float = 0.0,
        trajectory_match: float = 0.0,
        final_answer_quality: float = 0.0,
        step_count: int = 0,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
        overall_score: float = 0.0,
        intent_details: Dict[str, Any] = None,
        plan_details: Dict[str, Any] = None,
        tool_details: Dict[str, Any] = None,
        trajectory_details: Dict[str, Any] = None,
        answer_details: Dict[str, Any] = None,
    ):
        """Log agent efficiency metrics to agent_metrics sheet (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "",
            node_type or "agent",
            agent_name or "",
            master_agent or "supervisor",
            step_number,
            model or "",
            round(intent_correctness, 4),
            round(plan_quality, 4),
            round(tool_choice_correctness, 4),
            round(tool_completeness, 4),
            round(trajectory_match, 4),
            round(final_answer_quality, 4),
            step_count,
            tool_calls,
            round(latency_ms, 2),
            round(overall_score, 4),
            self._get_eval_status(overall_score),
            self._safe_str(intent_details or {}, max_length=5000),
            self._safe_str(plan_details or {}, max_length=5000),
            self._safe_str(tool_details or {}, max_length=5000),
            self._safe_str(trajectory_details or {}, max_length=5000),
            self._safe_str(answer_details or {}, max_length=5000),
            status,
            datetime.utcnow().isoformat(),
            "Us",
        ]

        def _write():
            try:
                sheet = self._get_sheet("agent_metrics")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log agent metrics: {e}")

        _sheets_executor.submit(_write)

    def log_llm_judge_result(
        self,
        run_id: str,
        company_name: str,
        model_used: str,
        node: str = "",
        node_type: str = "llm",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        temperature: float = None,
        duration_ms: float = 0,
        status: str = "ok",
        accuracy_score: float = 0.0,
        completeness_score: float = 0.0,
        consistency_score: float = 0.0,
        actionability_score: float = 0.0,
        data_utilization_score: float = 0.0,
        overall_score: float = 0.0,
        accuracy_reasoning: str = "",
        completeness_reasoning: str = "",
        consistency_reasoning: str = "",
        actionability_reasoning: str = "",
        data_utilization_reasoning: str = "",
        overall_reasoning: str = "",
        benchmark_alignment: float = 0.0,
        benchmark_comparison: str = "",
        suggestions: List[str] = None,
        tokens_used: int = 0,
        evaluation_cost: float = 0.0,
    ):
        """Log LLM-as-a-judge evaluation result to llm_judge_results sheet (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "",
            node_type or "llm",
            agent_name or "",
            master_agent or "supervisor",
            step_number,
            model_used,
            temperature if temperature is not None else 0.1,
            round(accuracy_score, 4),
            round(completeness_score, 4),
            round(consistency_score, 4),
            round(actionability_score, 4),
            round(data_utilization_score, 4),
            round(overall_score, 4),
            self._get_eval_status(overall_score),
            self._safe_str(accuracy_reasoning, max_length=2000),
            self._safe_str(completeness_reasoning, max_length=2000),
            self._safe_str(consistency_reasoning, max_length=2000),
            self._safe_str(actionability_reasoning, max_length=2000),
            self._safe_str(data_utilization_reasoning, max_length=2000),
            self._safe_str(overall_reasoning, max_length=5000),
            round(benchmark_alignment, 4) if benchmark_alignment else 0,
            self._safe_str(benchmark_comparison, max_length=5000),
            self._safe_str(suggestions or [], max_length=5000),
            tokens_used,
            round(evaluation_cost, 6),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),
            "Us",
        ]

        def _write():
            try:
                sheet = self._get_sheet("llm_judge_results")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log LLM judge result: {e}")

        _sheets_executor.submit(_write)

    def log_coalition(
        self,
        run_id: str,
        company_name: str,
        node: str = "evaluate",
        node_type: str = "evaluator",
        agent_name: str = "coalition_evaluator",
        master_agent: str = "supervisor",
        step_number: int = 0,
        is_correct: bool = False,
        correctness_score: float = 0.0,
        confidence: float = 0.0,
        correctness_category: str = "low",
        efficiency_score: float = 0.0,
        quality_score: float = 0.0,
        tool_score: float = 0.0,
        consistency_score: float = 0.0,
        agreement_score: float = 0.0,
        num_evaluators: int = 0,
        votes: List[Dict[str, Any]] = None,
        evaluation_time_ms: float = 0.0,
        status: str = "ok",
    ):
        """
        Log coalition evaluation result to coalition sheet (non-blocking).

        The coalition evaluator combines multiple evaluation methods:
        - Agent Efficiency (intent, plan, tools, trajectory)
        - LLM Quality (completeness, validity)
        - Tool Selection (precision, recall, F1)
        - Consistency (cross-run comparison)

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            node: Graph node (default: evaluate)
            node_type: Type of node (default: evaluator)
            agent_name: Name of agent (default: coalition_evaluator)
            step_number: Step number in workflow
            is_correct: Whether the run is considered correct
            correctness_score: Overall correctness score (0-1)
            confidence: Confidence in the assessment (0-1)
            correctness_category: Category (high/medium/low)
            efficiency_score: Agent efficiency score (0-1)
            quality_score: LLM quality score (0-1)
            tool_score: Tool selection score (0-1)
            consistency_score: Cross-run consistency score (0-1)
            agreement_score: How much evaluators agree (0-1)
            num_evaluators: Number of evaluators in coalition
            votes: List of individual evaluator votes
            evaluation_time_ms: Time taken for evaluation
            status: Status (ok/error)
        """
        if not self.is_connected():
            return

        # Coalition sheet columns:
        # run_id, company_name, node, node_type, agent_name, master_agent, step_number,
        # is_correct, correctness_score, confidence, correctness_category,
        # efficiency_score, quality_score, tool_score, consistency_score,
        # agreement_score, num_evaluators, votes_json,
        # evaluation_time_ms, status, timestamp, generated_by
        row = [
            run_id,
            company_name,
            node or "evaluate",
            node_type or "evaluator",
            agent_name or "coalition_evaluator",
            master_agent or "supervisor",
            step_number,
            "yes" if is_correct else "no",
            round(correctness_score, 4),
            round(confidence, 4),
            correctness_category or "low",
            round(efficiency_score, 4),
            round(quality_score, 4),
            round(tool_score, 4),
            round(consistency_score, 4),
            round(agreement_score, 4),
            num_evaluators,
            self._safe_str(votes or [], max_length=10000),
            round(evaluation_time_ms, 2),
            status,
            datetime.utcnow().isoformat(),
            "Us",
        ]

        def _write():
            try:
                sheet = self._get_sheet("coalition")
                sheet.append_row(row)
                logger.debug(f"Logged coalition result for run {run_id}: score={correctness_score:.2%}, category={correctness_category}")
            except Exception as e:
                logger.error(f"Failed to log coalition result: {e}")

        _sheets_executor.submit(_write)

    def log_state_dump(
        self,
        run_id: str,
        company_name: str,
        node: str = "evaluate",
        master_agent: str = "supervisor",
        step_number: int = 0,
        # Company info
        company_info: Dict[str, Any] = None,
        # Plan
        plan: List[Dict[str, Any]] = None,
        plan_tasks_count: int = 0,
        # API data
        api_data: Dict[str, Any] = None,
        api_sources: List[str] = None,
        # Search data
        search_data: Dict[str, Any] = None,
        # Assessment
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        assessment: Dict[str, Any] = None,
        # Evaluation scores
        coalition_score: float = 0.0,
        agent_metrics_score: float = 0.0,
        evaluation: Dict[str, Any] = None,
        # Errors
        errors: List[str] = None,
        # Metadata
        total_state_size_bytes: int = 0,
        duration_ms: float = 0.0,
        status: str = "completed",
    ):
        """
        Log a full workflow state dump to the state_dumps sheet (non-blocking).

        This captures the complete state of the workflow at a given point,
        similar to LangSmith's state tracking but with full visibility.

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            node: Graph node where state was captured
            step_number: Step number in workflow
            company_info: Parsed company information
            plan: Task plan list
            plan_tasks_count: Number of tasks in plan
            api_data: Data from API sources
            api_sources: List of API sources used
            search_data: Web search results
            risk_level: Assessed risk level
            credit_score: Credit score estimate
            confidence: Confidence score
            assessment: Full assessment dict
            coalition_score: Coalition evaluator score
            agent_metrics_score: Agent efficiency score
            evaluation: Full evaluation dict
            errors: List of errors
            total_state_size_bytes: Total size of state in bytes
            duration_ms: Time to reach this state
            status: Status (completed/failed)
        """
        if not self.is_connected():
            return

        # Calculate sizes
        company_info_json = self._safe_str(company_info or {}, max_length=5000)
        plan_json = self._safe_str(plan or [], max_length=10000)
        plan_size = len(json.dumps(plan or [], default=str).encode('utf-8'))

        api_data_summary = self._safe_str({k: f"{len(str(v))} bytes" for k, v in (api_data or {}).items()}, max_length=5000)
        api_data_size = len(json.dumps(api_data or {}, default=str).encode('utf-8'))

        search_data_summary = self._safe_str({"keys": list((search_data or {}).keys()), "size": f"{len(json.dumps(search_data or {}, default=str))} bytes"} if search_data else {}, max_length=5000)
        search_data_size = len(json.dumps(search_data or {}, default=str).encode('utf-8'))

        assessment_json = self._safe_str(assessment or {}, max_length=10000)
        evaluation_json = self._safe_str(evaluation or {}, max_length=10000)
        errors_json = self._safe_str(errors or [], max_length=5000)

        # state_dumps sheet columns:
        # run_id, company_name, node, master_agent, step_number,
        # company_info_json, plan_json, plan_size_bytes, plan_tasks_count,
        # api_data_summary, api_data_size_bytes, api_sources_count,
        # search_data_summary, search_data_size_bytes,
        # risk_level, credit_score, confidence, assessment_json,
        # coalition_score, agent_metrics_score, evaluation_json,
        # errors_json, error_count,
        # total_state_size_bytes, duration_ms, status,
        # timestamp, generated_by
        row = [
            run_id,
            company_name,
            node or "evaluate",
            master_agent or "supervisor",
            step_number,
            company_info_json,
            plan_json,
            plan_size,
            plan_tasks_count,
            api_data_summary,
            api_data_size,
            len(api_sources or []),
            search_data_summary,
            search_data_size,
            risk_level or "",
            credit_score or 0,
            round(confidence or 0, 4),
            assessment_json,
            round(coalition_score or 0, 4),
            round(agent_metrics_score or 0, 4),
            evaluation_json,
            errors_json,
            len(errors or []),
            total_state_size_bytes or 0,
            round(duration_ms or 0, 2),
            status,
            datetime.utcnow().isoformat(),
            "Us",
        ]

        def _write():
            try:
                sheet = self._get_sheet("state_dumps")
                sheet.append_row(row)
                logger.info(f"Logged state dump for run {run_id}: {total_state_size_bytes} bytes, {len(errors or [])} errors")
            except Exception as e:
                logger.error(f"Failed to log state dump: {e}")

        _sheets_executor.submit(_write)

    def log_model_consistency(self, *args, **kwargs):
        """DEPRECATED: Model consistency now logged via consistency_scores sheet."""
        pass

    def log_cross_model_eval(
        self,
        eval_id: str,
        company_name: str,
        models_compared: List[str],
        num_models: int,
        node: str = "",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        duration_ms: float = 0,
        status: str = "ok",
        risk_level_agreement: float = 0.0,
        credit_score_mean: float = 0.0,
        credit_score_std: float = 0.0,
        credit_score_range: float = 0.0,
        confidence_agreement: float = 0.0,
        best_model: str = "",
        best_model_reasoning: str = "",
        cross_model_agreement: float = 0.0,
        llm_judge_analysis: str = "",
        model_recommendations: List[str] = None,
        model_results: Dict[str, Dict[str, Any]] = None,
        pairwise_comparisons: List[Dict[str, Any]] = None,
    ):
        """Log cross-model evaluation result to cross_model_eval sheet (non-blocking)."""
        if not self.is_connected():
            return

        row = [
            eval_id,
            company_name,
            node or "",
            node_type or "agent",
            agent_name or "",
            master_agent or "supervisor",
            step_number,
            ", ".join(models_compared) if models_compared else "",
            num_models,
            round(risk_level_agreement, 4),
            round(credit_score_mean, 2),
            round(credit_score_std, 2),
            round(credit_score_range, 2),
            round(confidence_agreement, 4),
            best_model,
            self._safe_str(best_model_reasoning, max_length=2000),
            round(cross_model_agreement, 4),
            self._get_eval_status(cross_model_agreement),
            self._safe_str(llm_judge_analysis, max_length=5000),
            self._safe_str(model_recommendations or [], max_length=2000),
            self._safe_str(model_results or {}, max_length=10000),
            self._safe_str(pairwise_comparisons or [], max_length=5000),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),
            "Us",
        ]

        def _write():
            try:
                sheet = self._get_sheet("cross_model_eval")
                sheet.append_row(row)
            except Exception as e:
                logger.error(f"Failed to log cross-model eval: {e}")

        _sheets_executor.submit(_write)

    def log_deepeval_metrics(self, *args, **kwargs):
        """DEPRECATED: DeepEval metrics not currently in use."""
        pass

    def log_plan(
        self,
        run_id: str,
        company_name: str,
        task_plan: List[Dict[str, Any]],
        # Common fields
        node: str = "create_plan",
        agent_name: str = "",
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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
        master_agent: str = "supervisor",
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
            master_agent or "supervisor",
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

        # Core sheets to check (matches log_tests columns exactly)
        core_sheets = [
            "runs", "langgraph_events", "llm_calls", "tool_calls",
            "assessments", "evaluations", "tool_selections",
            "consistency_scores", "data_sources", "plans", "prompts",
            "cross_model_eval", "llm_judge_results", "agent_metrics", "coalition"
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
                # Must match log_tests columns exactly:
                # run_id, company_name, runs, langgraph_events, llm_calls, tool_calls,
                # assessments, evaluations, tool_selections, consistency_scores,
                # data_sources, plans, prompts, cross_model_eval, llm_judge_results,
                # agent_metrics, coalition, total_sheets_logged, verification_status,
                # timestamp, generated_by
                row = [
                    run_id,
                    company_name,
                    # Per-sheet: show count (or 0) - must match header order
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
                    sheet_counts.get("cross_model_eval", 0),
                    sheet_counts.get("llm_judge_results", 0),
                    sheet_counts.get("agent_metrics", 0),
                    sheet_counts.get("coalition", 0),
                    # Summary
                    sheets_with_data,
                    status,
                    datetime.utcnow().isoformat(),
                    "Us",
                ]

                sheet = self._get_sheet("log_tests")
                sheet.append_row(row)
                logger.info(f"Log verification for run {run_id}: {status} ({sheets_with_data}/{len(core_sheets)} sheets)")

            except Exception as e:
                logger.error(f"Failed to log verification: {e}")

        _sheets_executor.submit(_verify_and_log)

    def recreate_sheet(self, sheet_name: str) -> Dict[str, Any]:
        """
        Delete and recreate a specific sheet with proper headers.

        This is useful when column headers get out of sync.

        Args:
            sheet_name: Name of the sheet to recreate

        Returns:
            Dict with status and details
        """
        if not self.is_connected():
            return {"error": "Not connected to Google Sheets", "success": False}

        # Define sheet configurations (same as _init_sheets)
        sheet_configs = {
            "runs": [
                "run_id", "company_name", "node", "agent_name", "model", "temperature",
                "status", "started_at", "completed_at",
                "risk_level", "credit_score", "confidence", "total_time_ms",
                "total_steps", "total_llm_calls", "tools_used", "evaluation_score",
                "timestamp", "generated_by"
            ],
            "tool_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "tool_name", "tool_input", "tool_output",
                "parent_node", "workflow_phase", "call_depth", "parent_tool_id",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            "assessments": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model", "temperature", "prompt",
                "risk_level", "credit_score", "confidence", "reasoning", "recommendations",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            "evaluations": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
                "tool_selection_score", "tool_reasoning",
                "data_quality_score", "data_reasoning",
                "synthesis_score", "synthesis_reasoning", "overall_score",
                "eval_status",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            "tool_selections": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
                "selected_tools", "expected_tools", "correct_tools", "missing_tools", "extra_tools",
                "precision", "recall", "f1_score", "reasoning",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            "llm_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "call_type", "model", "temperature",
                "prompt", "response", "reasoning",
                "context", "current_task",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_cost", "output_cost", "total_cost",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            "consistency_scores": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model_name", "evaluation_type", "num_runs",
                "risk_level_consistency", "score_consistency", "score_std",
                "overall_consistency", "eval_status",
                "risk_levels", "credit_scores",
                "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            "data_sources": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "source_name", "records_found", "data_summary",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            "langgraph_events": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "event_type", "event_name", "model", "temperature", "tokens",
                "input_preview", "output_preview",
                "duration_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            "plans": [
                "run_id", "company_name", "node", "agent_name",
                "num_tasks", "plan_summary",
                "full_plan",
                "task_1", "task_2", "task_3", "task_4", "task_5",
                "task_6", "task_7", "task_8", "task_9", "task_10",
                "created_at", "status", "generated_by"
            ],
            "prompts": [
                "run_id", "company_name", "node", "agent_name", "step_number",
                "prompt_id", "prompt_name", "category",
                "system_prompt", "user_prompt",
                "variables_json",
                "model", "temperature",
                "timestamp", "generated_by"
            ],
            "cross_model_eval": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "models_compared", "num_models",
                "risk_level_agreement", "credit_score_mean", "credit_score_std", "credit_score_range",
                "confidence_agreement", "best_model", "best_model_reasoning",
                "cross_model_agreement", "eval_status",
                "llm_judge_analysis", "model_recommendations",
                "model_results", "pairwise_comparisons",
                "duration_ms", "status", "timestamp", "generated_by"
            ],
            "llm_judge_results": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model_used", "temperature",
                "accuracy_score", "completeness_score", "consistency_score",
                "actionability_score", "data_utilization_score", "overall_score", "eval_status",
                "accuracy_reasoning", "completeness_reasoning", "consistency_reasoning",
                "actionability_reasoning", "data_utilization_reasoning", "overall_reasoning",
                "benchmark_alignment", "benchmark_comparison", "suggestions",
                "tokens_used", "evaluation_cost", "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            "agent_metrics": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
                "intent_correctness", "plan_quality", "tool_choice_correctness",
                "tool_completeness", "trajectory_match", "final_answer_quality",
                "step_count", "tool_calls", "latency_ms",
                "overall_score", "eval_status",
                "intent_details", "plan_details", "tool_details", "trajectory_details", "answer_details",
                "status", "timestamp", "generated_by"
            ],
            "coalition": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "is_correct", "correctness_score", "confidence", "correctness_category",
                "efficiency_score", "quality_score", "tool_score", "consistency_score",
                "agreement_score", "num_evaluators", "votes_json",
                "evaluation_time_ms", "status",
                "timestamp", "generated_by"
            ],
            "log_tests": [
                "run_id", "company_name",
                "runs", "langgraph_events", "llm_calls", "tool_calls",
                "assessments", "evaluations", "tool_selections",
                "consistency_scores", "data_sources", "plans", "prompts",
                "cross_model_eval", "llm_judge_results", "agent_metrics", "coalition",
                "total_sheets_logged", "verification_status",
                "timestamp", "generated_by"
            ],
            "state_dumps": [
                "run_id", "company_name", "node", "step_number",
                "company_info_json",
                "plan_json", "plan_size_bytes", "plan_tasks_count",
                "api_data_summary", "api_data_size_bytes", "api_sources_count",
                "search_data_summary", "search_data_size_bytes",
                "risk_level", "credit_score", "confidence", "assessment_json",
                "coalition_score", "agent_metrics_score", "evaluation_json",
                "errors_json", "error_count",
                "total_state_size_bytes", "duration_ms", "status",
                "timestamp", "generated_by"
            ]
        }

        if sheet_name not in sheet_configs:
            return {"error": f"Unknown sheet: {sheet_name}", "success": False}

        headers = sheet_configs[sheet_name]

        try:
            # Delete the existing sheet if it exists
            try:
                existing_sheet = self.spreadsheet.worksheet(sheet_name)
                self.spreadsheet.del_worksheet(existing_sheet)
                logger.info(f"Deleted existing sheet: {sheet_name}")
            except gspread.WorksheetNotFound:
                logger.info(f"Sheet {sheet_name} does not exist, will create new")

            # Create new sheet with proper headers
            worksheet = self.spreadsheet.add_worksheet(
                title=sheet_name, rows=1000, cols=len(headers)
            )
            worksheet.append_row(headers)

            # Update cache
            self._sheets_cache[sheet_name] = worksheet

            logger.info(f"Recreated sheet: {sheet_name} with {len(headers)} columns")
            return {
                "success": True,
                "sheet_name": sheet_name,
                "columns": len(headers),
                "headers": headers
            }

        except Exception as e:
            logger.error(f"Failed to recreate sheet {sheet_name}: {e}")
            return {"error": str(e), "success": False}

    def recreate_sheets(self, sheet_names: List[str]) -> Dict[str, Any]:
        """
        Delete and recreate multiple sheets with proper headers.

        Args:
            sheet_names: List of sheet names to recreate

        Returns:
            Dict with results for each sheet
        """
        results = {}
        for sheet_name in sheet_names:
            results[sheet_name] = self.recreate_sheet(sheet_name)
        return results

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
            "consistency_scores", "data_sources", "plans", "prompts",
            "cross_model_eval", "llm_judge_results", "agent_metrics", "coalition", "log_tests"
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
