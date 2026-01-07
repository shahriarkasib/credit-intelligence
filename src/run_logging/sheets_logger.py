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
            # Sheet 2: Tool execution logs
            "tool_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "tool_name", "tool_input", "tool_output",
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
            # Sheet 6: Detailed step-by-step logs
            "step_logs": [
                "run_id", "company_name", "node", "node_type", "agent_name",
                "step_name", "step_number", "model", "temperature",
                "input_summary", "output_summary",
                "execution_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 7: LLM call logs (Mixed: we log, but tokens/response from LLM)
            "llm_calls": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "call_type", "model", "temperature", "context",
                "prompt_summary", "response_summary",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_cost", "output_cost", "total_cost",
                "execution_time_ms", "status",
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
            # Sheet 12: Task 17 - Detailed LLM call logs (Mixed: we log, but tokens/response from LLM)
            "llm_calls_detailed": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "llm_provider", "model", "temperature",
                "prompt", "context", "response", "reasoning",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_cost", "output_cost", "total_cost",
                "response_time_ms", "status", "error",
                "timestamp", "generated_by"
            ],
            # Sheet 13: Task 17 - Comprehensive run summaries
            "run_summaries": [
                "run_id", "company_name", "node", "node_type", "agent_name", "model", "temperature",
                "status", "risk_level", "credit_score", "confidence", "reasoning",
                "tool_selection_score", "data_quality_score", "synthesis_score", "overall_score",
                "final_decision", "decision_reasoning",
                "errors", "warnings", "tools_used", "agents_used", "total_steps",
                "started_at", "completed_at", "duration_ms",
                "total_tokens", "total_cost", "llm_calls_count",
                "timestamp", "generated_by"
            ],
            # Sheet 14: Task 4 - Agent efficiency metrics
            "agent_metrics": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number", "model",
                # Core agent metrics (0-1 scores)
                "intent_correctness", "plan_quality", "tool_choice_correctness",
                "tool_completeness", "trajectory_match", "final_answer_quality",
                # Execution metrics
                "step_count", "tool_calls", "latency_ms",
                # Overall score
                "overall_score", "eval_status",  # good/average/bad
                # Details (JSON)
                "intent_details", "plan_details", "tool_details",
                "trajectory_details", "answer_details",
                "status", "timestamp", "generated_by"
            ],
            # Sheet 15: Task 21 - LLM-as-a-judge evaluations
            "llm_judge_results": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model_used", "temperature",
                # Dimension scores (0-1)
                "accuracy_score", "completeness_score", "consistency_score",
                "actionability_score", "data_utilization_score", "overall_score",
                "eval_status",  # good/average/bad
                # Reasoning
                "accuracy_reasoning", "completeness_reasoning", "consistency_reasoning",
                "actionability_reasoning", "data_utilization_reasoning", "overall_reasoning",
                # Benchmark comparison
                "benchmark_alignment", "benchmark_comparison",
                # Suggestions
                "suggestions",
                # Metadata
                "tokens_used", "evaluation_cost", "duration_ms", "status",
                "timestamp", "generated_by"
            ],
            # Sheet 17: Task 21 Enhanced - Model consistency evaluation
            "model_consistency": [
                "eval_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model_name", "num_runs",
                # Core consistency metrics
                "risk_level_consistency", "credit_score_mean", "credit_score_std",
                "confidence_variance", "reasoning_similarity",
                "risk_factors_overlap", "recommendations_overlap",
                # Overall
                "overall_consistency", "is_consistent", "consistency_grade",
                "eval_status",  # good/average/bad
                # LLM Judge analysis
                "llm_judge_analysis", "llm_judge_concerns",
                # Run details
                "run_details",
                "duration_ms", "status", "timestamp", "generated_by"
            ],
            # Sheet 18: Task 21 Enhanced - Cross-model comparison
            "cross_model_eval": [
                "eval_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "models_compared", "num_models",
                # Agreement metrics
                "risk_level_agreement", "credit_score_mean", "credit_score_std",
                "credit_score_range", "confidence_agreement",
                # Best model
                "best_model", "best_model_reasoning",
                # Overall
                "cross_model_agreement", "eval_status",  # good/average/bad
                # LLM analysis
                "llm_judge_analysis", "model_recommendations",
                # Per-model results
                "model_results", "pairwise_comparisons",
                "duration_ms", "status", "timestamp", "generated_by"
            ],
            # Sheet 19: DeepEval Metrics - LLM-powered evaluation
            # Source: DeepEval library (https://docs.confident-ai.com/)
            "deepeval_metrics": [
                "run_id", "company_name", "node", "node_type", "agent_name", "step_number",
                "model_used",  # e.g., gpt-4, gpt-4o-mini
                # Core DeepEval metrics (0-1 scores)
                "answer_relevancy",       # Is the answer relevant to the question?
                "faithfulness",           # Is the answer grounded in the provided context?
                "hallucination",          # Does the answer contain hallucinated information? (lower is better)
                "contextual_relevancy",   # Is the retrieval context relevant?
                "bias",                   # Does the answer contain bias? (lower is better)
                "toxicity",               # Does the answer contain toxic content? (lower is better)
                # Overall score
                "overall_score", "eval_status",  # good/average/bad
                # Reasoning from DeepEval
                "answer_relevancy_reason", "faithfulness_reason", "hallucination_reason",
                "contextual_relevancy_reason", "bias_reason",
                # Input/output for reproducibility
                "input_query", "context_summary", "assessment_summary",
                # Metadata
                "evaluation_model", "evaluation_time_ms", "status",
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
    ):
        """Log a tool call (non-blocking)."""
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
        step_number: int,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: float,
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
        """Log a workflow step execution (non-blocking)."""
        if not self.is_connected():
            return

        # Determine status from success flag
        final_status = status if status != "ok" else ("ok" if success else "fail")

        # Prepare data synchronously
        row = [
            run_id,
            company_name,
            node or step_name,  # Use step_name as node if not provided
            node_type or "agent",
            agent_name or "",
            step_name,
            step_number,
            model or "",
            temperature if temperature is not None else 0.1,  # Default temperature
            self._safe_str(input_data),  # Full data (up to 50k)
            self._safe_str(output_data),  # Full data (up to 50k)
            execution_time_ms,
            final_status,
            error or "",  # Full error message
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We log steps via @log_node decorator
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
        # New common fields
        node: str = "",
        node_type: str = "llm",
        agent_name: str = "",
        step_number: int = 0,
        temperature: float = None,
        context: str = "",
        status: str = "ok",
        # Original fields
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
            node or "",
            node_type or "llm",
            agent_name or "",
            step_number,
            call_type,
            model,
            temperature if temperature is not None else 0.1,  # Default temperature
            self._safe_str(context, max_length=5000),
            self._safe_str(prompt),  # Full prompt (up to 50k)
            self._safe_str(response),  # Full response (up to 50k)
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            round(input_cost, 6),
            round(output_cost, 6),
            round(total_cost, 6),
            execution_time_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Mixed",  # generated_by: We log, but tokens/response from LLM (FW)
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

    def log_llm_call_detailed(
        self,
        run_id: str,
        company_name: str,
        llm_provider: str,
        agent_name: str,
        model: str,
        prompt: str,
        # New common fields
        node: str = "",
        node_type: str = "llm",
        step_number: int = 0,
        temperature: float = None,
        status: str = "ok",
        # Original fields
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
            node or "",
            node_type or "llm",
            agent_name,
            step_number,
            llm_provider,
            model,
            temperature if temperature is not None else 0.1,  # Default temperature
            self._safe_str(prompt, max_length=10000),
            self._safe_str(context, max_length=5000),
            self._safe_str(response, max_length=10000),
            self._safe_str(reasoning, max_length=2000),
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            round(input_cost, 6),
            round(output_cost, 6),
            round(total_cost, 6),
            response_time_ms,
            status,
            error or "",
            datetime.utcnow().isoformat(),  # timestamp
            "Mixed",  # generated_by: We log, but tokens/response from LLM (FW)
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
        # New common fields
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        model: str = "",
        temperature: float = None,
        total_steps: int = 0,
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
            node or "",
            node_type or "",
            agent_name or "",
            model or "",
            temperature if temperature is not None else 0.1,  # Default temperature
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
            total_steps,
            started_at,
            completed_at,
            duration_ms,
            total_tokens,
            round(total_cost, 6),
            llm_calls_count,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We aggregate and log run summaries
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
        # New common fields
        node: str = "",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        model: str = "",
        status: str = "ok",
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
            node or "",
            node_type or "agent",
            agent_name or "",
            step_number,
            model or "",
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
            self._get_eval_status(overall_score),  # eval_status
            # Details as JSON
            self._safe_str(intent_details or {}, max_length=5000),
            self._safe_str(plan_details or {}, max_length=5000),
            self._safe_str(tool_details or {}, max_length=5000),
            self._safe_str(trajectory_details or {}, max_length=5000),
            self._safe_str(answer_details or {}, max_length=5000),
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We calculate agent efficiency metrics
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
        # New common fields
        node: str = "",
        node_type: str = "llm",
        agent_name: str = "",
        step_number: int = 0,
        temperature: float = None,
        duration_ms: float = 0,
        status: str = "ok",
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
            node or "",
            node_type or "llm",
            agent_name or "",
            step_number,
            model_used,
            temperature if temperature is not None else 0.1,  # Default temperature
            # Dimension scores
            round(accuracy_score, 4),
            round(completeness_score, 4),
            round(consistency_score, 4),
            round(actionability_score, 4),
            round(data_utilization_score, 4),
            round(overall_score, 4),
            self._get_eval_status(overall_score),  # eval_status
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
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We run LLM-as-a-judge evaluation
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
        # New common fields
        node: str = "",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        duration_ms: float = 0,
        status: str = "ok",
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
            node or "",
            node_type or "agent",
            agent_name or "",
            step_number,
            model_name,
            num_runs,
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
            self._get_eval_status(overall_consistency),  # eval_status
            # LLM analysis
            self._safe_str(llm_judge_analysis, max_length=5000),
            self._safe_str(llm_judge_concerns or [], max_length=2000),
            # Run details
            self._safe_str(run_details or [], max_length=10000),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We calculate model consistency
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
        # New common fields
        node: str = "",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        duration_ms: float = 0,
        status: str = "ok",
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
            node or "",
            node_type or "agent",
            agent_name or "",
            step_number,
            ", ".join(models_compared) if models_compared else "",
            num_models,
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
            self._get_eval_status(cross_model_agreement),  # eval_status
            # LLM analysis
            self._safe_str(llm_judge_analysis, max_length=5000),
            self._safe_str(model_recommendations or [], max_length=2000),
            # Per-model results
            self._safe_str(model_results or {}, max_length=10000),
            self._safe_str(pairwise_comparisons or [], max_length=5000),
            duration_ms,
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We calculate cross-model evaluation
        ]

        def _write():
            try:
                sheet = self._get_sheet("cross_model_eval")
                sheet.append_row(row)
                logger.info(f"Logged cross-model eval for: {company_name} ({len(models_compared)} models)")
            except Exception as e:
                logger.error(f"Failed to log cross-model eval to sheets: {e}")

        _sheets_executor.submit(_write)

    # ==================== DEDICATED EVAL SHEETS ====================

    def log_deepeval_metrics(
        self,
        run_id: str,
        company_name: str,
        model_used: str,
        # New common fields
        node: str = "evaluate",
        node_type: str = "agent",
        agent_name: str = "",
        step_number: int = 0,
        status: str = "ok",
        # Core DeepEval metrics (0-1 scores)
        answer_relevancy: float = 0.0,
        faithfulness: float = 0.0,
        hallucination: float = 0.0,       # Lower is better
        contextual_relevancy: float = 0.0,
        bias: float = 0.0,                # Lower is better
        toxicity: float = 0.0,            # Lower is better
        # Overall
        overall_score: float = 0.0,
        # Reasoning
        answer_relevancy_reason: str = "",
        faithfulness_reason: str = "",
        hallucination_reason: str = "",
        contextual_relevancy_reason: str = "",
        bias_reason: str = "",
        # Input/output
        input_query: str = "",
        context_summary: str = "",
        assessment_summary: str = "",
        # Metadata
        evaluation_model: str = "",
        evaluation_time_ms: float = 0.0,
    ):
        """
        Log DeepEval evaluation metrics.

        DeepEval provides LLM-powered evaluation metrics:
        - answer_relevancy: Is the answer relevant to the question? (0-1, higher is better)
        - faithfulness: Is the answer grounded in the provided context? (0-1, higher is better)
        - hallucination: Does the answer contain hallucinated info? (0-1, LOWER is better)
        - contextual_relevancy: Is the retrieval context relevant? (0-1, higher is better)
        - bias: Does the answer contain bias? (0-1, LOWER is better)
        - toxicity: Does the answer contain toxic content? (0-1, LOWER is better)

        Overall score calculation:
        overall = (answer_relevancy * 0.25 + faithfulness * 0.30 +
                  (1 - hallucination) * 0.25 + contextual_relevancy * 0.10 +
                  (1 - bias) * 0.10)

        Non-blocking (async write).
        """
        if not self.is_connected():
            return

        row = [
            run_id,
            company_name,
            node or "evaluate",
            node_type or "agent",
            agent_name or "",
            step_number,
            model_used,
            # Core metrics
            round(answer_relevancy, 4),
            round(faithfulness, 4),
            round(hallucination, 4),
            round(contextual_relevancy, 4),
            round(bias, 4),
            round(toxicity, 4),
            round(overall_score, 4),
            self._get_eval_status(overall_score),  # eval_status
            # Reasoning
            self._safe_str(answer_relevancy_reason, max_length=2000),
            self._safe_str(faithfulness_reason, max_length=2000),
            self._safe_str(hallucination_reason, max_length=2000),
            self._safe_str(contextual_relevancy_reason, max_length=2000),
            self._safe_str(bias_reason, max_length=2000),
            # Input/output
            self._safe_str(input_query, max_length=1000),
            self._safe_str(context_summary, max_length=5000),
            self._safe_str(assessment_summary, max_length=5000),
            # Metadata
            evaluation_model or model_used,
            round(evaluation_time_ms, 2),
            status,
            datetime.utcnow().isoformat(),  # timestamp
            "Us",  # generated_by: We run DeepEval evaluation
        ]

        def _write():
            try:
                sheet = self._get_sheet("deepeval_metrics")
                sheet.append_row(row)
                logger.info(f"Logged DeepEval metrics for: {company_name} (run: {run_id}, overall: {overall_score:.2f})")
            except Exception as e:
                logger.error(f"Failed to log DeepEval metrics to sheets: {e}")

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
