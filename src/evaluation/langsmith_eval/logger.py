"""
LangSmith Evaluation Logger - Separate logging for LangSmith evaluations.

Logs evaluation results to:
- Google Sheets (langsmith_evaluations tab) - SEPARATE from other logs
- MongoDB (langsmith_evaluations collection)
- Local JSON files

This is completely separate from the existing evaluation logging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for async logging
_eval_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="langsmith_eval_logger")

# Try to import gspread
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logger.warning("gspread not installed. Google Sheets logging disabled.")

# Try to import MongoDB
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("pymongo not installed. MongoDB logging disabled.")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvaluationRun:
    """Represents a single evaluation run."""
    eval_id: str
    dataset_name: str
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Counts
    total_examples: int = 0
    passed_examples: int = 0
    failed_examples: int = 0

    # Aggregate Scores
    avg_risk_accuracy: float = 0.0
    avg_score_accuracy: float = 0.0
    avg_tool_selection_f1: float = 0.0
    avg_synthesis_quality: float = 0.0
    avg_trajectory_match: float = 0.0
    overall_score: float = 0.0

    # Metadata
    model_config: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleResult:
    """Represents evaluation result for a single example."""
    eval_id: str
    example_id: str
    company_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Individual Scores
    risk_level_correct: bool = False
    credit_score_accuracy: float = 0.0
    tool_selection_f1: float = 0.0
    synthesis_quality: float = 0.0
    trajectory_match: float = 0.0
    llm_judge_score: float = 0.0

    # Outputs
    actual_risk_level: str = ""
    expected_risk_level: str = ""
    actual_credit_score: int = 0
    expected_credit_score_range: str = ""
    actual_tools: str = ""
    expected_tools: str = ""

    # Comments
    tool_selection_comment: str = ""
    synthesis_comment: str = ""
    trajectory_comment: str = ""
    llm_judge_comment: str = ""

    # Status
    passed: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# LANGSMITH EVALUATION LOGGER
# =============================================================================

class LangSmithEvalLogger:
    """
    Logger specifically for LangSmith evaluation results.

    Logs to a SEPARATE Google Sheets tab (langsmith_evaluations) and
    MongoDB collection (langsmith_evaluations).

    This is independent from other logging in the system.
    """

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Sheet configurations for LangSmith evaluations
    SHEET_CONFIGS = {
        "langsmith_eval_runs": [
            "eval_id", "dataset_name", "experiment_name", "timestamp",
            "total_examples", "passed_examples", "failed_examples",
            "avg_risk_accuracy", "avg_score_accuracy", "avg_tool_selection_f1",
            "avg_synthesis_quality", "avg_trajectory_match", "overall_score",
            "duration_seconds", "model_config"
        ],
        "langsmith_eval_examples": [
            "eval_id", "example_id", "company_name", "timestamp",
            "risk_level_correct", "credit_score_accuracy", "tool_selection_f1",
            "synthesis_quality", "trajectory_match", "llm_judge_score",
            "actual_risk_level", "expected_risk_level",
            "actual_credit_score", "expected_credit_score_range",
            "actual_tools", "expected_tools",
            "tool_selection_comment", "synthesis_comment",
            "passed", "error"
        ],
    }

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        spreadsheet_id: Optional[str] = None,
        mongodb_uri: Optional[str] = None,
    ):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SPREADSHEET_ID")
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")

        self.sheets_client = None
        self.spreadsheet = None
        self.mongo_client = None
        self.mongo_db = None
        self._sheets_cache: Dict[str, Any] = {}

        # Local storage for offline logging
        self.local_path = Path(__file__).parent.parent.parent.parent / "data" / "langsmith_eval_logs"
        self.local_path.mkdir(parents=True, exist_ok=True)

        # Initialize connections
        if GSPREAD_AVAILABLE and self.credentials_path:
            self._connect_sheets()

        if MONGODB_AVAILABLE and self.mongodb_uri:
            self._connect_mongodb()

        logger.info(f"LangSmithEvalLogger initialized: sheets={self.sheets_connected()}, mongodb={self.mongodb_connected()}")

    def _connect_sheets(self):
        """Connect to Google Sheets."""
        try:
            creds_path = Path(self.credentials_path)
            if not creds_path.is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent
                creds_path = project_root / self.credentials_path

            if not creds_path.exists():
                logger.warning(f"Credentials file not found: {creds_path}")
                return

            creds = Credentials.from_service_account_file(
                str(creds_path),
                scopes=self.SCOPES
            )
            self.sheets_client = gspread.authorize(creds)

            if self.spreadsheet_id:
                self.spreadsheet = self.sheets_client.open_by_key(self.spreadsheet_id)
            else:
                try:
                    self.spreadsheet = self.sheets_client.open("Credit Intelligence Logs")
                except gspread.SpreadsheetNotFound:
                    self.spreadsheet = self.sheets_client.create("Credit Intelligence Logs")

            self._init_sheets()
            logger.info(f"LangSmithEvalLogger connected to Sheets: {self.spreadsheet.title}")

        except Exception as e:
            logger.warning(f"Failed to connect to Google Sheets: {e}")

    def _init_sheets(self):
        """Initialize LangSmith evaluation sheets."""
        if not self.spreadsheet:
            return

        existing_sheets = [ws.title for ws in self.spreadsheet.worksheets()]

        for sheet_name, headers in self.SHEET_CONFIGS.items():
            if sheet_name not in existing_sheets:
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name, rows=1000, cols=len(headers)
                )
                worksheet.append_row(headers)
                logger.info(f"Created sheet: {sheet_name}")
            self._sheets_cache[sheet_name] = self.spreadsheet.worksheet(sheet_name)

    def _connect_mongodb(self):
        """Connect to MongoDB."""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.mongo_client.admin.command('ping')
            self.mongo_db = self.mongo_client.credit_intelligence
            logger.info("LangSmithEvalLogger connected to MongoDB")
        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {e}")

    def sheets_connected(self) -> bool:
        return self.spreadsheet is not None

    def mongodb_connected(self) -> bool:
        return self.mongo_db is not None

    def _get_sheet(self, name: str):
        if name not in self._sheets_cache and self.spreadsheet:
            try:
                self._sheets_cache[name] = self.spreadsheet.worksheet(name)
            except Exception:
                return None
        return self._sheets_cache.get(name)

    def _safe_str(self, value: Any, max_length: int = 5000) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            s = json.dumps(value, default=str)
        else:
            s = str(value)
        return s[:max_length] if len(s) > max_length else s

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def log_evaluation_run(self, run: EvaluationRun):
        """
        Log an evaluation run summary.

        Args:
            run: EvaluationRun object with aggregate results
        """
        # Log to sheets (non-blocking)
        if self.sheets_connected():
            row = [
                run.eval_id,
                run.dataset_name,
                run.experiment_name,
                run.timestamp,
                run.total_examples,
                run.passed_examples,
                run.failed_examples,
                round(run.avg_risk_accuracy, 4),
                round(run.avg_score_accuracy, 4),
                round(run.avg_tool_selection_f1, 4),
                round(run.avg_synthesis_quality, 4),
                round(run.avg_trajectory_match, 4),
                round(run.overall_score, 4),
                round(run.duration_seconds, 2),
                self._safe_str(run.model_config),
            ]

            def _write():
                try:
                    sheet = self._get_sheet("langsmith_eval_runs")
                    if sheet:
                        sheet.append_row(row)
                except Exception as e:
                    logger.error(f"Failed to log eval run to sheets: {e}")

            _eval_executor.submit(_write)

        # Log to MongoDB (non-blocking)
        if self.mongodb_connected():
            def _write_mongo():
                try:
                    self.mongo_db.langsmith_eval_runs.insert_one(run.to_dict())
                except Exception as e:
                    logger.error(f"Failed to log eval run to MongoDB: {e}")

            _eval_executor.submit(_write_mongo)

        # Always log locally
        self._log_local("runs", run.to_dict())

    def log_example_result(self, result: ExampleResult):
        """
        Log a single example evaluation result.

        Args:
            result: ExampleResult object
        """
        # Log to sheets (non-blocking)
        if self.sheets_connected():
            row = [
                result.eval_id,
                result.example_id,
                result.company_name,
                result.timestamp,
                "Yes" if result.risk_level_correct else "No",
                round(result.credit_score_accuracy, 4),
                round(result.tool_selection_f1, 4),
                round(result.synthesis_quality, 4),
                round(result.trajectory_match, 4),
                round(result.llm_judge_score, 4),
                result.actual_risk_level,
                result.expected_risk_level,
                result.actual_credit_score,
                result.expected_credit_score_range,
                result.actual_tools,
                result.expected_tools,
                self._safe_str(result.tool_selection_comment, 500),
                self._safe_str(result.synthesis_comment, 500),
                "Pass" if result.passed else "Fail",
                result.error or "",
            ]

            def _write():
                try:
                    sheet = self._get_sheet("langsmith_eval_examples")
                    if sheet:
                        sheet.append_row(row)
                except Exception as e:
                    logger.error(f"Failed to log example result to sheets: {e}")

            _eval_executor.submit(_write)

        # Log to MongoDB (non-blocking)
        if self.mongodb_connected():
            def _write_mongo():
                try:
                    self.mongo_db.langsmith_eval_examples.insert_one(result.to_dict())
                except Exception as e:
                    logger.error(f"Failed to log example result to MongoDB: {e}")

            _eval_executor.submit(_write_mongo)

        # Always log locally
        self._log_local("examples", result.to_dict())

    def log_batch_results(
        self,
        eval_id: str,
        dataset_name: str,
        experiment_name: str,
        results: List[Dict[str, Any]],
        duration_seconds: float = 0.0,
        model_config: Dict = None,
    ):
        """
        Log a batch of evaluation results.

        Args:
            eval_id: Unique evaluation ID
            dataset_name: Name of the dataset used
            experiment_name: Name of the experiment
            results: List of result dicts from LangSmith evaluate()
            duration_seconds: Total duration
            model_config: Model configuration used
        """
        # Calculate aggregates
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))

        avg_risk = sum(r.get("risk_level_correct", 0) for r in results) / total if total else 0
        avg_score = sum(r.get("credit_score_accuracy", 0) for r in results) / total if total else 0
        avg_f1 = sum(r.get("tool_selection_f1", 0) for r in results) / total if total else 0
        avg_synthesis = sum(r.get("synthesis_quality", 0) for r in results) / total if total else 0
        avg_trajectory = sum(r.get("trajectory_match", 0) for r in results) / total if total else 0

        overall = (avg_risk + avg_score + avg_f1 + avg_synthesis + avg_trajectory) / 5

        # Log run summary
        run = EvaluationRun(
            eval_id=eval_id,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            total_examples=total,
            passed_examples=passed,
            failed_examples=total - passed,
            avg_risk_accuracy=avg_risk,
            avg_score_accuracy=avg_score,
            avg_tool_selection_f1=avg_f1,
            avg_synthesis_quality=avg_synthesis,
            avg_trajectory_match=avg_trajectory,
            overall_score=overall,
            duration_seconds=duration_seconds,
            model_config=model_config or {},
        )
        self.log_evaluation_run(run)

        # Log individual results
        for i, r in enumerate(results):
            example_result = ExampleResult(
                eval_id=eval_id,
                example_id=r.get("example_id", f"example_{i}"),
                company_name=r.get("company_name", ""),
                risk_level_correct=r.get("risk_level_correct", False),
                credit_score_accuracy=r.get("credit_score_accuracy", 0),
                tool_selection_f1=r.get("tool_selection_f1", 0),
                synthesis_quality=r.get("synthesis_quality", 0),
                trajectory_match=r.get("trajectory_match", 0),
                llm_judge_score=r.get("llm_judge_score", 0),
                actual_risk_level=r.get("actual_risk_level", ""),
                expected_risk_level=r.get("expected_risk_level", ""),
                actual_credit_score=r.get("actual_credit_score", 0),
                expected_credit_score_range=str(r.get("expected_credit_score_range", "")),
                actual_tools=", ".join(r.get("actual_tools", [])),
                expected_tools=", ".join(r.get("expected_tools", [])),
                tool_selection_comment=r.get("tool_selection_comment", ""),
                synthesis_comment=r.get("synthesis_comment", ""),
                passed=r.get("passed", False),
                error=r.get("error", ""),
            )
            self.log_example_result(example_result)

        logger.info(f"Logged batch results: {passed}/{total} passed (overall: {overall:.2f})")

    def _log_local(self, log_type: str, data: Dict):
        """Log to local JSON file."""
        try:
            date_str = datetime.utcnow().strftime("%Y%m%d")
            filepath = self.local_path / f"{log_type}_{date_str}.jsonl"
            with open(filepath, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log locally: {e}")

    def get_spreadsheet_url(self) -> Optional[str]:
        """Get the URL of the spreadsheet."""
        if self.spreadsheet:
            return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet.id}"
        return None

    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluation runs from MongoDB."""
        if not self.mongodb_connected():
            return []

        try:
            runs = list(
                self.mongo_db.langsmith_eval_runs
                .find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            return runs
        except Exception as e:
            logger.error(f"Failed to get recent runs: {e}")
            return []


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_eval_logger: Optional[LangSmithEvalLogger] = None


def get_eval_logger(force_new: bool = False) -> LangSmithEvalLogger:
    """Get the global LangSmithEvalLogger instance."""
    global _eval_logger
    if _eval_logger is None or force_new:
        _eval_logger = LangSmithEvalLogger()
    return _eval_logger
