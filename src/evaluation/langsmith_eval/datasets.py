"""
Dataset Management for LangSmith Evaluations.

Provides utilities for creating, managing, and populating evaluation datasets.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    logger.warning("LangSmith SDK not installed")


# =============================================================================
# SAMPLE TEST CASES
# =============================================================================

SAMPLE_TEST_CASES = [
    # Public US Companies
    {
        "inputs": {"company_name": "Apple Inc"},
        "outputs": {
            "expected_risk_level": "low",
            "expected_credit_score_range": [80, 95],
            "expected_tools": ["fetch_sec_data", "fetch_market_data", "web_search"],
            "is_public_company": True,
            "expected_trajectory": ["parse_input", "create_plan", "fetch_api_data", "search_web", "synthesize"],
        },
        "metadata": {"company_type": "public_us", "ticker": "AAPL"},
    },
    {
        "inputs": {"company_name": "Microsoft Corporation"},
        "outputs": {
            "expected_risk_level": "low",
            "expected_credit_score_range": [80, 95],
            "expected_tools": ["fetch_sec_data", "fetch_market_data", "web_search"],
            "is_public_company": True,
            "expected_trajectory": ["parse_input", "create_plan", "fetch_api_data", "search_web", "synthesize"],
        },
        "metadata": {"company_type": "public_us", "ticker": "MSFT"},
    },
    {
        "inputs": {"company_name": "Tesla Inc"},
        "outputs": {
            "expected_risk_level": "medium",
            "expected_credit_score_range": [60, 80],
            "expected_tools": ["fetch_sec_data", "fetch_market_data", "web_search"],
            "is_public_company": True,
            "expected_trajectory": ["parse_input", "create_plan", "fetch_api_data", "search_web", "synthesize"],
        },
        "metadata": {"company_type": "public_us", "ticker": "TSLA"},
    },
    # Private Companies
    {
        "inputs": {"company_name": "Unknown Private Company LLC"},
        "outputs": {
            "expected_risk_level": "high",
            "expected_credit_score_range": [30, 55],
            "expected_tools": ["web_search", "fetch_legal_data"],
            "is_public_company": False,
            "expected_trajectory": ["parse_input", "create_plan", "search_web", "synthesize"],
        },
        "metadata": {"company_type": "private"},
    },
    {
        "inputs": {"company_name": "Small Business Corp"},
        "outputs": {
            "expected_risk_level": "medium",
            "expected_credit_score_range": [40, 65],
            "expected_tools": ["web_search"],
            "is_public_company": False,
            "expected_trajectory": ["parse_input", "create_plan", "search_web", "synthesize"],
        },
        "metadata": {"company_type": "private"},
    },
    # Edge Cases
    {
        "inputs": {"company_name": ""},
        "outputs": {
            "expected_risk_level": "error",
            "expected_credit_score_range": [0, 0],
            "expected_tools": [],
            "is_public_company": False,
            "should_fail": True,
        },
        "metadata": {"company_type": "invalid", "test_type": "edge_case"},
    },
    {
        "inputs": {"company_name": "Enron Corporation"},
        "outputs": {
            "expected_risk_level": "high",
            "expected_credit_score_range": [0, 30],
            "expected_tools": ["web_search"],
            "is_public_company": False,  # Defunct
            "expected_trajectory": ["parse_input", "create_plan", "search_web", "synthesize"],
        },
        "metadata": {"company_type": "defunct", "test_type": "edge_case"},
    },
]


# =============================================================================
# DATASET MANAGER
# =============================================================================

class DatasetManager:
    """
    Manages LangSmith datasets for credit intelligence evaluation.

    Provides methods to:
    - Create datasets
    - Add test cases
    - Import/export datasets
    - Sync with local JSON files
    """

    def __init__(self, client: Optional[Any] = None):
        """Initialize with optional LangSmith client."""
        self.client = client
        if LANGSMITH_AVAILABLE and client is None:
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if api_key:
                try:
                    self.client = Client(api_key=api_key)
                    logger.info("DatasetManager connected to LangSmith")
                except Exception as e:
                    logger.warning(f"Failed to connect to LangSmith: {e}")

        # Local storage path
        self.local_path = Path(__file__).parent.parent.parent.parent / "data" / "eval_datasets"
        self.local_path.mkdir(parents=True, exist_ok=True)

    def is_connected(self) -> bool:
        """Check if connected to LangSmith."""
        return self.client is not None

    def create_dataset(
        self,
        name: str,
        description: str = "",
        test_cases: List[Dict] = None,
    ) -> Optional[str]:
        """
        Create a new dataset.

        Args:
            name: Dataset name
            description: Dataset description
            test_cases: Optional list of test cases to add

        Returns:
            Dataset ID if created in LangSmith, else None
        """
        dataset_id = None

        # Create in LangSmith if available
        if self.is_connected():
            try:
                dataset = self.client.create_dataset(
                    dataset_name=name,
                    description=description or f"Credit Intelligence evaluation dataset - {name}",
                )
                dataset_id = str(dataset.id)
                logger.info(f"Created LangSmith dataset: {name} (ID: {dataset_id})")

                # Add test cases
                if test_cases:
                    for case in test_cases:
                        self.client.create_example(
                            inputs=case.get("inputs", {}),
                            outputs=case.get("outputs", {}),
                            metadata=case.get("metadata", {}),
                            dataset_id=dataset.id,
                        )
                    logger.info(f"Added {len(test_cases)} examples to dataset")

            except Exception as e:
                logger.error(f"Failed to create LangSmith dataset: {e}")

        # Also save locally
        self._save_local_dataset(name, {
            "name": name,
            "description": description,
            "dataset_id": dataset_id,
            "created_at": datetime.utcnow().isoformat(),
            "examples": test_cases or [],
        })

        return dataset_id

    def add_example(
        self,
        dataset_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Add an example to a dataset.

        Args:
            dataset_name: Name of the dataset
            inputs: Input data
            outputs: Expected output data
            metadata: Optional metadata

        Returns:
            True if successful
        """
        success = False

        # Add to LangSmith if available
        if self.is_connected():
            try:
                # Find dataset by name
                datasets = list(self.client.list_datasets(dataset_name=dataset_name))
                if datasets:
                    self.client.create_example(
                        inputs=inputs,
                        outputs=outputs,
                        metadata=metadata or {},
                        dataset_id=datasets[0].id,
                    )
                    success = True
                    logger.info(f"Added example to LangSmith dataset: {dataset_name}")
                else:
                    logger.warning(f"Dataset not found in LangSmith: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to add example to LangSmith: {e}")

        # Also save locally
        local_data = self._load_local_dataset(dataset_name)
        if local_data:
            local_data["examples"].append({
                "inputs": inputs,
                "outputs": outputs,
                "metadata": metadata or {},
            })
            self._save_local_dataset(dataset_name, local_data)
            success = True

        return success

    def get_dataset(self, name: str) -> Optional[Dict]:
        """
        Get a dataset by name.

        Args:
            name: Dataset name

        Returns:
            Dataset dict with examples, or None
        """
        # Try LangSmith first
        if self.is_connected():
            try:
                datasets = list(self.client.list_datasets(dataset_name=name))
                if datasets:
                    dataset = datasets[0]
                    examples = list(self.client.list_examples(dataset_id=dataset.id))
                    return {
                        "name": dataset.name,
                        "id": str(dataset.id),
                        "description": dataset.description,
                        "examples": [
                            {
                                "inputs": ex.inputs,
                                "outputs": ex.outputs,
                                "metadata": ex.metadata,
                            }
                            for ex in examples
                        ],
                    }
            except Exception as e:
                logger.warning(f"Failed to get dataset from LangSmith: {e}")

        # Fallback to local
        return self._load_local_dataset(name)

    def list_datasets(self) -> List[Dict]:
        """List all available datasets."""
        datasets = []

        # From LangSmith
        if self.is_connected():
            try:
                for ds in self.client.list_datasets():
                    datasets.append({
                        "name": ds.name,
                        "id": str(ds.id),
                        "description": ds.description,
                        "source": "langsmith",
                    })
            except Exception as e:
                logger.warning(f"Failed to list LangSmith datasets: {e}")

        # From local storage
        for file in self.local_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    # Avoid duplicates
                    if not any(d["name"] == data["name"] for d in datasets):
                        datasets.append({
                            "name": data["name"],
                            "id": data.get("dataset_id"),
                            "description": data.get("description", ""),
                            "source": "local",
                        })
            except Exception:
                pass

        return datasets

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset."""
        success = False

        # Delete from LangSmith
        if self.is_connected():
            try:
                datasets = list(self.client.list_datasets(dataset_name=name))
                if datasets:
                    self.client.delete_dataset(dataset_id=datasets[0].id)
                    success = True
                    logger.info(f"Deleted LangSmith dataset: {name}")
            except Exception as e:
                logger.warning(f"Failed to delete from LangSmith: {e}")

        # Delete local
        local_file = self.local_path / f"{name}.json"
        if local_file.exists():
            local_file.unlink()
            success = True
            logger.info(f"Deleted local dataset: {name}")

        return success

    def create_sample_dataset(self, name: str = "credit_intel_eval") -> Optional[str]:
        """
        Create a sample dataset with predefined test cases.

        Args:
            name: Dataset name

        Returns:
            Dataset ID
        """
        return self.create_dataset(
            name=name,
            description="Sample dataset for Credit Intelligence evaluation with public, private, and edge case companies.",
            test_cases=SAMPLE_TEST_CASES,
        )

    def export_to_json(self, name: str, output_path: str = None) -> str:
        """Export a dataset to JSON file."""
        dataset = self.get_dataset(name)
        if not dataset:
            raise ValueError(f"Dataset not found: {name}")

        output_path = output_path or str(self.local_path / f"{name}_export.json")
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, default=str)

        logger.info(f"Exported dataset to: {output_path}")
        return output_path

    def import_from_json(self, json_path: str, name: str = None) -> Optional[str]:
        """Import a dataset from JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        name = name or data.get("name", Path(json_path).stem)
        return self.create_dataset(
            name=name,
            description=data.get("description", ""),
            test_cases=data.get("examples", []),
        )

    def _save_local_dataset(self, name: str, data: Dict):
        """Save dataset to local JSON file."""
        filepath = self.local_path / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_local_dataset(self, name: str) -> Optional[Dict]:
        """Load dataset from local JSON file."""
        filepath = self.local_path / f"{name}.json"
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton manager
_manager: Optional[DatasetManager] = None


def get_dataset_manager() -> DatasetManager:
    """Get the global DatasetManager instance."""
    global _manager
    if _manager is None:
        _manager = DatasetManager()
    return _manager


def create_credit_dataset(
    name: str = "credit_intel_eval",
    include_samples: bool = True,
) -> Optional[str]:
    """
    Create a credit intelligence evaluation dataset.

    Args:
        name: Dataset name
        include_samples: Whether to include sample test cases

    Returns:
        Dataset ID
    """
    manager = get_dataset_manager()
    if include_samples:
        return manager.create_sample_dataset(name)
    else:
        return manager.create_dataset(name, "Credit Intelligence evaluation dataset")


def add_test_case(
    dataset_name: str,
    company_name: str,
    expected_risk_level: str,
    expected_score_range: tuple = (50, 70),
    expected_tools: List[str] = None,
    is_public: bool = False,
    metadata: Dict = None,
) -> bool:
    """
    Add a test case to a dataset.

    Args:
        dataset_name: Name of the dataset
        company_name: Company to test
        expected_risk_level: Expected risk level (low, medium, high)
        expected_score_range: Expected credit score range (min, max)
        expected_tools: Expected tools to be used
        is_public: Whether it's a public company
        metadata: Additional metadata

    Returns:
        True if successful
    """
    manager = get_dataset_manager()
    return manager.add_example(
        dataset_name=dataset_name,
        inputs={"company_name": company_name},
        outputs={
            "expected_risk_level": expected_risk_level,
            "expected_credit_score_range": list(expected_score_range),
            "expected_tools": expected_tools or ["web_search"],
            "is_public_company": is_public,
        },
        metadata=metadata,
    )


def get_dataset(name: str) -> Optional[Dict]:
    """Get a dataset by name."""
    return get_dataset_manager().get_dataset(name)


def list_datasets() -> List[Dict]:
    """List all available datasets."""
    return get_dataset_manager().list_datasets()
