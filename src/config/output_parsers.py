"""LangChain Output Parsers for Credit Intelligence.

Provides:
- PydanticOutputParser factories for each schema type
- Format instructions for LLM prompts
- Fallback parsing with legacy JSON extraction
- Robust error handling
"""

import json
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Try to import LangChain output parsers
try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.exceptions import OutputParserException
    LANGCHAIN_PARSERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_PARSERS_AVAILABLE = False
    PydanticOutputParser = None
    OutputParserException = Exception
    logger.warning("langchain-core output parsers not available")

# Import our schemas
from .output_schemas import (
    ToolSelection,
    CreditAssessment,
    ParsedCompany,
    ValidationResult,
)

T = TypeVar('T', bound=BaseModel)


# =============================================================================
# Parser Factory Functions
# =============================================================================

def get_tool_selection_parser() -> Optional["PydanticOutputParser"]:
    """Get parser for ToolSelection responses."""
    if not LANGCHAIN_PARSERS_AVAILABLE:
        return None
    return PydanticOutputParser(pydantic_object=ToolSelection)


def get_credit_assessment_parser() -> Optional["PydanticOutputParser"]:
    """Get parser for CreditAssessment responses."""
    if not LANGCHAIN_PARSERS_AVAILABLE:
        return None
    return PydanticOutputParser(pydantic_object=CreditAssessment)


def get_parsed_company_parser() -> Optional["PydanticOutputParser"]:
    """Get parser for ParsedCompany responses."""
    if not LANGCHAIN_PARSERS_AVAILABLE:
        return None
    return PydanticOutputParser(pydantic_object=ParsedCompany)


def get_validation_result_parser() -> Optional["PydanticOutputParser"]:
    """Get parser for ValidationResult responses."""
    if not LANGCHAIN_PARSERS_AVAILABLE:
        return None
    return PydanticOutputParser(pydantic_object=ValidationResult)


# =============================================================================
# Format Instructions
# =============================================================================

def get_format_instructions(schema_type: str) -> str:
    """
    Get format instructions for a schema type.

    Args:
        schema_type: One of "tool_selection", "credit_assessment",
                    "parsed_company", "validation_result"

    Returns:
        Format instructions string to append to prompts
    """
    parsers = {
        "tool_selection": get_tool_selection_parser,
        "credit_assessment": get_credit_assessment_parser,
        "parsed_company": get_parsed_company_parser,
        "validation_result": get_validation_result_parser,
    }

    parser_fn = parsers.get(schema_type)
    if not parser_fn:
        logger.warning(f"Unknown schema type: {schema_type}")
        return ""

    parser = parser_fn()
    if parser:
        return parser.get_format_instructions()

    # Fallback: Generate basic instructions from schema
    schemas = {
        "tool_selection": ToolSelection,
        "credit_assessment": CreditAssessment,
        "parsed_company": ParsedCompany,
        "validation_result": ValidationResult,
    }
    schema_class = schemas.get(schema_type)
    if schema_class:
        return f"Respond with valid JSON matching this schema:\n{schema_class.model_json_schema()}"

    return ""


# =============================================================================
# Fallback JSON Extraction
# =============================================================================

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response text.

    Handles:
    - ```json code blocks
    - ``` code blocks
    - Raw JSON
    - JSON embedded in text

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dict or None if extraction fails
    """
    if not response:
        return None

    # Try to find JSON block
    json_str = response.strip()

    # Check for ```json block
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            json_str = response[start:end].strip()

    # Check for ``` block
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            json_str = response[start:end].strip()

    # Try to find JSON object
    elif "{" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        if end > start:
            json_str = response[start:end]

    # Parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode failed: {e}")
        return None


# =============================================================================
# Parse with Fallback
# =============================================================================

def parse_with_fallback(
    response: str,
    schema_class: Type[T],
    legacy_parser: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Union[T, Dict[str, Any]]:
    """
    Parse LLM response with fallback to legacy parsing.

    Tries in order:
    1. PydanticOutputParser.parse() - strict validation
    2. Extract JSON + Pydantic validation - lenient
    3. Legacy parser function - compatibility
    4. Raw JSON extraction - last resort

    Args:
        response: Raw LLM response text
        schema_class: Pydantic model class to parse into
        legacy_parser: Optional legacy parsing function for fallback

    Returns:
        Parsed Pydantic model or dict (if all parsing fails)

    Example:
        from config.output_parsers import parse_with_fallback
        from config.output_schemas import CreditAssessment

        result = parse_with_fallback(
            response=llm_response,
            schema_class=CreditAssessment,
            legacy_parser=self._parse_llm_response,  # Keep old parser as fallback
        )
    """
    # Try 1: LangChain PydanticOutputParser (strict)
    if LANGCHAIN_PARSERS_AVAILABLE:
        try:
            parser = PydanticOutputParser(pydantic_object=schema_class)
            result = parser.parse(response)
            logger.debug(f"Parsed with PydanticOutputParser: {schema_class.__name__}")
            return result
        except (OutputParserException, ValidationError) as e:
            logger.debug(f"PydanticOutputParser failed: {e}")

    # Try 2: Extract JSON + Pydantic validation (lenient)
    json_data = extract_json_from_response(response)
    if json_data:
        try:
            result = schema_class.model_validate(json_data)
            logger.debug(f"Parsed with JSON extraction + Pydantic: {schema_class.__name__}")
            return result
        except ValidationError as e:
            logger.debug(f"Pydantic validation failed: {e}")

    # Try 3: Legacy parser function
    if legacy_parser:
        try:
            legacy_result = legacy_parser(response)
            if legacy_result and not legacy_result.get("error"):
                logger.debug(f"Parsed with legacy parser")
                # Try to convert to Pydantic model
                try:
                    return schema_class.model_validate(legacy_result)
                except ValidationError:
                    return legacy_result
        except Exception as e:
            logger.debug(f"Legacy parser failed: {e}")

    # Try 4: Return raw JSON as dict
    if json_data:
        logger.warning(f"Returning raw JSON dict (validation failed)")
        return json_data

    # All parsing failed
    logger.error(f"All parsing methods failed for {schema_class.__name__}")
    return {"error": "Failed to parse response", "raw_response": response[:500]}


# =============================================================================
# Convenience Functions for Each Schema Type
# =============================================================================

def parse_tool_selection(
    response: str,
    legacy_parser: Optional[Callable] = None,
) -> Union[ToolSelection, Dict[str, Any]]:
    """Parse tool selection response."""
    return parse_with_fallback(response, ToolSelection, legacy_parser)


def parse_credit_assessment(
    response: str,
    legacy_parser: Optional[Callable] = None,
) -> Union[CreditAssessment, Dict[str, Any]]:
    """Parse credit assessment response."""
    return parse_with_fallback(response, CreditAssessment, legacy_parser)


def parse_company(
    response: str,
    legacy_parser: Optional[Callable] = None,
) -> Union[ParsedCompany, Dict[str, Any]]:
    """Parse company information response."""
    return parse_with_fallback(response, ParsedCompany, legacy_parser)


def parse_validation_result(
    response: str,
    legacy_parser: Optional[Callable] = None,
) -> Union[ValidationResult, Dict[str, Any]]:
    """Parse validation result response."""
    return parse_with_fallback(response, ValidationResult, legacy_parser)


# =============================================================================
# Utility Functions
# =============================================================================

def is_parsers_available() -> bool:
    """Check if LangChain output parsers are available."""
    return LANGCHAIN_PARSERS_AVAILABLE


def result_to_dict(result: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert parsing result to dict.

    Handles both Pydantic models and raw dicts.
    """
    if isinstance(result, BaseModel):
        return result.model_dump()
    return result


__all__ = [
    # Parser factories
    "get_tool_selection_parser",
    "get_credit_assessment_parser",
    "get_parsed_company_parser",
    "get_validation_result_parser",
    # Format instructions
    "get_format_instructions",
    # Parsing functions
    "parse_with_fallback",
    "parse_tool_selection",
    "parse_credit_assessment",
    "parse_company",
    "parse_validation_result",
    # Utilities
    "extract_json_from_response",
    "is_parsers_available",
    "result_to_dict",
]
