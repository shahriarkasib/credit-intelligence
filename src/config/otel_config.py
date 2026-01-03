"""
OpenTelemetry Integration for Credit Intelligence - Task 6

Lightweight OTEL integration for exporting traces to standard
OpenTelemetry-compatible backends (Jaeger, Zipkin, Grafana Tempo, etc.).

This module is OPTIONAL - the project already has Langfuse integration
which is OTEL-native. Use this module if you need to export traces
to additional OTEL backends.

Setup:
1. pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
2. Set environment variables:
   - OTEL_EXPORTER_OTLP_ENDPOINT (default: http://localhost:4317)
   - OTEL_SERVICE_NAME (default: credit-intelligence)
   - OTEL_TRACES_ENABLED (default: false)
"""

import os
import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    OTEL_SDK_AVAILABLE = True
except ImportError:
    OTEL_SDK_AVAILABLE = False
    trace = None
    TracerProvider = None
    logger.debug("opentelemetry-sdk not installed. OTEL tracing unavailable.")

# Check if OTLP exporter is available
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_EXPORTER_AVAILABLE = True
except ImportError:
    OTLP_EXPORTER_AVAILABLE = False
    OTLPSpanExporter = None
    logger.debug("opentelemetry-exporter-otlp not installed. Using console exporter.")


# Global state
_tracer_provider: Optional["TracerProvider"] = None
_tracer: Optional[Any] = None
_is_initialized = False


def is_otel_enabled() -> bool:
    """Check if OTEL tracing is enabled via environment variable."""
    return os.getenv("OTEL_TRACES_ENABLED", "false").lower() == "true"


def is_otel_available() -> bool:
    """Check if OpenTelemetry SDK is installed."""
    return OTEL_SDK_AVAILABLE


def setup_otel(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    use_console_exporter: bool = False,
) -> bool:
    """
    Set up OpenTelemetry tracing.

    Args:
        service_name: Name of the service (default: OTEL_SERVICE_NAME or "credit-intelligence")
        otlp_endpoint: OTLP collector endpoint (default: OTEL_EXPORTER_OTLP_ENDPOINT)
        use_console_exporter: If True, also export to console (for debugging)

    Returns:
        True if setup was successful, False otherwise
    """
    global _tracer_provider, _tracer, _is_initialized

    if _is_initialized:
        logger.debug("OTEL already initialized")
        return True

    if not OTEL_SDK_AVAILABLE:
        logger.warning("OpenTelemetry SDK not available. Install with: pip install opentelemetry-sdk")
        return False

    if not is_otel_enabled():
        logger.debug("OTEL tracing not enabled. Set OTEL_TRACES_ENABLED=true to enable.")
        return False

    try:
        # Get configuration
        service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "credit-intelligence")
        otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        # Create resource
        resource = Resource.create({SERVICE_NAME: service_name})

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Add OTLP exporter if available
        if OTLP_EXPORTER_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTEL OTLP exporter configured: {otlp_endpoint}")
        else:
            logger.warning("OTLP exporter not available. Install: pip install opentelemetry-exporter-otlp")

        # Add console exporter if requested (useful for debugging)
        if use_console_exporter:
            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            logger.info("OTEL console exporter enabled")

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Create tracer
        _tracer = trace.get_tracer(service_name)

        _is_initialized = True
        logger.info(f"OpenTelemetry initialized for service: {service_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        return False


def get_tracer() -> Optional[Any]:
    """
    Get the OpenTelemetry tracer.

    Returns:
        OpenTelemetry tracer or None if not initialized
    """
    global _tracer

    if not _is_initialized:
        if is_otel_enabled():
            setup_otel()

    return _tracer


def get_tracer_provider() -> Optional["TracerProvider"]:
    """Get the OpenTelemetry tracer provider."""
    return _tracer_provider


@contextmanager
def otel_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None,
):
    """
    Context manager for creating OTEL spans.

    Usage:
        with otel_span("credit_assessment", {"company.name": "Apple Inc"}) as span:
            # ... do work ...
            span.set_attribute("result.score", 0.95)

    Args:
        name: Name of the span
        attributes: Optional attributes to set on the span
        kind: Optional span kind (SpanKind.INTERNAL, SpanKind.CLIENT, etc.)

    Yields:
        The span object (or a no-op if OTEL not enabled)
    """
    tracer = get_tracer()

    if tracer is None:
        # Yield a no-op span-like object
        yield NoOpSpan()
        return

    # Get span kind
    if kind is None:
        from opentelemetry.trace import SpanKind
        kind = SpanKind.INTERNAL

    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


class NoOpSpan:
    """No-op span for when OTEL is not enabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


def record_llm_call(
    model: str,
    prompt: str,
    response: str,
    tokens: int = 0,
    latency_ms: float = 0,
    success: bool = True,
) -> None:
    """
    Record an LLM call as an OTEL span.

    Args:
        model: Model name/ID
        prompt: Input prompt
        response: Model response
        tokens: Total tokens used
        latency_ms: Latency in milliseconds
        success: Whether the call succeeded
    """
    tracer = get_tracer()
    if tracer is None:
        return

    from opentelemetry.trace import SpanKind, Status, StatusCode

    with tracer.start_as_current_span("llm.call", kind=SpanKind.CLIENT) as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt.length", len(prompt))
        span.set_attribute("llm.response.length", len(response))
        span.set_attribute("llm.tokens.total", tokens)
        span.set_attribute("llm.latency_ms", latency_ms)
        span.set_attribute("llm.success", success)

        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, "LLM call failed"))


def record_tool_call(
    tool_name: str,
    input_data: Dict[str, Any],
    output_data: Optional[Dict[str, Any]] = None,
    latency_ms: float = 0,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Record a tool call as an OTEL span.

    Args:
        tool_name: Name of the tool
        input_data: Tool input
        output_data: Tool output
        latency_ms: Latency in milliseconds
        success: Whether the call succeeded
        error: Error message if failed
    """
    tracer = get_tracer()
    if tracer is None:
        return

    from opentelemetry.trace import SpanKind, Status, StatusCode

    with tracer.start_as_current_span(f"tool.{tool_name}", kind=SpanKind.CLIENT) as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.latency_ms", latency_ms)
        span.set_attribute("tool.success", success)

        if error:
            span.set_attribute("tool.error", error)
            span.set_status(Status(StatusCode.ERROR, error))
        else:
            span.set_status(Status(StatusCode.OK))


def shutdown_otel() -> None:
    """Shutdown OpenTelemetry and flush any pending spans."""
    global _tracer_provider, _tracer, _is_initialized

    if _tracer_provider:
        try:
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down OpenTelemetry: {e}")
        finally:
            _tracer_provider = None
            _tracer = None
            _is_initialized = False


# Export public API
__all__ = [
    "is_otel_enabled",
    "is_otel_available",
    "setup_otel",
    "get_tracer",
    "get_tracer_provider",
    "otel_span",
    "record_llm_call",
    "record_tool_call",
    "shutdown_otel",
    "NoOpSpan",
]
