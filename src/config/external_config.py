"""
External Configuration Loader for Credit Intelligence.

Provides:
- YAML-based configuration loading
- Environment variable substitution
- Hot-reload capability (file watching)
- Runtime configuration updates
- Type-safe access to config values

Usage:
    from config.external_config import get_config, get_llm_config, get_prompt_config

    # Get full config
    config = get_config()

    # Get specific sections
    llm_config = get_llm_config()
    prompt = get_prompt_config("company_parser")

    # Reload config manually
    reload_config()
"""

import os
import re
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Run: pip install pyyaml")

# Try to import watchdog for file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.info("watchdog not installed. Hot-reload disabled. Run: pip install watchdog")


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    enabled: bool = False
    api_key: str = ""
    base_url: Optional[str] = None
    models: Dict[str, str] = field(default_factory=dict)
    default_model: str = "primary"


@dataclass
class PromptConfig:
    """Configuration for a single prompt."""
    id: str
    name: str
    description: str = ""
    category: str = "general"
    variables: List[str] = field(default_factory=list)
    system_prompt: str = ""
    user_template: str = ""


@dataclass
class CredentialsConfig:
    """Credentials configuration (resolved from env vars)."""
    mongodb_uri: str = ""
    google_credentials_path: str = ""
    google_spreadsheet_id: str = ""
    api_keys: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================

class ConfigManager:
    """
    Manages external configuration with hot-reload support.

    Features:
    - Loads configuration from YAML file
    - Substitutes environment variables
    - Watches for file changes (optional)
    - Provides typed access to config sections
    - Supports runtime updates
    """

    def __init__(self, config_path: Optional[str] = None):
        # Find config file
        self.config_path = self._find_config_path(config_path)
        self._config: Dict[str, Any] = {}
        self._last_modified: float = 0
        self._callbacks: List[Callable[[Dict], None]] = []
        self._lock = threading.RLock()
        self._observer = None
        self._watching = False

        # Load initial config
        self.reload()

    def _find_config_path(self, config_path: Optional[str]) -> Path:
        """Find the configuration file."""
        if config_path:
            return Path(config_path)

        # Search paths in order
        search_paths = [
            Path(__file__).parent.parent.parent / "config" / "settings.yaml",  # project_root/config/
            Path(__file__).parent.parent.parent / "settings.yaml",  # project_root/
            Path.cwd() / "config" / "settings.yaml",  # cwd/config/
            Path.cwd() / "settings.yaml",  # cwd/
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found config at: {path}")
                return path

        # Default to first path (will create if needed)
        return search_paths[0]

    def _substitute_env_vars(self, value: Any) -> Any:
        """Recursively substitute ${ENV_VAR} patterns with environment values."""
        if isinstance(value, str):
            # Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
            pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

            def replacer(match):
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.getenv(var_name, default)

            return re.sub(pattern, replacer, value)

        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]

        return value

    def reload(self) -> bool:
        """
        Reload configuration from file.

        Returns:
            True if config was reloaded, False if unchanged or error
        """
        if not YAML_AVAILABLE:
            logger.error("YAML not available, cannot load config")
            return False

        with self._lock:
            try:
                if not self.config_path.exists():
                    logger.warning(f"Config file not found: {self.config_path}")
                    return False

                # Check if file was modified
                mtime = self.config_path.stat().st_mtime
                if mtime == self._last_modified:
                    return False

                # Load YAML
                with open(self.config_path, 'r') as f:
                    raw_config = yaml.safe_load(f) or {}

                # Substitute environment variables
                self._config = self._substitute_env_vars(raw_config)
                self._last_modified = mtime

                logger.info(f"Loaded config from: {self.config_path}")

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(self._config)
                    except Exception as e:
                        logger.error(f"Config callback error: {e}")

                return True

            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return False

    def start_watching(self, interval_seconds: int = 5) -> bool:
        """
        Start watching config file for changes.

        Args:
            interval_seconds: How often to check for changes (if watchdog not available)

        Returns:
            True if watching started
        """
        if self._watching:
            return True

        if WATCHDOG_AVAILABLE:
            return self._start_watchdog()
        else:
            return self._start_polling(interval_seconds)

    def _start_watchdog(self) -> bool:
        """Start file watching using watchdog."""
        try:
            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(handler_self, config_manager):
                    handler_self.config_manager = config_manager

                def on_modified(handler_self, event):
                    if isinstance(event, FileModifiedEvent):
                        if Path(event.src_path).name == self.config_path.name:
                            logger.info("Config file changed, reloading...")
                            handler_self.config_manager.reload()

            self._observer = Observer()
            self._observer.schedule(
                ConfigFileHandler(self),
                str(self.config_path.parent),
                recursive=False
            )
            self._observer.start()
            self._watching = True
            logger.info(f"Started watching config file: {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start watchdog: {e}")
            return False

    def _start_polling(self, interval_seconds: int) -> bool:
        """Start polling for config changes."""
        def poll_loop():
            while self._watching:
                time.sleep(interval_seconds)
                if self.reload():
                    logger.info("Config reloaded via polling")

        self._watching = True
        thread = threading.Thread(target=poll_loop, daemon=True)
        thread.start()
        logger.info(f"Started polling config file every {interval_seconds}s")
        return True

    def stop_watching(self):
        """Stop watching for config changes."""
        self._watching = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        logger.info("Stopped watching config file")

    def on_change(self, callback: Callable[[Dict], None]):
        """Register a callback to be called when config changes."""
        self._callbacks.append(callback)

    # =========================================================================
    # CONFIG ACCESSORS
    # =========================================================================

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        with self._lock:
            return self._config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.

        Example:
            config.get("llm.default_provider")
            config.get("prompts.company_parser.system_prompt")
        """
        with self._lock:
            keys = key.split(".")
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    def get_app_config(self) -> Dict[str, Any]:
        """Get application metadata config."""
        return self.get("app", {})

    def get_llm_config(self) -> Dict[str, Any]:
        """Get full LLM configuration."""
        return self.get("llm", {})

    def get_llm_provider(self, provider: str = None) -> Optional[LLMProviderConfig]:
        """
        Get configuration for a specific LLM provider.

        Args:
            provider: Provider name (groq, openai, anthropic). Defaults to default_provider.

        Returns:
            LLMProviderConfig or None
        """
        provider = provider or self.get("llm.default_provider", "groq")
        provider_config = self.get(f"llm.providers.{provider}")

        if not provider_config:
            return None

        return LLMProviderConfig(
            name=provider,
            enabled=provider_config.get("enabled", False),
            api_key=provider_config.get("api_key", ""),
            base_url=provider_config.get("base_url"),
            models=provider_config.get("models", {}),
            default_model=provider_config.get("default_model", "primary"),
        )

    def get_prompt_config(self, prompt_id: str) -> Optional[PromptConfig]:
        """
        Get configuration for a specific prompt.

        Args:
            prompt_id: Prompt identifier

        Returns:
            PromptConfig or None
        """
        prompt_data = self.get(f"prompts.{prompt_id}")
        if not prompt_data:
            return None

        return PromptConfig(
            id=prompt_id,
            name=prompt_data.get("name", prompt_id),
            description=prompt_data.get("description", ""),
            category=prompt_data.get("category", "general"),
            variables=prompt_data.get("variables", []),
            system_prompt=prompt_data.get("system_prompt", ""),
            user_template=prompt_data.get("user_template", ""),
        )

    def get_all_prompts(self) -> Dict[str, PromptConfig]:
        """Get all prompt configurations."""
        prompts = {}
        prompts_data = self.get("prompts", {})

        for prompt_id, prompt_data in prompts_data.items():
            prompts[prompt_id] = PromptConfig(
                id=prompt_id,
                name=prompt_data.get("name", prompt_id),
                description=prompt_data.get("description", ""),
                category=prompt_data.get("category", "general"),
                variables=prompt_data.get("variables", []),
                system_prompt=prompt_data.get("system_prompt", ""),
                user_template=prompt_data.get("user_template", ""),
            )

        return prompts

    def get_credentials(self) -> CredentialsConfig:
        """Get credentials configuration (with env vars resolved)."""
        creds = self.get("credentials", {})

        return CredentialsConfig(
            mongodb_uri=creds.get("mongodb", {}).get("uri", ""),
            google_credentials_path=creds.get("google_sheets", {}).get("credentials_path", ""),
            google_spreadsheet_id=creds.get("google_sheets", {}).get("spreadsheet_id", ""),
            api_keys=creds.get("api_keys", {}),
        )

    def get_data_sources_config(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self.get("data_sources", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get("evaluation", {})

    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime settings."""
        return self.get("runtime", {})

    # =========================================================================
    # CONFIG UPDATES (Runtime)
    # =========================================================================

    def update(self, key: str, value: Any) -> bool:
        """
        Update a config value at runtime.

        Note: This does NOT persist to file. Use update_and_save() for persistence.

        Args:
            key: Dot-notation key
            value: New value

        Returns:
            True if updated
        """
        with self._lock:
            keys = key.split(".")
            config = self._config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value
            return True

    def update_and_save(self, key: str, value: Any) -> bool:
        """
        Update a config value and save to file.

        Args:
            key: Dot-notation key
            value: New value

        Returns:
            True if saved successfully
        """
        if not YAML_AVAILABLE:
            return False

        self.update(key, value)

        with self._lock:
            try:
                with open(self.config_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"Saved config update: {key}")
                return True
            except Exception as e:
                logger.error(f"Failed to save config: {e}")
                return False


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_config_manager: Optional[ConfigManager] = None


def get_config_manager(force_new: bool = False) -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None or force_new:
        _config_manager = ConfigManager()

        # Start watching if enabled in config
        runtime = _config_manager.get_runtime_config()
        if runtime.get("hot_reload", True):
            interval = runtime.get("watch_interval_seconds", 5)
            _config_manager.start_watching(interval)

    return _config_manager


def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return get_config_manager().config


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration."""
    return get_config_manager().get_llm_config()


def get_llm_provider(provider: str = None) -> Optional[LLMProviderConfig]:
    """Get configuration for a specific LLM provider."""
    return get_config_manager().get_llm_provider(provider)


def get_prompt_config(prompt_id: str) -> Optional[PromptConfig]:
    """Get configuration for a specific prompt."""
    return get_config_manager().get_prompt_config(prompt_id)


def get_all_prompts_from_config() -> Dict[str, PromptConfig]:
    """Get all prompt configurations from external config."""
    return get_config_manager().get_all_prompts()


def get_credentials() -> CredentialsConfig:
    """Get credentials configuration."""
    return get_config_manager().get_credentials()


def reload_config() -> bool:
    """Reload configuration from file."""
    return get_config_manager().reload()


def on_config_change(callback: Callable[[Dict], None]):
    """Register a callback for config changes."""
    get_config_manager().on_change(callback)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def get_prompt_text_from_config(prompt_id: str, **variables) -> tuple[str, str]:
    """
    Get formatted prompt text from external config.

    Args:
        prompt_id: Prompt identifier
        **variables: Variables to substitute

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    prompt = get_prompt_config(prompt_id)
    if not prompt:
        raise ValueError(f"Prompt not found in config: {prompt_id}")

    user_prompt = prompt.user_template.format(**variables)
    return prompt.system_prompt, user_prompt


def get_model_for_provider(provider: str = None, model_alias: str = "primary") -> Optional[str]:
    """
    Get the actual model ID for a provider and alias.

    Args:
        provider: Provider name (defaults to default_provider)
        model_alias: Model alias (primary, fast, etc.)

    Returns:
        Model ID string or None
    """
    provider_config = get_llm_provider(provider)
    if not provider_config:
        return None

    return provider_config.models.get(model_alias, provider_config.models.get("primary"))


def get_llm_model(provider: str = None, model: str = "primary") -> Dict[str, Any]:
    """
    Get LLM configuration for a specific provider and model.

    Args:
        provider: Provider name (groq, openai, anthropic). Defaults to default_provider.
        model: Model alias (primary, fast, balanced) or actual model ID

    Returns:
        Dict with provider config and resolved model ID:
        {
            "provider": "groq",
            "model_id": "llama-3.3-70b-versatile",
            "model_alias": "primary",
            "api_key": "...",
            "base_url": None,
            "temperature": 0.1,
            "max_tokens": 2000,
            "enabled": True
        }

    Example:
        config = get_llm_model("groq", "fast")
        # Returns: {"provider": "groq", "model_id": "llama-3.1-8b-instant", ...}

        config = get_llm_model(model="primary")  # Uses default provider
    """
    manager = get_config_manager()
    llm_config = manager.get_llm_config()

    # Get provider (default if not specified)
    provider = provider or llm_config.get("default_provider", "groq")
    provider_config = manager.get_llm_provider(provider)

    if not provider_config:
        return {
            "provider": provider,
            "model_id": None,
            "error": f"Provider '{provider}' not found in config"
        }

    if not provider_config.enabled:
        return {
            "provider": provider,
            "model_id": None,
            "error": f"Provider '{provider}' is disabled"
        }

    # Resolve model alias to actual model ID
    model_alias = model
    if model in provider_config.models:
        model_id = provider_config.models[model]
    else:
        # Assume it's already an actual model ID
        model_id = model
        # Try to find the alias
        for alias, mid in provider_config.models.items():
            if mid == model:
                model_alias = alias
                break

    return {
        "provider": provider,
        "model_id": model_id,
        "model_alias": model_alias,
        "api_key": provider_config.api_key,
        "base_url": provider_config.base_url,
        "temperature": llm_config.get("default_temperature", 0.1),
        "max_tokens": llm_config.get("default_max_tokens", 2000),
        "enabled": provider_config.enabled,
    }


def list_available_models(provider: str = None) -> Dict[str, str]:
    """
    List all available models for a provider.

    Args:
        provider: Provider name. If None, lists models for all enabled providers.

    Returns:
        Dict mapping alias -> model_id

    Example:
        models = list_available_models("groq")
        # Returns: {"primary": "llama-3.3-70b-versatile", "fast": "llama-3.1-8b-instant", ...}
    """
    manager = get_config_manager()

    if provider:
        provider_config = manager.get_llm_provider(provider)
        if provider_config and provider_config.enabled:
            return provider_config.models
        return {}

    # List all models from all enabled providers
    all_models = {}
    providers_config = manager.get("llm.providers", {})

    for prov_name, prov_data in providers_config.items():
        if prov_data.get("enabled", False):
            models = prov_data.get("models", {})
            for alias, model_id in models.items():
                all_models[f"{prov_name}:{alias}"] = model_id

    return all_models


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test loading
    manager = get_config_manager()

    print("\n=== App Config ===")
    print(manager.get_app_config())

    print("\n=== LLM Config ===")
    llm = manager.get_llm_config()
    print(f"Default provider: {llm.get('default_provider')}")

    print("\n=== Groq Provider ===")
    groq = manager.get_llm_provider("groq")
    if groq:
        print(f"Enabled: {groq.enabled}")
        print(f"Models: {groq.models}")

    print("\n=== Prompts ===")
    prompts = manager.get_all_prompts()
    for pid, p in prompts.items():
        print(f"- {pid}: {p.name} ({p.category})")

    print("\n=== Credentials ===")
    creds = manager.get_credentials()
    print(f"MongoDB URI set: {bool(creds.mongodb_uri)}")
    print(f"API Keys: {list(creds.api_keys.keys())}")
