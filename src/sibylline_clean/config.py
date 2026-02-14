"""Configuration loader for multilingual prompt injection patterns.

Loads pattern and motif configurations from YAML files with priority resolution:
1. User config: ~/.config/{app_name}/patterns/ (highest priority)
2. Project config: .{app_name}/patterns/ in current directory
3. Package defaults: shipped with clean (fallback)
"""

import re
from pathlib import Path

# Lazy import yaml to avoid startup cost
_yaml = None


def _get_yaml():
    """Lazy-load PyYAML."""
    global _yaml
    if _yaml is None:
        import yaml

        _yaml = yaml
    return _yaml


def _get_package_defaults_path() -> Path:
    """Get path to package default patterns using importlib.resources."""
    # Use importlib.resources for Python 3.9+ compatibility
    try:
        from importlib.resources import files

        return files("sibylline_clean.pattern_data") / "_defaults"
    except (ImportError, TypeError):
        # Fallback for older Python or editable installs
        return Path(__file__).parent / "pattern_data" / "_defaults"


class PatternConfig:
    """Load patterns from config files with priority resolution.

    Config locations are checked in priority order:
    1. ~/.config/{app_name}/patterns/ - User overrides
    2. .{app_name}/patterns/ - Project-specific patterns
    3. Package defaults - Shipped with clean

    Users can override individual language files or add new languages
    by placing YAML files in the config directories.
    """

    # Available languages in package defaults
    AVAILABLE_LANGUAGES = [
        "en",
        "es",
        "fr",
        "de",
        "zh",
        "ja",  # Original languages
        "ko",
        "ru",
        "ar",
        "pt",
        "it",
        "hi",
        "nl",  # Added languages
    ]

    def __init__(self, languages: list[str] | None = None, app_name: str = "clean"):
        """Initialize with specified languages.

        Args:
            languages: List of language codes to load. Use ["all"] for all
                      available languages. Defaults to ["en"].
            app_name: Application name for config directory resolution.
                     Controls where user overrides are loaded from
                     (e.g., ~/.config/{app_name}/patterns/).
        """
        if languages is None:
            languages = ["en"]

        # Expand "all" to all available languages
        if "all" in languages:
            self.languages = self.AVAILABLE_LANGUAGES.copy()
        else:
            self.languages = languages

        self._app_name = app_name
        self._config_locations = [
            Path.home() / ".config" / app_name / "patterns",  # User overrides
            Path.cwd() / f".{app_name}" / "patterns",  # Project config
        ]

        self._patterns: dict[str, list[str]] = {}
        self._motifs: dict[str, list[str]] = {}
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        self._settings: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load patterns for all enabled languages."""
        for lang in self.languages:
            self._load_language(lang)

    def _find_config_file(self, lang: str) -> Path | None:
        """Find config file for a language, checking locations in priority order.

        Returns:
            Path to the config file, or None if not found.
        """
        filename = f"{lang}.yaml"

        # Check user/project configs first (in priority order)
        for config_dir in self._config_locations:
            config_file = config_dir / filename
            if config_file.exists():
                return config_file

        # Fall back to package defaults
        defaults_path = _get_package_defaults_path()
        default_file = defaults_path / filename

        # Handle both Path objects and importlib traversables
        if hasattr(default_file, "is_file"):
            if default_file.is_file():
                return default_file
        elif hasattr(default_file, "read_text"):
            # importlib.resources Traversable
            return default_file

        return None

    def _load_language(self, lang: str) -> None:
        """Load patterns and motifs for a single language."""
        config_file = self._find_config_file(lang)
        if config_file is None:
            return

        yaml = _get_yaml()

        # Read content (handle both Path and Traversable)
        if hasattr(config_file, "read_text"):
            content = config_file.read_text(encoding="utf-8")
        else:
            content = config_file.read_text(encoding="utf-8")

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError:
            return

        if not data:
            return

        # Load settings (for CJK word boundary handling)
        if "settings" in data:
            self._settings[lang] = data["settings"]

        # Load patterns
        if "patterns" in data:
            for category, patterns in data["patterns"].items():
                if category not in self._patterns:
                    self._patterns[category] = []
                self._patterns[category].extend(patterns)

        # Load motifs
        if "motifs" in data:
            for category, motifs in data["motifs"].items():
                if category not in self._motifs:
                    self._motifs[category] = []
                self._motifs[category].extend(motifs)

    def get_patterns(self) -> dict[str, list[re.Pattern]]:
        """Get compiled regex patterns for all categories.

        Returns:
            Dict mapping category names to lists of compiled patterns.
        """
        if self._compiled_patterns:
            return self._compiled_patterns

        for category, patterns in self._patterns.items():
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error:
                    # Skip invalid patterns
                    pass
            if compiled:
                self._compiled_patterns[category] = compiled

        return self._compiled_patterns

    def get_motifs(self) -> dict[str, list[str]]:
        """Get motifs for all categories.

        Returns:
            Dict mapping category names to lists of motif strings.
        """
        return self._motifs.copy()

    def get_raw_patterns(self) -> dict[str, list[str]]:
        """Get raw (uncompiled) patterns for all categories.

        Returns:
            Dict mapping category names to lists of pattern strings.
        """
        return self._patterns.copy()

    def uses_word_boundaries(self, lang: str) -> bool:
        """Check if a language uses word boundaries in patterns.

        CJK languages (Chinese, Japanese) don't use spaces between words,
        so word boundary patterns (\\b) don't work correctly.

        Returns:
            False for CJK languages, True otherwise.
        """
        settings = self._settings.get(lang, {})
        return settings.get("word_boundaries", True)

    @classmethod
    def list_available_languages(cls) -> list[str]:
        """List all available language codes.

        Returns:
            List of language codes that have default patterns.
        """
        return cls.AVAILABLE_LANGUAGES.copy()
