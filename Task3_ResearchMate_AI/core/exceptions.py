"""Custom exceptions used across ResearchMate AI."""


class ResearchMateError(Exception):
    """Base class for all application errors."""


class ConfigError(ResearchMateError):
    """Invalid or missing configuration."""


class ToolInputError(ResearchMateError):
    """The arguments supplied to a tool were invalid."""


class CalculatorError(ToolInputError):
    """The expression could not be safely evaluated."""


class FileAccessError(ResearchMateError):
    """A file could not be read (missing, forbidden, unsupported...)."""


class SearchError(ResearchMateError):
    """Web search failed (network, timeout, provider error)."""


class ToolTimeoutError(ResearchMateError):
    """A tool did not finish within the configured time limit."""


class LLMError(ResearchMateError):
    """The language model could not be reached or returned an error."""
