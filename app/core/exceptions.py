"""
Custom exception classes for the application.
"""

class AppException(Exception):
    """Base class for application-specific exceptions."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class PromptNotFoundError(AppException):
    """Raised when a prompt is not found."""
    pass

class LLMGenerationError(AppException):
    """Raised when an LLM fails to generate content."""
    pass

class DatabaseError(AppException):
    """Raised for database-related errors."""
    pass

class ConfigurationError(AppException):
    """Raised for configuration errors."""
    pass
