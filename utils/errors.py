from typing import Optional

class ChannelSummarizerError(Exception):
    """Base exception class for ChannelSummarizer"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

class APIError(ChannelSummarizerError):
    """Raised when API calls (Discord, Claude, etc) fail"""
    pass

class DiscordError(ChannelSummarizerError):
    """Raised when Discord operations fail"""
    pass

class SummaryError(ChannelSummarizerError):
    """Raised when summary generation fails"""
    pass

class ConfigurationError(ChannelSummarizerError):
    """Raised when there are configuration/setup issues"""
    pass

class MediaProcessingError(ChannelSummarizerError):
    """Raised when processing images/videos fails"""
    pass

class DatabaseError(ChannelSummarizerError):
    """Raised when database operations fail"""
    pass