import functools
import logging
import traceback
from typing import Optional

logger = logging.getLogger('DiscordBot')

def handle_errors(operation_name: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}")
                logger.debug(traceback.format_exc())
                raise
        return wrapper
    return decorator

class ErrorHandler:
    def __init__(self, notification_channel=None, admin_user=None):
        self.notification_channel = notification_channel
        self.admin_user = admin_user
        self.logger = logging.getLogger('DiscordBot') 