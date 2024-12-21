import functools
import logging
import traceback

logger = logging.getLogger('ErrorHandler')

class ErrorHandler:
    def __init__(self, notification_channel=None, admin_user=None):
        self.notification_channel = notification_channel
        self.admin_user = admin_user

    async def notify(self, title, error):
        message = f"**{title}**\n```\n{error}\n```"
        if self.notification_channel:
            await self.notification_channel.send(message)
        elif self.admin_user:
            await self.admin_user.send(message)
        else:
            logger.error("No valid notification target found for errors")

    async def handle_error(self, error: Exception):
        """Handle an error by logging it and optionally notifying admin"""
        error_msg = f"Error occurred: {str(error)}"
        logger.error(error_msg)
        
        if self.notification_channel:
            try:
                await self.notification_channel.send(f"🚨 {error_msg}")
                
                if self.admin_user:
                    await self.notification_channel.send(f"<@{self.admin_user}> Please check the logs.")
                    
            except Exception as e:
                logger.error(f"Failed to send error notification: {str(e)}")

def handle_errors(context_label=""):
    """
    Decorator to handle errors that occur in async functions.
    Sends a notification to admins or error channel if something goes wrong.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {context_label} of {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                if self.error_handler:
                    await self.error_handler.notify(f"Error in {context_label}", str(e))
                raise
        return wrapper
    return decorator