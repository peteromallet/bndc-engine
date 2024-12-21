import logging
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from utils.errors import DiscordError

logger = logging.getLogger('SchedulerService')


class SchedulerService:
    """Handles all scheduling operations."""
    def __init__(self, bot):
        self.bot = bot
        self.scheduler = AsyncIOScheduler()

    async def schedule_daily_summary(self):
        """
        Schedule daily summary at 10:00 UTC (or any time needed).
        Retries on failure.
        """
        consecutive_failures = 0
        max_consecutive_failures = 3

        while True:
            try:
                now = datetime.utcnow()
                target = now.replace(hour=10, minute=0, second=0, microsecond=0)
                if now.hour >= 10:
                    target += timedelta(days=1)

                delay = (target - now).total_seconds()
                logger.info(f"Waiting {delay/3600:.2f} hours until next summary at {target} UTC")
                await asyncio.sleep(delay)

                await self.bot.generate_summary()
                logger.info("Summary generated successfully.")
                consecutive_failures = 0

            except DiscordError as e:
                consecutive_failures += 1
                logger.warning(f"Discord error (attempt {consecutive_failures}/{max_consecutive_failures}): {e}")
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical("Too many Discord errors. Stopping scheduler.")
                    break
                await asyncio.sleep(300)  # Attempt another run after delay

            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Unexpected error in scheduler (attempt {consecutive_failures}/{max_consecutive_failures}): {e}")
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical("Too many unexpected errors. Stopping scheduler.")
                    # Raise or break out, depending on desired behavior
                    break
                await asyncio.sleep(300)


    async def execute_scheduled_task(self):
        """
        Example: If you need to schedule additional tasks, place them here.
        """
        # This could be used for other cron tasks, like weekly summaries, etc.
        pass 