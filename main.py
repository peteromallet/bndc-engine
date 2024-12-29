import os
import sys
import argparse
import asyncio
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import traceback
import time

from src.summaries.summarizer import ChannelSummarizer

# Add logger setup at the top level
def setup_logging(dev_mode=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if dev_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('DiscordBot')

# Add at top of file
MAX_RETRIES = 3
READY_TIMEOUT = 60
INITIAL_RETRY_DELAY = 3600  # 1 hour
MAX_RETRY_WAIT = 24 * 3600  # 24 hours in seconds

async def run_bot(bot, token, run_now):
    """Run the bot with optional immediate summary generation"""
    try:
        # Create task for bot connection
        bot_task = asyncio.create_task(bot.start(token))
        
        # Wait for bot to be ready
        start_time = time.time()
        while not bot.is_ready():
            if time.time() - start_time > READY_TIMEOUT:
                raise TimeoutError("Bot failed to become ready within timeout period")
            await asyncio.sleep(1)
            
        bot.logger.info("Bot is ready and fully connected")
            
        if run_now:
            bot.logger.info("Running immediate summary...")
            await asyncio.sleep(2)  # Extra sleep
            await bot.generate_summary()
            bot._shutdown_flag = True  # Immediately shutdown after
            await bot.close()
            bot_task.cancel()
            await cleanup_tasks([bot_task])
        else:
            bot.logger.info("Starting scheduled mode...")
            # Create and start the scheduler task
            scheduler_task = asyncio.create_task(schedule_daily_summary(bot))
            
            # Wait for either the bot task or scheduler task to complete
            done, pending = await asyncio.wait(
                [bot_task, scheduler_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            await cleanup_tasks(pending)
            
    except Exception as e:
        bot.logger.error(f"Error running bot: {e}")
        traceback.print_exc()
        sys.exit(1)

async def schedule_daily_summary(bot):
    """
    Run daily summaries on schedule. Only exits if there's an error or explicit shutdown.
    """
    try:
        while not bot._shutdown_flag:
            retry_count = 0  # Reset retry count for each day's attempt
            # Get current UTC time
            now = datetime.utcnow()
            
            # Set target time to 10:00 UTC today
            target = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
            # If it's already past 10:00 UTC today, schedule for tomorrow
            if now.hour >= 10:
                target += timedelta(days=1)
            
            # Calculate how long to wait
            delay = (target - now).total_seconds()
            bot.logger.info(f"Scheduler: Waiting {delay/3600:.2f} hours until next summary at {target} UTC")
            
            # Wait until the target time
            try:
                await asyncio.sleep(delay)
                if not bot._shutdown_flag:
                    await bot.generate_summary()
                    # Success - clear retry count
                    retry_count = 0
            except asyncio.CancelledError:
                bot.logger.info("Scheduler: Summary schedule cancelled - shutting down")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    bot.logger.error(f"Scheduler: Failed after {MAX_RETRIES} attempts")
                    bot._shutdown_flag = True
                    raise
                wait_time = min(INITIAL_RETRY_DELAY * (2 ** retry_count), MAX_RETRY_WAIT)
                await asyncio.sleep(wait_time)
    except Exception as e:
        bot.logger.error(f"Scheduler: Fatal error in scheduler: {e}")
        bot.logger.debug(traceback.format_exc())
        bot._shutdown_flag = True
        raise

async def cleanup_tasks(tasks):
    """Properly cleanup any pending tasks"""
    for task in tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Discord Channel Summarizer Bot')
    parser.add_argument('--run-now', action='store_true', help='Run the summary process immediately')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging first
    logger = setup_logging(args.dev)
    
    # Create and configure the bot
    bot = ChannelSummarizer(logger=logger)
    
    # Set dev mode if specified
    if args.dev:
        bot.dev_mode = True
        logger.info("Running in DEVELOPMENT mode")
    
    # Load configuration
    bot.load_config()
    
    # Add error handling and logging for channel processing
    try:
        # Get bot token
        bot_token = os.getenv('DISCORD_BOT_TOKEN')
        if not bot_token:
            raise ValueError("Discord bot token not found in environment variables")
        
        # Run the bot with the new async runner
        asyncio.run(run_bot(bot, bot_token, args.run_now))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 