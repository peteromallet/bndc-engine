import os
import sys
import argparse
import asyncio
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import traceback
import time

from src.features.curation.curator import ArtCurator
from src.features.summarisation.summarizer import ChannelSummarizer
from src.common.log_handler import LogHandler

def setup_logging(dev_mode=False):
    """Setup shared logging configuration for all bots"""
    # Create log handler with proper mode
    log_handler = LogHandler(
        logger_name='DiscordBot',
        prod_log_file='discord_bot.log',
        dev_log_file='discord_bot_dev.log'
    )
    
    # Setup logging with proper mode
    logger = log_handler.setup_logging(dev_mode)
    
    # Verify logger was created successfully
    if not logger:
        print("ERROR: Failed to create logger")
        sys.exit(1)
        
    return logger

# Constants
MAX_RETRIES = 3
READY_TIMEOUT = 60
INITIAL_RETRY_DELAY = 3600  # 1 hour
MAX_RETRY_WAIT = 24 * 3600  # 24 hours in seconds

async def run_summarizer(bot, token, run_now):
    """Run the summarizer bot with optional immediate summary generation"""
    try:
        # Create task for bot connection
        bot_task = asyncio.create_task(bot.start(token))
        
        # Wait for bot to be ready
        start_time = time.time()
        while not bot.is_ready():
            if time.time() - start_time > READY_TIMEOUT:
                raise TimeoutError("Summarizer bot failed to become ready within timeout period")
            await asyncio.sleep(1)
            
        bot.logger.info("Summarizer bot is ready and fully connected")
            
        if run_now:
            bot.logger.info("Running immediate summary generation...")
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
        bot.logger.error(f"Error running summarizer bot: {e}")
        traceback.print_exc()
        raise

async def run_curator(bot, token):
    """Run the curator bot"""
    try:
        # Create task for bot connection
        bot_task = asyncio.create_task(bot.start(token))
        
        # Wait for bot to be ready
        start_time = time.time()
        while not bot.is_ready():
            if time.time() - start_time > READY_TIMEOUT:
                raise TimeoutError("Curator bot failed to become ready within timeout period")
            await asyncio.sleep(1)
            
        bot.logger.info("Curator bot is ready and fully connected")
            
        # Wait for the bot task to complete or be cancelled
        await bot_task
            
    except Exception as e:
        bot.logger.error(f"Error running curator bot: {e}")
        traceback.print_exc()
        raise

async def schedule_daily_summary(bot):
    """Run daily summaries on schedule. Only exits if there's an error or explicit shutdown."""
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
            hours_until_next = delay/3600
            bot.logger.info(f"Next summary scheduled for {target} UTC ({hours_until_next:.1f} hours from now)")
            
            # Wait until the target time
            try:
                await asyncio.sleep(delay)
                if not bot._shutdown_flag:
                    bot.logger.info("Starting scheduled summary generation")
                    await bot.generate_summary()
                    # Success - clear retry count
                    retry_count = 0
                    bot.logger.info("Scheduled summary generation completed successfully")
            except asyncio.CancelledError:
                bot.logger.info("Summary schedule cancelled - shutting down")
                break
            except Exception as e:
                retry_count += 1
                bot.logger.error(f"Summary generation attempt {retry_count}/{MAX_RETRIES} failed: {e}")
                if retry_count >= MAX_RETRIES:
                    bot.logger.error(f"Failed after {MAX_RETRIES} attempts - shutting down scheduler")
                    bot._shutdown_flag = True
                    raise
                wait_time = min(INITIAL_RETRY_DELAY * (2 ** retry_count), MAX_RETRY_WAIT)
                bot.logger.info(f"Retrying in {wait_time/3600:.1f} hours")
                await asyncio.sleep(wait_time)
    except Exception as e:
        bot.logger.error(f"Fatal error in scheduler: {e}")
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

async def run_all_bots(curator_bot, summarizer_bot, token, run_now, logger):
    """Run both bots concurrently"""
    try:
        # Create tasks for both bots
        curator_task = asyncio.create_task(
            run_curator(curator_bot, token)
        )
        summarizer_task = asyncio.create_task(
            run_summarizer(summarizer_bot, token, run_now)
        )
        
        curator_bot.logger.info("Starting curator bot")
        summarizer_bot.logger.info("Starting summarizer bot")
        
        # Wait for either task to complete (or fail)
        done, pending = await asyncio.wait(
            [curator_task, summarizer_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # If one bot fails, cancel the other and raise the exception
        for task in done:
            try:
                await task
            except Exception as e:
                # Cancel remaining tasks
                await cleanup_tasks(pending)
                raise
                
        # Clean up any remaining tasks
        await cleanup_tasks(pending)
        
    except Exception as e:
        curator_bot.logger.error(f"Error in bot operation: {e}")
        summarizer_bot.logger.error(f"Error in bot operation: {e}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Discord Bots')
    parser.add_argument('--summary-now', action='store_true', help='Run the summary process immediately')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup shared logging for both bots
    logger = setup_logging(args.dev)
    logger.info("Starting bot initialization")
    
    # Create and configure both bots with shared logger
    curator_bot = ArtCurator(logger=logger)
    summarizer_bot = ChannelSummarizer(logger=logger)
    
    # Set dev mode if specified
    if args.dev:
        curator_bot.dev_mode = True
        summarizer_bot.dev_mode = True
        logger.info("Running in DEVELOPMENT mode")
    else:
        logger.info("Running in PRODUCTION mode")
    
    # Load configuration for summarizer
    summarizer_bot.load_config()
    
    try:
        # Get bot token
        token = os.getenv('DISCORD_BOT_TOKEN')
        if not token:
            raise ValueError("Discord bot token not found in environment variables")
        
        logger.info("Configuration loaded successfully, starting bots")
        
        # Run both bots
        asyncio.run(run_all_bots(
            curator_bot,
            summarizer_bot,
            token,
            args.summary_now,
            logger
        ))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running bots: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 