import os
import sys
import argparse
import asyncio
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import traceback

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

async def run_bot(bot, token, run_now):
    """Run the bot with optional immediate summary generation"""
    try:
        # Create task for bot connection
        bot_task = asyncio.create_task(bot.start(token))
        
        # Wait for bot to be ready
        while not bot.is_ready():
            await asyncio.sleep(1)
            
        bot.logger.info("Bot is ready and fully connected")
            
        if run_now:
            bot.logger.info("Running immediate summary...")
            # Wait a moment for the bot to fully connect
            await asyncio.sleep(2)
            # Trigger immediate summary processing
            await bot.generate_summary()
            # Signal shutdown after summary completes
            bot._shutdown_flag = True
            # Close the bot connection
            await bot.close()
            # Cancel the bot task
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass
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
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
    except Exception as e:
        bot.logger.error(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def schedule_daily_summary(bot):
    """
    Run daily summaries on schedule. Only exits if there's an error or explicit shutdown.
    """
    try:
        while not bot._shutdown_flag:
            now = datetime.utcnow()
            target = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
            if now.hour >= 10:
                target += timedelta(days=1)
            
            delay = (target - now).total_seconds()
            bot.logger.info(f"Scheduler: Waiting {delay/3600:.2f} hours until next summary at {target} UTC")
            
            try:
                await asyncio.sleep(delay)
                if not bot._shutdown_flag:
                    bot.logger.info("Scheduler: Starting daily summary generation")
                    await bot.generate_summary()
                    bot.logger.info(f"Scheduler: Summary generated successfully at {datetime.utcnow()} UTC")
            except asyncio.CancelledError:
                bot.logger.info("Scheduler: Summary schedule cancelled - shutting down")
                break
            except Exception as e:
                bot.logger.error(f"Scheduler: Error generating summary: {e}")
                bot.logger.debug(traceback.format_exc())
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)
                
    except Exception as e:
        bot.logger.error(f"Scheduler: Fatal error in scheduler: {e}")
        bot.logger.debug(traceback.format_exc())
        bot._shutdown_flag = True
        raise

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