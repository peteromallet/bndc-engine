import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

from src.summaries.summarizer import ChannelSummarizer

async def run_bot(bot, token, run_now):
    """Run the bot with optional immediate summary generation"""
    try:
        if run_now:
            # Create a task for the bot's normal operation
            bot_task = asyncio.create_task(bot.start(token))
            # Wait a moment for the bot to connect
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
            await bot.start(token)
    except Exception as e:
        print(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Discord Channel Summarizer Bot')
    parser.add_argument('--run-now', action='store_true', help='Run the summary process immediately')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create and configure the bot
    bot = ChannelSummarizer()
    
    # Set dev mode if specified
    if args.dev:
        bot.dev_mode = True
        print("Running in DEVELOPMENT mode")
    
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
    except Exception as e:
        print(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 