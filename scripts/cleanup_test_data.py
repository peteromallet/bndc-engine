import discord
from discord.ext import commands
import asyncio
import logging
from dotenv import load_dotenv, set_key
import os
import sys
import argparse
import traceback

# Add parent directory to Python path BEFORE importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.constants import get_database_path
from src.common.db_handler import DatabaseHandler
from src.common.base_bot import BaseDiscordBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable discord.py debug logging
discord_logger = logging.getLogger('discord')
discord_logger.setLevel(logging.WARNING)

def main():
    # Always use dev database for test data cleanup
    db_path = get_database_path(True)  # Force dev database
    logger.info(f"Using database at: {db_path}")

    # Delete the dev database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Deleted database at: {db_path}")
    else:
        logger.info(f"No database found at: {db_path}")

    # Load environment variables from .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("No Discord bot token found in environment variables")
        sys.exit(1)

    # Initialize and run the bot
    bot = ChannelCleaner()

    async def cleanup():
        try:
            await bot.start(token)
        except KeyboardInterrupt:
            await bot.close()
        finally:
            await bot.close()

    # Add event handlers
    @bot.event
    async def on_ready():
        logger.info(f'Bot is ready: {bot.user.name}')
        
        # Get the dev channel IDs from environment
        dev_channels = os.getenv('DEV_CHANNELS_TO_MONITOR', '').split(',')
        dev_summary = os.getenv('DEV_SUMMARY_CHANNEL_ID')
        
        channels_to_clean = []
        if dev_summary:
            channels_to_clean.append(dev_summary.strip())
        channels_to_clean.extend([ch.strip() for ch in dev_channels if ch.strip()])
        
        # Process all dev channels
        for channel_id in channels_to_clean:
            try:
                channel = bot.get_channel(int(channel_id))
                if not channel:
                    logger.error(f"Could not find channel with ID {channel_id}")
                    continue

                logger.info(f"Cleaning messages from channel: #{channel.name}")
                deleted = await bot.delete_messages(channel)
                logger.info(f"Deleted {deleted} messages from #{channel.name}")
            except Exception as e:
                logger.error(f"Error cleaning channel {channel_id}: {e}")

        await bot.close()

    # Run the bot
    asyncio.run(cleanup())

class ChannelCleaner(BaseDiscordBot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.voice_states = False
        super().__init__(
            command_prefix="!",
            intents=intents,
            heartbeat_timeout=120.0,
            guild_ready_timeout=30.0,
            gateway_queue_size=512,
            logger=logger
        )
        self.target_user_id = 301463647895683072

    async def delete_messages(self, channel):
        """Delete all messages from a channel."""
        deleted_count = 0
        try:
            messages = []
            async for message in channel.history(limit=None):
                if message.author.id in [self.user.id, self.target_user_id]:
                    messages.append(message)

            if not messages:
                logger.info(f"No messages to delete in #{channel.name}")
                return 0

            logger.info(f"Found {len(messages)} messages to delete in #{channel.name}")
            
            # Try bulk deletion first
            if len(messages) > 1:
                try:
                    chunks = [messages[i:i + 100] for i in range(0, len(messages), 100)]
                    for chunk in chunks:
                        await channel.delete_messages(chunk)
                        deleted_count += len(chunk)
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Bulk deletion failed: {e}")
                    # Fall back to individual deletion
                    deleted_count = 0
                    for message in messages:
                        try:
                            await message.delete()
                            deleted_count += 1
                            await asyncio.sleep(1)
                        except Exception as e:
                            logger.error(f"Failed to delete message {message.id}: {e}")
            else:
                # Single message case
                await messages[0].delete()
                deleted_count = 1

            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning channel #{channel.name}: {e}")
            return deleted_count

if __name__ == "__main__":
    main()