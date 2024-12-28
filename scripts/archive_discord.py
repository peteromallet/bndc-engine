import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discord
from discord.ext import commands
import asyncio
from datetime import datetime
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
from src.common.db_handler import DatabaseHandler
from src.common.constants import DATABASE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('archive_discord.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MessageArchiver(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Load environment variables
        load_dotenv()
        
        # Don't create DB connection in __init__
        self.db_path = DATABASE_PATH  # Consider moving to env var
        self.db = None
        
        # Check if token exists
        if not os.getenv('DISCORD_BOT_TOKEN'):
            raise ValueError("DISCORD_BOT_TOKEN not found in environment variables")
        
        # Add reconnect settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Channel configurations
        self.archive_configs = {
            1145677539738665020: {  # comfyui channel
                'name': 'comfyui',
                'batch_size': 100,  # Make sure this is 100
                'delay': 1.0  # Delay between batches in seconds
            }
        }

    async def setup_hook(self):
        """Setup hook to initialize database and start archiving."""
        try:
            # Create fresh connection when starting up
            self.db = DatabaseHandler(self.db_path)
            self.db.init_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def archive_channel(self, channel_id: int) -> None:
        """Archive all messages from a channel."""
        try:
            # Verify DB connection is healthy
            if not self.db or not self.db.conn:
                self.db = DatabaseHandler(self.db_path)
            
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Could not find channel {channel_id}")
                return

            config = self.archive_configs.get(channel_id)
            if not config:
                logger.error(f"No configuration found for channel {channel_id}")
                return

            logger.info(f"Starting archive of #{channel.name}")
            
            try:
                # Get date range of archived messages
                earliest_date, latest_date = self.db.get_message_date_range(channel_id)
            except AttributeError as e:
                logger.error(f"Database method not found: {e}")
                return
            except Exception as e:
                logger.error(f"Error getting message date range: {e}")
                return

            archived_message_ids = set(self.db.get_all_message_ids(channel_id))
            logger.info(f"Found {len(archived_message_ids)} previously archived messages")
            
            if earliest_date and latest_date:
                logger.info(f"Archived messages range from {earliest_date} to {latest_date}")
            
            message_count = 0
            batch = []
            
            # Get all message dates to check for gaps
            message_dates = self.db.get_message_dates(channel_id)
            if message_dates:
                # Sort dates and find gaps larger than 1 day
                message_dates.sort()
                gaps = []
                for i in range(len(message_dates) - 1):
                    current = datetime.fromisoformat(message_dates[i])
                    next_date = datetime.fromisoformat(message_dates[i + 1])
                    if (next_date - current).days > 1:
                        gaps.append((current, next_date))
                
                if gaps:
                    logger.info(f"Found {len(gaps)} gaps in message history")
                    for start, end in gaps:
                        logger.info(f"Searching for messages between {start} and {end}")
                        async for message in channel.history(limit=None, after=start, before=end, oldest_first=True):
                            if message.id not in archived_message_ids:
                                message_count = await self._process_message(message, channel_id, batch, message_count, config)

            # Still check before earliest and after latest
            earliest_date, latest_date = self.db.get_message_date_range(channel_id)
            if earliest_date:
                logger.info("Searching for older messages...")
                async for message in channel.history(limit=None, before=earliest_date, oldest_first=True):
                    if message.id not in archived_message_ids:
                        message_count = await self._process_message(message, channel_id, batch, message_count, config)
            
            if latest_date:
                logger.info("Searching for newer messages...")
                async for message in channel.history(limit=None, after=latest_date):
                    if message.id not in archived_message_ids:
                        message_count = await self._process_message(message, channel_id, batch, message_count, config)
            
            # If no archived messages exist, get all messages
            if not earliest_date and not latest_date:
                logger.info("No existing archives found. Getting all messages...")
                async for message in channel.history(limit=None, oldest_first=True):
                    if message.id not in archived_message_ids:
                        message_count = await self._process_message(message, channel_id, batch, message_count, config)
            
            logger.info(f"Found {message_count} new messages to archive")
            
            # Store any remaining messages
            if batch:
                try:
                    self.db.store_messages(batch)
                    logger.info(f"Stored final batch of {len(batch)} messages")
                except Exception as e:
                    logger.error(f"Failed to store final batch: {e}")
            
            logger.info(f"Archive complete - processed {message_count} new messages")
            
        except Exception as e:
            logger.error(f"Error archiving channel {channel.name if channel else channel_id}: {e}")
        finally:
            # Don't close the connection here as it's reused across channels
            pass

    async def _process_message(self, message, channel_id, batch, message_count, config):
        """Helper method to process a single message."""
        try:
            message_data = {
                'id': message.id,
                'message_id': message.id,
                'channel_id': channel_id,
                'author_id': message.author.id,
                'author_name': message.author.name,
                'author_discriminator': message.author.discriminator,
                'author_avatar_url': str(message.author.avatar.url) if message.author.avatar else None,
                'content': message.content,
                'created_at': message.created_at.isoformat(),
                'attachments': [
                    {
                        'url': attachment.url,
                        'filename': attachment.filename
                    } for attachment in message.attachments
                ],
                'embeds': [embed.to_dict() for embed in message.embeds],
                'reactions': [
                    {
                        'emoji': str(reaction.emoji),
                        'count': reaction.count
                    } for reaction in message.reactions
                ] if message.reactions else [],
                'reference_id': message.reference.message_id if message.reference else None,
                'edited_at': message.edited_at.isoformat() if message.edited_at else None,
                'is_pinned': message.pinned,
                'thread_id': message.thread.id if hasattr(message, 'thread') and message.thread else None,
                'message_type': str(message.type),
                'flags': message.flags.value,
                'jump_url': message.jump_url,
                'channel_name': message.channel.name,
            }
            
            batch.append(message_data)
            message_count += 1
            
            # Process batch when it reaches batch_size
            if len(batch) >= config['batch_size']:
                try:
                    initial_batch_size = len(batch)
                    self.db.store_messages(batch)
                    logger.info(f"Processed batch of {initial_batch_size} messages from #{message.channel.name} (total processed: {message_count})")
                    batch.clear()  # Clear the batch instead of creating new list
                    await asyncio.sleep(config['delay'])
                except Exception as e:
                    logger.error(f"Failed to store batch: {e}")
            
            return message_count
                    
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            return message_count

    async def on_ready(self):
        """Called when bot is ready."""
        try:
            logger.info(f"Logged in as {self.user}")
            
            # Archive configured channels
            for channel_id in self.archive_configs:
                await self.archive_channel(channel_id)
            
            logger.info("Archiving complete, shutting down bot")
            # Close the bot after archiving
            await self.close()
        except Exception as e:
            logger.error(f"Error in on_ready: {e}")
            await self.close()

    async def close(self):
        """Properly close the bot and database connection."""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
                self.db = None
            await super().close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point."""
    bot = None
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        bot = MessageArchiver()
        
        # Start the bot and keep it running until archiving is complete
        async def runner():
            await bot.start(os.getenv('DISCORD_BOT_TOKEN'))
            # Wait for the bot to be ready and complete archiving
            while not bot.is_closed():
                await asyncio.sleep(1)
        
        # Run the bot until it completes
        loop.run_until_complete(runner())
        
    except discord.LoginFailure:
        logger.error("Failed to login. Please check your Discord token.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Ensure everything is cleaned up properly
        if bot:
            if bot.db:
                bot.db.close()
            if not loop.is_closed():
                loop.run_until_complete(bot.close())
            
        # Clean up the event loop
        try:
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                remaining_tasks = asyncio.all_tasks(loop)
                if remaining_tasks:
                    loop.run_until_complete(asyncio.gather(*remaining_tasks))
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            if loop.is_running():
                loop.stop()
            if not loop.is_closed():
                loop.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot shutdown initiated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}") 