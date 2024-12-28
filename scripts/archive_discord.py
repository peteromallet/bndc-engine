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
                'batch_size': 100,  # Messages per batch
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
            
            # Get last archived message ID
            last_message_id = self.db.get_last_message_id(channel_id)
            
            message_count = 0
            batch = []
            
            # Use history() with after parameter if we have a last message
            history_kwargs = {'limit': None}  # No limit to get all messages
            if last_message_id:
                history_kwargs['after'] = discord.Object(id=last_message_id)
            
            async for message in channel.history(**history_kwargs):
                try:
                    # Convert message to storable format
                    message_data = {
                        'id': message.id,
                        'message_id': message.id,
                        'channel_id': channel_id,
                        'author_id': message.author.id,
                        'author_name': message.author.name,
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
                        'flags': message.flags.value
                    }
                    
                    batch.append(message_data)
                    message_count += 1
                    
                    # Process batch when it reaches batch_size
                    if len(batch) >= config['batch_size']:
                        try:
                            self.db.store_messages(batch)
                            logger.info(f"Archived {message_count} messages from #{channel.name}")
                            batch = []
                            await asyncio.sleep(config['delay'])
                        except Exception as e:
                            logger.error(f"Failed to store batch: {e}")
                            # Consider implementing retry logic here
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing message {message.id}: {e}")
                    continue
            
            # Store any remaining messages
            if batch:
                try:
                    self.db.store_messages(batch)
                except Exception as e:
                    logger.error(f"Failed to store final batch: {e}")
            
        except Exception as e:
            logger.error(f"Error archiving channel {channel.name if channel else channel_id}: {e}")
        finally:
            # Don't close the connection here as it's reused across channels
            pass

    async def on_ready(self):
        """Called when bot is ready."""
        logger.info(f"Logged in as {self.user}")
        
        # Archive configured channels
        for channel_id in self.archive_configs:
            await self.archive_channel(channel_id)
        
        # Close the bot after archiving
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
        bot = MessageArchiver()
        bot.run(os.getenv('DISCORD_BOT_TOKEN'), reconnect=True)
    except discord.LoginFailure:
        logger.error("Failed to login. Please check your Discord token.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Ensure everything is cleaned up
        if bot and bot.db:
            bot.db.close()
        # Ensure event loop is properly closed
        loop = asyncio.get_event_loop()
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