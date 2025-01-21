import discord
from discord.ext import commands
import asyncio
import logging
from dotenv import load_dotenv, set_key
import os
import sys
import argparse

# Add parent directory to Python path BEFORE importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.constants import get_database_path
from src.common.db_handler import DatabaseHandler

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
    parser = argparse.ArgumentParser(description='Cleanup Test Data')
    parser.add_argument('--dev', action='store_true', help='Run cleanup on development database')
    args = parser.parse_args()
    
    db_path = get_database_path(args.dev)
    logger.info(f"Using database at: {db_path}")

class ChannelCleaner(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.voice_states = False  # Explicitly disable voice states
        
        super().__init__(command_prefix="!", intents=intents)
        self.db = DatabaseHandler()
        logger.info(f"Connected to database at: {self.db.db_path}")
        self.target_user_id = 301463647895683072  # Add this line to store the target user ID
        
    async def delete_database_entries(self, channel_id: int):
        """Delete all database entries for a channel."""
        try:
            # Get counts before deletion
            self.db.cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE channel_id = ?",
                (channel_id,)
            )
            before_messages = self.db.cursor.fetchone()[0]
            
            self.db.cursor.execute(
                "SELECT COUNT(*) FROM daily_summaries WHERE channel_id = ?",
                (channel_id,)
            )
            before_daily = self.db.cursor.fetchone()[0]
            
            self.db.cursor.execute(
                "SELECT COUNT(*) FROM channel_summary WHERE channel_id = ?",
                (channel_id,)
            )
            before_summary = self.db.cursor.fetchone()[0]
            
            # Delete from all tables
            self.db.cursor.execute(
                "DELETE FROM messages WHERE channel_id = ?",
                (channel_id,)
            )
            logger.info(f"Deleted {before_messages} entries from messages for channel {channel_id}")
            
            # Clean up FTS table
            self.db.cursor.execute(
                "DELETE FROM messages_fts WHERE rowid IN (SELECT id FROM messages WHERE channel_id = ?)",
                (channel_id,)
            )
            logger.info(f"Cleaned up FTS entries for channel {channel_id}")
            
            self.db.cursor.execute(
                "DELETE FROM daily_summaries WHERE channel_id = ?",
                (channel_id,)
            )
            logger.info(f"Deleted {before_daily} entries from daily_summaries for channel {channel_id}")
            
            self.db.cursor.execute(
                "DELETE FROM channel_summary WHERE channel_id = ?",
                (channel_id,)
            )
            logger.info(f"Deleted {before_summary} entries from channel_summary for channel {channel_id}")
            
            # Commit all changes
            self.db.conn.commit()
            
            # Verify deletions
            self.db.cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE channel_id = ?",
                (channel_id,)
            )
            after_messages = self.db.cursor.fetchone()[0]
            
            if after_messages > 0:
                logger.warning(f"Still {after_messages} messages remaining for channel {channel_id}")
            else:
                logger.info(f"Successfully deleted all entries for channel {channel_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting database entries for channel {channel_id}: {e}")
            self.db.conn.rollback()
            return False

    async def delete_messages(self, channel):
        """Delete all messages from a channel or thread."""
        deleted_count = 0
        
        try:
            # Modified logging for initial message check
            logger.info(f"Starting message deletion in channel: #{channel.name}")
            
            # Check for deletable messages from both bot and target user
            has_deletable_messages = False
            message_count = 0
            deletable_message_count = 0
            async for message in channel.history(limit=None):
                message_count += 1
                if message.author.id in [self.user.id, self.target_user_id]:
                    has_deletable_messages = True
                    deletable_message_count += 1
            
            logger.info(f"Channel #{channel.name} has {message_count} total messages, {deletable_message_count} deletable messages")
            
            if not has_deletable_messages:
                if isinstance(channel, discord.Thread):
                    logger.info(f"Skipping thread (no deletable messages): #{channel.parent.name}/{channel.name}")
                else:
                    logger.info(f"Skipping channel (no deletable messages): #{channel.name}")
                return 0
                
            while True:
                messages = []
                async for message in channel.history(limit=None):
                    # Modified to check for both bot and target user messages
                    if message.author.id in [self.user.id, self.target_user_id]:
                        messages.append(message)
                
                if not messages:
                    break
                
                # Process in chunks of 100
                chunks = [messages[i:i + 100] for i in range(0, len(messages), 100)]
                
                for chunk in chunks:
                    if len(chunk) > 1:
                        await channel.delete_messages(chunk)
                        deleted_count += len(chunk)
                        if isinstance(channel, discord.Thread):
                            logger.info(f"Deleted {len(chunk)} messages from thread: #{channel.parent.name}/{channel.name}")
                        else:
                            logger.info(f"Deleted {len(chunk)} messages from channel: #{channel.name}")
                    else:
                        await chunk[0].delete()
                        deleted_count += 1
                        if isinstance(channel, discord.Thread):
                            logger.info(f"Deleted 1 message from thread: #{channel.parent.name}/{channel.name}")
                        else:
                            logger.info(f"Deleted 1 message from channel: #{channel.name}")
                    
                    await asyncio.sleep(1)  # Rate limiting
                
                if len(chunks) == 0:
                    break
                
        except Exception as e:
            if isinstance(channel, discord.Thread):
                logger.error(f"Error cleaning thread #{channel.parent.name}/{channel.name}: {e}")
            else:
                logger.error(f"Error cleaning channel #{channel.name}: {e}")
        
        return deleted_count

    async def clean_channel_and_threads(self, channel):
        """Clean a channel and all its threads."""
        total_deleted = 0
        
        # If it's a category, process all channels in it
        if isinstance(channel, discord.CategoryChannel):
            logger.info(f"Processing category: {channel.name}")
            for subchannel in channel.channels:
                deleted = await self.clean_channel_and_threads(subchannel)
                total_deleted += deleted
            return total_deleted
        
        # For text channels and threads
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            # Clean main channel messages
            deleted = await self.delete_messages(channel)
            total_deleted += deleted
            
            # Clean database entries
            db_deleted = await self.delete_database_entries(channel.id)
            if db_deleted:
                logger.info(f"Deleted database entries for channel #{channel.name}")
            
            # Only process threads if it's a text channel (not a thread)
            if isinstance(channel, discord.TextChannel):
                # Clean all threads in the channel
                threads = channel.threads
                logger.info(f"Found {len(threads)} active threads in #{channel.name}")
                
                # Get archived threads
                archived_threads = []
                async for thread in channel.archived_threads():
                    archived_threads.append(thread)
                
                all_threads = threads + archived_threads
                logger.info(f"Processing {len(all_threads)} total threads in #{channel.name}")
                
                for thread in all_threads:
                    try:
                        # Only delete threads created by the bot
                        if thread.owner_id == self.user.id:
                            await thread.delete()
                            logger.info(f"Deleted thread: {thread.name} in #{channel.name}")
                        else:
                            logger.info(f"Skipping thread not owned by bot: {thread.name} in #{channel.name}")
                    except Exception as e:
                        logger.error(f"Could not delete thread {thread.name}: {e}")
        
        return total_deleted

    async def close(self):
        """Cleanup when bot is shutting down."""
        self.db.conn.close()  # Remove await - sqlite connection doesn't use async
        await super().close()

if __name__ == "__main__":
    main()