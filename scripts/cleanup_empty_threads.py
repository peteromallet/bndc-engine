import os
import sys
import asyncio
import discord
from dotenv import load_dotenv
import logging
from pathlib import Path
import aioconsole

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.constants import get_database_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_thread(channel, thread):
    """Process a single thread and return True if changes were made."""
    try:
        # Get ALL messages including system messages to find and delete the thread creation message
        all_messages = [msg async for msg in thread.history(limit=None)]
        
        # Get non-system messages (excluding the bot's messages)
        regular_messages = [msg for msg in all_messages 
                          if not msg.type == discord.MessageType.thread_created
                          and msg.author.id != 1316765722738688030]
        
        # If thread is empty (only has system messages or messages from ignored user)
        if not regular_messages:
            logger.info(f"Found empty thread: {thread.name}")
            
            try:
                # Get the parent message
                parent_message = await channel.fetch_message(thread.id)
                
                # Add love letter reaction if it doesn't exist
                has_love_letter = any(r.emoji == '💌' for r in parent_message.reactions)
                if not has_love_letter:
                    await parent_message.add_reaction('💌')
                    logger.info(f"Added love letter reaction to message {parent_message.id}")
                
                # Find and delete the thread creation system message
                thread_creation_msg = next((msg for msg in all_messages 
                                          if msg.type == discord.MessageType.thread_created), None)
                if thread_creation_msg:
                    await thread_creation_msg.delete()
                    logger.info(f"Deleted thread creation system message")
                
                # Delete the thread
                await thread.delete()
                logger.info(f"Deleted empty thread: {thread.name}")
                return True
                
            except discord.NotFound:
                logger.warning(f"Could not find parent message for thread {thread.id}")
            except Exception as e:
                logger.error(f"Error processing thread {thread.id}: {e}")
    
    except Exception as e:
        logger.error(f"Error checking thread {thread.id}: {e}")
    
    return False

async def cleanup_empty_threads(channel_id: int, dev_mode: bool = False):
    """Clean up empty threads and add reactions to their parent messages."""
    # Load environment variables
    load_dotenv()
    
    # Get bot token based on mode
    token = os.getenv('DEV_DISCORD_BOT_TOKEN' if dev_mode else 'DISCORD_BOT_TOKEN')
    if not token:
        raise ValueError("Discord bot token not found in environment variables")

    # Set up Discord client with necessary intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guild_messages = True
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True
    intents = discord.Intents.all()  # Enable all intents to ensure we have thread access
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        try:
            logger.info(f"Bot connected as {client.user}")
            
            # Get the channel
            channel = client.get_channel(channel_id)
            if not channel:
                logger.error(f"Could not find channel with ID {channel_id}")
                await client.close()
                return
                
            logger.info(f"Processing threads in channel: {channel.name}")
            
            # Get all threads in the channel
            threads = await channel.guild.active_threads()
            art_channel_threads = [t for t in threads if t.parent_id == channel_id]
            
            # Sort threads by creation time (newest first)
            art_channel_threads.sort(key=lambda t: t.created_at, reverse=True)
            
            logger.info(f"Found {len(art_channel_threads)} threads to process")
            
            # Process each thread
            first_change = True
            for i, thread in enumerate(art_channel_threads, 1):
                logger.info(f"\nProcessing thread {i}/{len(art_channel_threads)}: {thread.name}")
                
                changes_made = await process_thread(channel, thread)
                
                if changes_made and first_change:
                    # Ask user if they want to continue, but only on first change
                    print("\nContinue processing threads? (y/n)")
                    response = await aioconsole.ainput()
                    if response.lower() != 'y':
                        logger.info("User chose to stop processing")
                        break
                    first_change = False
                
                # Add a small delay to avoid rate limits
                await asyncio.sleep(1)
            
            logger.info("Finished processing threads")
            await client.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            await client.close()

    try:
        await client.start(token)
    except KeyboardInterrupt:
        await client.close()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        await client.close()

def main():
    parser = argparse.ArgumentParser(description='Clean up empty threads and add reactions')
    parser.add_argument('--dev', action='store_true', help='Use development mode')
    args = parser.parse_args()
    
    # Use the specific channel ID
    channel_id = 1138865343314530324
    
    try:
        asyncio.run(cleanup_empty_threads(channel_id, args.dev))
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    main() 