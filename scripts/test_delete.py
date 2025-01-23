import discord
from discord.ext import commands
import asyncio
import logging
from dotenv import load_dotenv
import os
import sys
import traceback

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable discord.py debug logging
discord_logger = logging.getLogger('discord')
discord_logger.setLevel(logging.WARNING)

class MessageDeleter(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        logger.info("Initializing bot with intents: %s", intents)
        super().__init__(command_prefix="!", intents=intents)
        self.target_user_id = 301463647895683072  # Add target user ID

    async def delete_message(self, channel, message_id):
        """Delete a specific message."""
        try:
            logger.info(f"Attempting to fetch message {message_id} from channel {channel.name} ({channel.id})")
            # Get the message
            message = await channel.fetch_message(message_id)
            logger.info(f"Found message details:")
            logger.info(f"  ID: {message.id}")
            logger.info(f"  Author: {message.author} ({message.author.id})")
            logger.info(f"  Type: {message.type}")
            logger.info(f"  Content: {message.content}")
            logger.info(f"  Created at: {message.created_at}")
            logger.info(f"  Channel: {message.channel.name} ({message.channel.id})")
            if isinstance(message.channel, discord.Thread):
                logger.info(f"  Parent Channel: {message.channel.parent.name} ({message.channel.parent.id})")
            
            # Check if we should delete this message
            if message.author.id not in [self.user.id, self.target_user_id]:
                logger.warning(f"Message author {message.author.id} is not bot or target user - skipping deletion")
                return False

            # Try to delete it
            logger.info("Attempting to delete message...")
            await message.delete()
            logger.info(f"Successfully deleted message {message_id}")
            return True
        except discord.NotFound:
            logger.error(f"Message {message_id} not found")
            return False
        except discord.Forbidden as e:
            logger.error(f"Forbidden to delete message: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            logger.error("Full traceback: %s", traceback.format_exc())
            return False

    async def setup_hook(self):
        self.loop.create_task(self.delete_task())

    async def delete_task(self):
        await self.wait_until_ready()
        logger.info(f'Bot is ready: {self.user.name} ({self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')
        
        # Get the channel (art channel in this case)
        channel_id = int(os.getenv('ART_CHANNEL_ID', 0))
        logger.info(f"Looking for channel with ID: {channel_id}")
        channel = self.get_channel(channel_id)
        if not channel:
            logger.error(f"Could not find channel with ID {channel_id}")
            await self.close()
            return
        
        logger.info(f"Found channel: {channel.name} ({channel.id})")

        # Try to delete the specific message
        message_id = 1331785978515554400
        logger.info(f"Attempting to delete message: {message_id}")
        success = await self.delete_message(channel, message_id)
        logger.info(f"Delete operation {'succeeded' if success else 'failed'}")
        
        # Close the bot
        logger.info("Closing bot...")
        await self.close()

async def main():
    # Load environment variables
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    logger.info(f"Loading environment from: {env_path}")
    load_dotenv(env_path)
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("No Discord bot token found in environment variables")
        return

    # Initialize bot
    logger.info("Creating bot instance...")
    bot = MessageDeleter()
    
    try:
        logger.info("Starting bot...")
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await bot.close()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        logger.error("Full traceback: %s", traceback.format_exc())
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 