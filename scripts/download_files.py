"""Script to download files from a Discord channel."""
import os
import sys
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv
import aiohttp
import logging
from typing import List
from pathlib import Path
import datetime

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from src
from src.common.rate_limiter import RateLimiter
from src.common.base_bot import BaseDiscordBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileDownloader(BaseDiscordBot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        super().__init__(
            command_prefix="!",
            intents=intents,
            heartbeat_timeout=120.0,
            guild_ready_timeout=30.0,
            gateway_queue_size=512,
            logger=logger
        )
        self.rate_limiter = RateLimiter()
        self.download_tasks = []
        self.selected_type = 'images'  # Change as needed
        self.selected_time = 'all'      # Change as needed
        self.download_all_channels = False
        self.channel_ids = [
            1315381196221321258,  # Your specified channel
            # Add more channel IDs here
        ]
        self.file_types = {
            'all': [],  # Empty list means all files
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.webp'],
            'videos': ['.mp4', '.mov', '.webm'],
            'json': ['.json']
        }
        self.time_filters = {
            'week': 7,
            'month': 30,
            'year': 365,
            'all': None  # None means no time limit
        }

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'Logged in as {self.user.name}')
        # Start the download process
        asyncio.create_task(self.run_downloads())

    async def run_downloads(self):
        try:
            logger.info("Starting download tasks...")
            
            # Calculate time limit if needed
            time_limit = None
            if self.time_filters[self.selected_time]:
                time_limit = discord.utils.utcnow() - datetime.timedelta(days=self.time_filters[self.selected_time])
            
            # Create a list to hold download tasks
            download_tasks = []
            
            if self.download_all_channels:
                for guild in self.guilds:
                    for channel in guild.text_channels:
                        download_tasks.append(
                            self.process_channel(
                                channel.id,
                                self.file_types[self.selected_type],
                                time_limit=time_limit
                            )
                        )
            else:
                for channel_id in self.channel_ids:
                    download_tasks.append(
                        self.process_channel(
                            channel_id,
                            self.file_types[self.selected_type],
                            time_limit=time_limit
                        )
                    )
            
            # Run download tasks concurrently with rate limiting
            await asyncio.gather(*download_tasks)
            
            logger.info("All download tasks completed.")
        except Exception as e:
            logger.error(f"Error during download tasks: {e}")
        finally:
            await self.close()

    async def download_file(self, url: str, filename: str, download_path: str):
        """Download a file from a URL."""
        try:
            # Wrap the download in rate limiter
            async def download():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            filepath = os.path.join(download_path, filename)
                            with open(filepath, 'wb') as f:
                                f.write(await response.read())
                            logger.info(f"Downloaded: {filename}")
                            return True
                        else:
                            logger.error(f"Failed to download {filename}: Status {response.status}")
                            return False
                            
            return await self.rate_limiter.execute(download_path, download())
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False

    async def process_channel(self, channel_id: int, file_types: List[str], time_limit=None):
        """Process channel and download files."""
        try:
            # Create absolute path to files directory
            base_dir = Path(__file__).parent.parent / 'files'
            download_path = base_dir / f"channel_{channel_id}"
            download_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving files to: {download_path}")
            
            # Wrap the channel fetch in rate limiter
            async def get_channel():
                return self.get_channel(channel_id)
                
            channel = await self.rate_limiter.execute(f"channel_{channel_id}", get_channel())
            if not channel:
                logger.warning(f"Could not find channel with ID {channel_id}")
                return
            
            logger.info(f"Downloading files from #{channel.name}")
            downloaded_count = 0
            
            # Track message attachment counts
            message_attachment_counts = {}
            
            # Wrap the history fetch in rate limiter
            async def get_history():
                messages = []
                async for message in channel.history(limit=None):
                    if time_limit and message.created_at < time_limit:
                        break
                    messages.append(message)
                return messages
                
            messages = await self.rate_limiter.execute(f"history_{channel_id}", get_history())
            
            for message in messages:
                if message.attachments:
                    message_attachment_counts[message.id] = 0
                    
                for attachment in message.attachments:
                    file_ext = os.path.splitext(attachment.filename)[1].lower()
                    
                    if not file_types or any(type in file_ext for type in file_types):
                        # Increment counter for this message's attachments
                        message_attachment_counts[message.id] += 1
                        
                        # Create filename in format: msg_{messageid}_{number}{extension}
                        unique_filename = f"msg_{message.id}_{message_attachment_counts[message.id]}{file_ext}"
                        
                        if await self.download_file(
                            attachment.url,
                            unique_filename,
                            str(download_path)
                        ):
                            downloaded_count += 1
            
            logger.info(f"Downloaded {downloaded_count} files from #{channel.name} to {download_path}")
            
        except Exception as e:
            logger.error(f"Error processing channel {channel_id}: {e}")

async def main():
    load_dotenv()
    bot = FileDownloader()
    try:
        await bot.start(os.getenv('DISCORD_BOT_TOKEN'))
    except KeyboardInterrupt:
        logger.info("Bot shutdown initiated by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 