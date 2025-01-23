# Standard library imports
import asyncio
import copy
import io
import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime, timedelta
from typing import List, Tuple, Set, Dict, Optional, Any, Union
import sqlite3
import argparse
import random

# Third-party imports
import aiohttp
import anthropic
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Local imports
from src.common.db_handler import DatabaseHandler
from src.common.errors import *
from src.common.error_handler import ErrorHandler, handle_errors
from src.common.rate_limiter import RateLimiter
from src.common.log_handler import LogHandler
from scripts.news_summary import NewsSummarizer

# Optional imports for media processing
try:
    from PIL import Image
    import moviepy.editor as mp
    MEDIA_PROCESSING_AVAILABLE = True
except ImportError:
    MEDIA_PROCESSING_AVAILABLE = False

################################################################################
# You may already have a scheduling function somewhere, but here is a simple stub:
################################################################################
async def schedule_daily_summary(bot):
    """
    Example stub for daily scheduled runs. 
    Adjust logic and scheduling library as appropriate to your environment.
    """
    while not bot._shutdown_flag:
        now_utc = datetime.utcnow()
        # Suppose we run at 10:00 UTC daily
        run_time = now_utc.replace(hour=10, minute=0, second=0, microsecond=0)
        if run_time < now_utc:
            run_time += timedelta(days=1)
        await asyncio.sleep((run_time - now_utc).total_seconds())
        
        if bot._shutdown_flag:
            break
        
        try:
            await bot.generate_summary()
        except Exception as e:
            bot.logger.error(f"Scheduled summary run failed: {e}")
        
        # Sleep 24h until next scheduled run:
        await asyncio.sleep(86400)

################################################################################

class ChannelSummarizerError(Exception):
    """Base exception class for ChannelSummarizer"""
    pass

class APIError(ChannelSummarizerError):
    """Raised when API calls fail"""
    pass

class DiscordError(ChannelSummarizerError):
    """Raised when Discord operations fail"""
    pass

class SummaryError(ChannelSummarizerError):
    """Raised when summary generation fails"""
    pass

class Attachment:
    def __init__(self, filename: str, data: bytes, content_type: str, reaction_count: int, username: str, content: str = "", jump_url: str = ""):
        self.filename = filename
        self.data = data
        self.content_type = content_type
        self.reaction_count = reaction_count
        self.username = username
        self.content = content
        self.jump_url = jump_url  # Add jump_url field

class AttachmentHandler:
    def __init__(self, max_size: int = 25 * 1024 * 1024):
        self.max_size = max_size
        self.attachment_cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger('ChannelSummarizer')
        
    def clear_cache(self):
        """Clear the attachment cache"""
        self.attachment_cache.clear()
        
    async def process_attachment(self, attachment: discord.Attachment, message: discord.Message, session: aiohttp.ClientSession, original_jump_url: str = None) -> Optional[Attachment]:
        """Process a single attachment with size and type validation."""
        try:
            cache_key = f"{message.channel.id}:{message.id}"
            
            # Use original_jump_url if provided (dev mode), otherwise use message.jump_url
            jump_url = original_jump_url if original_jump_url else message.jump_url

            async with session.get(attachment.url, timeout=300) as response:
                if response.status != 200:
                    raise APIError(f"Failed to download attachment: HTTP {response.status}")

                file_data = await response.read()
                if len(file_data) > self.max_size:
                    self.logger.warning(f"Skipping large file {attachment.filename} ({len(file_data)/1024/1024:.2f}MB)")
                    return None

                total_reactions = sum(reaction.count for reaction in message.reactions) if message.reactions else 0
                
                # Get guild display name (nickname) if available, otherwise use display name
                author_name = message.author.display_name
                if hasattr(message.author, 'guild'):
                    member = message.guild.get_member(message.author.id)
                    if member:
                        author_name = member.nick or member.display_name

                processed_attachment = Attachment(
                    filename=attachment.filename,
                    data=file_data,
                    content_type=attachment.content_type,
                    reaction_count=total_reactions,
                    username=author_name,  # Use the determined name
                    content=message.content,
                    jump_url=jump_url  # Use the correct jump URL
                )

                # Ensure the cache key structure is consistent
                if cache_key not in self.attachment_cache:
                    self.attachment_cache[cache_key] = {
                        'attachments': [],
                        'reaction_count': total_reactions,
                        'username': author_name,
                        'channel_id': str(message.channel.id)
                    }
                self.attachment_cache[cache_key]['attachments'].append(processed_attachment)

                return processed_attachment

        except Exception as e:
            self.logger.error(f"Failed to process attachment {attachment.filename}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def prepare_files(self, message_ids: List[str], channel_id: str) -> List[Tuple[discord.File, int, str, str]]:
        """Prepare Discord files from cached attachments."""
        files = []
        for message_id in message_ids:
            # Use composite key to look up attachments
            cache_key = f"{channel_id}:{message_id}"
            if cache_key in self.attachment_cache:
                for attachment in self.attachment_cache[cache_key]['attachments']:
                    try:
                        file = discord.File(
                            io.BytesIO(attachment.data),
                            filename=attachment.filename,
                            description=f"From message ID: {message_id} (🔥 {attachment.reaction_count} reactions)"
                        )
                        files.append((
                            file,
                            attachment.reaction_count,
                            message_id,
                            attachment.username
                        ))
                    except Exception as e:
                        self.logger.error(f"Failed to prepare file {attachment.filename}: {e}")
                        continue

        return sorted(files, key=lambda x: x[1], reverse=True)[:10]

    def get_all_files_sorted(self) -> List[Attachment]:
        """
        Retrieve all attachments sorted by reaction count in descending order.
        """
        all_attachments = []
        for channel_data in self.attachment_cache.values():
            all_attachments.extend(channel_data['attachments'])
        
        # Sort attachments by reaction_count in descending order
        sorted_attachments = sorted(all_attachments, key=lambda x: x.reaction_count, reverse=True)
        return sorted_attachments

class MessageFormatter:
    @staticmethod
    def format_usernames(usernames: List[str]) -> str:
        """Format a list of usernames with proper grammar and bold formatting."""
        unique_usernames = list(dict.fromkeys(usernames))
        if not unique_usernames:
            return ""
        
        # Add bold formatting if not already present
        formatted_usernames = []
        for username in unique_usernames:
            if not username.startswith('**'):
                username = f"**{username}**"
            formatted_usernames.append(username)
        
        if len(formatted_usernames) == 1:
            return formatted_usernames[0]
        
        return f"{', '.join(formatted_usernames[:-1])} and {formatted_usernames[-1]}"

    @staticmethod
    def chunk_content(content: str, max_length: int = 1900) -> List[Tuple[str, Set[str]]]:
        """Split content into chunks while preserving message links."""
        chunks = []
        current_chunk = ""
        current_chunk_links = set()

        for line in content.split('\n'):
            message_links = set(re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', line))
            
            # Start new chunk if we hit emoji or length limit
            if (any(line.startswith(emoji) for emoji in ['🎥', '💻', '🎬', '🤖', '📱', '🔧', '🎨', '📊']) and 
                current_chunk):
                if current_chunk:
                    chunks.append((current_chunk, current_chunk_links))
                current_chunk = ""
                current_chunk_links = set()
                current_chunk += '\n---\n\n'

            if len(current_chunk) + len(line) + 2 <= max_length:
                current_chunk += line + '\n'
                current_chunk_links.update(message_links)
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_chunk_links))
                current_chunk = line + '\n'
                current_chunk_links = set(message_links)

        if current_chunk:
            chunks.append((current_chunk, current_chunk_links))

        return chunks

    def chunk_long_content(self, content: str, max_length: int = 1900) -> List[str]:
        """Split content into chunks that respect Discord's length limits."""
        chunks = []
        current_chunk = ""
        
        # Split by lines to avoid breaking mid-sentence
        lines = content.split('\n')
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 <= max_length:
                current_chunk += line + '\n'
            else:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class ChannelSummarizer(commands.Bot):
    def __init__(self, logger=None, dev_mode=False):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        intents.members = True
        intents.presences = True
        intents.reactions = True  # Add reactions intent

        super().__init__(
            command_prefix="!", 
            intents=intents,
            heartbeat_timeout=60.0,
            guild_ready_timeout=10.0,
            gateway_queue_size=512
        )
        
        self._dev_mode = None
        self.logger = logger or logging.getLogger(__name__)
        self.log_handler = LogHandler(
            logger_name='ChannelSummarizer',
            prod_log_file='discord_bot.log',
            dev_log_file='discord_bot_dev.log'
        )
        
        self.claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.session = None
        
        self.rate_limiter = RateLimiter()
        self.attachment_handler = AttachmentHandler()
        self.message_formatter = MessageFormatter()
        self.db = DatabaseHandler(dev_mode=dev_mode)
        self.error_handler = ErrorHandler()
        
        self.guild_id = None
        self.summary_channel_id = None
        self.channels_to_monitor = []
        self.dev_channels_to_monitor = []
        self.first_message = None
        self._summary_lock = asyncio.Lock()
        self._shutdown_flag = False
        self.current_summary_attachments = []
        self.approved_channels = []
        self.original_urls = {}  # Add this to store original URLs
        
        # Set initial dev mode (this will trigger the setter to load correct IDs)
        self.dev_mode = dev_mode
        
        # Initialize discord_client as self since we inherit from commands.Bot
        self.discord_client = self

    def setup_logger(self, dev_mode):
        """Initialize or update logger configuration"""
        self.logger = self.log_handler.setup_logging(dev_mode)
        
        if self.logger:
            self.logger.info("Bot initializing...")
            if dev_mode:
                self.logger.debug("Development mode enabled")

    @property
    def dev_mode(self):
        return self._dev_mode

    @dev_mode.setter
    def dev_mode(self, value):
        """Set development mode and reconfigure logger"""
        if self._dev_mode != value:
            self._dev_mode = value
            self.setup_logger(value)

    def load_config(self):
        """Load configuration based on mode"""
        self.logger.debug("Loading configuration...")
        self.logger.debug(f"Current TEST_DATA_CHANNEL: {os.getenv('TEST_DATA_CHANNEL')}")
        
        load_dotenv(override=True)
        self.logger.debug(f"After reload TEST_DATA_CHANNEL: {os.getenv('TEST_DATA_CHANNEL')}")
        
        self.logger.debug("All channel-related environment variables:")
        for key, value in os.environ.items():
            if 'CHANNEL' in key:
                self.logger.debug(f"{key}: {value}")
        
        try:
            if self.dev_mode:
                self.logger.info("Loading development configuration")
                self.guild_id = int(os.getenv('DEV_GUILD_ID'))
                self.summary_channel_id = int(os.getenv('DEV_SUMMARY_CHANNEL_ID'))
                channels_str = os.getenv('DEV_CHANNELS_TO_MONITOR')
                if not channels_str:
                    raise ConfigurationError("DEV_CHANNELS_TO_MONITOR not found in environment")
                try:
                    self.dev_channels_to_monitor = [int(chan.strip()) for chan in channels_str.split(',') if chan.strip()]
                    self.logger.info(f"DEV_CHANNELS_TO_MONITOR: {self.dev_channels_to_monitor}")
                except ValueError as e:
                    raise ConfigurationError(f"Invalid channel ID in DEV_CHANNELS_TO_MONITOR: {e}")
            else:
                self.logger.info("Loading production configuration")
                self.guild_id = int(os.getenv('GUILD_ID'))
                self.summary_channel_id = int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))
                channels_str = os.getenv('CHANNELS_TO_MONITOR')
                if not channels_str:
                    raise ConfigurationError("CHANNELS_TO_MONITOR not found in environment")
                try:
                    self.channels_to_monitor = [int(chan.strip()) for chan in channels_str.split(',') if chan.strip()]
                    self.logger.info(f"CHANNELS_TO_MONITOR: {self.channels_to_monitor}")
                except ValueError as e:
                    raise ConfigurationError(f"Invalid ID in CHANNELS_TO_MONITOR: {e}")
            
            self.logger.info(
                f"Configured with guild_id={self.guild_id}, "
                f"summary_channel={self.summary_channel_id}, "
                f"channels={self.channels_to_monitor if not self.dev_mode else self.dev_channels_to_monitor}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from test.json (not heavily used in current approach)."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_data_path = os.path.join(script_dir, 'test.json')
            
            if not os.path.exists(test_data_path):
                test_data = {
                    "messages": [
                        {
                            "content": "Test message 1",
                            "author": "test_user",
                            "timestamp": datetime.utcnow().isoformat(),
                            "attachments": [],
                            "reactions": 0,
                            "id": "1234567890"
                        }
                    ]
                }
                with open(test_data_path, 'w') as f:
                    json.dump(test_data, f, indent=2)
                self.logger.info(f"Created default test data at {test_data_path}")
                return test_data["messages"]
                
            with open(test_data_path, 'r') as f:
                data = json.load(f)
                return data.get("messages", [])
                
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            return []

    async def setup_hook(self):
        """Called when the bot is starting up."""
        try:
            self.session = aiohttp.ClientSession()
        except Exception as e:
            raise ConfigurationError("Failed to initialize bot", e)

    async def on_ready(self):
        """Called when the bot is fully connected."""
        try:
            self.logger.info(f"Logged in as {self.user}")
            
            notification_channel = self.get_channel(self.summary_channel_id)
            if not notification_channel:
                self.logger.error(f"Could not find summary channel with ID {self.summary_channel_id}")
                self.logger.info("Available channels:")
                for guild in self.guilds:
                    for channel in guild.channels:
                        self.logger.info(f"- {channel.name} (ID: {channel.id})")
                return
            
            admin_user = await self.fetch_user(int(os.getenv('ADMIN_USER_ID')))
            self.error_handler = ErrorHandler(notification_channel, admin_user)
            self.logger.info(f"Successfully initialized with summary channel: {notification_channel.name}")
            
        except Exception as e:
            self.logger.error(f"Error in on_ready: {e}")
            self.logger.debug(traceback.format_exc())

    async def get_channel_history(self, channel_id: int) -> List[dict]:
        """Retrieve recent message history for a channel from the database (past 24h)."""
        self.logger.info(f"Getting message history for channel {channel_id} from database")
        try:
            # Skip date check in dev mode
            date_condition = "" if self.dev_mode else "AND m.created_at > datetime('now', '-1 day')"
            
            # Just get messages from this specific channel
            query = f"""
                SELECT 
                    m.message_id, m.channel_id, m.author_id, m.content,
                    m.created_at, m.attachments, m.embeds, m.reaction_count,
                    m.reactors, m.reference_id, m.edited_at, m.is_pinned,
                    m.thread_id, m.message_type, m.flags, m.jump_url,
                    m.indexed_at,
                    COALESCE(mem.server_nick, mem.global_name, mem.username) as author_name,
                    c.channel_name
                FROM messages m
                LEFT JOIN members mem ON m.author_id = mem.member_id
                LEFT JOIN channels c ON m.channel_id = c.channel_id
                WHERE m.channel_id = ?
                {date_condition}
                ORDER BY m.created_at DESC
            """
            
            # Set row factory to return dictionaries
            self.db.conn.row_factory = sqlite3.Row
            cursor = self.db.conn.cursor()
            cursor.execute(query, (channel_id,))
            messages = [dict(row) for row in cursor.fetchall()]
            self.db.conn.row_factory = None
            
            self.logger.info(f"Retrieved {len(messages)} messages for channel {channel_id}")
            return messages

        except Exception as e:
            self.logger.error(f"Error getting messages from database: {e}")
            self.logger.debug(traceback.format_exc())
            return []


    async def generate_short_summary(self, full_summary: str, message_count: int) -> str:
        """
        Get a short summary using Claude with proper async handling.
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                conversation = f"""Create exactly 3 bullet points summarizing key developments. STRICT format requirements:
1. The FIRST LINE MUST BE EXACTLY: 📨 __{message_count} messages sent__
2. Then three bullet points that:
   - Start with -
   - Give a short summary of one of the main topics from the full summary - priotise topics that are related to the channel and are likely to be useful to others.
   - Bold the most important finding/result/insight using **
   - Keep each to a single line
4. DO NOT MODIFY THE MESSAGE COUNT OR FORMAT IN ANY WAY

Required format:
"📨 __{message_count} messages sent__
• [Main topic 1] 
• [Main topic 2]
• [Main topic 3]"
DO NOT CHANGE THE MESSAGE COUNT LINE. IT MUST BE EXACTLY AS SHOWN ABOVE. DO NOT ADD INCLUDE ELSE IN THE MESSAGE OTHER THAN THE ABOVE.

Full summary to work from:
{full_summary}"""

                loop = asyncio.get_running_loop()
                
                # Define a helper function to call the synchronous Claude API method
                def create_short_summary():
                    return self.claude.messages.create(
                        model="claude-3-5-haiku-latest",
                        max_tokens=8192,
                        messages=[
                            {
                                "role": "user",
                                "content": conversation
                            }
                        ]
                    )
                
                # Run the synchronous create_summary in a separate thread to avoid blocking
                response = await loop.run_in_executor(None, create_short_summary)
                
                return response.content[0].text.strip()
                    
            except asyncio.TimeoutError:
                retry_count += 1
                self.logger.error(f"Timeout attempt {retry_count}/{max_retries} while generating short summary")
                if retry_count < max_retries:
                    self.logger.info(f"Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    self.logger.error("All retry attempts failed")
                    return f"__📨 {message_count} messages sent__\n• Error generating summary\u200B"
                    
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error attempt {retry_count}/{max_retries} while generating short summary: {e}")
                if retry_count < max_retries:
                    self.logger.info(f"Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    self.logger.error("All retry attempts failed")
                    return f"__📨 {message_count} messages sent__\n• Error generating summary\u200B"

    async def safe_send_message(self, channel, content=None, embed=None, file=None, files=None, reference=None):
        """Safely send a message with concurrency-limited retry logic."""
        try:
            return await self.rate_limiter.execute(
                f"channel_{channel.id}",
                lambda: channel.send(
                    content=content,
                    embed=embed,
                    file=file,
                    files=files,
                    reference=reference
                )
            )
        except discord.HTTPException as e:
            self.logger.error(f"HTTP error sending message: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            raise

    async def create_media_content(self, files: List[Tuple[discord.File, int, str, str]], max_media: int = 4) -> Optional[discord.File]:
        """Create a collage of images or a combined video, depending on attachments."""
        try:
            if not MEDIA_PROCESSING_AVAILABLE:
                self.logger.error("Media processing libraries are not available")
                return None
            
            self.logger.info(f"Starting media content creation with {len(files)} files")
            
            images = []
            videos = []
            has_audio = False
            
            for file_tuple, _, _, _ in files[:max_media]:
                file_tuple.fp.seek(0)
                data = file_tuple.fp.read()
                
                if file_tuple.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.logger.debug(f"Processing image: {file_tuple.filename}")
                    img = Image.open(io.BytesIO(data))
                    images.append(img)
                elif file_tuple.filename.lower().endswith(('.mp4', '.mov', '.webm')):
                    self.logger.debug(f"Processing video: {file_tuple.filename}")
                    temp_path = f'temp_{len(videos)}.mp4'
                    with open(temp_path, 'wb') as f:
                        f.write(data)
                    video = mp.VideoFileClip(temp_path)
                    if video.audio is not None:
                        has_audio = True
                        self.logger.debug(f"Video {file_tuple.filename} has audio")
                    videos.append(video)
            
            self.logger.info(f"Processed {len(images)} images and {len(videos)} videos. Has audio: {has_audio}")
                
            if videos and has_audio:
                self.logger.info("Creating combined video with audio")
                final_video = mp.concatenate_videoclips(videos)
                output_path = 'combined_video.mp4'
                final_video.write_videofile(output_path)
                
                for video in videos:
                    video.close()
                final_video.close()
                
                self.logger.info("Video combination complete")
                
                with open(output_path, 'rb') as f:
                    return discord.File(f, filename='combined_video.mp4')
                
            elif images or (videos and not has_audio):
                self.logger.info("Creating image/GIF collage")
                
                # Convert silent videos to GIF
                for i, video in enumerate(videos):
                    self.logger.debug(f"Converting silent video {i+1} to GIF")
                    gif_path = f'temp_gif_{len(images)}.gif'
                    video.write_gif(gif_path)
                    gif_img = Image.open(gif_path)
                    images.append(gif_img)
                    video.close()
                
                if not images:
                    self.logger.warning("No images available for collage")
                    return None
                
                n = len(images)
                if n == 1:
                    cols, rows = 1, 1
                elif n == 2:
                    cols, rows = 2, 1
                else:
                    cols, rows = 2, 2
                
                self.logger.debug(f"Creating {cols}x{rows} collage for {n} images")
                
                target_size = (800 // cols, 800 // rows)
                resized_images = []
                for i, img in enumerate(images):
                    self.logger.debug(f"Resizing image {i+1}/{len(images)} to {target_size}")
                    img = img.convert('RGB')
                    img.thumbnail(target_size)
                    resized_images.append(img)
                
                collage = Image.new('RGB', (800, 800))
                
                for idx, img in enumerate(resized_images):
                    x = (idx % cols) * (800 // cols)
                    y = (idx // cols) * (800 // rows)
                    collage.paste(img, (x, y))
                
                self.logger.info("Collage creation complete")
                
                buffer = io.BytesIO()
                collage.save(buffer, format='JPEG')
                buffer.seek(0)
                return discord.File(buffer, filename='collage.jpg')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating media content: {e}")
            self.logger.debug(traceback.format_exc())
            return None
        finally:
            # Cleanup
            import os
            self.logger.debug("Cleaning up temporary files")
            for f in os.listdir():
                if f.startswith('temp_'):
                    try:
                        os.remove(f)
                        self.logger.debug(f"Removed temporary file: {f}")
                    except Exception as ex:
                        self.logger.warning(f"Failed to remove temporary file {f}: {ex}")

    async def create_summary_thread(self, message, thread_name, is_top_generations=False):
        """Create a thread attached to `message`."""
        try:
            thread = await message.create_thread(
                name=thread_name,
                auto_archive_duration=1440  # 24 hours
            )
            self.logger.info(f"Created thread: {thread.name}")
            return thread
        except Exception as e:
            self.logger.error(f"Error creating thread: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def post_top_art_share(self, summary_channel):
        try:
            self.logger.info("Starting post_top_art_share")
            art_channel_id = int(os.getenv('DEV_ART_CHANNEL_ID' if self.dev_mode else 'ART_CHANNEL_ID', 0))
            if art_channel_id == 0:
                self.logger.error("Invalid art channel ID (0)")
                return
                
            self.logger.info(f"Art channel ID: {art_channel_id}")
            
            # We'll use the DB approach for both dev and prod to keep consistent
            yesterday = datetime.utcnow() - timedelta(days=1)
            query = """
                SELECT 
                    m.message_id,
                    m.content,
                    m.attachments,
                    m.jump_url,
                    COALESCE(mem.server_nick, mem.global_name, mem.username) as author_name,
                    m.reactors,
                    m.embeds,
                    m.author_id,
                    CASE 
                        WHEN m.reactors IS NULL OR m.reactors = '[]' THEN 0
                        ELSE json_array_length(m.reactors)
                    END as unique_reactor_count
                FROM messages m
                JOIN members mem ON m.author_id = mem.member_id
                WHERE m.channel_id = ?
                AND m.created_at > ?
                AND json_valid(m.attachments)
                AND m.attachments != '[]'
                ORDER BY CASE 
                    WHEN m.reactors IS NULL OR m.reactors = '[]' THEN 0
                    ELSE json_array_length(m.reactors)
                END DESC
                LIMIT 1
            """
            try:
                self.db.conn.row_factory = sqlite3.Row
                cursor = self.db.conn.cursor()
                cursor.execute(query, (art_channel_id, yesterday.isoformat()))
                top_art = cursor.fetchone()
                
                if not top_art:
                    self.logger.info("No art posts found in database for the last 24 hours.")
                    return
                    
                top_art = dict(top_art)
                attachments = json.loads(top_art['attachments'])
                if not attachments:
                    self.logger.warning("No attachments found in top art post query result.")
                    return
                
                # Just pick the first attachment
                attachment = attachments[0]
                attachment_url = attachment.get('url')
                if not attachment_url:
                    self.logger.error("No URL found in attachment for top art share.")
                    return
                
                # Check if there's a video link in the content
                has_video_link = False
                if top_art['content']:
                    has_video_link = any(x in top_art['content'].lower() for x in ['youtu.be', 'youtube.com', 'vimeo.com'])
                
                # Always use display name, never tag
                author_display = top_art['author_name']
                
                content = [
                    f"# Top Art Sharing Post by {author_display}"
                ]
                
                if top_art['content'] and top_art['content'].strip():
                    # Only add the content if it's not just a video link when we have an attachment
                    if not has_video_link or len(top_art['content'].split()) > 1:
                        content.append(f"💭 *\"{top_art['content']}\"*")
                
                # Only add the attachment URL if we don't have a video link in the content
                # or if the content has more than just the video link
                if not has_video_link or len(top_art['content'].split()) > 1:
                    content.append(attachment_url)
                
                content.append(f"🔗 Original post: {top_art['jump_url']}")
                
                formatted_content = "\n".join(content)
                await self.safe_send_message(summary_channel, formatted_content)
                self.logger.info("Posted top art share successfully")

            except Exception as e:
                self.logger.error(f"Database error in post_top_art_share: {e}")
                self.logger.debug(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"Error posting top art share: {e}")
            self.logger.debug(traceback.format_exc())

    async def post_top_x_generations(self, summary_channel, limit=5, channel_id: Optional[int] = None, ignore_message_ids: Optional[List[int]] = None):
        """
        (4) Send the top X gens post. 
        We'll just pick top `limit` video-type messages with >= 3 unique reactors in the last 24 hours,
        and post them in a thread.
        
        Args:
            summary_channel: The channel to post the summary to
            limit: Maximum number of generations to show
            channel_id: Optional specific channel to get generations from. If None, searches all channels.
            ignore_message_ids: Optional list of message IDs to exclude from the results
        
        Returns:
            The top generation if any exist.
        """
        try:
            self.logger.info("Starting post_top_x_generations")
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            # Get art sharing channel ID to exclude
            art_channel_id = int(os.getenv('DEV_ART_CHANNEL_ID' if self.dev_mode else 'ART_CHANNEL_ID', 0))
            
            # Build the channel condition
            channel_condition = ""
            query_params = []
            
            # Skip date check in dev mode
            if not self.dev_mode:
                query_params.append(yesterday.isoformat())
                date_condition = "m.created_at > ?"
            else:
                date_condition = "1=1"  # Always true condition
            
            # Get test channel IDs if in dev mode
            test_channel_ids = None
            test_data_channel_id = None
            if self.dev_mode:
                test_channels = os.getenv("TEST_DATA_CHANNEL", "")
                if not test_channels:
                    self.logger.error("TEST_DATA_CHANNEL not set")
                    return
                
                test_channel_ids = [int(cid.strip()) for cid in test_channels.split(',') if cid.strip()]
                if not test_channel_ids:
                    self.logger.error("No valid channel IDs found in TEST_DATA_CHANNEL")
                    return
                
                # Select the first test channel ID for individual summaries
                test_data_channel_id = test_channel_ids[0]
                
                # For main summary, get messages from all test channels
                self.logger.info(f"Getting messages from all test channels for main summary: {test_channel_ids}")
                channels_str = ','.join(str(c) for c in test_channel_ids)
                channel_condition = f" AND m.channel_id IN ({channels_str})"
            else:
                # In prod mode, use the provided channel or all monitored channels
                if channel_id:
                    channel_condition = "AND m.channel_id = ?"
                    query_params.append(channel_id)
                else:
                    if self.channels_to_monitor:
                        channels_str = ','.join(str(c) for c in self.channels_to_monitor)
                        channel_condition = f" AND (m.channel_id IN ({channels_str}) OR EXISTS (SELECT 1 FROM channels c2 WHERE c2.channel_id = m.channel_id AND c2.category_id IN ({channels_str})))"
            
            # Always exclude art sharing channel if it's valid
            if art_channel_id != 0:
                channel_condition += f" AND m.channel_id != {art_channel_id}"
            
            # Add message ID exclusion if provided
            ignore_condition = ""
            if ignore_message_ids and len(ignore_message_ids) > 0:
                ignore_ids_str = ','.join(str(mid) for mid in ignore_message_ids)
                ignore_condition = f" AND m.message_id NOT IN ({ignore_ids_str})"
            
            query = f"""
                WITH video_messages AS (
                    SELECT 
                        m.message_id,
                        m.channel_id,
                        m.content,
                        m.attachments,
                        m.reactors,
                        m.jump_url,
                        c.channel_name,
                        COALESCE(mem.server_nick, mem.global_name, mem.username) as author_name,
                        CASE 
                            WHEN m.reactors IS NULL OR m.reactors = '[]' THEN 0
                            ELSE json_array_length(m.reactors)
                        END as unique_reactor_count
                    FROM messages m
                    JOIN channels c ON m.channel_id = c.channel_id
                    JOIN members mem ON m.author_id = mem.member_id
                    WHERE {date_condition}
                    {channel_condition}
                    {ignore_condition}
                    AND json_valid(m.attachments)
                    AND m.attachments != '[]'
                    AND LOWER(c.channel_name) NOT LIKE '%nsfw%'
                    AND EXISTS (
                        SELECT 1
                        FROM json_each(m.attachments)
                        WHERE LOWER(json_extract(value, '$.filename')) LIKE '%.mp4'
                           OR LOWER(json_extract(value, '$.filename')) LIKE '%.mov'
                           OR LOWER(json_extract(value, '$.filename')) LIKE '%.webm'
                    )
                )
                SELECT *
                FROM video_messages
                WHERE unique_reactor_count >= 3
                ORDER BY unique_reactor_count DESC
                LIMIT {limit}
            """
            
            # Set row factory before creating cursor
            self.db.conn.row_factory = sqlite3.Row
            cursor = self.db.conn.cursor()
            cursor.execute(query, query_params)
            top_generations = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            self.db.conn.row_factory = None
            
            if not top_generations:
                self.logger.info(f"No qualifying videos found - skipping top {limit} gens post.")
                return None
            
            # Get the first generation to include in the header
            first_gen = top_generations[0]
            attachments = json.loads(first_gen['attachments'])
            
            # Get the first video attachment
            video_attachment = next(
                (a for a in attachments if any(a.get('filename', '').lower().endswith(ext) for ext in ('.mp4', '.mov', '.webm'))),
                None
            )
            if not video_attachment:
                return None
                
            desc = [
                f"## {'Top Generation' if len(top_generations) == 1 else f'Top {len(top_generations)} Generations (last 24h)'}" + (f" in #{first_gen['channel_name']}" if channel_id else "") + "\n",
                f"1. By **{first_gen['author_name']}**" + (f" in #{first_gen['channel_name']}" if not channel_id else "")
            ]
            
            if first_gen['content'] and first_gen['content'].strip():
                desc.append(f"💭 *\"{first_gen['content'][:150]}\"*")
            
            desc.append(f"🔥 {first_gen['unique_reactor_count']} unique reactions")
            desc.append(video_attachment['url'])
            desc.append(f"🔗 Original post: {first_gen['jump_url']}")
            msg_text = "\n".join(desc)
            
            # Create the header message
            header_message = await self.safe_send_message(summary_channel, msg_text)
            
            # Only create a thread if there's more than one generation
            if len(top_generations) > 1:
                thread = await self.create_summary_thread(
                    header_message,
                    f"Top Generations - {datetime.utcnow().strftime('%B %d, %Y')}"
                )
                
                if not thread:
                    self.logger.error("Failed to create thread for top generations")
                    return None
                
                # Post remaining generations in the thread
                for i, row in enumerate(top_generations[1:], 2):
                    gen = dict(row)
                    attachments = json.loads(gen['attachments'])
                    
                    # Just pick the first video attachment for demonstration
                    video_attachment = next(
                        (a for a in attachments if any(a.get('filename', '').lower().endswith(ext) for ext in ('.mp4', '.mov', '.webm'))),
                        None
                    )
                    if not video_attachment:
                        continue
                    
                    desc = [
                        f"**{i}.** By **{gen['author_name']}**" + (f" in #{gen['channel_name']}" if not channel_id else "")
                    ]
                    
                    if gen['content'] and gen['content'].strip():
                        desc.append(f"💭 *\"{gen['content'][:150]}\"*")
                    
                    desc.append(f"🔥 {gen['unique_reactor_count']} unique reactions")
                    desc.append(video_attachment['url'])
                    desc.append(f"🔗 Original post: {gen['jump_url']}")
                    msg_text = "\n".join(desc)
                    
                    await self.safe_send_message(thread, msg_text)
                    await asyncio.sleep(1)
            
            self.logger.info("Posted top X gens successfully.")
            return top_generations[0] if top_generations else None

        except Exception as e:
            self.logger.error(f"Error in post_top_x_generations: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def post_top_gens_for_channel(self, thread: discord.Thread, channel_id: int):
        """
        (5)(iv) Post the top gens from that channel that haven't yet been included,
        i.e., with over 3 reactions, in the last 24 hours.
        """
        try:
            self.logger.info(f"Posting top gens for channel {channel_id} in thread {thread.name}")
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            query = """
                SELECT 
                    m.message_id,
                    m.content,
                    m.attachments,
                    m.jump_url,
                    COALESCE(mem.server_nick, mem.global_name, mem.username) as author_name,
                    CASE 
                        WHEN m.reactors IS NULL OR m.reactors = '[]' THEN 0
                        ELSE json_array_length(m.reactors)
                    END as unique_reactor_count
                FROM messages m
                JOIN members mem ON m.author_id = mem.member_id
                JOIN channels c ON m.channel_id = c.channel_id
                WHERE m.channel_id = ?
                AND m.created_at > ?
                AND json_valid(m.attachments)
                AND m.attachments != '[]'
                AND LOWER(c.channel_name) NOT LIKE '%nsfw%'
                AND EXISTS (
                    SELECT 1
                    FROM json_each(m.attachments)
                    WHERE LOWER(json_extract(value, '$.filename')) LIKE '%.mp4'
                       OR LOWER(json_extract(value, '$.filename')) LIKE '%.mov'
                       OR LOWER(json_extract(value, '$.filename')) LIKE '%.webm'
                )
                AND (
                    CASE 
                        WHEN m.reactors IS NULL OR m.reactors = '[]' THEN 0
                        ELSE json_array_length(m.reactors)
                    END
                ) >= 3
                ORDER BY unique_reactor_count DESC
                LIMIT 5
            """
            
            # Set row factory before creating cursor
            self.db.conn.row_factory = sqlite3.Row
            cursor = self.db.conn.cursor()
            cursor.execute(query, (channel_id, yesterday.isoformat()))
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            self.db.conn.row_factory = None
            
            if not results:
                self.logger.info(f"No top generations found for channel {channel_id}")
                return

            await self.safe_send_message(thread, "\n## 🎬 Top Generations Last 24h\n")
            
            for i, row in enumerate(results, 1):
                try:
                    attachments = json.loads(row['attachments'])
                    
                    # Just pick the first video attachment
                    video_attachment = next(
                        (a for a in attachments if any(a.get('filename', '').lower().endswith(ext) for ext in ('.mp4', '.mov', '.webm'))),
                        None
                    )
                    if not video_attachment:
                        continue
                    
                    desc = [
                        f"**{i}.** By **{row['author_name']}**",
                        f"🔥 {row['unique_reactor_count']} unique reactions"
                    ]
                    
                    if row['content'] and row['content'].strip():
                        desc.append(f"💭 *\"{row['content'][:150]}\"*")
                    
                    desc.append(video_attachment['url'])
                    desc.append(f"🔗 Original post: {row['jump_url']}")
                    msg_text = "\n".join(desc)
                    
                    await self.safe_send_message(thread, msg_text)
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing generation {i}: {e}")
                    self.logger.debug(traceback.format_exc())
                    continue

            self.logger.info(f"Successfully posted top generations for channel {channel_id}")

        except Exception as e:
            self.logger.error(f"Error in post_top_gens_for_channel: {e}")
            self.logger.debug(traceback.format_exc())

        finally:
            # Reset row factory
            self.db.conn.row_factory = None

    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            self.logger.info("Starting cleanup...")
            
            # Close aiohttp session first
            if hasattr(self, 'session') and self.session and not self.session.closed:
                self.logger.info("Closing aiohttp session...")
                await self.session.close()
                await asyncio.sleep(0.5)  # Give it a moment to close cleanly
            
            # Clean up Claude client
            if hasattr(self, 'claude'):
                self.logger.info("Cleaning up Claude client...")
                self.claude = None
            
            # Close Discord client last
            if self.discord_client and not self.discord_client.is_closed():
                self.logger.info("Closing Discord client...")
                await self.discord_client.close()
                await asyncio.sleep(0.5)  # Give it a moment to close cleanly
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.debug(traceback.format_exc())
            # Still try to close Discord client even if we had an error
            if self.discord_client and not self.discord_client.is_closed():
                await self.discord_client.close()

    async def generate_summary(self):
        """
        Generate and post summaries following these steps:
        1) Post daily header to summary channel
        2) Overall summary of ALL messages (past 24h)
        3) Top Art Sharing Post
        4) Top X gens post
        5) Individual channel summaries
        """
        try:
            async with self._summary_lock:
                self.logger.info("Generating requested summary...")

                # (1) Post the exact daily header to summary channel
                summary_channel = self.get_channel(self.summary_channel_id)
                if not summary_channel:
                    self.logger.error(f"Could not find summary channel {self.summary_channel_id}")
                    return
                
                current_date = datetime.utcnow()

                # Generate the overall summary using all messages at once
                news_summarizer = NewsSummarizer(
                    dev_mode=self.dev_mode,
                    discord_client=self,
                    monitored_channels=self.channels_to_monitor
                )
                
                # Get messages from all monitored channels
                messages = news_summarizer.get_channel_messages()
                if messages:
                    self.logger.info(f"Generating overall summary for {len(messages)} messages...")
                    overall_summary = await news_summarizer.generate_news_summary(messages)
                    if overall_summary not in ["[NOTHING OF NOTE]", "[NO SIGNIFICANT NEWS]", "[NO MESSAGES TO ANALYZE]"]:
                        formatted_summary = news_summarizer.format_news_for_discord(overall_summary)
                        # Add a header before the main summary
                        self.first_message = await self.safe_send_message(summary_channel, f"\n\n# Daily Summary - {current_date.strftime('%A, %B %d, %Y')}\n\n")
                        # Post the main summary directly in the summary channel
                        self.logger.info("Posting main summary to summary channel")
                        await news_summarizer.post_to_discord(formatted_summary, summary_channel)
                    else:
                        await self.safe_send_message(summary_channel, "_No significant activity to summarize in the last 24 hours._")
                else:
                    await self.safe_send_message(summary_channel, "_No messages found in the last 24 hours for overall summary._")

                # (4) Top X gens post and get the top generation
                await self.post_top_x_generations(summary_channel, limit=4)

                # (3) Top Art Sharing Post
                await self.post_top_art_share(summary_channel)
                
                # Add link back to the start
                self.logger.info("Attempting to send link back to start...")
                if self.first_message:
                    self.logger.info(f"First message exists with ID: {self.first_message.id}")
                    link_to_start = f"https://discord.com/channels/{self.first_message.guild.id}/{self.first_message.channel.id}/{self.first_message.id}"
                    self.logger.info(f"Generated link: {link_to_start}")
                    await self.safe_send_message(summary_channel, f"\n---\n\n***Click here to jump to the beginning of today's summary:*** {link_to_start}\u200B\n")
                    self.logger.info("Sent link back to start message")
                else:
                    self.logger.warning("No first_message found, cannot send link back")
                
                # (5) Now process each channel individually
                if self.dev_mode:
                    # Get test channel IDs
                    test_channels = os.getenv("TEST_DATA_CHANNEL", "")
                    if not test_channels:
                        self.logger.error("TEST_DATA_CHANNEL not set")
                        return
                    
                    test_channel_ids = [int(cid.strip()) for cid in test_channels.split(',') if cid.strip()]
                    if not test_channel_ids:
                        self.logger.error("No valid channel IDs found in TEST_DATA_CHANNEL")
                        return
                    
                    # Select the first test channel ID for individual summaries
                    test_data_channel_id = test_channel_ids[0]
                    
                    # Already selected test_data_channel_id earlier
                    self.logger.info(f"Using test channel {test_data_channel_id} for individual channel summaries")
                    
                    # First check if selected test channel has enough messages
                    self.db.conn.row_factory = sqlite3.Row
                    cursor = self.db.conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) as msg_count
                        FROM messages
                        WHERE channel_id = ?
                        AND created_at > datetime('now', '-1 day')
                    """, (test_data_channel_id,))
                    result = cursor.fetchone()
                    msg_count = result['msg_count'] if result else 0
                    
                    if msg_count >= 25:
                        # Get messages from the selected test channel
                        test_messages = await self.get_channel_history(test_data_channel_id)
                        self.logger.info(f"Retrieved {len(test_messages)} messages from test channel {test_data_channel_id}")
                        
                        # Process each dev channel
                        for dev_channel_id in self.dev_channels_to_monitor:
                            try:
                                channel_obj = self.get_channel(dev_channel_id)
                                if not channel_obj:
                                    self.logger.error(f"Could not find channel object for {dev_channel_id}")
                                    continue
                                    
                                channel_name = channel_obj.name
                                self.logger.info(f"Processing dev channel {channel_name} ({dev_channel_id})")
                                
                                # Create/find monthly thread
                                current_date = datetime.utcnow()
                                month_year = current_date.strftime('%B %Y')
                                thread_name = f"Monthly Summary - {month_year}"
                                
                                # Search for existing thread
                                existing_thread = None
                                try:
                                    if channel_obj.guild:
                                        active_threads = await channel_obj.guild.active_threads()
                                        if active_threads:
                                            for th in active_threads:
                                                if th.parent_id == channel_obj.id and th.name == thread_name:
                                                    existing_thread = th
                                                    break
                                except Exception as e:
                                    self.logger.error(f"Error searching for existing thread in {channel_name}: {e}")
                                
                                if not existing_thread:
                                    header_message = await self.safe_send_message(
                                        channel_obj,
                                        f"# Monthly Summary Thread - {month_year}"
                                    )
                                    thread = await self.create_summary_thread(header_message, thread_name)
                                else:
                                    thread = existing_thread
                                
                                if not thread:
                                    self.logger.error(f"Failed to create/find monthly thread for {channel_name}")
                                    continue
                                
                                # Generate summaries using test data
                                channel_summary = await news_summarizer.generate_news_summary(test_messages)
                                if channel_summary in ["[NOTHING OF NOTE]", "[NO SIGNIFICANT NEWS]", "[NO MESSAGES TO ANALYZE]"]:
                                    continue
                                
                                # Get short summary
                                short_summary = await self.generate_short_summary(channel_summary, msg_count)
                                
                                # Post full summary in thread
                                formatted_channel_summary = news_summarizer.format_news_for_discord(channel_summary)
                                thread_title = f"# Summary for #{channel_name} for {current_date.strftime('%A, %B %d, %Y')}\n\n"
                                title_message = await self.safe_send_message(thread, thread_title)
                                for message in formatted_channel_summary:
                                    await news_summarizer.post_to_discord([message], thread)

                                # Post short summary in the channel with link to thread
                                link_to_thread = f"https://discord.com/channels/{thread.guild.id}/{thread.id}/{title_message.id}"
                                to_post = [
                                    f"### Channel summary for {current_date.strftime('%A, %B %d, %Y')}",
                                    short_summary,
                                    f"[Go to monthly thread for more details]({link_to_thread})"
                                ]
                                await self.safe_send_message(channel_obj, "\n".join(to_post))

                                # Post top gens in thread
                                await self.post_top_gens_for_channel(thread, test_data_channel_id)
                                
                                # Add link back to the start of today's summary in this thread
                                if title_message:
                                    link_to_start = f"https://discord.com/channels/{thread.guild.id}/{thread.id}/{title_message.id}"
                                    await self.safe_send_message(thread, f"\n---\n\n***Click here to jump to the beginning of today's summary:*** {link_to_start}\u200B\n")
                            except Exception as e:
                                self.logger.error(f"Error processing dev channel {dev_channel_id}: {e}")
                                self.logger.debug(traceback.format_exc())
                                continue
                else:
                    # Production mode - process actual channels
                    # Query to get active channels with message counts
                    channel_query = """
                        SELECT 
                            c.channel_id,
                            c.channel_name,
                            COALESCE(c2.channel_name, 'direct') as source,
                            COUNT(m.message_id) as msg_count
                        FROM channels c
                        LEFT JOIN channels c2 ON c.category_id = c2.channel_id
                        LEFT JOIN messages m ON c.channel_id = m.channel_id
                            AND m.created_at > datetime('now', '-1 day')
                        WHERE c.channel_id IN ({})
                            OR c.category_id IN ({})
                        GROUP BY c.channel_id, c.channel_name, source
                        HAVING COUNT(m.message_id) >= 25
                        ORDER BY msg_count DESC
                    """.format(
                        ','.join(str(cid) for cid in self.channels_to_monitor),
                        ','.join(str(cid) for cid in self.channels_to_monitor)
                    )
                    
                    # Set row factory before creating cursor
                    self.db.conn.row_factory = sqlite3.Row
                    cursor = self.db.conn.cursor()
                    cursor.execute(channel_query)
                    active_channels = cursor.fetchall()
                    cursor.close()
                    self.db.conn.row_factory = None
                    
                    self.logger.info("Processing individual summaries for channels with 25+ messages:")
                    for channel in active_channels:
                        try:
                            self.logger.info(f"  - {channel['channel_name']} ({channel['channel_id']}) [{channel['source']}]: {channel['msg_count']} messages")
                            
                            channel_id = channel['channel_id']
                            messages = await self.get_channel_history(channel_id)
                            
                            channel_obj = self.get_channel(channel_id)
                            if not channel_obj:
                                self.logger.error(f"Could not find channel object for {channel_id}")
                                continue
                                
                            channel_name = channel_obj.name
                                
                                # Create/find monthly thread
                            current_date = datetime.utcnow()
                            month_year = current_date.strftime('%B %Y')
                            thread_name = f"Monthly Summary - {month_year}"
                            
                            existing_thread = None
                            try:
                                if channel_obj.guild:
                                    active_threads = await channel_obj.guild.active_threads()
                                    if active_threads:
                                        for th in active_threads:
                                            if th.parent_id == channel_obj.id and th.name == thread_name:
                                                existing_thread = th
                                                break
                            except Exception as e:
                                self.logger.error(f"Error searching for existing thread: {e}")
                                
                            if not existing_thread:
                                header_message = await self.safe_send_message(
                                    channel_obj,
                                    f"# Monthly Summary Thread - {month_year}"
                                )
                                thread = await self.create_summary_thread(header_message, thread_name)
                            else:
                                thread = existing_thread

                            if not thread:
                                self.logger.error(f"Failed to create/find monthly thread for {channel_name}")
                                continue

                            # (ii) Get a summary of that channel for the past 24 hours
                            channel_summary = await news_summarizer.generate_news_summary(messages)
                            if channel_summary in ["[NOTHING OF NOTE]", "[NO SIGNIFICANT NEWS]", "[NO MESSAGES TO ANALYZE]"]:
                                continue

                            # (iii) Get a summary-of-summary (short summary)
                            short_summary = await self.generate_short_summary(channel_summary, len(messages))

                            # Format and post the full channel summary inside the thread
                            formatted_channel_summary = news_summarizer.format_news_for_discord(channel_summary)
                            
                            # Add the title and post the full formatted summary in the thread
                            thread_title = f"\n---\n\n# Summary for #{channel_name} for {current_date.strftime('%A, %B %d, %Y')}\n\n"
                            
                            # Post each message in the thread, starting with the title
                            title_message = await self.safe_send_message(thread, thread_title)
                            for message in formatted_channel_summary:
                                await news_summarizer.post_to_discord([message], thread)

                            # Then post just the short summary in the channel itself
                            link_to_thread = f"https://discord.com/channels/{thread.guild.id}/{thread.id}"
                            to_post = [
                                f"### Channel summary for {current_date.strftime('%A, %B %d, %Y')}",
                                short_summary,
                                f"[Go to monthly thread for more details]({link_to_thread})"
                            ]
                            await self.safe_send_message(channel_obj, "\n".join(to_post))

                            # (iv) Post top gens in that thread
                            await self.post_top_gens_for_channel(thread, channel_id)
                            
                            # Add link back to the start of today's summary in this thread
                            if title_message:
                                link_to_start = f"https://discord.com/channels/{thread.guild.id}/{thread.id}/{title_message.id}"
                                await self.safe_send_message(thread, f"\n---\n\n***Click here to jump to the beginning of today's summary:*** {link_to_start}")
                        except Exception as e:
                            self.logger.error(f"Error processing channel {channel_id}: {e}")
                            self.logger.debug(traceback.format_exc())
                            continue
        except Exception as e:
            self.logger.error(f"Critical error in summary generation: {e}")
            self.logger.debug(traceback.format_exc())
            raise
        finally:
            # Ensure cleanup happens after summary generation
            await self.cleanup()

if __name__ == "__main__":
    main()