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
                    raise ConfigurationError(f"Invalid channel ID in CHANNELS_TO_MONITOR: {e}")
            
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
            
            query = f"""
                SELECT 
                    m.message_id, m.channel_id, m.author_id, m.content,
                    m.created_at, m.attachments, m.embeds, m.reaction_count,
                    m.reactors, m.reference_id, m.edited_at, m.is_pinned,
                    m.thread_id, m.message_type, m.flags, m.jump_url,
                    m.indexed_at,
                    mem.username, mem.server_nick, mem.global_name
                FROM messages m
                LEFT JOIN members mem ON m.author_id = mem.member_id
                WHERE m.channel_id = ?
                {date_condition}
                ORDER BY m.created_at DESC
            """
            
            results = self.db.execute_query(query, (channel_id,))
            self.logger.debug(f"Raw query returned {len(results)} results for channel {channel_id}")
            if results:
                self.logger.debug(f"Sample row: {results[0]}")
            
            messages = []
            for row in results:
                try:
                    created_at = row[4]
                    if created_at:
                        created_at = created_at.replace('Z', '+00:00')
                        created_at = datetime.fromisoformat(created_at)
                    else:
                        created_at = datetime.utcnow()

                    edited_at = row[10]
                    if edited_at:
                        edited_at = edited_at.replace('Z', '+00:00')
                        edited_at = datetime.fromisoformat(edited_at)

                    indexed_at = row[16]
                    if indexed_at:
                        indexed_at = datetime.fromisoformat(indexed_at.replace('Z', '+00:00'))

                    attachments = []
                    if row[5]:
                        try:
                            if row[5] != '[]':
                                attachments = json.loads(row[5])
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid attachments JSON for message {row[0]}: {row[5]}")

                    embeds = []
                    if row[6]:
                        try:
                            if row[6] != '[]':
                                embeds = json.loads(row[6])
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid embeds JSON for message {row[0]}: {row[6]}")

                    reactors = []
                    if row[8]:
                        try:
                            if row[8] != '[]':
                                reactors = json.loads(row[8])
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid reactors JSON for message {row[0]}: {row[8]}")

                    author_name = row[19] or row[18] or row[17] or 'Unknown User'
                    
                    message = {
                        'message_id': row[0],
                        'channel_id': row[1],
                        'author_id': row[2],
                        'content': row[3] or '',
                        'created_at': created_at,
                        'attachments': attachments,
                        'embeds': embeds,
                        'reaction_count': row[7] or 0,
                        'reactors': reactors,
                        'reference_id': row[9],
                        'edited_at': edited_at,
                        'is_pinned': bool(row[11]),
                        'thread_id': row[12],
                        'message_type': row[13].replace('MessageType.', '') if row[13] else 'default',
                        'flags': row[14] or 0,
                        'jump_url': row[15],
                        'indexed_at': indexed_at,
                        'author_name': author_name
                    }
                    messages.append(message)
                    if len(messages) <= 2:
                        self.logger.debug(f"Successfully processed message: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message {row[0]}: {e}")
                    self.logger.debug(f"Problematic row: {row}")
                    self.logger.debug(traceback.format_exc())
                    continue
            
            self.logger.info(f"Retrieved {len(messages)} messages from database for channel {channel_id}")
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
        """
        (3) Send the top Art Sharing Post from the last 24 hours.
        In dev mode, this uses the DB for the art sharing channel,
        in prod mode, tries to get it live from Discord or from DB (depending on environment).
        """
        try:
            self.logger.info("Starting post_top_art_share")
            art_channel_id = int(os.getenv('DEV_ART_SHARING_CHANNEL_ID' if self.dev_mode else 'ART_SHARING_CHANNEL_ID', 0))
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
                
                # In production, use a mention. In dev/staging, just use the name
                author_display = f"<@{top_art['author_id']}>" if not self.dev_mode else top_art['author_name']
                
                content = [
                    f"# Top Art Sharing Post by {author_display}"
                ]
                
                if top_art['content'] and top_art['content'].strip():
                    # Only add the content if it's not just a video link when we have an attachment
                    if not has_video_link or len(top_art['content'].split()) > 1:
                        content.append(f"💭 *\"{top_art['content']}\"*")
                
                content.append(f"🔥 Reactions: {top_art['unique_reactor_count']}")
                
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
            finally:
                self.db.conn.row_factory = None

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
            art_channel_id = int(os.getenv('DEV_ART_SHARING_CHANNEL_ID' if self.dev_mode else 'ART_SHARING_CHANNEL_ID', 0))
            
            # Build the channel condition
            channel_condition = ""
            query_params = []
            
            # Skip date check in dev mode
            if not self.dev_mode:
                query_params.append(yesterday.isoformat())
                date_condition = "m.created_at > ?"
            else:
                date_condition = "1=1"  # Always true condition
            
            if self.dev_mode:
                # In dev mode, always use test_data_channel unless a specific channel is provided
                test_data_channel_id = int(os.getenv("TEST_DATA_CHANNEL", "0"))
                if test_data_channel_id == 0:
                    self.logger.warning("TEST_DATA_CHANNEL not set or invalid. Skipping top generations.")
                    return None
                if channel_id:
                    channel_condition = "AND m.channel_id = ?"
                    query_params.append(channel_id)
                else:
                    channel_condition = f"AND m.channel_id = {test_data_channel_id}"
            else:
                # In prod mode, use the provided channel or all monitored channels
                if channel_id:
                    channel_condition = "AND m.channel_id = ?"
                    query_params.append(channel_id)
                else:
                    if self.channels_to_monitor:
                        channels_str = ','.join(str(c) for c in self.channels_to_monitor)
                        channel_condition = f" AND m.channel_id IN ({channels_str})"
            
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
            
            self.db.conn.row_factory = sqlite3.Row
            cursor = self.db.conn.cursor()
            cursor.execute(query, query_params)
            top_generations = cursor.fetchall()
            cursor.close()
            self.db.conn.row_factory = None
            
            if not top_generations:
                self.logger.info(f"No qualifying videos found - skipping top {limit} gens post.")
                return None
            
            # Get the first generation to include in the header
            first_gen = dict(top_generations[0])
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
        (For simplicity, we won't store which were 'already included' in the DB in this example;
         you can adapt that logic if needed.)
        """
        try:
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
            """
            
            results = self.db.execute_query(query, (channel_id, yesterday.isoformat()))
            
            if not results:
                return

            await self.safe_send_message(thread, "\n## 🎬 Top Generations Last 24h\n")
            
            for row in results:
                try:
                    message_id, content, attachments_json, jump_url, author_name, reaction_count = row
                    attachments = json.loads(attachments_json)
                    
                    # Just pick the first video attachment
                    video_attachment = next(
                        (a for a in attachments if any(a.get('filename', '').lower().endswith(ext) for ext in ('.mp4', '.mov', '.webm'))),
                        None
                    )
                    if not video_attachment:
                        continue
                    
                    desc = [
                        f"By **{author_name}**",
                        f"🔥 {reaction_count} unique reactions"
                    ]
                    
                    if content and content.strip():
                        desc.append(f"💭 Message text: `{content[:100]}...`")
                    
                    desc.append(f"🔗 {jump_url}")
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(video_attachment['url']) as response:
                            if response.status == 200:
                                file_data = await response.read()
                                file = discord.File(
                                    io.BytesIO(file_data),
                                    filename=video_attachment['filename']
                                )
                                await self.safe_send_message(thread, '\n'.join(desc), file=file)
                                await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"Error posting generation for channel {channel_id}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in post_top_gens_for_channel: {e}")
            self.logger.debug(traceback.format_exc())

    ###########################################################################
    # The updated generate_summary flow per the request
    ###########################################################################
    async def generate_summary(self):
        """
        Please do the exact steps below (returning all code):
        
        1) Post "Daily Summary - Tuesday, January 21, 2025" to the summary channel.
        2) Send an overall summary of ALL messages over the past 24 hours to the summary channel using NewsSummarizer.
        3) Send the top Art Sharing Post (from the last 24 hours).
        4) Send the top X gens post (with limit of your choice).
        
        5) For each channel in CHANNELS_TO_MONITOR (or DEV_CHANNELS_TO_MONITOR in dev) that has over 25 messages in the last 24 hours:
           (i) Start or continue a monthly thread
           (ii) Get a summary of that channel for the past 24 hours 
               - In production, use the actual channel
               - In dev mode, use TEST_DATA_CHANNEL for the actual data (but post results in the dev channel's monthly thread)
           (iii) Get a summary of that summary using create_short_summary
           (iii) Post the summary in the summary channel, linking to the monthly thread
           (iv) Post the top gens (3+ reactions) for that channel that haven't yet been included, at the end of the monthly thread one-by-one.
        """
        try:
            async with self._summary_lock:
                self.logger.info("Generating requested summary...")

                # (1) Post the exact daily header to summary channel
                summary_channel = self.get_channel(self.summary_channel_id)
                if not summary_channel:
                    self.logger.error(f"Could not find summary channel {self.summary_channel_id}")
                    return
                
                daily_header = "# Daily Summary - Tuesday, January 21, 2025"
                first_message = await self.safe_send_message(summary_channel, daily_header)

                # Decide which channels to check for data
                # In dev mode, we'll gather messages from the TEST_DATA_CHANNEL for summarizing,
                # but we will still *post* to the dev channels' threads if they pass the threshold.
                if self.dev_mode:
                    test_data_channel_id = int(os.getenv("TEST_DATA_CHANNEL", "0"))
                    channels_to_use = self.dev_channels_to_monitor  # We'll iterate over dev channels for posting
                    # We'll read data from test_data_channel_id
                else:
                    channels_to_use = self.channels_to_monitor

                # (2) Overall summary of ALL messages from the past 24h
                #    In prod: gather from all channels_to_monitor.
                #    In dev: gather from test_data_channel only (once).
                all_messages = []
                if self.dev_mode:
                    if test_data_channel_id == 0:
                        self.logger.warning("TEST_DATA_CHANNEL not set or invalid. Skipping overall summary.")
                    else:
                        # Grab from that single channel
                        all_messages = await self.get_channel_history(test_data_channel_id)
                else:
                    for cid in channels_to_use:
                        msgs = await self.get_channel_history(cid)
                        all_messages.extend(msgs)

                news_summarizer = NewsSummarizer(dev_mode=self.dev_mode, discord_client=self)
                if all_messages:
                    overall_summary = await news_summarizer.generate_news_summary(all_messages)
                    if overall_summary not in ["[NOTHING OF NOTE]", "[NO SIGNIFICANT NEWS]", "[NO MESSAGES TO ANALYZE]"]:
                        formatted_summary = news_summarizer.format_news_for_discord(overall_summary)
                        await news_summarizer.post_to_discord(formatted_summary, summary_channel)
                else:
                    await self.safe_send_message(summary_channel, "_No messages found in the last 24 hours for overall summary._")

                # (4) Top X gens post and get the top generation
                await self.post_top_x_generations(summary_channel, limit=4)

                # (3) Top Art Sharing Post
                await self.post_top_art_share(summary_channel)
                
                # (5) For each channel that has > 25 messages in last 24h:
                #     i) monthly thread
                #     ii) channel summary
                #     iii) short summary
                #     iv) post short summary in summary channel, linking to thread
                #     v) post top gens in that thread
                for channel_id in channels_to_use:
                    # Gather messages for this channel (or from the test_data channel if dev_mode).
                    if self.dev_mode:
                        if test_data_channel_id == 0:
                            continue
                        messages = await self.get_channel_history(test_data_channel_id)
                    else:
                        messages = await self.get_channel_history(channel_id)

                    # Filter only messages actually from the last 24 hours
                    # (get_channel_history already does a 24h filter, but let's be sure).
                    if len(messages) < 25:
                        self.logger.info(f"Channel {channel_id} has fewer than 25 messages - skipping.")
                        continue
                    
                    # We proceed since it has 25+ messages
                    channel_obj = self.get_channel(channel_id)
                    if not channel_obj:
                        self.logger.error(f"Could not find channel object for {channel_id}")
                        continue
                        
                    channel_name = channel_obj.name
                    
                    # (i) Start or continue monthly thread in the channel itself
                    current_date = datetime.utcnow()
                    month_year = current_date.strftime('%B %Y')
                    thread_name = f"Monthly Summary - {month_year}"
                    
                    # Search existing threads in the channel
                    existing_thread = None
                    try:
                        if channel_obj.guild:
                            active_threads = await channel_obj.guild.active_threads()
                            if active_threads:
                                for th in active_threads:
                                    if th.parent_id == channel_obj.id and th.name == thread_name:
                                        existing_thread = th
                                        break
                        else:
                            self.logger.warning(f"Could not access guild for channel {channel_name}")
                    except Exception as e:
                        self.logger.error(f"Error searching for existing thread: {e}")
                    
                    if not existing_thread:
                        # Create a "header message" in the channel that we'll attach the thread to
                        header_message = await self.safe_send_message(
                            channel_obj,
                            f"# Monthly Summary Thread - {month_year}"
                        )
                        thread = await self.create_summary_thread(header_message, thread_name)
                    else:
                        thread = existing_thread

                    if not thread:
                        self.logger.error(f"Failed to create/find a monthly thread for {channel_name}")
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
                    thread_title = f"# Summary for #{channel_name} for {current_date.strftime('%A, %B %d, %Y')}\n\n"
                    
                    # Post each message in the thread, starting with the title
                    await self.safe_send_message(thread, thread_title)
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

                # Add link back to the start of today's summary
                if first_message:
                    link_to_start = f"https://discord.com/channels/{first_message.guild.id}/{first_message.channel.id}/{first_message.id}"
                    await self.safe_send_message(summary_channel, f"\n***Click here to jump to the beginning of today's summary: {link_to_start}***")

        except Exception as e:
            self.logger.error(f"Critical error in summary generation: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    async def close(self):
        """Override close to handle cleanup."""
        try:
            self.logger.info("Starting bot cleanup...")
            if hasattr(self, 'session') and self.session and not self.session.closed:
                self.logger.info("Closing aiohttp session...")
                await self.session.close()
            if hasattr(self, 'claude'):
                self.logger.info("Cleaning up Claude client...")
                self.claude = None
            
            self.logger.info("Cleanup completed, calling parent close...")
            await super().close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.debug(traceback.format_exc())
            await super().close()

def main():
    parser = argparse.ArgumentParser(description='Discord Channel Summarizer Bot')
    parser.add_argument('--summary-now', action='store_true', help='Run the summary process immediately')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    bot = ChannelSummarizer()
    bot.dev_mode = args.dev
    
    bot.load_config()
    
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        raise ValueError("Discord bot token not found in environment variables")
    
    if args.dev:
        bot.logger.info("Running in DEVELOPMENT mode")
    
    loop = asyncio.get_event_loop()
    
    @bot.event
    async def on_ready():
        bot.logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
        bot.logger.info("Connected to servers: %s", [guild.name for guild in bot.guilds])
        
        if args.summary_now:
            bot.logger.info("Running summary process immediately...")
            try:
                await bot.generate_summary()
                bot.logger.info("Summary process completed. Shutting down...")
            finally:
                bot._shutdown_flag = True
                await bot.close()
        else:
            if not hasattr(bot, '_scheduler_started'):
                bot._scheduler_started = True
                bot.logger.info("Starting scheduled mode - will run daily at 10:00 UTC")
                loop.create_task(schedule_daily_summary(bot))
    
    try:
        loop.run_until_complete(bot.start(bot_token))
    except KeyboardInterrupt:
        bot.logger.info("Keyboard interrupt received - shutting down...")
    finally:
        try:
            loop.run_until_complete(asyncio.sleep(1))
            tasks = [t for t in asyncio.all_tasks(loop) 
                     if not t.done() and t != asyncio.current_task(loop)]
            if tasks:
                bot.logger.info(f"Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            if not loop.is_closed():
                bot.logger.info("Closing bot connection...")
                loop.run_until_complete(bot.close())
        except Exception as e:
            bot.logger.error(f"Error during shutdown: {e}")
            bot.logger.debug(traceback.format_exc())
        finally:
            if not loop.is_closed():
                bot.logger.info("Closing event loop...")
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            
        if getattr(bot, '_shutdown_flag', False):
            sys.exit(0)

if __name__ == "__main__":
    main()