import discord
import anthropic
from datetime import datetime, timedelta, timezone
import asyncio
import os
from discord.ext import commands
from dotenv import load_dotenv
import io
import aiohttp
import argparse
import re
import logging
import traceback
import random
from typing import List, Tuple, Set, Dict, Optional, Any, Union
from dataclasses import dataclass
from src.db_handler import DatabaseHandler
from utils.errors import *
from utils.error_handler import ErrorHandler, handle_errors
import json
from logging.handlers import RotatingFileHandler
from utils.log_handler import LogHandler
import sys
from utils.rate_limiter import RateLimiter
import copy

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
            self.logger.debug(f"Processing attachment for cache key: {cache_key}")
            
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
                self.logger.debug(f"Attachment {attachment.filename} has {total_reactions} reactions")

                processed_attachment = Attachment(
                    filename=attachment.filename,
                    data=file_data,
                    content_type=attachment.content_type,
                    reaction_count=total_reactions,
                    username=message.author.name,
                    content=message.content,
                    jump_url=jump_url  # Use the correct jump URL
                )

                # Ensure the cache key structure is consistent
                if cache_key not in self.attachment_cache:
                    self.attachment_cache[cache_key] = {
                        'attachments': [],
                        'reaction_count': total_reactions,
                        'username': message.author.name,
                        'channel_id': str(message.channel.id)  # Ensure channel_id is stored as string
                    }
                self.attachment_cache[cache_key]['attachments'].append(processed_attachment)
                self.logger.debug(f"Successfully cached attachment {attachment.filename} for key {cache_key}")

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
        
        Returns:
            List[Attachment]: Sorted list of Attachment objects.
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
            if (any(line.startswith(emoji) for emoji in ['🎥', '💻', '🎬', '🤖', '📱', '�������', '🔧', '🎨', '📊']) and 
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
    def __init__(self):
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
        self.logger = None
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
        self.db = DatabaseHandler()
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
        
    def setup_logger(self, dev_mode):
        """Initialize or update logger configuration"""
        self.logger = self.log_handler.setup_logging(dev_mode)
        
        # Now that logger is set up, we can log initialization info
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
            self.setup_logger(value)  # Reconfigure logger with new mode

    def load_config(self):
        """Load configuration based on mode"""
        self.logger.debug("Loading configuration...")
        self.logger.debug(f"Current TEST_DATA_CHANNEL: {os.getenv('TEST_DATA_CHANNEL')}")
        
        # Force reload the .env file
        load_dotenv(override=True)
        self.logger.debug(f"After reload TEST_DATA_CHANNEL: {os.getenv('TEST_DATA_CHANNEL')}")
        
        # Log all channel-related env vars
        self.logger.debug("All channel-related environment variables:")
        for key, value in os.environ.items():
            if 'CHANNEL' in key:
                self.logger.debug(f"{key}: {value}")
        
        try:
            if self.dev_mode:
                self.logger.info("Loading development configuration")  # Changed
                self.guild_id = int(os.getenv('DEV_GUILD_ID'))
                self.summary_channel_id = int(os.getenv('DEV_SUMMARY_CHANNEL_ID'))
                channels_str = os.getenv('DEV_CHANNELS_TO_MONITOR')
                if not channels_str:
                    raise ConfigurationError("DEV_CHANNELS_TO_MONITOR not found in environment")
                self.dev_channels_to_monitor = [chan.strip() for chan in channels_str.split(',')]
                self.logger.info(f"DEV_CHANNELS_TO_MONITOR: {self.dev_channels_to_monitor}")  # Changed
            else:
                self.logger.info("Loading production configuration")  # Changed
                self.guild_id = int(os.getenv('GUILD_ID'))
                self.summary_channel_id = int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))
                channels_str = os.getenv('CHANNELS_TO_MONITOR')
                if not channels_str:
                    raise ConfigurationError("CHANNELS_TO_MONITOR not found in environment")
                self.channels_to_monitor = [int(chan.strip()) for chan in channels_str.split(',')]
                self.logger.info(f"CHANNELS_TO_MONITOR: {self.channels_to_monitor}")  # Changed
            
            self.logger.info(f"Configured with guild_id={self.guild_id}, "  # Changed
                        f"summary_channel={self.summary_channel_id}, "
                        f"channels={self.channels_to_monitor if not self.dev_mode else self.dev_channels_to_monitor}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")  # Changed
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from test.json"""
        try:
            # Use an absolute path relative to the script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_data_path = os.path.join(script_dir, 'test.json')
            
            if not os.path.exists(test_data_path):
                # Create default test data if file doesn't exist
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
        """This is called when the bot is starting up"""
        try:
            self.session = aiohttp.ClientSession()
            
            notification_channel = self.get_channel(self.summary_channel_id)
            admin_user = await self.fetch_user(int(os.getenv('ADMIN_USER_ID')))
            self.error_handler = ErrorHandler(notification_channel, admin_user)
            
        except Exception as e:
            raise ConfigurationError("Failed to initialize bot", e)

    async def get_channel_history(self, channel_id: int) -> List[Union[discord.Message, dict]]:
        """Get message history for a channel, using test data channel in dev mode"""
        if self.dev_mode:
            test_channel_id = int(os.getenv('TEST_DATA_CHANNEL'))
            self.logger.debug(f"TEST_DATA_CHANNEL: {test_channel_id}")
            
            # Get the actual test channel
            test_channel = self.get_channel(test_channel_id)
            if not test_channel:
                self.logger.error(f"Could not find test channel {test_channel_id}")
                return []
            
            # Get real messages from the test channel
            messages = []
            async for message in test_channel.history(
                limit=100,
                after=datetime.utcnow() - timedelta(days=1)
            ):
                # Create a copy of the message with modified channel but preserve original URL
                message_copy = copy.copy(message)
                original_jump_url = message.jump_url  # Store original URL
                message_copy.channel = self.get_channel(channel_id)  # Use the target channel
                
                # Store both original URL and channel ID
                self.original_urls[message_copy.id] = {
                    'url': original_jump_url,
                    'channel_id': str(channel_id)  # Use target channel ID instead of original
                }
                
                messages.append(message_copy)
                
                # Process attachments with target channel info
                if message.attachments:
                    async with aiohttp.ClientSession() as session:
                        for attachment in message.attachments:
                            # Use target channel ID for cache key
                            processed = await self.attachment_handler.process_attachment(
                                attachment, 
                                message_copy,  # Use modified message with target channel
                                session,
                                original_jump_url=original_jump_url
                            )
            
            self.logger.info(f"Loaded {len(messages)} messages from test data channel")
            return messages
        
        else:
            # Production mode remains unchanged
            channel = self.get_channel(channel_id)
            if not channel:
                raise DiscordError(f"Could not access channel {channel_id}")
            
            messages = []
            async for message in channel.history(
                limit=100,
                after=datetime.utcnow() - timedelta(days=1)
            ):
                messages.append(message)
            
            return messages

    def get_original_url(self, message_id: int) -> str:
        """Get the original URL for a message ID"""
        return self.original_urls.get(message_id, "")  # Return empty string if not found

    @handle_errors("get_claude_summary")
    async def get_claude_summary(self, messages):
        """
        Generate summary using Claude API with comprehensive error handling.
        """
        if not messages:
            self.logger.info("No messages to summarize")
            return "[NOTHING OF NOTE]"
        
        self.logger.info(f"Generating summary for {len(messages)} messages")
        
        try:
            # Build the conversation prompt
            # Build the conversation prompt
            conversation = """Please summarize the interesting and noteworthy Discord happenings and ideas in bullet points. ALWAYS include Discord links and external links. You should extract ideas and information that may be useful to others from conversations. Avoid stuff like bug reports that are circumstantial or not useful to others. Break them into topics and sub-topics.

If there's nothing significant or noteworthy in the messages - if it's just casual discussion with no meaningful news just respond with exactly "[NOTHING OF NOTE]" (and no other text). Always include external links and Discord links wherever possible.

Requirements:
1. Make sure to ALWAYS include Discord links and external links 
2. Use Discord's markdown format (not regular markdown)
3. Use - for top-level points (no bullet for the header itself). Only use - for clear sub-points that directly relate to the point above. You should generally just create a new point for each new topic.
4. Make each main topic a ## header with an emoji
5. Use ** for bold text (especially for usernames and main topics)
6. Keep it simple - just bullet points and sub-points for each topic, no headers or complex formatting
7. ALWAYS include the message author's name in bold (**username**) for each point if there's a specific person who did something, said something important, or seemed to be helpful - mention their username, don't tag them. Call them "Banodocians" instead of "users".
8. Always include a funny or relevant emoji in the topic title

Here's one example of what a good summary and topic should look like:

## 🤏 **H264/H265 Compression Techniques for Video Generation Improves Img2Vid**
- **zeevfarbman** recommended h265 compression for frame degradation with less perceptual impact: https://discord.com/channels/1076117621407223829/1309520535012638740/1316462339318354011
- **johndopamine** suggested using h264 node in MTB node pack for better video generation: https://discord.com/channels/564534534/1316786801247260672
- Codec compression can help "trick" current workflows/models: https://github.com/tdrussell/codex-pipe
- melmass confirmed adding h265 support to their tools: https://discord.com/channels/1076117621407223829/1309520535012638740/1316786801247260672

And here's another example of a good summary and topic:

## 🏋 **Person Training for Hunyuan Video is Now Possible**    
- **Kytra** explained that training can be done on relatively modest hardware (24GB VRAM): https://discord.com/channels/1076117621407223829/1316079815446757396/1316418253102383135
- **TDRussell** provided the repository link: https://github.com/tdrussell/diffusion-pipe
- Banodocians are generally experimenting with training LoRAs using images and videos

While here are bad topics to cover:
- Bug reports that seem unremarkable and not useful to others
- Messages that are not helpful or informative to others
- Discussions that ultimately have no real value to others
- Humourous messages that are not helpful or informative to others
- Personal messages, commentary or information that are not likely to be helpful or informative to others

Remember:

1. You MUST ALWAYS include relevant Discord links and external links
2. Only include topics that are likely to be interesting and noteworthy to others
3. You MUST ALWAYS include the message author's name in bold (**username**) for each point if there's a specific person who did something, said something important, or seemed to be helpful - mention their username, don't tag them. Call them "Banodocians" instead of "users".
4. You MUST ALWAYS use Discord's markdown format (not regular markdown)
5. Keep most information at the top bullet level. Only use sub-points for direct supporting details
6. Make topics clear headers with ##
7. Remember: if there's nothing of note, just respond with "[NOTHING OF NOTE]" (and no other text).

IMPORTANT: For each bullet point, use the EXACT message URL provided in the data - do not write <message_url> but instead use the actual URL from the message data.

Please provide the summary now - don't include any other introductory text, ending text, or explanation of the summary:\n\n"""

            for msg in messages:
                timestamp = msg.created_at
                author = msg.author.name
                content = msg.content
                reactions = sum(reaction.count for reaction in msg.reactions) if msg.reactions else 0
                
                # Use original URL in dev mode, otherwise use current message URL
                if self.dev_mode:
                    jump_url = self.original_urls.get(msg.id, msg.jump_url)
                else:
                    jump_url = msg.jump_url

                conversation += f"{timestamp} - {author}: {content}"
                if reactions:
                    conversation += f"\nReactions: {reactions}"
                conversation += f"\nDiscord link: {jump_url}\n\n"

            loop = asyncio.get_running_loop()
            
            # Define a helper function to call the synchronous Claude API method
            def create_summary():
                return self.claude.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=8192,
                    messages=[
                        {
                            "role": "user",
                            "content": conversation
                        }
                    ],
                    timeout=120  # 120-second timeout
                )
            
            # Run the synchronous create_summary in a separate thread to avoid blocking
            response = await loop.run_in_executor(None, create_summary)
            
            self.logger.debug(f"Response type: {type(response)}")
            self.logger.debug(f"Response content: {response.content}")
            
            # Ensure the response has the expected structure
            if not hasattr(response, 'content') or not response.content:
                raise ValueError("Invalid response format from Claude API.")

            summary_text = response.content[0].text.strip()
            self.logger.info("Summary generated successfully")
            return summary_text
                
        except asyncio.CancelledError:
            self.logger.info("Summary generation cancelled - shutting down gracefully")
            raise
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt detected - shutting down gracefully")
            raise
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            self.logger.debug(traceback.format_exc())
            raise

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

    async def send_initial_message(self, channel, short_summary: str, source_channel_name: str) -> Tuple[discord.Message, Optional[str]]:
        """
        Send the initial summary message to the channel with top reacted media if available.
        Returns tuple of (message, used_message_id if media was attached, else None)
        """
        try:
            source_channel = discord.utils.get(self.get_all_channels(), name=source_channel_name.strip('#'))
            
            message_content = (f"## <#{source_channel.id}>\n{short_summary}" 
                             if source_channel else f"## {source_channel_name}\n{short_summary}")
            
            # Find the most reacted media file
            top_media = None
            top_reactions = 2  # Minimum reaction threshold
            top_message_id = None
            
            for message_id, cache_data in self.attachment_handler.attachment_cache.items():
                if cache_data['reaction_count'] > top_reactions:
                    for attachment in cache_data['attachments']:
                        if any(attachment.filename.lower().endswith(ext) 
                              for ext in ('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.mov', '.webm')):
                            top_media = attachment
                            top_reactions = cache_data['reaction_count']
                            top_message_id = message_id
            
            # If we found popular media, attach it to the message
            initial_message = None
            if top_media:
                file = discord.File(
                    io.BytesIO(top_media.data),
                    filename=top_media.filename,
                    description=f"Most popular media (🔥 {top_reactions} reactions)"
                )
                initial_message = await self.safe_send_message(channel, message_content, file=file)
            else:
                initial_message = await self.safe_send_message(channel, message_content)
            
            self.logger.info(f"Initial message sent successfully: {initial_message.id if initial_message else 'None'}")
            return initial_message, top_message_id
            
        except Exception as e:
            self.logger.error(f"Failed to send initial message: {e}")
            self.logger.debug(traceback.format_exc())
            raise DiscordError(f"Failed to send initial message: {e}")

    async def create_summary_thread(self, message, thread_name, is_top_generations=False):
        """Create a thread for a summary message."""
        try:
            self.logger.info(f"Creating thread for message {message.id}, attempt 1")
            
            # Format thread name based on type
            if is_top_generations:
                # Keep original name for top generations
                thread_name_with_prefix = thread_name
            else:
                current_date = datetime.utcnow().strftime('%B %d, %Y')
                
                # For source/monthly thread, use month format
                if thread_name.startswith('#'):
                    thread_name_with_prefix = f"{datetime.utcnow().strftime('%b')} - {thread_name} Summary"
                else:
                    # For main summary thread, include channel name and date
                    thread_name_with_prefix = f"Summary for {thread_name} - {current_date}"
            
            thread = await message.create_thread(
                name=thread_name_with_prefix,
                auto_archive_duration=1440  # 24 hours
            )
            
            self.logger.info(f"Thread created successfully: {message.id} with name '{thread_name}'")
            return thread
                
        except Exception as e:
            self.logger.error(f"Error creating thread: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    async def prepare_topic_files(self, topic: str) -> List[Tuple[discord.File, int, str, str]]:
        """
        Prepare files for a given topic.
        
        Args:
            topic: The topic text to process
            
        Returns:
            List of tuples containing (discord.File, reaction_count, message_id, username)
        """
        topic_files = []
        message_links = re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', topic)
        
        for message_id in message_links:
            if message_id in self.attachment_handler.attachment_cache:
                for attachment in self.attachment_handler.attachment_cache[message_id]['attachments']:
                    try:
                        if len(attachment.data) <= 25 * 1024 * 1024:  # 25MB limit
                            file = discord.File(
                                io.BytesIO(attachment.data),
                                filename=attachment.filename,
                                description=f"From message ID: {message_id} (🔥 {self.attachment_handler.attachment_cache[message_id]['reaction_count']} reactions)"
                            )
                            topic_files.append((
                                file,
                                self.attachment_handler.attachment_cache[message_id]['reaction_count'],
                                message_id,
                                self.attachment_handler.attachment_cache[message_id].get('username', 'Unknown')  # Add username
                            ))
                    except Exception as e:
                        self.logger.error(f"Failed to prepare file {attachment.filename}: {e}")
                        continue
                        
        return sorted(topic_files, key=lambda x: x[1], reverse=True)[:10]  # Limit to top 10 files

    async def send_topic_chunk(self, thread: discord.Thread, chunk: str, files: List[Tuple[discord.File, int, str, str]] = None):
        try:
            if files:
                # Get list of usernames in order of their attachments
                usernames = [f"**{file_tuple[3]}**" for file_tuple in files]  # Access username from tuple
                unique_usernames = []
                [unique_usernames.append(x) for x in usernames if x not in unique_usernames]
                
                # Format usernames with 'and' and ':'
                if len(unique_usernames) > 1:
                    formatted_usernames = f"{', '.join(unique_usernames[:-1])} and {unique_usernames[-1]}"
                else:
                    formatted_usernames = unique_usernames[0]
                
                # Only add the header if this chunk has files
                if "Related attachments" not in chunk:
                    chunk = f"{chunk}\nRelated attachments from {formatted_usernames} / "
                
                await self.safe_send_message(thread, chunk, files=[file_tuple[0] for file_tuple in files])
            else:
                await self.safe_send_message(thread, chunk)
        except Exception as e:
            self.logger.error(f"Failed to send topic chunk: {e}")
            self.logger.debug(traceback.format_exc())

    async def process_topic(self, thread: discord.Thread, topic: str, is_first: bool = False, is_source_thread: bool = False) -> Tuple[Set[str], List[str]]:
        """Process and send a single topic to the thread."""
        try:
            if not topic.strip():
                return set(), []
                
            # Format topic header - remove the initial separator
            formatted_topic = topic
            self.logger.info(f"Processing topic: {formatted_topic[:100]}...")

            # Split into header and bullet points
            lines = topic.split('\n')
            header = []
            current_bullet = []
            bullets = []
            in_header = True
            chunk_message_ids = set()
            all_files = set()  # Use set to track used filenames

            # First pass - organize into header and bullets
            for line in lines:
                if line.strip().startswith('-'):
                    if in_header:
                        in_header = False
                        # Add horizontal line with zero-width space if not first topic
                        header = '\n'.join(header) if is_first else "---\n\u200B\n" + '\n'.join(header)
                    if current_bullet:
                        bullets.append(current_bullet)
                    current_bullet = [line]
                else:
                    if in_header:
                        header.append(line)
                    elif current_bullet:
                        current_bullet.append(line)

            if current_bullet:
                bullets.append(current_bullet)

            # Send header first
            if header:
                await self.safe_send_message(thread, header)
                await asyncio.sleep(1)

            # Process each bullet point
            for bullet in bullets:
                try:
                    bullet_text = '\n'.join(bullet)
                    bullet_files = []
                    
                    # Find message IDs in this bullet point - simplified regex
                    message_ids = re.findall(r'discord\.com/channels/\d+/\d+/(\d+)', bullet_text)
                    
                    if message_ids:  # Only process if we have message links
                        self.logger.debug(f"Found message IDs in bullet: {message_ids}")
                        
                        # Get attachments for all message IDs in this bullet
                        for message_id in message_ids:
                            chunk_message_ids.add(message_id)
                            
                            # Search all cache keys for this message ID
                            for cache_key in self.attachment_handler.attachment_cache:
                                if cache_key.endswith(f":{message_id}"):
                                    cache_data = self.attachment_handler.attachment_cache[cache_key]
                                    self.logger.debug(f"Found cache data for message {message_id} with key {cache_key}: {cache_data}")
                                    
                                    for attachment in cache_data['attachments']:
                                        try:
                                            # Only skip if we've already used this exact file
                                            if attachment.filename in all_files:
                                                self.logger.debug(f"Skipping duplicate file: {attachment.filename}")
                                                continue
                                            
                                            file = discord.File(
                                                io.BytesIO(attachment.data),
                                                filename=attachment.filename,
                                                description=f"From message ID: {message_id} (🔥 {cache_data['reaction_count']} reactions)"
                                            )
                                            file_tuple = (
                                                file,
                                                cache_data['reaction_count'],
                                                message_id,
                                                cache_data['username']
                                            )
                                            bullet_files.append(file_tuple)
                                            all_files.add(attachment.filename)
                                            self.logger.debug(f"Added file {attachment.filename} to bullet point")
                                        except Exception as e:
                                            self.logger.error(f"Failed to prepare file: {e}")
                                            continue

                    # Send bullet point with its files
                    if bullet_files:
                        # Sort files by reaction count but include all of them
                        bullet_files.sort(key=lambda x: x[1], reverse=True)
                        
                        # Format usernames for files
                        usernames = [f"**{f[3]}**" for f in bullet_files]
                        unique_usernames = []
                        [unique_usernames.append(x) for x in usernames if x not in unique_usernames]
                        
                        if len(unique_usernames) > 1:
                            formatted_usernames = f"{', '.join(unique_usernames[:-1])} and {unique_usernames[-1]}"
                        else:
                            formatted_usernames = unique_usernames[0]
                                                                        
                        # Log what we're about to send
                        self.logger.debug(f"Sending bullet point with {len(bullet_files)} files")
                        for f in bullet_files:
                            self.logger.debug(f"- File: {f[0].filename} from {f[3]} with {f[1]} reactions")
                        
                        await self.safe_send_message(
                            thread,
                            bullet_text,
                            files=[f[0] for f in bullet_files]
                        )
                    else:
                        await self.safe_send_message(thread, bullet_text)
                
                except Exception as e:
                    self.logger.error(f"Failed to send bullet point: {e}")
                    self.logger.debug(traceback.format_exc())
                    continue
                
                await asyncio.sleep(1)  # Prevent rate limiting

            return chunk_message_ids, list(all_files)
            
        except Exception as e:
            self.logger.error(f"Error processing topic: {e}")
            self.logger.debug(traceback.format_exc())
            return set(), []

    async def process_unused_attachments(self, thread, used_message_ids: Set[str], max_attachments: int = 10, 
                                       previous_thread_id: Optional[str] = None, used_files: List[discord.File] = None,
                                       is_source_thread: bool = False):
        """Process and post any unused attachments from the channel."""
        try:
            # Track filenames that have been used
            used_filenames = set()
            if used_files:
                for file in used_files:
                    if isinstance(file, str):
                        used_filenames.add(file)
                    else:
                        used_filenames.add(file.filename)

            # Get unused attachments sorted by reaction count
            unused_attachments = []
            for cache_key, cache_data in self.attachment_handler.attachment_cache.items():
                channel_part, message_id = cache_key.split(":", 1)
                
                # Skip if message was already used or has low reactions
                if message_id not in used_message_ids and cache_data['reaction_count'] >= 3:
                    for attachment in cache_data['attachments']:
                        try:
                            # Skip if this file was already used
                            if attachment.filename in used_filenames:
                                self.logger.debug(f"Skipping already used file: {attachment.filename}")
                                continue
                            
                            if len(attachment.data) <= 25 * 1024 * 1024:  # 25MB limit
                                file = discord.File(
                                    io.BytesIO(attachment.data),
                                    filename=attachment.filename,
                                    description=f"From {attachment.username} ( {attachment.reaction_count} reactions)"
                                )
                                unused_attachments.append((
                                    file,
                                    attachment.reaction_count,
                                    message_id,
                                    attachment.username,
                                    attachment.jump_url
                                ))
                                used_filenames.add(attachment.filename)
                        except Exception as e:
                            self.logger.error(f"Failed to prepare unused attachment: {e}")
                            continue

            if unused_attachments:
                # Sort by reaction count
                unused_attachments.sort(key=lambda x: x[1], reverse=True)
                
                # Limit attachments based on thread type
                max_files = 3 if is_source_thread else max_attachments
                unused_attachments = unused_attachments[:max_files]
                
                # Post header
                await self.safe_send_message(thread, "\n\n---\n# 📎 Other Popular Attachments")
                
                # Post each attachment with both file and link
                for file, reaction_count, message_id, username, jump_url in unused_attachments:
                    message_content = f"By **{username}**: {jump_url}"
                    await self.safe_send_message(thread, message_content, file=file)

            # Always add navigation links, regardless of whether there were attachments
            try:
                footer_text = "\n---\n\u200B\n"

                if is_source_thread:  # If we're in the monthly/source thread
                    # Only show one link - to the beginning of this thread
                    async for first_message in thread.history(oldest_first=True, limit=1):
                        footer_text += f"***Click here to jump to the beginning of this thread: {first_message.jump_url}***"
                        await self.safe_send_message(thread, footer_text)
                        break
                else:  # If we're in the main summary thread
                    # Show both links - to monthly thread and to beginning of this thread
                    if previous_thread_id and str(previous_thread_id) != str(thread.id):
                        try:
                            previous_thread = await self.fetch_channel(previous_thread_id)
                            if previous_thread and isinstance(previous_thread, discord.Thread):
                                async for first_message in previous_thread.history(oldest_first=True, limit=1):
                                    thread_date = previous_thread.created_at.strftime('%B %Y')
                                    footer_text += (
                                        f"**Click here to jump to the beginning of this month's summary:** {first_message.jump_url}\n\n"
                                    )
                                    break
                        except Exception as e:
                            self.logger.error(f"Error processing previous thread: {e}")

                    # Add current thread jump link
                    async for first_message in thread.history(oldest_first=True, limit=1):
                        footer_text += f"***Click here to jump to the beginning of today's summary: {first_message.jump_url}***"
                        await self.safe_send_message(thread, footer_text)
                        break

            except Exception as e:
                self.logger.error(f"Failed to add thread links: {e}")

        except Exception as e:
            self.logger.error(f"Error processing unused attachments: {e}")
            self.logger.debug(traceback.format_exc())

    async def post_summary(self, channel_id, summary: str, source_channel_name: str, message_count: int):
        try:
            source_channel = self.get_channel(channel_id)
            if not source_channel:
                raise DiscordError(f"Could not access source channel {channel_id}")

            self.logger.info(f"Starting post_summary for channel {source_channel.name} (ID: {channel_id})")
            
            # Add debug logging for content lengths
            self.logger.debug(f"Summary length: {len(summary)} characters")
            self.logger.debug(f"First 500 chars of summary: {summary[:500]}")
            
            # Get existing thread ID from DB
            existing_thread_id = self.db.get_summary_thread_id(channel_id)
            self.logger.debug(f"Existing thread ID: {existing_thread_id}")
            
            source_thread = None
            thread_existed = False  # Track if thread existed before
            if existing_thread_id:
                try:
                    self.logger.info(f"Attempting to fetch existing thread with ID: {existing_thread_id}")
                    source_thread = await self.fetch_channel(existing_thread_id)
                    
                    # Add check for orphaned thread
                    if isinstance(source_thread, discord.Thread):
                        try:
                            # Try to fetch the parent message
                            parent_message = await source_channel.fetch_message(source_thread.id)
                            self.logger.info(f"Successfully fetched existing thread: {source_thread.name}")
                            thread_existed = True
                        except discord.NotFound:
                            self.logger.warning(f"Parent message for thread {existing_thread_id} was deleted. Creating new thread.")
                            source_thread = None
                            self.db.update_summary_thread(channel_id, None)
                    else:
                        self.logger.error(f"Fetched channel is not a thread: {existing_thread_id}")
                        source_thread = None
                        self.db.update_summary_thread(channel_id, None)
                except discord.NotFound:
                    self.logger.error(f"Thread {existing_thread_id} not found")
                    source_thread = None
                    self.db.update_summary_thread(channel_id, None)
                except Exception as e:
                    self.logger.error(f"Failed to fetch existing thread: {e}")
                    source_thread = None
                    self.db.update_summary_thread(channel_id, None)

            if not source_thread:
                # Only unpin messages when creating a new thread
                pins = await source_channel.pins()
                for pin in pins:
                    if pin.author.id == self.user.id:  # Check if the pin was made by the bot
                        await pin.unpin()
                    
                self.logger.info("Creating new thread for channel")
                current_date = datetime.utcnow()
                short_month = current_date.strftime('%b')
                thread_message = await source_channel.send(
                    f"📝 A new summary thread has been created for <#{source_channel.id}> for {datetime.utcnow().strftime('%B %Y')}.\n\n"
                    "All of the messages in this channel will be summarised here for your convenience:"
                )
                await thread_message.pin()  # Pin the new thread message
                
                thread_name = f"{short_month} - #{source_channel_name} Summary"
                source_thread = await thread_message.create_thread(name=thread_name)
                
                self.logger.info(f"Created new thread with ID: {source_thread.id}")
                self.db.update_summary_thread(channel_id, source_thread.id)
                self.logger.info(f"Updated DB with new thread ID: {source_thread.id}")

            # Generate short summary and handle main summary channel post
            short_summary = await self.generate_short_summary(summary, message_count)
            
            # Get top attachment for initial message - MODIFIED THIS SECTION
            initial_file = None
            all_files = []
            
            # First try channel-specific attachments
            for cache_key, cache_data in self.attachment_handler.attachment_cache.items():
                try:
                    channel_part, _ = cache_key.split(":", 1)
                    # Convert both to integers for comparison to handle string/int mismatches
                    if int(channel_part) == channel_id:  # Only look at attachments from this channel
                        for attachment in cache_data['attachments']:
                            # Add debug logging
                            self.logger.debug(f"Found attachment {attachment.filename} in channel {channel_id}")
                            if any(attachment.filename.lower().endswith(ext) 
                                  for ext in ('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.mov', '.webm')):
                                all_files.append((
                                    attachment,
                                    cache_data.get('reaction_count', 0)  # Use get() with default
                                ))
                                self.logger.debug(f"Added {attachment.filename} to files list with {cache_data.get('reaction_count', 0)} reactions")
                except (ValueError, AttributeError) as e:
                    self.logger.error(f"Error processing cache key {cache_key}: {e}")
                    continue

            # Add debug logging
            self.logger.debug(f"Found {len(all_files)} files for channel {channel_id}")

            # Sort by reaction count and get top file
            if all_files:
                top_attachment, reaction_count = sorted(all_files, key=lambda x: x[1], reverse=True)[0]
                self.logger.debug(f"Selected top attachment {top_attachment.filename} with {reaction_count} reactions")
                initial_file = discord.File(
                    io.BytesIO(top_attachment.data),
                    filename=top_attachment.filename,
                    description=f"Most reacted media from channel (🔥 {reaction_count} reactions)"
                )
            else:
                self.logger.debug(f"No attachments found for channel {channel_id}")

            channel_mention = f"<#{source_channel.id}>"
            summary_channel = self.get_channel(self.summary_channel_id)
            
            # Add debug logging
            self.logger.debug(f"Sending initial message to summary channel. Has file: {initial_file is not None}")
            
            initial_message = await self.safe_send_message(
                summary_channel,
                f"## {channel_mention}\n{short_summary}",
                file=initial_file
            )

            # Track all files used in any part of the summary
            used_files = []
            if initial_file:
                used_files.append(initial_file)

            # Create and process main summary thread
            thread = await self.create_summary_thread(initial_message, source_channel_name)
            topics = summary.split("## ")
            topics = [t.strip().rstrip('#').strip() for t in topics if t.strip()]
            used_message_ids = set()

            # Add "Detailed Summary" headline at the beginning (only in main summary thread)
            await self.safe_send_message(thread, "# Detailed Summary")

            for i, topic in enumerate(topics):
                topic_used_ids, topic_files = await self.process_topic(
                    thread, 
                    topic, 
                    is_first=(i == 0),
                    is_source_thread=False  # Main summary thread
                )
                used_message_ids.update(topic_used_ids)
                used_files.extend(topic_files)  # Track files used in topics
                
            await self.process_unused_attachments(
                thread, 
                used_message_ids, 
                previous_thread_id=existing_thread_id,
                used_files=used_files,
                is_source_thread=False  # Main summary thread
            )

            # Post to source channel thread (with limited attachments)
            current_date = datetime.utcnow().strftime('%A, %B %d, %Y')

            # Create a new file object for the source thread
            if initial_file:
                initial_file.fp.seek(0)
                source_file = discord.File(
                    initial_file.fp,
                    filename=initial_file.filename,
                    description=initial_file.description
                )
            else:
                source_file = None

            # Keep original header for source thread (without "Detailed Summary")
            await self.safe_send_message(source_thread, f"# Summary from {current_date}")

            # Process topics for source thread
            for i, topic in enumerate(topics):
                topic_used_ids, topic_files = await self.process_topic(
                    source_thread, 
                    topic, 
                    is_first=(i == 0),
                    is_source_thread=True  # Source channel thread
                )
                used_message_ids.update(topic_used_ids)
                used_files.extend(topic_files)  # Track files used in topics

            # Only difference: limit attachments to 3 in the final section
            await self.process_unused_attachments(
                source_thread, 
                used_message_ids, 
                max_attachments=3,
                previous_thread_id=existing_thread_id,
                is_source_thread=True  # Source channel thread
            )

            # After posting the summary to the thread, notify the channel if it was an existing thread
            if thread_existed:
                # Build proper Discord deep link using channel and thread IDs
                thread_link = f"https://discord.com/channels/{self.guild_id}/{source_channel.id}/{source_thread.id}"
                notification_message = f"📝 A new daily summary has been added for <#{source_channel.id}>.\n\nYou can see all of {datetime.utcnow().strftime('%B')}'s activity here: {thread_link}"
                await self.safe_send_message(source_channel, notification_message)
                self.logger.info(f"Posted thread update notification to {source_channel.name}")

            # Add footer with thread links
            try:
                footer_text = "\n---\n\u200B\n"

                if existing_thread_id and str(existing_thread_id) != str(source_thread.id):
                    try:
                        self.logger.info(f"Attempting to fetch previous thread {existing_thread_id}")
                        previous_thread = None
                        
                        try:
                            previous_thread = self.get_channel(existing_thread_id)
                            if not previous_thread:
                                previous_thread = await self.fetch_channel(existing_thread_id)
                        except discord.NotFound:
                            self.logger.warning(f"Could not find thread with ID {existing_thread_id}")
                        
                        if previous_thread and isinstance(previous_thread, discord.Thread):
                            try:
                                async for first_message in previous_thread.history(oldest_first=True, limit=1):
                                    thread_date = previous_thread.created_at.strftime('%B %Y')
                                    footer_text += (
                                        f"**You can find a summary of this channel's activity from "
                                        f"{thread_date} here:** {first_message.jump_url}\n\n"
                                    )
                                    break
                            except Exception as e:
                                self.logger.error(f"Error creating previous thread link: {e}")
                        else:
                            self.logger.error(
                                f"Retrieved channel is not a thread: {type(previous_thread)}"
                                f" for ID {existing_thread_id}"
                            )
                            # Clear invalid thread ID from database
                            self.db.update_summary_thread(source_thread.parent.id, None)
                    except Exception as e:
                        self.logger.error(f"Error processing previous thread: {e}")

                # Add current thread jump link
                try:
                    async for first_message in source_thread.history(oldest_first=True, limit=1):
                        footer_text += f"***Click here to jump to the beginning of this thread: {first_message.jump_url}***"
                        await self.safe_send_message(source_thread, footer_text)
                        self.logger.info("Successfully added thread navigation links")
                        break
                except Exception as e:
                    self.logger.error(f"Failed to add thread jump link: {e}")

            except Exception as e:
                self.logger.error(f"Failed to add thread links: {e}")
                self.logger.debug(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"Error in post_summary: {e}")
            self.logger.debug(traceback.format_exc())

    async def generate_summary(self):
        if self._summary_lock.locked():
            self.logger.warning("Summary generation already in progress, skipping...")
            return
            
        async with self._summary_lock:
            self.logger.info("generate_summary started")
            self.logger.info("\nStarting summary generation")
            
            try:
                self.current_summary_attachments = []  # Reset for new summary
                self.attachment_handler.clear_cache()  # Clear at start
                
                # Remove test mode check and use production channel directly
                summary_channel = self.get_channel(self.summary_channel_id)
                if not summary_channel:
                    raise DiscordError(f"Could not access summary channel {self.summary_channel_id}")
                
                self.logger.info(f"Found summary channel: {summary_channel.name} "
                           f"({'DEV' if self.dev_mode else 'PRODUCTION'} mode)")
                
                active_channels = False
                date_header_posted = False
                self.first_message = None

                # Process channels based on whether they are categories or specific channels
                channels_to_process = []
                if self.dev_mode:
                    if "everyone" in self.dev_channels_to_monitor:
                        guild = self.get_guild(self.guild_id)
                        if not guild:
                            raise DiscordError(f"Could not access guild {self.guild_id}")
                        
                        # Only exclude support channels
                        channels_to_process = [channel.id for channel in guild.text_channels 
                                             if 'support' not in channel.name.lower()]
                    else:
                        # Monitor specified channels or categories
                        for item in self.dev_channels_to_monitor:
                            try:
                                item_id = int(item)
                                channel = self.get_channel(item_id)
                                if isinstance(channel, discord.CategoryChannel):
                                    # If it's a category, add all text channels within it, excluding support channels
                                    self.logger.info(f"Processing category: {channel.name}")
                                    channels_to_process.extend([
                                        c.id for c in channel.channels 
                                        if isinstance(c, discord.TextChannel) and 'support' not in c.name.lower()
                                    ])
                                else:
                                    # If it's a regular channel, add it if it doesn't have "support" in its name
                                    if 'support' not in channel.name.lower():
                                        channels_to_process.append(item_id)
                            except ValueError:
                                self.logger.warning(f"Invalid channel/category ID: {item}")
                else:
                    # Production mode - same logic for handling categories and channels
                    for item in self.channels_to_monitor:
                        channel = self.get_channel(item)
                        if isinstance(channel, discord.CategoryChannel):
                            # If it's a category, add all text channels within it, excluding support channels
                            self.logger.info(f"Processing category: {channel.name}")
                            channels_to_process.extend([
                                c.id for c in channel.channels 
                                if isinstance(c, discord.TextChannel) and 'support' not in c.name.lower()
                            ])
                        else:
                            # If it's a regular channel, add it if it doesn't have "support" in its name
                            if 'support' not in channel.name.lower():
                                channels_to_process.append(item)
                
                self.logger.info(f"Final list of channels to process: {channels_to_process}")
                
                # Process channels sequentially
                for channel_id in channels_to_process:
                    try:
                        channel = self.get_channel(channel_id)
                        if not channel:
                            self.logger.error(f"Could not access channel {channel_id}")
                            continue
                        
                        self.logger.info(f"Processing channel: {channel.name}")
                        messages = await self.get_channel_history(channel.id)
                        
                        # Store attachments from this channel
                        for message in messages:
                            if message.attachments:
                                self.logger.debug(f"Processing {len(message.attachments)} attachments from message {message.id}")
                                async with aiohttp.ClientSession() as session:
                                    for attachment in message.attachments:
                                        try:
                                            processed = await self.attachment_handler.process_attachment(
                                                attachment, message, session
                                            )
                                            if processed:
                                                self.logger.debug(f"Successfully processed attachment {attachment.filename}")
                                            else:
                                                self.logger.debug(f"Failed to process attachment {attachment.filename}")
                                        except Exception as e:
                                            self.logger.error(f"Error processing attachment: {e}")
                                            continue
                        
                        # Process messages for caching attachments regardless of channel type
                        if len(messages) >= 10:
                            # For gens channels, just process attachments without creating summary
                            if 'gens' in channel.name.lower():
                                self.logger.info(f"Processing attachments for gens channel: {channel.name}")
                                continue  # Skip to next channel after attachments are cached
                            
                            # For non-gens channels, generate summary
                            summary = await self.get_claude_summary(messages)
                            self.logger.info(f"Generated summary for {channel.name}: {summary[:100]}...")
                                    
                            if "[NOTHING OF NOTE]" not in summary:
                                self.logger.info(f"Noteworthy activity found in {channel.name}")
                                active_channels = True
                                
                                if not date_header_posted:
                                    current_date = datetime.utcnow()
                                    header = f"#  Daily Summary for {current_date.strftime('%A, %B %d, %Y')}"
                                    self.first_message = await summary_channel.send(header)
                                    date_header_posted = True
                                    self.logger.info("Posted date header")
                                
                                short_summary = await self.generate_short_summary(summary, len(messages))
                                
                                # Store in database regardless of mode
                                self.db.store_daily_summary(
                                    channel_id=channel.id,
                                    channel_name=channel.name,
                                    messages=messages,
                                    full_summary=summary,
                                    short_summary=short_summary
                                )
                                self.logger.info(f"Stored summary in database for {channel.name}")
                                
                                await self.post_summary(
                                    channel.id,
                                    summary,
                                    channel.name,
                                    len(messages)
                                )
                                await asyncio.sleep(2)
                            else:
                                self.logger.info(f"No noteworthy activity in {channel.name}")
                        else:
                            self.logger.warning(f"Skipping {channel.name} - only {len(messages)} messages")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing channel {channel.name}: {e}")
                        self.logger.debug(traceback.format_exc())
                        continue
                
                if not active_channels:
                    self.logger.info("No channels had significant activity - sending notification")
                    await summary_channel.send("No channels had significant activity in the past 24 hours.")
                else:
                    # Create top generations thread after all channels are processed
                    self.logger.info("Active channels found, attempting to create top generations thread")
                    try:
                        await self.create_top_generations_thread(summary_channel)
                        self.logger.info("Top generations thread creation completed")
                    except Exception as e:
                        self.logger.error(f"Failed to create top generations thread: {e}")
                        self.logger.debug(traceback.format_exc())
                        # Don't raise here to allow the rest of the summary to complete

                # Add the top art share post regardless of active channels
                try:
                    await self.post_top_art_share(summary_channel)
                    self.logger.info("Top art share posted")
                except Exception as e:
                    self.logger.error(f"Failed to post top art share: {e}")
                    self.logger.debug(traceback.format_exc())

                # Add footer with jump link at the very end
                if self.first_message:
                    footer_text = f"""---

**_Click here to jump to the top of today's summary:_** {self.first_message.jump_url}"""
                    await self.safe_send_message(summary_channel, footer_text)
                    self.logger.info("Footer message added to summary channel")
                else:
                    self.logger.error("first_message is not defined. Cannot add footer.")
                
            except Exception as e:
                self.logger.error(f"Critical error in summary generation: {e}")
                self.logger.debug(traceback.format_exc())
                raise
            self.logger.info("generate_summary completed")

    async def on_ready(self):
        self.logger.info(f"Logged in as {self.user}")

    async def safe_send_message(self, channel, content=None, embed=None, file=None, files=None, reference=None):
        """Safely send a message with retry logic and error handling."""
        try:
            return await self.rate_limiter.execute(
                f"channel_{channel.id}",
                channel.send(
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
        """Create either a collage of images or a combined video based on media type."""
        try:
            from PIL import Image
            import moviepy.editor as mp
            import io
            
            self.logger.info(f"Starting media content creation with {len(files)} files")
            
            images = []
            videos = []
            has_audio = False
            
            # First pass - categorize media and check for audio
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
                
            # If we have videos with audio, combine them sequentially
            if videos and has_audio:
                self.logger.info("Creating combined video with audio")
                final_video = mp.concatenate_videoclips(videos)
                output_path = 'combined_video.mp4'
                final_video.write_videofile(output_path)
                
                # Clean up
                for video in videos:
                    video.close()
                final_video.close()
                
                self.logger.info("Video combination complete")
                
                # Convert to discord.File
                with open(output_path, 'rb') as f:
                    return discord.File(f, filename='combined_video.mp4')
                
            # If we have videos without audio or images, create a collage
            elif images or (videos and not has_audio):
                self.logger.info("Creating image/GIF collage")
                
                # Convert silent videos to GIFs for the collage
                for i, video in enumerate(videos):
                    self.logger.debug(f"Converting video {i+1}/{len(videos)} to GIF")
                    gif_path = f'temp_gif_{len(images)}.gif'
                    video.write_gif(gif_path)
                    gif_img = Image.open(gif_path)
                    images.append(gif_img)
                    video.close()
                
                if not images:
                    self.logger.warning("No images available for collage")
                    return None
                
                # Calculate collage dimensions
                n = len(images)
                if n == 1:
                    cols, rows = 1, 1
                elif n == 2:
                    cols, rows = 2, 1
                else:
                    cols, rows = 2, 2
                
                self.logger.debug(f"Creating {cols}x{rows} collage for {n} images")
                
                # Create collage
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
                
                # Convert to discord.File
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
            # Clean up temporary files
            import os
            self.logger.debug("Cleaning up temporary files")
            for f in os.listdir():
                if f.startswith('temp_'):
                    try:
                        os.remove(f)
                        self.logger.debug(f"Removed temporary file: {f}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temporary file {f}: {e}")

    async def create_top_generations_thread(self, summary_channel):
        """Creates a thread showcasing the top video generations with >3 reactors."""
        try:
            self.logger.info("Starting create_top_generations_thread")
            
            # Get the correct channel to monitor in dev mode
            if self.dev_mode:
                # CHANGE: Fix the channel ID extraction from cache keys
                monitored_channels = list(set(
                    int(cache_key.split(':')[0])  # Only take the channel ID part before the colon
                    for cache_key in self.attachment_handler.attachment_cache.keys()
                ))
                
                self.logger.info(f"Using cached channels in dev mode: {monitored_channels}")
                
                # Don't update channels_to_monitor in dev mode
                self.approved_channels = monitored_channels  # Allow all channels with cached attachments
                
            else:
                monitored_channels = [int(chan) for chan in self.channels_to_monitor]
            
            self.logger.info(f"Monitoring channels: {monitored_channels}")
            self.logger.debug(f"Approved channels: {self.approved_channels}")
            
            # Get all video attachments from stored attachments
            video_attachments = []
            seen_videos = set()  # Track unique videos by filename + username
            
            # Track all videos from the attachment cache
            for cache_key, cache_data in self.attachment_handler.attachment_cache.items():
                try:
                    channel_id = int(cache_key.split(':')[0])
                    
                    # CHANGE: In dev mode, accept all channels that have cached attachments
                    if self.dev_mode:
                        channel = self.get_channel(channel_id)
                        if not channel:
                            continue
                        
                        # Skip if channel name contains 'support' or 'art'
                        if any(term in channel.name.lower() for term in ['support', 'art']):
                            continue
                    else:
                        # Production mode checks
                        if channel_id not in monitored_channels:
                            self.logger.debug(f"Skipping channel {channel_id} - not in monitored list")
                            continue
                        
                        if channel_id not in self.approved_channels:
                            self.logger.debug(f"Skipping channel {channel_id} - not in approved list")
                            continue
                        
                        channel = self.get_channel(channel_id)
                        if not channel:
                            continue
                        
                        if 'support' in channel.name.lower():
                            continue

                    for attachment in cache_data['attachments']:
                        # Create a unique identifier for this video
                        video_id = f"{attachment.filename}:{cache_data['username']}"
                        
                        # Skip if we've already seen this video from this user
                        if video_id in seen_videos:
                            self.logger.debug(f"Skipping duplicate video: {video_id}")
                            continue
                        
                        # More lenient video check
                        is_video = any(attachment.filename.lower().endswith(ext) 
                                     for ext in ('.mp4', '.mov', '.webm', '.avi'))
                        
                        if is_video and cache_data['reaction_count'] >= 3:
                            self.logger.info(f"Found video in {channel.name} with {cache_data['reaction_count']} reactions")
                            
                            video_attachments.append({
                                'attachment': attachment,
                                'channel_name': channel.name,
                                'reaction_count': cache_data['reaction_count'],
                                'username': cache_data['username']
                            })
                            seen_videos.add(video_id)  # Mark this video as seen
                            self.logger.info(f"Added video to top generations list")

                except Exception as e:
                    self.logger.error(f"Error processing cache key {cache_key}: {e}")
                    continue

            # Rest of the method remains the same...

            # Check if we have any videos before proceeding
            if not video_attachments:
                self.logger.info("No qualifying videos found - skipping top generations thread")
                return
            
            # Sort by unique reactor count
            video_attachments.sort(key=lambda x: x['reaction_count'], reverse=True)
            
            # Take top 10 but include ALL videos that meet criteria
            top_generations = video_attachments[:10]

            # Rest of the method remains the same...

            # Get the top generation for the header message
            top_gen = top_generations[0]
            top_message = top_gen['attachment'].content[:100] + "..." if len(top_gen['attachment'].content) > 100 else top_gen['attachment'].content
            
            # Create dynamic header based on number of generations
            header_title = (
                "Today's Top Generation" if len(top_generations) == 1 
                else f"Today's Top {len(top_generations)} Generations"
            )
            
            # Create the initial message and thread with top generation details
            header_content = [
                f"# {header_title}\n\n"                
                f"## 1. By **{top_gen['username']}** in {top_gen['channel_name']}\n"
                f"🔥 {top_gen['reaction_count']} unique reactions\n"
            ]
            
            # Only add message text if it's not empty
            if top_message.strip():
                header_content.append(f"💭 Message text: `{top_message}`\n")
            
            # Add plain URL link
            header_content.append(f"🔗 {top_gen['attachment'].jump_url}")
            
            # Join all content parts
            header_content = ''.join(header_content)
            
            # Create header message with the top generation's video
            file = discord.File(
                io.BytesIO(top_gen['attachment'].data),
                filename=top_gen['attachment'].filename,
                description=f"Top Generation - {top_gen['reaction_count']} unique reactions"
            )
            
            self.logger.info("Creating initial thread message with top generation")
            header_message = await self.safe_send_message(
                summary_channel,
                header_content,
                file=file
            )

            self.logger.info("Creating thread")
            thread = await self.create_summary_thread(
                header_message,
                f"Top Generations for {datetime.utcnow().strftime('%B %d, %Y')}",
                is_top_generations=True
            )

            # Remove the introduction text and start directly with the generations
            # Post each generation (starting from index 1 to skip the top one we already posted)
            for i, gen in enumerate(top_generations[1:], 2):
                try:
                    attachment = gen['attachment']
                    channel_name = gen['channel_name']
                    
                    # Create message link
                    message_link = attachment.jump_url
                    
                    # Get message content (first 100 chars)
                    msg_text = attachment.content[:100] + "..." if len(attachment.content) > 100 else attachment.content
                    
                    # Build description content
                    description = [
                        f"## {i}. By **{gen['username']}** in #{channel_name}\n"
                        f"🔥 {gen['reaction_count']} unique reactions\n"
                    ]
                    
                    # Only add message text if it's not empty
                    if msg_text.strip():
                        description.append(f"💭 Message text: `{msg_text}`\n")
                    
                    # Add plain URL link
                    description.append(f"🔗 {message_link}")
                    
                    # Join all description parts
                    description = ''.join(description)

                    file = discord.File(
                        io.BytesIO(attachment.data),
                        filename=attachment.filename,
                        description=f"#{i} - {gen['reaction_count']} unique reactions"
                    )

                    await self.safe_send_message(thread, description, file=file)
                    await asyncio.sleep(1)  # Prevent rate limiting

                except Exception as e:
                    self.logger.error(f"Error posting generation #{i}: {e}")
                    continue

            # Add footer
            footer = (
                "\n---\n"
                "**These generations represent the most popular non-#art_sharing videos "
                "from the past 24 hours, ranked by unique reactions from the community.**\n\n"
                "_Only generations with more than 3 unique reactions are included in this list._"
            )
            await self.safe_send_message(thread, footer)

            self.logger.info("Top generations thread created successfully")

        except Exception as e:
            self.logger.error(f"Error creating top generations thread: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    async def post_top_art_share(self, summary_channel):
        """Posts the most reacted-to art share from the art sharing channel in the last 24 hours."""
        try:
            self.logger.info("Starting post_top_art_share")
            
            # Get correct art sharing channel ID based on mode
            art_channel_id = int(os.getenv('DEV_ART_SHARING_CHANNEL_ID' if self.dev_mode else 'ART_SHARING_CHANNEL_ID'))
            self.logger.info(f"Art channel ID: {art_channel_id}")
            
            art_channel = self.get_channel(art_channel_id)
            
            if not art_channel:
                self.logger.error(f"Could not access art sharing channel {art_channel_id}")
                return
            
            self.logger.info(f"Processing art channel: {art_channel.name}")
            
            # Get last 24 hours of messages
            yesterday = datetime.utcnow() - timedelta(days=1)
            top_art = None
            max_unique_reactors = 0
            
            message_count = 0
            async for message in art_channel.history(after=yesterday, limit=None):
                message_count += 1
                if not message.attachments:
                    continue
                
                # Count unique reactors
                unique_reactors = set()
                for reaction in message.reactions:
                    async for user in reaction.users():
                        unique_reactors.add(user.id)
                
                if len(unique_reactors) > max_unique_reactors:
                    max_unique_reactors = len(unique_reactors)
                    top_art = message
            
            self.logger.info(f"Processed {message_count} messages from art channel")
            
            if top_art and max_unique_reactors > 0:
                self.logger.info(f"Found top art with {max_unique_reactors} reactions")
                
                try:
                    # Get the first attachment URL
                    attachment_url = top_art.attachments[0].url
                    
                    # Format the message content
                    content = [
                        f"# Top <#{art_channel.id}> Post Today",
                        "",  # Empty line for spacing
                        f"By: **{top_art.author.name}**" if self.dev_mode else f"By: <@{top_art.author.id}>"
                    ]
                    
                    # Add comment if there is one
                    if top_art.content:
                        content.append(f"💭 *\"{top_art.content}\"*")
                    
                    # Add attachment URL and jump link
                    content.append(f"{attachment_url}")  # Direct attachment URL
                    content.append(f"🔗 Original post: {top_art.jump_url}")
                    
                    # Join with newlines and send as one message
                    formatted_content = "\n".join(content)
                    await self.safe_send_message(summary_channel, formatted_content)
                    self.logger.info("Posted top art share successfully")

                except Exception as e:
                    self.logger.error(f"Error handling art post: {e}")
                    self.logger.debug(traceback.format_exc())

            else:
                self.logger.info("No art posts with reactions found in the last 24 hours")

        except Exception as e:
            self.logger.error(f"Error posting top art share: {e}")
            self.logger.debug(traceback.format_exc())

    async def close(self):
        """Override close to handle cleanup"""
        try:
            self.logger.info("Starting bot cleanup...")
            
            # Close aiohttp session if it exists and isn't closed
            if hasattr(self, 'session') and self.session and not self.session.closed:
                self.logger.info("Closing aiohttp session...")
                await self.session.close()
            
            # Close any remaining connections
            if hasattr(self, 'claude'):
                self.logger.info("Cleaning up Claude client...")
                self.claude = None
            
            self.logger.info("Cleanup completed, calling parent close...")
            await super().close()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.debug(traceback.format_exc())
            # Still try to close parent even if we had an error
            await super().close()

async def schedule_daily_summary(bot):
    """
    Run daily summaries on schedule. Only exits if there's an error or explicit shutdown.
    """
    try:
        while not bot._shutdown_flag:  # Add loop to keep running
            now = datetime.utcnow()
            target = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
            if now.hour >= 10:
                target += timedelta(days=1)
            
            delay = (target - now).total_seconds()
            bot.logger.info(f"Waiting {delay/3600:.2f} hours until next summary at {target} UTC")
            
            try:
                await asyncio.sleep(delay)
                await bot.generate_summary()
                bot.logger.info(f"Summary generated successfully at {datetime.utcnow()} UTC")
            except asyncio.CancelledError:
                bot.logger.info("Summary schedule cancelled - shutting down")
                break
            except Exception as e:
                bot.logger.error(f"Error generating summary: {e}")
                bot.logger.debug(traceback.format_exc())
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)
                
    except Exception as e:
        bot.logger.error(f"Fatal error in scheduler: {e}")
        bot.logger.debug(traceback.format_exc())
        bot._shutdown_flag = True  # Signal shutdown on fatal error
        await bot.close()
        raise

def main():
    # Parse command line arguments FIRST
    parser = argparse.ArgumentParser(description='Discord Channel Summarizer Bot')
    parser.add_argument('--run-now', action='store_true', help='Run the summary process immediately')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    # Create and configure the bot
    bot = ChannelSummarizer()
    
    # Set dev mode before any other initialization
    bot.dev_mode = args.dev
    
    # Now load config and continue initialization
    bot.load_config()
    
    # Get bot token after loading config
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        raise ValueError("Discord bot token not found in environment variables")
    
    if args.dev:
        bot.logger.info("Running in DEVELOPMENT mode - using test data")  # Changed from main_logger to bot.logger
    
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Modify the on_ready event to handle immediate execution if requested
    @bot.event
    async def on_ready():
        bot.logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
        bot.logger.info("Connected to servers: %s", [guild.name for guild in bot.guilds])
        
        if args.run_now:
            bot.logger.info("Running summary process immediately...")
            try:
                await bot.generate_summary()
                bot.logger.info("Summary process completed. Shutting down...")
            finally:
                bot._shutdown_flag = True  # Signal for shutdown
                await bot.close()
        else:
            # Add a flag to prevent multiple scheduler starts
            if not hasattr(bot, '_scheduler_started'):
                bot._scheduler_started = True
                bot.logger.info("Starting scheduled mode - will run daily at 10:00 UTC")
                loop.create_task(schedule_daily_summary(bot))
    
    # Run the bot with proper cleanup
    try:
        loop.run_until_complete(bot.start(bot_token))
    except KeyboardInterrupt:
        bot.logger.info("Keyboard interrupt received - shutting down...")
    finally:
        # Only do full cleanup on shutdown
        try:
            loop.run_until_complete(asyncio.sleep(1))  # Give tasks a moment to complete
            
            # Cancel all running tasks
            tasks = [t for t in asyncio.all_tasks(loop) 
                    if not t.done() and t != asyncio.current_task(loop)]
            if tasks:
                bot.logger.info(f"Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            # Close the bot connection
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
            
        # Only exit if we're in immediate mode
        if getattr(bot, '_shutdown_flag', False):
            sys.exit(0)

if __name__ == "__main__":
    main()

