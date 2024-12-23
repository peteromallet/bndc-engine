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
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
from src.db_handler import DatabaseHandler
from utils.errors import *
from utils.error_handler import ErrorHandler, handle_errors
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('discord_bot.log')  # File handler
    ]
)
logger = logging.getLogger('ChannelSummarizer')

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

class RateLimiter:
    """Manages rate limiting for Discord API calls with exponential backoff."""
    
    def __init__(self):
        self.backoff_times = {}  # Store backoff times per channel
        self.base_delay = 1.0    # Base delay in seconds
        self.max_delay = 64.0    # Maximum delay in seconds
        self.jitter = 0.1        # Random jitter factor
        
    async def execute(self, key, coroutine):
        """
        Executes a coroutine with rate limit handling.
        
        Args:
            key: Identifier for the rate limit (e.g., channel_id)
            coroutine: The coroutine to execute
            
        Returns:
            The result of the coroutine execution
        """
        max_retries = 5
        attempt = 0
        
        while attempt < max_retries:
            try:
                # Add jitter to prevent thundering herd
                if key in self.backoff_times:
                    jitter = random.uniform(-self.jitter, self.jitter)
                    await asyncio.sleep(self.backoff_times[key] * (1 + jitter))
                
                result = await coroutine
                
                # Reset backoff on success
                self.backoff_times[key] = self.base_delay
                return result
                
            except discord.HTTPException as e:
                attempt += 1
                
                if e.status == 429:  # Rate limit hit
                    retry_after = e.retry_after if hasattr(e, 'retry_after') else None
                    
                    if retry_after:
                        logger.warning(f"Rate limit hit for {key}. Retry after {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        # Calculate exponential backoff
                        current_delay = self.backoff_times.get(key, self.base_delay)
                        next_delay = min(current_delay * 2, self.max_delay)
                        self.backoff_times[key] = next_delay
                        
                        logger.warning(f"Rate limit hit for {key}. Using exponential backoff: {next_delay}s")
                        await asyncio.sleep(next_delay)
                        
                elif attempt == max_retries:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Discord API error (attempt {attempt}/{max_retries}): {e}")
                    # Calculate exponential backoff for other errors
                    current_delay = self.backoff_times.get(key, self.base_delay)
                    next_delay = min(current_delay * 2, self.max_delay)
                    self.backoff_times[key] = next_delay
                    await asyncio.sleep(next_delay)
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                logger.debug(traceback.format_exc())
                raise

class Attachment:
    def __init__(self, filename: str, data: bytes, content_type: str, reaction_count: int, username: str):
        self.filename = filename
        self.data = data
        self.content_type = content_type
        self.reaction_count = reaction_count
        self.username = username

class MessageData:
    def __init__(self, content: str, author: str, timestamp: datetime, jump_url: str, reactions: int, id: str, attachments: List[Attachment]):
        self.content = content
        self.author = author
        self.timestamp = timestamp
        self.jump_url = jump_url
        self.reactions = reactions
        self.id = id
        self.attachments = attachments

class AttachmentHandler:
    def __init__(self, max_size: int = 25 * 1024 * 1024):
        self.max_size = max_size
        # Add channel_id to make the cache structure clearer
        self.attachment_cache: Dict[str, Dict[str, Any]] = {}
        
    def clear_cache(self):
        """Clear the attachment cache"""
        self.attachment_cache.clear()
        
    async def process_attachment(self, attachment: discord.Attachment, message: discord.Message, session: aiohttp.ClientSession) -> Optional[Attachment]:
        """Process a single attachment with size and type validation."""
        try:
            logger.debug(f"Processing attachment {attachment.filename} from message {message.id} in channel {message.channel.id}")
            
            # Create composite key that includes channel ID
            cache_key = f"{message.channel.id}:{message.id}"
            
            async with session.get(attachment.url, timeout=300) as response:
                if response.status != 200:
                    raise APIError(f"Failed to download attachment: HTTP {response.status}")

                file_data = await response.read()
                if len(file_data) > self.max_size:
                    logger.warning(f"Skipping large file {attachment.filename} ({len(file_data)/1024/1024:.2f}MB)")
                    return None

                total_reactions = sum(reaction.count for reaction in message.reactions) if message.reactions else 0

                processed_attachment = Attachment(
                    filename=attachment.filename,
                    data=file_data,
                    content_type=attachment.content_type,
                    reaction_count=total_reactions,
                    username=message.author.name
                )

                # Store with channel context
                if cache_key not in self.attachment_cache:
                    self.attachment_cache[cache_key] = {
                        'attachments': [],
                        'reaction_count': total_reactions,
                        'username': message.author.name,
                        'channel_id': str(message.channel.id)
                    }
                self.attachment_cache[cache_key]['attachments'].append(processed_attachment)

                return processed_attachment

        except Exception as e:
            logger.error(f"Failed to process attachment {attachment.filename}: {e}")
            logger.debug(traceback.format_exc())
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
                        logger.error(f"Failed to prepare file {attachment.filename}: {e}")
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
            if (any(line.startswith(emoji) for emoji in ['🎥', '💻', '🎬', '🤖', '📱', '💡', '🔧', '🎨', '📊']) and 
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
        
        self.claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.session = None

        self.summary_channel_id = int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))
        self.guild_id = int(os.getenv('GUILD_ID'))

        logger.info("Bot initialized with:")
        logger.info(f"Guild ID: {self.guild_id}")
        logger.info(f"Summary Channel: {self.summary_channel_id}")

        self.rate_limiter = RateLimiter()

        self.attachment_handler = AttachmentHandler()
        self.message_formatter = MessageFormatter()

        self.db = DatabaseHandler()

        self.error_handler = ErrorHandler()

        self.dev_mode = False
        
        self.guild_id = None
        self.summary_channel_id = None
        self.category_ids = []
        self.first_message = None

    def load_config(self):
        """Load configuration based on current mode"""
        try:
            if self.dev_mode:
                logger.info("Loading development configuration")
                self.guild_id = int(os.getenv('DEV_GUILD_ID'))
                self.summary_channel_id = int(os.getenv('DEV_SUMMARY_CHANNEL_ID'))
                categories_str = os.getenv('DEV_CATEGORIES_TO_MONITOR')
                if not categories_str:
                    raise ConfigurationError("DEV_CATEGORIES_TO_MONITOR not found in environment")
            else:
                logger.info("Loading production configuration")
                self.guild_id = int(os.getenv('GUILD_ID'))
                self.summary_channel_id = int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))
                categories_str = os.getenv('CATEGORIES_TO_MONITOR')
                if not categories_str:
                    raise ConfigurationError("CATEGORIES_TO_MONITOR not found in environment")
            
            self.category_ids = [int(cat_id) for cat_id in categories_str.split(',')]
            logger.info(f"Configured with guild_id={self.guild_id}, "
                       f"summary_channel={self.summary_channel_id}, "
                       f"categories={self.category_ids}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def load_test_data(self) -> Dict[str, Any]:
        """Load test data from test.json"""
        try:
            with open('test.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {"messages": []}

    async def setup_hook(self):
        """This is called when the bot is starting up"""
        try:
            self.session = aiohttp.ClientSession()
            
            notification_channel = self.get_channel(self.summary_channel_id)
            admin_user = await self.fetch_user(int(os.getenv('ADMIN_USER_ID')))
            self.error_handler = ErrorHandler(notification_channel, admin_user)
            
        except Exception as e:
            raise ConfigurationError("Failed to initialize bot", e)

    async def get_channel_history(self, channel_id):
        """Get channel history with support for test data channel in dev mode"""
        if self.dev_mode:
            try:
                test_channel_id = int(os.getenv('TEST_DATA_CHANNEL'))
                logger.info(f"Using test data channel {test_channel_id} for development")
                
                test_channel = self.get_channel(test_channel_id)
                if not test_channel:
                    raise ChannelSummarizerError(f"Could not access test data channel {test_channel_id}")
                
                yesterday = datetime.now(timezone.utc) - timedelta(days=1)
                messages = []
                
                async for message in test_channel.history(after=yesterday, limit=None):
                    total_reactions = sum(reaction.count for reaction in message.reactions) if message.reactions else 0
                    
                    # Process attachments
                    if message.attachments:
                        async with aiohttp.ClientSession() as session:
                            for attachment in message.attachments:
                                processed = await self.attachment_handler.process_attachment(
                                    attachment, message, session
                                )
                    
                    messages.append({
                        'content': message.content,
                        'author': message.author.name,
                        'timestamp': message.created_at,
                        'jump_url': message.jump_url,
                        'reactions': total_reactions,
                        'id': str(message.id),
                        'attachments': [
                            {
                                'url': a.url,
                                'filename': a.filename,
                                'content_type': a.content_type
                            } for a in message.attachments
                        ]
                    })
                
                logger.info(f"Loaded {len(messages)} messages from test data channel")
                return messages
                
            except Exception as e:
                logger.error(f"Error loading test data channel: {e}")
                logger.debug(traceback.format_exc())
                return []
        
        # Production code path
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                raise ChannelSummarizerError(f"Could not access channel {channel_id}")
            
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            messages = []
            
            async for message in channel.history(after=yesterday, limit=None):
                total_reactions = sum(reaction.count for reaction in message.reactions) if message.reactions else 0
                
                # Process attachments
                if message.attachments:
                    async with aiohttp.ClientSession() as session:
                        for attachment in message.attachments:
                            processed = await self.attachment_handler.process_attachment(
                                attachment, message, session
                            )
                
                messages.append({
                    'content': message.content,
                    'author': message.author.name,
                    'timestamp': message.created_at,
                    'jump_url': message.jump_url,
                    'reactions': total_reactions,
                    'id': str(message.id),
                    'attachments': [
                        {
                            'url': a.url,
                            'filename': a.filename,
                            'content_type': a.content_type
                        } for a in message.attachments
                    ]
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting channel history: {e}")
            logger.debug(traceback.format_exc())
            raise

    @handle_errors("get_claude_summary")
    async def get_claude_summary(self, messages):
        """
        Generate summary using Claude API with comprehensive error handling.
        """
        if not messages:
            logger.info("No messages to summarize")
            return "[NOTHING OF NOTE]"
        
        logger.info(f"Generating summary for {len(messages)} messages")
        
        try:
            # Build the conversation prompt
            conversation = """Please summarize the interesting and noteworthy Discord happenings and ideas in bullet points. ALWAYS include Discord links and external links. You should extract ideas and information that may be useful to others from conversations. Avoid stuff like bug reports that are circumstantial or not useful to others. Break them into topics and sub-topics.

If there's nothing significant or noteworthy in the messages, just respond with exactly "[NOTHING OF NOTE]" (and no other text). Always include external links and Discord links wherever possible.

Requirements:
1. Make sure to ALWAYS include Discord links and external links 
2. Use Discord's markdown format (not regular markdown)
3. Use - for top-level points (no bullet for the header itself). Only use - for clear sub-points that directly relate to the point above. You should generally just create a new point for each new topic.
4. Make each main topic a ### header with an emoji
5. Use ** for bold text (especially for usernames and main topics)
6. Keep it simple - just bullet points and sub-points for each topic, no headers or complex formatting
7. ALWAYS include the message author's name in bold (**username**) for each point if there's a specific person who did something, said something important, or seemed to be helpful - mention their username, don't tag them. Call them "Banodocians" instead of "users".
8. Always include a funny or relevant emoji in the topic title

Here's one example of what a good summary and topic should look like:

### 🤏 **H264/H265 Compression Techniques for Video Generation Improves Img2Vid**
- **zeevfarbman** recommended h265 compression for frame degradation with less perceptual impact: https://discord.com/channels/1076117621407223829/1309520535012638740/1316462339318354011
- **johndopamine** suggested using h264 node in MTB node pack for better video generation: https://discord.com/channels/564534534/1316786801247260672
- Codec compression can help "trick" current workflows/models: https://github.com/tdrussell/codex-pipe
- melmass confirmed adding h265 support to their tools: https://discord.com/channels/1076117621407223829/1309520535012638740/1316786801247260672

And here's another example of a good summary and topic:

### 🏋 **Person Training for Hunyuan Video is Now Possible**    
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
6. Make topics clear headers with ###

IMPORTANT: For each bullet point, use the EXACT message URL provided in the data - do not write <message_url> but instead use the actual URL from the message data.

Please provide the summary now - don't include any other introductory text, ending text, or explanation of the summary:\n\n"""
        
            for msg in messages:
                conversation += f"{msg['timestamp']} - {msg['author']}: {msg['content']}"
                if msg['reactions']:
                    conversation += f"\nReactions: {msg['reactions']}"
                conversation += f"\nDiscord link: {msg['jump_url']}\n\n"

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
            
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content: {response.content}")
            
            # Ensure the response has the expected structure
            if not hasattr(response, 'content') or not response.content:
                raise ValueError("Invalid response format from Claude API.")

            summary_text = response.content[0].text.strip()
            logger.info("Summary generated successfully")
            return summary_text
                
        except asyncio.CancelledError:
            logger.info("Summary generation cancelled - shutting down gracefully")
            raise
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected - shutting down gracefully")
            raise
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            logger.debug(traceback.format_exc())
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
                logger.error(f"Timeout attempt {retry_count}/{max_retries} while generating short summary")
                if retry_count < max_retries:
                    logger.info(f"Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    logger.error("All retry attempts failed")
                    return f"__📨 {message_count} messages sent__\n• Error generating summary\u200B"
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Error attempt {retry_count}/{max_retries} while generating short summary: {e}")
                if retry_count < max_retries:
                    logger.info(f"Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    logger.error("All retry attempts failed")
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
            
            logger.info(f"Initial message sent successfully: {initial_message.id if initial_message else 'None'}")
            return initial_message, top_message_id
            
        except Exception as e:
            logger.error(f"Failed to send initial message: {e}")
            logger.debug(traceback.format_exc())
            raise DiscordError(f"Failed to send initial message: {e}")

    async def create_summary_thread(self, message: discord.Message, source_channel_name: str) -> discord.Thread:
        """
        Create a thread for the summary.
        
        Args:
            message: Discord message to create thread from
            source_channel_name: Name of the channel being summarized
            
        Returns:
            discord.Thread: The created thread
        """
        max_retries = 3  # Maximum number of retry attempts
        for attempt in range(1, max_retries + 1):
            try:
                if not message:
                    logger.error("Cannot create thread: message is None")
                    raise ValueError("Message cannot be None")
                    
                logger.info(f"Creating thread for message {message.id}, attempt {attempt}")
                current_date = datetime.utcnow()
                
                # Check if this is the main summary channel
                is_summary_channel = message.channel.id == self.summary_channel_id
                
                if is_summary_channel:
                    # For main summary channel, use short month format
                    thread_name = f"Summary for #{source_channel_name} for {current_date.strftime('%A, %B %d')}"

                else:
                    short_month = current_date.strftime('%b')
                    thread_name = f"{short_month} - #{source_channel_name} Summary"
                    # For topic channels, use the original format
                                    
                thread = await message.create_thread(
                    name=thread_name
                )
                
                logger.info(f"Thread created successfully: {thread.id} with name '{thread_name}'")
                
                # Send a separator message in the thread
                await self.safe_send_message(thread, "## 📝 Detailed Summary")
                
                # Store the thread ID in the database
                self.db.update_summary_thread(message.channel.id, thread.id)
                logger.info(f"Stored thread ID {thread.id} in database for channel {message.channel.id}")
                
                return thread  # Successfully created thread
                
            except discord.Forbidden as e:
                logger.error(f"Bot lacks permissions to create thread: {e}")
                raise DiscordError("Insufficient permissions to create thread") from e
                
            except discord.HTTPException as e:
                logger.warning(f"Failed to create thread on attempt {attempt}: {e}")
                
                if attempt < max_retries:
                    retry_delay = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Exceeded max retries ({max_retries}) for creating thread")
                    raise DiscordError(f"Failed to create thread after {max_retries} attempts") from e
                
            except Exception as e:
                logger.error(f"Unexpected error during thread creation: {e}")
                logger.debug(traceback.format_exc())
                raise DiscordError(f"Unexpected error during thread creation: {e}") from e

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
                        logger.error(f"Failed to prepare file {attachment.filename}: {e}")
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
            logger.error(f"Failed to send topic chunk: {e}")
            logger.debug(traceback.format_exc())

    async def process_topic(self, thread: discord.Thread, topic: str, is_first: bool = False) -> Tuple[Set[str], List[discord.File]]:
        """Process and send a single topic to the thread. Returns used message IDs and files."""
        try:
            if not topic.strip():
                return set(), []

            # Format topic header
            formatted_topic = ("---\n" if not is_first else "") + f"### {topic}"
            logger.info(f"Processing topic: {formatted_topic[:100]}...")

            # Get message IDs from this chunk
            chunk_message_ids = set(re.findall(r'https://discord\.com/channels/\d+/(\d+)/(\d+)', topic))
            used_message_ids = set()
            used_files = []  # Track files used in this topic

            # Prepare files for just this chunk
            chunk_files = []
            for channel_id, message_id in chunk_message_ids:
                cache_key = f"{channel_id}:{message_id}"
                if cache_key in self.attachment_handler.attachment_cache:
                    cache_data = self.attachment_handler.attachment_cache[cache_key]
                    for attachment in cache_data['attachments']:
                        try:
                            if len(attachment.data) <= 25 * 1024 * 1024:  # 25MB limit
                                file = discord.File(
                                    io.BytesIO(attachment.data),
                                    filename=attachment.filename,
                                    description=f"From {cache_data['username']} (🔥 {cache_data['reaction_count']} reactions)"
                                )
                                chunk_files.append((
                                    file,
                                    cache_data['reaction_count'],
                                    message_id,
                                    cache_data['username']
                                ))
                                used_files.append(file)  # Track the file
                        except Exception as e:
                            logger.error(f"Failed to prepare file: {e}")
                            continue

            # Sort files by reaction count and limit to top 10
            chunk_files.sort(key=lambda x: x[1], reverse=True)
            chunk_files = chunk_files[:10]

            # Send the chunk with its specific attachments
            if chunk_files:
                # Send content first
                await self.safe_send_message(thread, formatted_topic)
                
                # Then send each attachment individually
                for file, reaction_count, message_id, username in chunk_files:
                    try:
                        await self.safe_send_message(
                            thread,
                            file=file
                        )
                    except Exception as e:
                        logger.error(f"Failed to send attachment: {e}")
            else:
                # If no attachments, just send the content
                await self.safe_send_message(thread, formatted_topic)

            return chunk_message_ids, used_files

        except Exception as e:
            logger.error(f"Error processing topic: {e}")
            logger.debug(traceback.format_exc())
            return set(), []
    
    async def process_unused_attachments(self, thread: discord.Thread, used_message_ids: Set[str], max_attachments: int = 10, previous_thread_id: Optional[int] = None, used_files: List[discord.File] = None):
        """Process and send unused but popular attachments."""
        unused_attachments = []
        
        # Create set of filenames that were already used
        used_filenames = {f.filename for f in (used_files or [])}
        
        # Get all attachments already in the thread
        try:
            async for message in thread.history(limit=None):
                for attachment in message.attachments:
                    used_filenames.add(attachment.filename)
                    logger.debug(f"Adding existing thread attachment to used files: {attachment.filename}")
        except Exception as e:
            logger.error(f"Error getting thread history for attachment tracking: {e}")
        
        for cache_key, cache_data in self.attachment_handler.attachment_cache.items():
            channel_part, message_part = cache_key.split(":", 1)

            # Skip if message was already used or if attachment was used as main image
            if message_part not in used_message_ids and cache_data['reaction_count'] >= 3:
                message_link = f"https://discord.com/channels/{self.guild_id}/{channel_part}/{message_part}"

                for attachment in cache_data['attachments']:
                    try:
                        # Skip if this file was already used anywhere in the thread
                        if attachment.filename in used_filenames:
                            logger.debug(f"Skipping already used file: {attachment.filename}")
                            continue
                        
                        if len(attachment.data) <= 25 * 1024 * 1024:  # 25MB limit
                            file = discord.File(
                                io.BytesIO(attachment.data),
                                filename=attachment.filename,
                                description=f"From {attachment.username} (🔥 {attachment.reaction_count} reactions)"
                            )
                            unused_attachments.append((
                                file, 
                                attachment.reaction_count,
                                message_part,
                                attachment.username,
                                message_link
                            ))
                    except Exception as e:
                        logger.error(f"Failed to prepare unused attachment: {e}")
                        continue

        if unused_attachments:
            # Sort by reaction count
            unused_attachments.sort(key=lambda x: x[1], reverse=True)
            
            # Post header
            await self.safe_send_message(thread, "\n\n---\n## 📎 Other Popular Attachments")
            
            # Post each attachment individually with a message link (limited by max_attachments)
            for file, reaction_count, message_id, username, message_link in unused_attachments[:max_attachments]:
                message_content = f"By **{username}**: {message_link}"
                await self.safe_send_message(thread, message_content, file=file)

        # Add links to previous thread and current thread beginning
        try:
            footer_text = "\n---\n\u200B\n"

            if previous_thread_id and str(previous_thread_id) != str(thread.id):  # Convert to strings for comparison
                try:
                    # Add more detailed logging
                    logger.info(f"Attempting to fetch previous thread {previous_thread_id}")
                    previous_thread = None
                    
                    # Try both fetch_channel and get_channel
                    try:
                        previous_thread = self.get_channel(previous_thread_id)
                        if not previous_thread:
                            previous_thread = await self.fetch_channel(previous_thread_id)
                    except discord.NotFound:
                        logger.warning(f"Could not find thread with ID {previous_thread_id}")
                    
                    if previous_thread and isinstance(previous_thread, discord.Thread):
                        try:
                            # Get first message of the thread for proper linking
                            async for first_message in previous_thread.history(oldest_first=True, limit=1):
                                thread_date = previous_thread.created_at.strftime('%B %Y')
                                footer_text += (
                                    f"**You can find a summary of this channel's activity from "
                                    f"{thread_date} here:** {first_message.jump_url}\n\n"
                                )
                                logger.info(f"Successfully added link to previous thread from {thread_date}")
                                break
                        except discord.Forbidden:
                            logger.error(f"Bot lacks permissions to access thread {previous_thread_id}")
                        except discord.NotFound:
                            logger.error(f"Thread {previous_thread_id} exists but cannot be accessed")
                        except Exception as e:
                            logger.error(f"Error creating previous thread link: {e}")
                    else:
                        logger.error(
                            f"Retrieved channel is not a thread: {type(previous_thread)}"
                            f" for ID {previous_thread_id}"
                        )
                        # Clear invalid thread ID from database
                        self.db.update_summary_thread(thread.parent.id, None)
                        logger.info(f"Cleared invalid thread ID {previous_thread_id} from database")
                except Exception as e:
                    logger.error(f"Error processing previous thread: {e}")
                    logger.debug(traceback.format_exc())

            # Add current thread jump link
            try:
                async for first_message in thread.history(oldest_first=True, limit=1):
                    footer_text += f"***Click here to jump to the beginning of this thread: {first_message.jump_url}***"
                    await self.safe_send_message(thread, footer_text)
                    logger.info("Successfully added thread navigation links")
                    break
            except Exception as e:
                logger.error(f"Failed to add thread jump link: {e}")
                logger.debug(traceback.format_exc())

        except Exception as e:
            logger.error(f"Failed to add thread links: {e}")
            logger.debug(traceback.format_exc())

    async def post_summary(self, channel_id, summary: str, source_channel_name: str, message_count: int):
        try:
            source_channel = self.get_channel(channel_id)
            if not source_channel:
                raise DiscordError(f"Could not access source channel {channel_id}")

            logger.info(f"Starting post_summary for channel {source_channel.name} (ID: {channel_id})")
            
            # Retrieve existing thread ID from DB before creating a new thread
            existing_thread_id = self.db.get_summary_thread_id(channel_id)
            previous_thread_id = existing_thread_id  # Store as previous_thread_id
            logger.info(f"Retrieved existing thread ID from DB: {existing_thread_id}")
            
            source_thread = None
            thread_existed = False  # Track if thread existed before
            if existing_thread_id:
                try:
                    logger.info(f"Attempting to fetch existing thread with ID: {existing_thread_id}")
                    source_thread = await self.fetch_channel(existing_thread_id)
                    if isinstance(source_thread, discord.Thread):
                        logger.info(f"Successfully fetched existing thread: {source_thread.name}")
                        thread_existed = True  # Mark that we found an existing thread
                    else:
                        logger.error(f"Fetched channel is not a thread: {existing_thread_id}")
                        source_thread = None
                        self.db.update_summary_thread(channel_id, None)
                except discord.NotFound:
                    logger.error(f"Thread {existing_thread_id} not found")
                    source_thread = None
                    self.db.update_summary_thread(channel_id, None)
                except Exception as e:
                    logger.error(f"Failed to fetch existing thread: {e}")
                    source_thread = None
                    self.db.update_summary_thread(channel_id, None)

            if not source_thread:
                # Only unpin messages when creating a new thread
                pins = await source_channel.pins()
                for pin in pins:
                    if pin.author.id == self.user.id:  # Check if the pin was made by the bot
                        await pin.unpin()
                    
                logger.info("Creating new thread for channel")
                current_date = datetime.utcnow()
                short_month = current_date.strftime('%b')
                thread_message = await source_channel.send(
                    f"📝 A new summary thread has been created for #{source_channel.name} for {datetime.utcnow().strftime('%B %Y')}.\n\n"
                    "All of the messages in this channel will be summarised here for your convenience:"
                )
                await thread_message.pin()  # Pin the new thread message
                
                thread_name = f"{short_month} - #{source_channel_name} Summary"
                source_thread = await thread_message.create_thread(name=thread_name)
                
                logger.info(f"Created new thread with ID: {source_thread.id}")
                self.db.update_summary_thread(channel_id, source_thread.id)
                logger.info(f"Updated DB with new thread ID: {source_thread.id}")

            # Generate short summary and handle main summary channel post
            short_summary = await self.generate_short_summary(summary, message_count)
            
            # Get top attachment for initial message
            all_files = self.attachment_handler.get_all_files_sorted()
            top_attachment = all_files[0] if all_files else None
            
            initial_file = None
            if top_attachment:
                initial_file = discord.File(
                    io.BytesIO(top_attachment.data),
                    filename=top_attachment.filename,
                    description=f"🔥 {top_attachment.reaction_count} reactions"
                )
            
            channel_mention = f"<#{source_channel.id}>"

            # Remove test mode check and use production channel directly
            summary_channel = self.get_channel(self.summary_channel_id)
            initial_message = await self.safe_send_message(
                summary_channel,
                f"## <#{source_channel.id}>\n{short_summary}",
                file=initial_file
            )

            # Track all files used in any part of the summary
            used_files = []
            if initial_file:
                used_files.append(initial_file)

            # Create and process main summary thread
            thread = await self.create_summary_thread(initial_message, source_channel_name)
            topics = summary.split("### ")
            topics = [t.strip().rstrip('#').strip() for t in topics if t.strip()]
            used_message_ids = set()
            
            for i, topic in enumerate(topics):
                topic_used_ids, topic_files = await self.process_topic(thread, topic, is_first=(i == 0))
                used_message_ids.update(topic_used_ids)
                used_files.extend(topic_files)  # Track files used in topics
                
            await self.process_unused_attachments(
                thread, 
                used_message_ids, 
                previous_thread_id=previous_thread_id,
                used_files=used_files  # Pass all used files
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

            await source_thread.send(
                f"## Summary from {current_date}\n\n{summary}",
                file=source_file
            )

            # Process topics for source thread
            for i, topic in enumerate(topics):
                topic_used_ids, topic_files = await self.process_topic(source_thread, topic, is_first=(i == 0))
                used_message_ids.update(topic_used_ids)
                used_files.extend(topic_files)  # Track files used in topics

            # Only difference: limit attachments to 3 in the final section
            await self.process_unused_attachments(source_thread, used_message_ids, max_attachments=3, previous_thread_id=previous_thread_id)

            # After posting the summary to the thread, notify the channel if it was an existing thread
            if thread_existed:
                # Build proper Discord deep link using channel and thread IDs
                thread_link = f"https://discord.com/channels/{self.guild_id}/{source_channel.id}/{source_thread.id}"
                notification_message = f"📝 A new summary has been added for #{source_channel.name}.\n\nYou can see all of {datetime.utcnow().strftime('%B')}'s activity here: {thread_link}"
                await self.safe_send_message(source_channel, notification_message)
                logger.info(f"Posted thread update notification to {source_channel.name}")

        except Exception as e:
            logger.error(f"Error in post_summary: {e}")
            logger.debug(traceback.format_exc())
            raise

    async def generate_summary(self):
        logger.info("generate_summary started")
        logger.info("\nStarting summary generation")
        
        try:
            # Remove test mode check and use production channel directly
            summary_channel = self.get_channel(self.summary_channel_id)
            if not summary_channel:
                raise DiscordError(f"Could not access summary channel {self.summary_channel_id}")
            
            logger.info(f"Found summary channel: {summary_channel.name} "
                       f"({'DEV' if self.dev_mode else 'PRODUCTION'} mode)")
            
            active_channels = False
            date_header_posted = False
            self.first_message = None  # Initialize the instance variable

            # Process categories sequentially
            for category_id in self.category_ids:
                try:
                    category = self.get_channel(category_id)
                    if not category:
                        logger.error(f"Could not access category {category_id}")
                        continue
                    
                    logger.info(f"\nProcessing category: {category.name}")
                    
                    channels = [channel for channel in category.channels 
                              if isinstance(channel, discord.TextChannel)]
                    
                    logger.info(f"Found {len(channels)} channels in category {category.name}")
                    
                    if not channels:
                        logger.warning(f"No text channels found in category {category.name}")
                        continue
                    
                    # Process channels sequentially
                    for channel in channels:
                        try:
                            # Clear attachment cache before processing each channel
                            self.attachment_handler.clear_cache()
                            
                            logger.info(f"Processing channel: {channel.name}")
                            messages = await self.get_channel_history(channel.id)
                            
                            logger.info(f"Channel {channel.name} has {len(messages)} messages")
                            
                            if len(messages) >= 20:  # Only process channels with sufficient activity
                                summary = await self.get_claude_summary(messages)
                                logger.info(f"Generated summary for {channel.name}: {summary[:100]}...")
                                
                                if "[NOTHING OF NOTE]" not in summary:
                                    logger.info(f"Noteworthy activity found in {channel.name}")
                                    active_channels = True
                                    
                                    if not date_header_posted:
                                        current_date = datetime.utcnow()
                                        header = f"# 📅 Daily Summary for {current_date.strftime('%A, %B %d, %Y')}"
                                        self.first_message = await summary_channel.send(header)
                                        date_header_posted = True
                                        logger.info("Posted date header")
                                    
                                    short_summary = await self.generate_short_summary(summary, len(messages))
                                    
                                    # Store in database regardless of mode
                                    self.db.store_daily_summary(
                                        channel_id=channel.id,
                                        channel_name=channel.name,
                                        messages=messages,
                                        full_summary=summary,
                                        short_summary=short_summary
                                    )
                                    logger.info(f"Stored summary in database for {channel.name}")
                                    
                                    await self.post_summary(
                                        channel.id,
                                        summary,
                                        channel.name,
                                        len(messages)
                                    )
                                    await asyncio.sleep(2)
                                else:
                                    logger.info(f"No noteworthy activity in {channel.name}")
                            else:
                                logger.warning(f"Skipping {channel.name} - only {len(messages)} messages")
                            
                        except Exception as e:
                            logger.error(f"Error processing channel {channel.name}: {e}")
                            logger.debug(traceback.format_exc())
                            continue
                        
                except Exception as e:
                    logger.error(f"Error processing category {category_id}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
            
            if not active_channels:
                logger.info("No channels had significant activity - sending notification")
                await summary_channel.send("No channels had significant activity in the past 24 hours.")
            
            # Add footer with jump link at the very end
            if self.first_message:
                footer_text = f"""---

**_Click here to jump to the top of today's summary:_** {self.first_message.jump_url}"""

                await self.safe_send_message(summary_channel, footer_text)
                logger.info("Footer message added to summary channel")
            else:
                logger.error("first_message is not defined. Cannot add footer.")
            
        except Exception as e:
            logger.error(f"Critical error in summary generation: {e}")
            logger.debug(traceback.format_exc())
            raise
        logger.info("generate_summary completed")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user}")

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
            logger.error(f"HTTP error sending message: {e}")
            raise
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def get_reddit_suggestion(self, summary: str) -> Optional[dict]:
        """
        Analyze summary for potential Reddit content.
        Returns dict with title and topic if found, None otherwise.
        """
        logger.info("Analyzing summary for Reddit potential")
        
        prompt = """Analyze this Discord summary and determine if any topic would make for an engaging Reddit post. 

Examples of good Reddit topics:
- Novel technical achievements or discoveries
- Unique creative techniques that others could learn from
- Interesting workflow improvements or optimizations
- Surprising or counter-intuitive findings
- New ways of combining existing tools

Examples of bad Reddit topics:
- Basic questions or troubleshooting
- Personal projects without broader interest
- Bug reports or issues
- General chat or discussions
- Already widely known information

If you find a suitable topic, respond with a proposed Reddit title and the relevant topic text. If not, respond with exactly "NO_REDDIT_CONTENT".

Summary to analyze:
{summary}"""

        try:
            loop = asyncio.get_running_loop()
            
            def create_reddit_analysis():
                return self.claude.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            logger.debug("Sending request to Claude for Reddit analysis")
            response = await loop.run_in_executor(None, create_reddit_analysis)
            result = response.content[0].text.strip()
            
            if result == "NO_REDDIT_CONTENT":
                logger.info("No Reddit-worthy content found")
                return None
            
            # Parse title and content from response
            lines = result.split('\n')
            if len(lines) >= 2:
                suggestion = {
                    'title': lines[0].strip(),
                    'content': '\n'.join(lines[1:]).strip()
                }
                logger.info(f"Found potential Reddit post: {suggestion['title']}")
                return suggestion
            
            logger.warning("Invalid response format from Claude")
            return None
            
        except Exception as e:
            logger.error(f"Error getting Reddit suggestion: {e}")
            logger.debug(traceback.format_exc())
            return None

    async def create_media_content(self, files: List[Tuple[discord.File, int, str, str]], max_media: int = 4) -> Optional[discord.File]:
        """Create either a collage of images or a combined video based on media type."""
        try:
            from PIL import Image
            import moviepy.editor as mp
            import io
            
            logger.info(f"Starting media content creation with {len(files)} files")
            
            images = []
            videos = []
            has_audio = False
            
            # First pass - categorize media and check for audio
            for file_tuple, _, _, _ in files[:max_media]:
                file_tuple.fp.seek(0)
                data = file_tuple.fp.read()
                
                if file_tuple.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    logger.debug(f"Processing image: {file_tuple.filename}")
                    img = Image.open(io.BytesIO(data))
                    images.append(img)
                elif file_tuple.filename.lower().endswith(('.mp4', '.mov', '.webm')):
                    logger.debug(f"Processing video: {file_tuple.filename}")
                    temp_path = f'temp_{len(videos)}.mp4'
                    with open(temp_path, 'wb') as f:
                        f.write(data)
                    video = mp.VideoFileClip(temp_path)
                    if video.audio is not None:
                        has_audio = True
                        logger.debug(f"Video {file_tuple.filename} has audio")
                    videos.append(video)
            
            logger.info(f"Processed {len(images)} images and {len(videos)} videos. Has audio: {has_audio}")
                
            # If we have videos with audio, combine them sequentially
            if videos and has_audio:
                logger.info("Creating combined video with audio")
                final_video = mp.concatenate_videoclips(videos)
                output_path = 'combined_video.mp4'
                final_video.write_videofile(output_path)
                
                # Clean up
                for video in videos:
                    video.close()
                final_video.close()
                
                logger.info("Video combination complete")
                
                # Convert to discord.File
                with open(output_path, 'rb') as f:
                    return discord.File(f, filename='combined_video.mp4')
                
            # If we have videos without audio or images, create a collage
            elif images or (videos and not has_audio):
                logger.info("Creating image/GIF collage")
                
                # Convert silent videos to GIFs for the collage
                for i, video in enumerate(videos):
                    logger.debug(f"Converting video {i+1}/{len(videos)} to GIF")
                    gif_path = f'temp_gif_{len(images)}.gif'
                    video.write_gif(gif_path)
                    gif_img = Image.open(gif_path)
                    images.append(gif_img)
                    video.close()
                
                if not images:
                    logger.warning("No images available for collage")
                    return None
                
                # Calculate collage dimensions
                n = len(images)
                if n == 1:
                    cols, rows = 1, 1
                elif n == 2:
                    cols, rows = 2, 1
                else:
                    cols, rows = 2, 2
                
                logger.debug(f"Creating {cols}x{rows} collage for {n} images")
                
                # Create collage
                target_size = (800 // cols, 800 // rows)
                resized_images = []
                for i, img in enumerate(images):
                    logger.debug(f"Resizing image {i+1}/{len(images)} to {target_size}")
                    img = img.convert('RGB')
                    img.thumbnail(target_size)
                    resized_images.append(img)
                
                collage = Image.new('RGB', (800, 800))
                
                for idx, img in enumerate(resized_images):
                    x = (idx % cols) * (800 // cols)
                    y = (idx // cols) * (800 // rows)
                    collage.paste(img, (x, y))
                
                logger.info("Collage creation complete")
                
                # Convert to discord.File
                buffer = io.BytesIO()
                collage.save(buffer, format='JPEG')
                buffer.seek(0)
                return discord.File(buffer, filename='collage.jpg')
                
            return None
            
        except Exception as e:
            logger.error(f"Error creating media content: {e}")
            logger.debug(traceback.format_exc())
            return None
        finally:
            # Clean up temporary files
            import os
            logger.debug("Cleaning up temporary files")
            for f in os.listdir():
                if f.startswith('temp_'):
                    try:
                        os.remove(f)
                        logger.debug(f"Removed temporary file: {f}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {f}: {e}")

    async def send_reddit_suggestion(self, suggestion: dict, files: List[Tuple[discord.File, int, str, str]]):
        """Send Reddit post suggestion to specified user."""
        try:
            logger.info("Attempting to send Reddit suggestion")
            user = await self.fetch_user(301463647895683072)
            if not user:
                logger.error("Could not find target user")
                return
            
            # Create media content
            media = None
            if files:
                logger.info(f"Creating media content from {len(files)} files")
                media = await self.create_media_content(files)
            
            message = f"**Potential Reddit Post Idea**\n\nTitle: {suggestion['title']}\n\nBased on: {suggestion['content']}"
            
            logger.debug(f"Sending message to user {user.name}")
            await user.send(message, file=media if media else None)
            logger.info("Reddit suggestion sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Reddit suggestion: {e}")
            logger.debug(traceback.format_exc())

async def schedule_daily_summary(bot):
    """
    Schedule daily summaries with error handling and recovery.
    """
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            now = datetime.utcnow()
            target = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
            if now.hour >= 10:
                target += timedelta(days=1)
            
            delay = (target - now).total_seconds()
            logger.info(f"Waiting {delay/3600:.2f} hours until next summary at {target} UTC")
            
            await asyncio.sleep(delay)
            
            await bot.generate_summary()
            logger.info(f"Summary generated successfully at {datetime.utcnow()} UTC")
            # Reset failure counter on success
            consecutive_failures = 0
            
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Error in scheduler (failure {consecutive_failures}): {e}")
            logger.debug(traceback.format_exc())
            
            if consecutive_failures >= max_consecutive_failures:
                logger.critical(f"Too many consecutive failures ({consecutive_failures}). "
                              f"Stopping scheduler.")
                # Optionally notify administrators
                raise
            
            # Wait for a short period before retrying
            await asyncio.sleep(300)  # 5 minutes

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Discord Channel Summarizer Bot')
    parser.add_argument('--run-now', action='store_true', help='Run the summary process immediately instead of waiting for scheduled time')
    parser.add_argument('--dev', action='store_true', help='Run in development mode using test data')
    args = parser.parse_args()
    
    # Load and validate environment variables
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    guild_id = os.getenv('GUILD_ID')
    production_channel_id = os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID')
    categories_to_monitor = os.getenv('CATEGORIES_TO_MONITOR')
    
    if not bot_token:
        raise ValueError("Discord bot token not found in environment variables")
    if not anthropic_key:
        raise ValueError("Anthropic API key not found in environment variables")
    if not guild_id:
        raise ValueError("Guild ID not found in environment variables")
    if not production_channel_id:
        raise ValueError("Production summary channel ID not found in environment variables")
    if not categories_to_monitor:
        raise ValueError("Categories to monitor not found in environment variables")
        
    # Create and run the bot
    bot = ChannelSummarizer()
    bot.dev_mode = args.dev
    bot.load_config()  # Explicitly load config after setting dev_mode
    
    if args.dev:
        logger.info("Running in DEVELOPMENT mode - using test data")
    
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Modify the on_ready event to handle immediate execution if requested
    @bot.event
    async def on_ready():
        logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
        logger.info("Connected to servers: %s", [guild.name for guild in bot.guilds])
        
        if args.run_now:
            logger.info("Running summary process immediately...")
            try:
                await bot.generate_summary()
                logger.info("Summary process completed. Shutting down...")
            finally:
                # Only close sessions and shutdown in immediate mode
                if bot.session and not bot.session.closed:
                    await bot.session.close()
                await bot.close()
        else:
            # Start the scheduler for regular operation
            loop.create_task(schedule_daily_summary(bot))
    
    # Run the bot with proper cleanup
    try:
        loop.run_until_complete(bot.start(bot_token))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - shutting down...")
        # Only do full cleanup on keyboard interrupt
        try:
            # Cancel all running tasks
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            # Close the bot connection
            if not loop.is_closed():
                loop.run_until_complete(bot.close())
        finally:
            logger.info("Closing event loop...")
            loop.close()
    else:
        # In normal operation, just keep running
        loop.run_forever()

if __name__ == "__main__":
    main()