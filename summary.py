import discord
import anthropic
from datetime import datetime, timedelta
import asyncio
import os
from discord.ext import commands
from dotenv import load_dotenv
import io
import time
import aiohttp
import argparse
import re
import logging
import traceback
import random
from typing import List, Tuple, Set
import ssl

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

class ChannelSummarizer(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        intents.members = True
        intents.presences = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Initialize Anthropic client
        self.claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Create aiohttp session
        self.session = None

        # Get configuration from environment variables
        categories_str = os.getenv('CATEGORIES_TO_MONITOR')
        self.category_ids = [int(cat_id) for cat_id in categories_str.split(',')]
        self.summary_channel_id = int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))
        self.test_summary_channel_id = int(os.getenv('TEST_SUMMARY_CHANNEL_ID'))
        self.guild_id = int(os.getenv('GUILD_ID'))
        self.is_test_mode = False  # Will be set via command line arg

        logger.info("Bot initialized with:")
        logger.info(f"Guild ID: {self.guild_id}")
        logger.info(f"Summary Channel: {self.summary_channel_id}")
        logger.info(f"Categories to monitor: {len(self.category_ids)} categories")

        # Add this to your existing __init__
        self.rate_limiter = RateLimiter()

    async def get_channel_history(self, channel_id):
        """
        Retrieve message history for a channel with comprehensive error handling.
        """
        try:
            logger.info(f"Attempting to get history for channel {channel_id}")
            channel = self.get_channel(channel_id)
            if not channel:
                raise DiscordError(f"Could not find channel with ID {channel_id}")
            
            yesterday = datetime.utcnow() - timedelta(days=1)
            messages = []
            self.attachment_cache = {}
            
            async for message in channel.history(after=yesterday, limit=None):
                try:
                    total_reactions = sum(reaction.count for reaction in message.reactions)
                    
                    # Handle attachments
                    if message.attachments and (
                        total_reactions >= 3 or 
                        any(attachment.filename.lower().endswith(('.mp4', '.mov', '.webm')) 
                            for attachment in message.attachments)
                    ):
                        attachments = []
                        for attachment in message.attachments:
                            try:
                                async with self.session.get(attachment.url, timeout=300) as response:
                                    if response.status == 200:
                                        file_data = await response.read()
                                        if len(file_data) <= 25 * 1024 * 1024:
                                            attachments.append({
                                                'filename': attachment.filename,
                                                'data': file_data,
                                                'content_type': attachment.content_type,
                                                'reaction_count': total_reactions
                                            })
                                        else:
                                            logger.warning(f"Skipping large file {attachment.filename} "
                                                         f"({len(file_data)/1024/1024:.2f}MB)")
                                    else:
                                        raise APIError(f"Failed to download attachment: HTTP {response.status}")
                            except aiohttp.ClientError as e:
                                logger.error(f"Network error downloading attachment {attachment.filename}: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"Unexpected error downloading attachment {attachment.filename}: {e}")
                                logger.debug(traceback.format_exc())
                                continue
                        
                        if attachments:
                            self.attachment_cache[str(message.id)] = {
                                'attachments': attachments,
                                'reaction_count': total_reactions
                            }
                    
                    # Process message content
                    content = message.content
                    if content:
                        content = '\n'.join(line for line in content.split('\n') 
                                          if not line.strip().startswith('http'))
                    
                    messages.append({
                        'content': content,
                        'author': message.author.name,
                        'timestamp': message.created_at,
                        'jump_url': message.jump_url,
                        'reactions': total_reactions,
                        'id': str(message.id)
                    })
                    
                except discord.NotFound:
                    logger.warning(f"Message was deleted while processing: {message.id}")
                    continue
                except discord.Forbidden:
                    logger.error(f"No permission to access message: {message.id}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message {message.id}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
            
            logger.info(f"Found {len(messages)} messages in channel {channel.name}")
            return messages
            
        except discord.Forbidden:
            logger.error(f"Bot lacks permissions to access channel {channel_id}")
            raise DiscordError("Insufficient permissions to access channel")
        except discord.NotFound:
            logger.error(f"Channel {channel_id} not found")
            raise DiscordError("Channel not found")
        except discord.HTTPException as e:
            logger.error(f"Discord API error while accessing channel {channel_id}: {e}")
            raise DiscordError(f"Discord API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error accessing channel {channel_id}: {e}")
            logger.debug(traceback.format_exc())
            raise ChannelSummarizerError(f"Unexpected error: {e}")

    async def get_claude_summary(self, messages):
        """
        Generate summary using Claude API with comprehensive error handling.
        """
        if not messages:
            logger.info("No messages to summarize")
            return "[NOTHING OF NOTE]"
        
        logger.info(f"Generating summary for {len(messages)} messages")
        conversation = """Please summarize the interesting and noteworthy Discord happenings and ideas in bullet points. You should extract ideas and information that may be useful to others from conversations AND discord links or URLS. Avoid stuff like bug reports that are circumstantial or not useful to others. Break them into topics and sub-topics with relevant links - Discord links or URLS.

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

Please provide the summary now - don't include any other text:\n\n"""
        
        for msg in messages:
            conversation += f"{msg['timestamp']} - {msg['author']}: {msg['content']}"
            if msg['reactions']:
                conversation += f"\nReactions: {msg['reactions']}"
            conversation += f"\Discord link: {msg['jump_url']}\n\n"
            
        max_retries = 3
        retry_count = 0
        backoff_time = 1  # Initial backoff time in seconds
        
        while retry_count < max_retries:
            try:
                # Add timeout parameter
                response = self.claude.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=8192,
                    messages=[
                        {
                            "role": "user",
                            "content": conversation
                        }
                    ],
                    timeout=60  # 60 second timeout
                )
                
                summary_text = response.content[0].text.strip()
                logger.info("Summary generated successfully")
                return summary_text
                
            except anthropic.APIError as e:
                retry_count += 1
                logger.error(f"Claude API error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    wait_time = backoff_time * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed")
                    raise APIError(f"Claude API failed after {max_retries} attempts: {e}")
                    
            except anthropic.RateLimitError as e:
                retry_count += 1
                logger.warning(f"Rate limit hit (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    wait_time = e.retry_after if hasattr(e, 'retry_after') else backoff_time * (2 ** (retry_count - 1))
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise APIError("Rate limit exceeded and max retries reached")
                    
            except (TimeoutError, ssl.SSLError, ConnectionError) as e:
                retry_count += 1
                logger.error(f"Network error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    wait_time = backoff_time * (2 ** (retry_count - 1))
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise APIError(f"Network error after {max_retries} attempts: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error generating summary: {e}")
                logger.debug(traceback.format_exc())
                raise SummaryError(f"Unexpected error during summary generation: {e}")

        # If we get here, all retries failed
        raise APIError(f"Failed to generate summary after {max_retries} attempts")

    def get_short_summary(self, full_summary, message_count):
        if "[NOTHING OF NOTE]" in full_summary:
            return f"__📨 {message_count} messages sent__\n\nNo significant activity in the past 24 hours"

        conversation = f"""Create exactly 3 bullet points summarizing key developments. STRICT format requirements:
1. The FIRST LINE MUST BE EXACTLY: __📨 {message_count} messages sent__
2. Then three bullet points that:
   - Start with -
   - Bold the most important finding/result/insight using **
   - Keep each to a single line
3. DO NOT MODIFY THE MESSAGE COUNT OR FORMAT IN ANY WAY

Required format:
"__📨 {message_count} messages sent__

• Video Generation shows **45% better performance with new prompting technique**
• Training Process now requires **only 8GB VRAM with optimized pipeline**
• Model Architecture changes **reduce hallucinations by 60%**
"
DO NOT CHANGE THE MESSAGE COUNT LINE. IT MUST BE EXACTLY AS SHOWN ABOVE. DO NOT ADD INCLUDE ELSE IN THE MESSAGE OTHER THAN THE ABOVE.

Full summary to work from:
{full_summary}"""

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.claude.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": conversation
                        }
                    ]
                )
                
                return response.content[0].text.strip()
            except Exception as e:
                retry_count += 1
                logger.error(f"Error attempt {retry_count}/{max_retries} while generating short summary: {e}")
                if retry_count < max_retries:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error("All retry attempts failed")
                    return f"Channel had {message_count} messages in the past 24 hours. Error generating detailed summary."

    async def send_initial_message(self, channel, short_summary: str, source_channel_name: str) -> discord.Message:
        """
        Send the initial summary message to the channel.
        
        Args:
            channel: Discord channel to send message to
            short_summary: Condensed version of the summary
            source_channel_name: Name of the channel being summarized
            
        Returns:
            discord.Message: The sent message object
        """
        try:
            # Get the source channel object to get its ID
            source_channel = discord.utils.get(self.get_all_channels(), name=source_channel_name.strip('#'))
            
            if source_channel:
                message_content = f"## <#{source_channel.id}>\n\n{short_summary}"
            else:
                message_content = f"## {source_channel_name}\n\n{short_summary}"
                
            return await self.safe_send_message(channel, message_content)
            
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
        try:
            current_date = datetime.utcnow()
            thread_name = f"Summary for #{source_channel_name} for {current_date.strftime('%A, %B %d')}"
            thread = await self.safe_create_thread(message, name=thread_name)
            await self.safe_send_message(thread, "──────────────────────")
            return thread
            
        except Exception as e:
            logger.error(f"Failed to create summary thread: {e}")
            logger.debug(traceback.format_exc())
            raise DiscordError(f"Failed to create summary thread: {e}")

    async def prepare_topic_files(self, topic: str) -> List[Tuple[discord.File, int, str]]:
        """
        Prepare files for a given topic.
        
        Args:
            topic: The topic text to process
            
        Returns:
            List of tuples containing (discord.File, reaction_count, message_id)
        """
        topic_files = []
        message_links = re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', topic)
        
        for message_id in message_links:
            if message_id in self.attachment_cache:
                for attachment in self.attachment_cache[message_id]['attachments']:
                    try:
                        if len(attachment['data']) <= 25 * 1024 * 1024:  # 25MB limit
                            file = discord.File(
                                io.BytesIO(attachment['data']),
                                filename=attachment['filename'],
                                description=f"From message ID: {message_id} (🔥 {self.attachment_cache[message_id]['reaction_count']} reactions)"
                            )
                            topic_files.append((
                                file,
                                self.attachment_cache[message_id]['reaction_count'],
                                message_id
                            ))
                    except Exception as e:
                        logger.error(f"Failed to prepare file {attachment['filename']}: {e}")
                        continue
                        
        return sorted(topic_files, key=lambda x: x[1], reverse=True)[:10]  # Limit to top 10 files

    async def send_topic_chunk(self, thread: discord.Thread, chunk: str, files: List[discord.File] = None):
        """
        Send a chunk of topic content to the thread.
        
        Args:
            thread: Thread to send content to
            chunk: Content to send
            files: Optional list of files to attach
        """
        try:
            if files:
                await self.safe_send_message(thread, chunk, files=files)
            else:
                await self.safe_send_message(thread, chunk)
        except Exception as e:
            logger.error(f"Failed to send topic chunk: {e}")
            logger.debug(traceback.format_exc())

    async def process_topic(self, thread: discord.Thread, topic: str, is_first: bool = False) -> None:
        """
        Process and send a single topic to the thread.
        
        Args:
            thread: Thread to send topic to
            topic: Topic content to process
            is_first: Whether this is the first topic
        """
        if not topic.strip():
            return

        # Format topic header
        if is_first:
            formatted_topic = f"### {topic}"
        else:
            formatted_topic = f"---\n### {topic}"

        # Split topic if it's too long
        max_length = 1900
        if len(formatted_topic) > max_length:
            chunks = []
            current_chunk = ""
            current_chunk_links = set()
            
            for line in formatted_topic.split('\n'):
                # Check if line contains a Discord message link
                message_links = re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', line)
                
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

            # Send chunks with their associated files
            for chunk, chunk_links in chunks:
                # Prepare files for this chunk
                chunk_files = []
                for message_id in chunk_links:
                    if message_id in self.attachment_cache:
                        for attachment in self.attachment_cache[message_id]['attachments']:
                            try:
                                if len(attachment['data']) <= 25 * 1024 * 1024:  # 25MB limit
                                    file = discord.File(
                                        io.BytesIO(attachment['data']),
                                        filename=attachment['filename'],
                                        description=f"From message ID: {message_id} (🔥 {self.attachment_cache[message_id]['reaction_count']} reactions)"
                                    )
                                    chunk_files.append(file)
                            except Exception as e:
                                logger.error(f"Failed to prepare file {attachment['filename']}: {e}")
                                continue
                
                # Send chunk with its associated files
                await self.send_topic_chunk(thread, chunk, files=chunk_files[:10] if chunk_files else None)
            
        else:
            # Format topic with horizontal rules
            lines = formatted_topic.split('\n')
            formatted_lines = []
            current_chunk = ""
            current_chunk_links = set()
            chunks = []
            
            for i, line in enumerate(lines):
                # Check for Discord message links in the line
                message_links = re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', line)
                
                if i > 0 and any(line.startswith(emoji) for emoji in ['🎥', '💻', '🎬', '🤖', '📱', '💡', '🔧', '🎨', '📊']):
                    if current_chunk:
                        chunks.append((current_chunk, current_chunk_links))
                    current_chunk = ""
                    current_chunk_links = set()
                    current_chunk += '\n---\n'
                
                current_chunk += line + '\n'
                current_chunk_links.update(message_links)
                
            if current_chunk:
                chunks.append((current_chunk, current_chunk_links))
                
            # Send each chunk with its associated files
            for chunk, chunk_links in chunks:
                # Prepare files for this chunk
                chunk_files = []
                for message_id in chunk_links:
                    if message_id in self.attachment_cache:
                        for attachment in self.attachment_cache[message_id]['attachments']:
                            try:
                                if len(attachment['data']) <= 25 * 1024 * 1024:  # 25MB limit
                                    file = discord.File(
                                        io.BytesIO(attachment['data']),
                                        filename=attachment['filename'],
                                        description=f"From message ID: {message_id} (🔥 {self.attachment_cache[message_id]['reaction_count']} reactions)"
                                    )
                                    chunk_files.append(file)
                            except Exception as e:
                                logger.error(f"Failed to prepare file {attachment['filename']}: {e}")
                                continue
                
                # Send chunk with its associated files
                await self.send_topic_chunk(thread, chunk, files=chunk_files[:10] if chunk_files else None)

        await asyncio.sleep(1)  # Prevent rate limiting

    async def process_unused_attachments(self, thread: discord.Thread, used_message_ids: Set[str]):
        """
        Process and send any unused but popular attachments.
        
        Args:
            thread: Thread to send attachments to
            used_message_ids: Set of message IDs that have already been processed
        """
        unused_attachments = []
        for message_id, cache_data in self.attachment_cache.items():
            if message_id not in used_message_ids and cache_data['reaction_count'] >= 3:
                for attachment in cache_data['attachments']:
                    try:
                        if len(attachment['data']) <= 25 * 1024 * 1024:
                            file = discord.File(
                                io.BytesIO(attachment['data']),
                                filename=attachment['filename'],
                                description=f"From message ID: {message_id} (🔥 {cache_data['reaction_count']} reactions)"
                            )
                            unused_attachments.append((file, cache_data['reaction_count']))
                    except Exception as e:
                        logger.error(f"Failed to prepare unused attachment: {e}")
                        continue

        if unused_attachments:
            unused_attachments.sort(key=lambda x: x[1], reverse=True)
            files = [file for file, _ in unused_attachments[:10]]
            await self.safe_send_message(thread, "**📎 Other Popular Attachments**", files=files)

    async def post_summary(self, channel_id: int, summary: str, source_channel_name: str, message_count: int):
        """
        Post a complete summary to a Discord channel.
        
        Args:
            channel_id: ID of the channel to post to
            summary: Full summary content
            source_channel_name: Name of the source channel
            message_count: Number of messages processed
        """
        logger.info(f"Attempting to post summary to channel {channel_id}")
        
        try:
            target_channel_id = self.test_summary_channel_id if self.is_test_mode else self.summary_channel_id
            channel = self.get_channel(target_channel_id)
            
            if not channel:
                raise DiscordError(f"Could not find channel with ID {target_channel_id}")

            # Generate and post initial summary
            short_summary = self.get_short_summary(summary, message_count)
            initial_message = await self.send_initial_message(channel, short_summary, source_channel_name)
            
            # Create thread for detailed summary
            thread = await self.create_summary_thread(initial_message, source_channel_name)

            # Process topics
            topics = summary.split("### ")
            topics = [topic.strip().rstrip('#').strip() for topic in topics if topic.strip()]
            
            used_message_ids = set()
            for i, topic in enumerate(topics):
                # Collect message IDs from this topic
                used_message_ids.update(re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', topic))
                
                # Process the topic
                await self.process_topic(thread, topic, is_first=(i == 0))

            # Handle unused attachments
            await self.process_unused_attachments(thread, used_message_ids)

            logger.info(f"Successfully posted summary to {channel.name}")
            
        except Exception as e:
            logger.error(f"Failed to post summary: {e}")
            logger.debug(traceback.format_exc())
            raise

    async def generate_summary(self):
        """
        Main summary generation loop with error handling.
        """
        logger.info("\nStarting summary generation")
        
        try:
            channel_id = self.test_summary_channel_id if self.is_test_mode else self.summary_channel_id
            summary_channel = self.get_channel(channel_id)
            if not summary_channel:
                raise DiscordError(f"Could not access summary channel {channel_id}")
            
            logger.info(f"Found summary channel: {summary_channel.name} "
                       f"({'TEST' if self.is_test_mode else 'PRODUCTION'} mode)")
            
            active_channels = False
            date_header_posted = False
            
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
                    
                    if not channels:
                        logger.warning(f"No text channels found in category {category.name}")
                        continue
                    
                    # Process channels sequentially
                    for channel in channels:
                        try:
                            messages = await self.get_channel_history(channel.id)
                            
                            if len(messages) >= 20:  # Only process channels with sufficient activity
                                summary = await self.get_claude_summary(messages)
                                
                                if "[NOTHING OF NOTE]" not in summary:
                                    if not date_header_posted:
                                        await self.post_date_header(summary_channel)
                                        date_header_posted = True
                                    
                                    active_channels = True
                                    await self.post_summary(
                                        self.summary_channel_id, 
                                        summary, 
                                        channel.name, 
                                        len(messages)
                                    )
                                    # Add delay between channels to prevent rate limiting
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
                await summary_channel.send("No channels had significant activity in the past 24 hours.")
            
        except Exception as e:
            logger.error(f"Critical error in summary generation: {e}")
            logger.debug(traceback.format_exc())
            raise

    async def setup_hook(self):
        logger.info("Setup hook started")
        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()
        logger.info("Setup hook completed")

    async def close(self):
        logger.info("Closing bot...")
        # Close aiohttp session if it exists
        if self.session:
            await self.session.close()
        # Call parent's close method
        await super().close()

    async def post_date_header(self, channel):
        """
        Posts a date header for the daily summary.
        
        Args:
            channel (discord.TextChannel): The channel to post the header in.
        """
        try:
            current_date = datetime.utcnow()
            header = f"# 📅 Daily Summary for {current_date.strftime('%A, %B %d, %Y')}"
            await channel.send(header)
            logger.info(f"Posted date header in {channel.name}")
        except discord.Forbidden:
            logger.error(f"No permission to post in channel {channel.name}")
            raise DiscordError("Cannot post date header - insufficient permissions")
        except discord.HTTPException as e:
            logger.error(f"Failed to post date header: {e}")
            raise DiscordError(f"Failed to post date header: {e}")
        except Exception as e:
            logger.error(f"Unexpected error posting date header: {e}")
            logger.debug(traceback.format_exc())
            raise ChannelSummarizerError(f"Failed to post date header: {e}")

    async def safe_send_message(self, channel, content=None, **kwargs):
        """
        Safely send a message with rate limit handling.
        """
        return await self.rate_limiter.execute(
            f"send_message_{channel.id}",
            channel.send(content, **kwargs)
        )

    async def safe_create_thread(self, message, **kwargs):
        """
        Safely create a thread with rate limit handling.
        """
        return await self.rate_limiter.execute(
            f"create_thread_{message.channel.id}",
            message.create_thread(**kwargs)
        )

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
    parser.add_argument('--test', action='store_true', help='Run in test mode using test channel')
    args = parser.parse_args()
    
    # Load and validate environment variables
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    guild_id = os.getenv('GUILD_ID')
    production_channel_id = os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID')
    test_channel_id = os.getenv('TEST_SUMMARY_CHANNEL_ID')
    categories_to_monitor = os.getenv('CATEGORIES_TO_MONITOR')
    
    if not bot_token:
        raise ValueError("Discord bot token not found in environment variables")
    if not anthropic_key:
        raise ValueError("Anthropic API key not found in environment variables")
    if not guild_id:
        raise ValueError("Guild ID not found in environment variables")
    if not production_channel_id:
        raise ValueError("Production summary channel ID not found in environment variables")
    if not test_channel_id:
        raise ValueError("Test summary channel ID not found in environment variables")
    if not categories_to_monitor:
        raise ValueError("Categories to monitor not found in environment variables")
        
    # Create and run the bot
    bot = ChannelSummarizer()
    bot.is_test_mode = args.test
    
    if args.test:
        logger.info("Running in TEST mode - summaries will be sent to test channel")
    
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Modify the on_ready event to handle immediate execution if requested
    @bot.event
    async def on_ready():
        logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
        logger.info("Connected to servers: %s", [guild.name for guild in bot.guilds])
        
        if args.run_now:
            logger.info("Running summary process immediately...")
            await bot.generate_summary()
            logger.info("Summary process completed. Shutting down...")
            await bot.close()
        else:
            # Start the scheduler for regular operation
            loop.create_task(schedule_daily_summary(bot))
    
    # Run the bot
    try:
        loop.run_until_complete(bot.start(bot_token))
    except KeyboardInterrupt:
        loop.run_until_complete(bot.close())
    finally:
        loop.close()

if __name__ == "__main__":
    main()