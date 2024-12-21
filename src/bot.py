import logging
import os
import re
import traceback
import random
import asyncio
from typing import Set

import discord
from discord.ext import commands
import aiohttp

import anthropic

from utils.errors import (
    ChannelSummarizerError,
    APIError,
    DiscordError,
    SummaryError,
    ConfigurationError,
    MediaProcessingError,
    DatabaseError
)
from utils.error_handler import ErrorHandler, handle_errors
from src.services.attachment_service import AttachmentService
from src.services.summary_service import SummaryService
from src.services.reddit_service import RedditService
from src.db_handler import DatabaseHandler

logger = logging.getLogger('ChannelSummarizer')

class RateLimiter:
    """Manages rate limiting for Discord API calls with exponential backoff."""
    def __init__(self):
        self.backoff_times = {}
        self.base_delay = 1.0
        self.max_delay = 64.0
        self.jitter = 0.1

    async def execute(self, key, coroutine):
        max_retries = 5
        attempt = 0

        while attempt < max_retries:
            try:
                # Add jitter to prevent thundering herd
                if key in self.backoff_times:
                    jitter = random.uniform(-self.jitter, self.jitter)
                    await asyncio.sleep(self.backoff_times[key] * (1 + jitter))

                result = await coroutine
                self.backoff_times[key] = self.base_delay  # Reset on success
                return result

            except discord.HTTPException as e:
                attempt += 1
                if e.status == 429:  # Rate limit
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        logger.warning(f"Rate limit hit for {key}. Retry after {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
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

        super().__init__(
            command_prefix="!",
            intents=intents,
            heartbeat_timeout=60.0,
            guild_ready_timeout=10.0,
            gateway_queue_size=512
        )

        # Initialize Anthropics client
        self.claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        # Create aiohttp session
        self.session = None

        # Get configuration
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

        self.rate_limiter = RateLimiter()

        # Services
        self.attachment_service = AttachmentService()
        self.summary_service = SummaryService(self.claude)
        self.reddit_service = RedditService(self.claude)

        # Database
        self.db = DatabaseHandler()

        # Error handler
        self.error_handler = None

    async def setup_hook(self):
        """Runs after bot is connected."""
        try:
            self.session = aiohttp.ClientSession()
            notification_channel = self.get_channel(self.summary_channel_id)
            admin_user = await self.fetch_user(301463647895683072)
            self.error_handler = ErrorHandler(notification_channel, admin_user)
        except Exception as e:
            raise ConfigurationError("Failed to initialize bot", e)

    @handle_errors("safe_send_message")
    async def safe_send_message(self, channel, content=None, embed=None, file=None, files=None, reference=None):
        """Safely send a message with retry logic."""
        async def send_message():
            return await channel.send(
                content=content,
                embed=embed,
                file=file,
                files=files,
                reference=reference
            )
        
        return await self.rate_limiter.execute(
            f"channel_{channel.id}",
            send_message()
        )

    @handle_errors("get_channel_history")
    async def get_channel_history(self, channel_id):
        """
        Retrieve message history for a channel with comprehensive error handling.
        """
        try:
            logger.info(f"Attempting to get history for channel {channel_id}")
            channel = self.get_channel(channel_id)
            if not channel:
                raise DiscordError(f"Could not find channel with ID {channel_id}")

            # Skip channels with 'support' in the name
            if 'support' in channel.name.lower():
                logger.info(f"Skipping support channel: {channel.name}")
                return []

            from datetime import datetime, timedelta
            yesterday = datetime.utcnow() - timedelta(days=1)
            messages = []

            logger.info(f"Fetching messages after {yesterday} for channel {channel.name}")

            async for message in channel.history(after=yesterday, limit=None):
                try:
                    if message.attachments:
                        for attachment in message.attachments:
                            await self.attachment_service.process_attachment(
                                attachment, message, self.session
                            )

                    # Count unique reactors
                    unique_reactors = set()
                    for reaction in message.reactions:
                        async for user in reaction.users():
                            unique_reactors.add(user.id)
                    total_unique_reactors = len(unique_reactors)

                    content = message.content
                    if content:
                        # Remove lines that are purely links
                        content = '\n'.join(
                            line for line in content.split('\n')
                            if not line.strip().startswith('http')
                        )

                    messages.append({
                        'content': content,
                        'author': message.author.name,
                        'timestamp': message.created_at,
                        'jump_url': message.jump_url,
                        'reactions': total_unique_reactors,
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

    async def create_summary_thread(self, message, source_channel_name):
        """Create a thread for the summary."""
        from datetime import datetime
        if not message:
            logger.error("Cannot create thread: message is None")
            raise DiscordError("Cannot create thread: message is None")

        current_date = datetime.utcnow()
        thread_name = f"Summary for #{source_channel_name} for {current_date.strftime('%A, %B %d')}"

        try:
            # Create thread with proper rate limiting
            async def create_thread():
                return await message.create_thread(
                    name=thread_name,
                    auto_archive_duration=1440,  # 24 hours
                    reason=f"Daily summary thread for {source_channel_name}"
                )
            
            thread = await self.rate_limiter.execute(
                f"thread_create_{message.channel.id}",
                create_thread()
            )
            
            logger.info(f"Thread created successfully: {thread.id}")
            
            # Wait briefly after thread creation
            await asyncio.sleep(1)
            
            return thread

        except discord.Forbidden as e:
            logger.error(f"Bot lacks permissions to create thread: {e}")
            logger.error(f"Bot permissions: {message.guild.me.guild_permissions}")
            raise DiscordError("Insufficient permissions to create thread")
        
        except discord.HTTPException as e:
            if e.code == 30033:  # Too many threads
                logger.error("Too many active threads in channel")
                raise DiscordError("Too many active threads in channel")
            elif e.code == 50035:  # Invalid thread name
                logger.error(f"Invalid thread name: {thread_name}")
                # Try with a simpler name
                thread_name = f"Summary for {source_channel_name}"
                return await self.create_summary_thread(message, source_channel_name)
            else:
                logger.error(f"Failed to create thread: {e} (Code: {e.code}, Status: {e.status})")
                raise DiscordError(f"Failed to create thread: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error creating thread: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def process_topic(self, thread: discord.Thread, topic: str, is_first: bool = False) -> Set[str]:
        """Process and send a single topic to the thread."""
        from src.services.summary_service import MessageFormatter  # or keep it in summary_service
        if not topic.strip():
            return set()

        formatted_topic = ("---\n" if not is_first else "") + f"### {topic}"
        logger.info(f"Processing topic: {formatted_topic[:100]}...")

        used_message_ids = set()
        chunks = MessageFormatter.chunk_content(formatted_topic)

        for chunk_content, chunk_links in chunks:
            message_ids = set(
                re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', chunk_content)
            )
            used_message_ids.update(message_ids)

            # Attachments for this chunk
            chunk_files = self.attachment_service.prepare_chunk_files(message_ids)
            if chunk_files:
                await self.safe_send_message(thread, chunk_content)
                for file_tuple in chunk_files:
                    try:
                        await self.safe_send_message(thread, file=file_tuple[0])
                    except Exception as e:
                        logger.error(f"Failed to send attachment: {e}")
            else:
                await self.safe_send_message(thread, chunk_content)

        return used_message_ids

    async def process_unused_attachments(self, thread: discord.Thread, used_message_ids: Set[str], main_attachment_id: str = None):
        """
        Process and send any unused but popular attachments.
        Excludes the main attachment that was used in the thread header.
        """
        unused_attachments = self.attachment_service.get_unused_popular_attachments(
            used_message_ids, 
            self.guild_id,
            main_attachment_id
        )

        if unused_attachments:
            await self.safe_send_message(thread, "\n\n---\n### 📎 Other Popular Attachments")

            for att in unused_attachments[:10]:
                msg_content = f"By **{att['username']}**: {att['message_link']}"
                await self.safe_send_message(thread, msg_content, file=att['file'])

        # Attempt to show start-of-thread jump link
        async for first_message in thread.history(oldest_first=True, limit=1):
            await self.safe_send_message(
                thread,
                f"---\n\u200B\n***Click here to jump to the beginning of this thread: {first_message.jump_url}***"
            )
            break

    async def post_summary(self, channel, summary: str, source_channel_name: str, message_count: int):
        """Post a complete summary to a Discord channel."""
        try:
            # Generate and post initial summary
            short_summary = await self.summary_service.generate_short_summary(summary, message_count)
            initial_message = await self.safe_send_message(
                channel,
                f"## #{source_channel_name}\n{short_summary}"
            )
            
            if not initial_message:
                logger.error("Failed to create initial message - cannot create thread")
                return
            
            # Create thread and add debug logging
            logger.info(f"Creating thread for channel {source_channel_name}")
            thread = await self.create_summary_thread(initial_message, source_channel_name)
            logger.info(f"Thread created: {thread.id}")
            
            # Process topics
            topics = summary.split("### ")
            topics = [t.strip().rstrip('#').strip() for t in topics if t.strip()]
            
            used_message_ids = set()

            # Post topics in the newly created thread
            for i, topic in enumerate(topics):
                logger.debug(f"Processing topic {i+1}/{len(topics)}")
                topic_used_ids = await self.process_topic(thread, topic, is_first=(i == 0))
                used_message_ids.update(topic_used_ids)

            # Process remaining attachments in the new thread
            await self.process_unused_attachments(thread, used_message_ids)
            
            logger.info(f"Successfully posted summary to thread for {source_channel_name}")

        except Exception as e:
            logger.error(f"Error in post_summary: {e}")
            logger.debug(traceback.format_exc())
            raise

    async def generate_summary(self):
        """Generate the daily summary."""
        from datetime import datetime, timedelta
        logger.info("generate_summary started")

        try:
            # Identify target channel
            channel_id = self.test_summary_channel_id if self.is_test_mode else self.summary_channel_id
            summary_channel = self.get_channel(channel_id)
            if not summary_channel:
                raise DiscordError(f"Could not access summary channel {channel_id}")

            logger.info(f"Found summary channel: {summary_channel.name} "
                        f"({'TEST' if self.is_test_mode else 'PRODUCTION'} mode)")

            active_channels = False
            date_header_posted = False

            for category_id in self.category_ids:
                category = self.get_channel(category_id)
                if not category:
                    logger.error(f"Could not access category {category_id}")
                    continue

                logger.info(f"\nProcessing category: {category.name}")

                channels = [ch for ch in category.channels if isinstance(ch, discord.TextChannel)]
                logger.info(f"Found {len(channels)} channels in category {category.name}")

                for channel in channels:
                    # Clear attachment cache each time
                    self.attachment_service.clear_cache()

                    logger.info(f"Processing channel: {channel.name}")
                    messages = await self.get_channel_history(channel.id)
                    logger.info(f"Channel {channel.name} has {len(messages)} messages")

                    if len(messages) < 20:
                        logger.info(f"Skipping channel {channel.name} (only {len(messages)} messages)")
                        continue

                    # Generate summary with the SummaryService
                    summary = await self.summary_service.generate_summary(messages)

                    if "[NOTHING OF NOTE]" not in summary:
                        active_channels = True

                        # If we haven't posted a date header, do it once
                        if not date_header_posted:
                            current_date = datetime.utcnow()
                            header = f"# 📅 Daily Summary for {current_date.strftime('%A, %B %d, %Y')}"
                            await summary_channel.send(header)
                            date_header_posted = True

                        short_summary = await self.summary_service.generate_short_summary(
                            summary, len(messages)
                        )

                        # Only store if not in test mode
                        if not self.is_test_mode:
                            self.db.store_daily_summary(
                                channel_id=channel.id,
                                channel_name=channel.name,
                                messages=messages,
                                full_summary=summary,
                                short_summary=short_summary
                            )
                            logger.info(f"Stored summary in DB for channel {channel.name}")

                        await self.post_summary(
                            summary_channel,
                            summary,
                            channel.name,
                            len(messages)
                        )
                    else:
                        logger.info(f"No noteworthy activity in channel {channel.name}")

            if not active_channels:
                logger.info("No channels had significant activity - sending notification")
                await summary_channel.send("No channels had significant activity in the past 24 hours.")

        except Exception as e:
            logger.error(f"Critical error in summary generation: {e}")
            logger.debug(traceback.format_exc())
            raise
        logger.info("generate_summary completed")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user}")