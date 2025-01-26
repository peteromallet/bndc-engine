import os
import sys
import argparse
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discord
from discord.ext import commands
import asyncio
from datetime import datetime, timedelta, timezone
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
from src.common.db_handler import DatabaseHandler
from src.common.constants import get_database_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('archive_discord.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MessageArchiver(commands.Bot):
    def __init__(self, dev_mode=False, order="newest", days=None, batch_size=500, in_depth=False):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.messages = True  # This is the correct attribute for message history
        intents.reactions = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Load environment variables
        load_dotenv()
        
        # Set database path based on mode
        self.db_path = get_database_path(dev_mode)
        self.db = None
        
        # Check if token exists
        if not os.getenv('DISCORD_BOT_TOKEN'):
            raise ValueError("DISCORD_BOT_TOKEN not found in environment variables")
        
        # Add reconnect settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Get bot user ID from env
        self.bot_user_id = int(os.getenv('BOT_USER_ID'))
        
        # Get guild ID based on mode
        self.guild_id = int(os.getenv('DEV_GUILD_ID' if dev_mode else 'GUILD_ID'))
        
        # Channels to skip
        self.skip_channels = {1076117621407223832}  # Welcome channel
        
        # Default config for all channels
        self.default_config = {
            'batch_size': batch_size,
            'delay': 0.25
        }
        
        # Set message ordering
        self.oldest_first = order.lower() == "oldest"
        logger.info(f"Message ordering: {'oldest to newest' if self.oldest_first else 'newest to oldest'}")
        
        # Set days limit
        self.days_limit = days
        if days:
            logger.info(f"Will fetch messages from the last {days} days")
        else:
            logger.info("Will fetch all available messages")
            
        # Set in-depth mode
        self.in_depth = in_depth
        if in_depth:
            logger.info("Running in in-depth mode - will perform thorough message checks")
        
        # Add rate limiting tracking
        self.last_api_call = datetime.now()
        self.api_call_count = 0
        self.rate_limit_reset = datetime.now()
        self.rate_limit_remaining = 50  # Conservative initial limit

    async def setup_hook(self):
        """Setup hook to initialize database and start archiving."""
        try:
            # Create fresh connection when starting up
            self.db = DatabaseHandler(self.db_path)
            self.db.init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _ensure_db_connection(self):
        """Ensure database connection is alive and reconnect if needed."""
        try:
            # Test the connection
            if not self.db or not self.db.conn:
                logger.info("Database connection lost, reconnecting...")
                self.db = DatabaseHandler(self.db_path)
                self.db.init_db()
                logger.info("Successfully reconnected to database")
            else:
                # Test if connection is actually working
                self.db.cursor.execute("SELECT 1")
        except Exception as e:
            logger.warning(f"Database connection test failed, reconnecting: {e}")
            try:
                if self.db:
                    self.db.close()
                self.db = DatabaseHandler(self.db_path)
                self.db.init_db()
                logger.info("Successfully reconnected to database")
            except Exception as e:
                logger.error(f"Failed to reconnect to database: {e}")
                raise

    async def on_ready(self):
        """Called when bot is ready."""
        try:
            logger.info(f"Logged in as {self.user}")
            
            # Get the guild
            guild = self.get_guild(self.guild_id)
            if not guild:
                logger.error(f"Could not find guild with ID {self.guild_id}")
                await self.close()
                return
            
            # Archive all text channels in the guild
            for channel in guild.text_channels:
                if channel.id not in self.skip_channels:
                    logger.info(f"Starting archive of text channel #{channel.name}")
                    await self.archive_channel(channel.id)
            
            # Debug logging for forum channels
            logger.info(f"Found {len(guild.forums)} forum channels")
            for forum in guild.forums:
                logger.info(f"Found forum channel: #{forum.name} (ID: {forum.id})")
            
            # Archive all forum channels and their threads
            for forum in guild.forums:
                if forum.id not in self.skip_channels:
                    logger.info(f"Starting archive of forum channel #{forum.name}")
                    # Archive the forum posts (threads)
                    thread_count = 0
                    async for thread in forum.archived_threads():
                        thread_count += 1
                        logger.info(f"Starting archive of forum thread #{thread.name} in {forum.name}")
                        await self.archive_channel(thread.id)
                    # Archive active threads
                    for thread in forum.threads:
                        thread_count += 1
                        logger.info(f"Starting archive of active forum thread #{thread.name} in {forum.name}")
                        await self.archive_channel(thread.id)
                    logger.info(f"Processed {thread_count} total threads in forum #{forum.name}")
            
            # Archive threads in text channels
            for channel in guild.text_channels:
                if channel.id not in self.skip_channels:
                    logger.info(f"Checking for threads in #{channel.name}")
                    # Archive archived threads
                    async for thread in channel.archived_threads():
                        logger.info(f"Starting archive of thread #{thread.name} in {channel.name}")
                        await self.archive_channel(thread.id)
                    # Archive active threads
                    for thread in channel.threads:
                        logger.info(f"Starting archive of active thread #{thread.name} in {channel.name}")
                        await self.archive_channel(thread.id)
            
            logger.info("Archiving complete, shutting down bot")
            # Close the bot after archiving
            await self.close()
        except Exception as e:
            logger.error(f"Error in on_ready: {e}")
            await self.close()

    async def _wait_for_rate_limit(self):
        """Handles rate limiting for Discord API calls."""
        now = datetime.now()
        self.api_call_count += 1
        
        time_since_last = (now - self.last_api_call).total_seconds()
        
        # Basic throttling - ensure at least 0.1s between calls
        if time_since_last < 0.1:
            await asyncio.sleep(0.1 - time_since_last)
            logger.debug(f"Basic throttle - {time_since_last:.3f}s since last call")
        
        # Only enforce rate limits if we're actually approaching them
        if self.api_call_count >= 45:  # Conservative buffer before hitting 50
            wait_time = 1.0  # Start with a 1s pause
            logger.info(f"Rate limit approaching - Current count: {self.api_call_count}, Remaining: {self.rate_limit_remaining}")
            await asyncio.sleep(wait_time)
            self.api_call_count = 0
            self.rate_limit_reset = datetime.now() + timedelta(seconds=60)
            self.rate_limit_remaining = 50
            logger.info(f"Rate limit reset - New remaining: {self.rate_limit_remaining}, Next reset: {self.rate_limit_reset}")
        
        self.last_api_call = now

    async def archive_channel(self, channel_id: int) -> None:
        """Archive all messages from a channel."""
        channel_start_time = datetime.now()
        try:
            # Skip welcome channel
            if channel_id in self.skip_channels:
                logger.info(f"Skipping welcome channel {channel_id}")
                return
            
            # Ensure DB connection is healthy
            self._ensure_db_connection()
            
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Could not find channel {channel_id}")
                return
            
            # Get the actual channel (parent forum if this is a thread)
            actual_channel = channel
            if hasattr(channel, 'parent') and channel.parent:
                actual_channel = channel.parent
                logger.debug(f"Using parent forum #{actual_channel.name} (ID: {actual_channel.id}) for thread #{channel.name}")
                # Update channel_id to use parent forum's ID
                channel_id = actual_channel.id
            
            logger.info(f"Starting archive of #{channel.name} at {channel_start_time}")
            
            # Calculate the cutoff date if days limit is set
            cutoff_date = None
            if self.days_limit:
                # Make sure to create timezone-aware datetime
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.days_limit)
                logger.info(f"Will only fetch messages after {cutoff_date}")
            
            # Initialize dates as None
            earliest_date = None
            latest_date = None
            
            try:
                # Get date range of archived messages
                earliest_date, latest_date = self.db.get_message_date_range(channel_id)
                # Make sure dates are timezone-aware
                if earliest_date:
                    earliest_date = earliest_date.replace(tzinfo=timezone.utc)
                    logger.info(f"Earliest message in DB for #{channel.name}: {earliest_date}")
                if latest_date:
                    latest_date = latest_date.replace(tzinfo=timezone.utc)
                    logger.info(f"Latest message in DB for #{channel.name}: {latest_date}")
            except Exception as e:
                logger.warning(f"Could not get message date range, will fetch all messages: {e}")
            
            message_count = 0
            batch = []
            
            # If no archived messages exist or we're in in-depth mode, get all messages in the time range
            if not earliest_date or not latest_date or self.in_depth:
                if self.in_depth:
                    logger.info(f"In-depth mode: Re-checking all messages in time range for #{channel.name}")
                else:
                    logger.info(f"No existing archives found for #{channel.name}. Getting all messages...")
                
                message_counter = 0
                logger.info(f"Starting message fetch for #{channel.name} from {'oldest to newest' if self.oldest_first else 'newest to oldest'}...")
                try:
                    # We'll paginate through messages using before/after
                    last_message = None
                    while True:
                        history_kwargs = {
                            'limit': None,  # No limit, we'll control the flow ourselves
                            'oldest_first': self.oldest_first,
                            'before': last_message.created_at if last_message else None,
                            'after': cutoff_date if cutoff_date else None
                        }
                        
                        logger.info(f"Fetching messages for #{channel.name} with kwargs: {history_kwargs}")
                        current_batch = []
                        
                        try:
                            got_messages = False
                            async for message in channel.history(**{k: v for k, v in history_kwargs.items() if v is not None}):
                                got_messages = True
                                last_message = message
                                
                                # Process message
                                message_counter += 1
                                if message_counter % 25 == 0:
                                    logger.info(f"Fetched {message_counter} messages so far from #{channel.name}, last message from {message.created_at}")
                                
                                try:
                                    # Skip messages from the bot
                                    if message.author.id == self.bot_user_id:
                                        continue
                                    
                                    # In in-depth mode, always process the message
                                    # In normal mode, only process if not already in DB
                                    if self.in_depth or not self.db.message_exists(message.id):
                                        current_batch.append(message)
                                    
                                    # Store batch when it reaches the threshold
                                    if len(current_batch) >= 100:
                                        try:
                                            processed_messages = []
                                            for msg in current_batch:
                                                processed_msg = await self._process_message(msg, channel_id)
                                                if processed_msg:
                                                    processed_messages.append(processed_msg)
                                            
                                            if processed_messages:
                                                logger.info(f"Storing batch of {len(processed_messages)} messages from #{channel.name}")
                                                self.db.store_messages(processed_messages)
                                                message_count += len(processed_messages)
                                                logger.info(f"Successfully stored batch of {len(processed_messages)} messages from #{channel.name} (total processed: {message_count})")
                                            
                                            current_batch = []
                                            await asyncio.sleep(0.1)
                                        except Exception as e:
                                            logger.error(f"Failed to store batch: {e}")
                                            logger.error(f"Error details: {str(e)}")
                                            
                                except Exception as e:
                                    logger.error(f"Error processing message {message.id}: {e}")
                                    logger.error(f"Error details: {str(e)}")
                                    continue
                                
                        except discord.Forbidden:
                            logger.warning(f"Missing permissions to read messages in #{channel.name}")
                            break
                        except Exception as e:
                            logger.error(f"Error fetching messages: {e}")
                            break
                        
                        # Process any remaining messages in the current batch
                        if current_batch:
                            try:
                                processed_messages = []
                                for msg in current_batch:
                                    processed_msg = await self._process_message(msg, channel_id)
                                    if processed_msg:
                                        processed_messages.append(processed_msg)
                                
                                if processed_messages:
                                    logger.info(f"Storing final batch of {len(processed_messages)} messages from #{channel.name}")
                                    self.db.store_messages(processed_messages)
                                    message_count += len(processed_messages)
                            except Exception as e:
                                logger.error(f"Failed to store final batch: {e}")
                                logger.error(f"Error details: {str(e)}")
                        
                        # If we didn't get any messages in this fetch, break the loop
                        if not got_messages:
                            logger.info(f"No more messages found in #{channel.name} for the current time range")
                            break
                            
                        await asyncio.sleep(0.1)
                    
                    logger.info(f"Finished initial fetch for #{channel.name}: {message_counter} messages fetched, last message from {last_message.created_at if last_message else 'N/A'}")
                except Exception as e:
                    logger.error(f"Error fetching message history: {e}")
                    logger.error(f"Error details: {str(e)}")
                    raise

            # Still check before earliest and after latest, respecting days limit
            if latest_date:
                logger.info(f"Searching for newer messages in #{channel.name} (after {latest_date})...")
                current_batch = []
                messages_found = 0
                async for message in channel.history(limit=None, after=latest_date, oldest_first=self.oldest_first):
                    messages_found += 1
                    if messages_found % 100 == 0:
                        logger.info(f"Found {messages_found} newer messages in #{channel.name}")
                    if cutoff_date and message.created_at < cutoff_date:
                        logger.debug(f"Reached cutoff date {cutoff_date}, stopping newer message search")
                        break
                    
                    # Skip messages from the bot
                    if message.author.id == self.bot_user_id:
                        continue
                        
                    current_batch.append(message)
                    message_counter += 1
                    
                    # Store batch when it reaches the threshold
                    if len(current_batch) >= 100:
                        try:
                            processed_messages = []
                            for msg in current_batch:
                                processed_msg = await self._process_message(msg, channel_id)
                                if processed_msg:
                                    processed_messages.append(processed_msg)
                            
                            if processed_messages:
                                logger.info(f"Storing batch of {len(processed_messages)} messages from #{channel.name}")
                                self.db.store_messages(processed_messages)
                                message_count += len(processed_messages)
                                logger.info(f"Successfully stored batch of {len(processed_messages)} messages from #{channel.name} (total processed: {message_count})")
                            
                            current_batch = []
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.error(f"Failed to store batch: {e}")
                            logger.error(f"Error details: {str(e)}")
                
                # Process any remaining messages
                if current_batch:
                    try:
                        processed_messages = []
                        for msg in current_batch:
                            processed_msg = await self._process_message(msg, channel_id)
                            if processed_msg:
                                processed_messages.append(processed_msg)
                        
                        if processed_messages:
                            logger.info(f"Storing batch of {len(processed_messages)} messages from #{channel.name}")
                            self.db.store_messages(processed_messages)
                            message_count += len(processed_messages)
                    except Exception as e:
                        logger.error(f"Failed to store batch: {e}")
                        logger.error(f"Error details: {str(e)}")
            
            # Only search for older messages if we're not using --days flag
            if not self.days_limit and earliest_date:
                logger.info(f"Searching for older messages in #{channel.name} (before {earliest_date})...")
                current_batch = []
                messages_found = 0
                async for message in channel.history(limit=None, before=earliest_date, oldest_first=self.oldest_first):
                    messages_found += 1
                    if messages_found % 100 == 0:
                        logger.info(f"Found {messages_found} older messages in #{channel.name}")
                    if cutoff_date and message.created_at < cutoff_date:
                        continue
                        
                    # Skip messages from the bot
                    if message.author.id == self.bot_user_id:
                        continue
                        
                    current_batch.append(message)
                    message_counter += 1
                    
                    # Store batch when it reaches the threshold
                    if len(current_batch) >= 100:
                        try:
                            processed_messages = []
                            for msg in current_batch:
                                processed_msg = await self._process_message(msg, channel_id)
                                if processed_msg:
                                    processed_messages.append(processed_msg)
                            
                            if processed_messages:
                                logger.info(f"Storing batch of {len(processed_messages)} messages from #{channel.name}")
                                self.db.store_messages(processed_messages)
                                message_count += len(processed_messages)
                                logger.info(f"Successfully stored batch of {len(processed_messages)} messages from #{channel.name} (total processed: {message_count})")
                            
                            current_batch = []
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.error(f"Failed to store batch: {e}")
                            logger.error(f"Error details: {str(e)}")
                
                # Process any remaining messages
                if current_batch:
                    try:
                        processed_messages = []
                        for msg in current_batch:
                            processed_msg = await self._process_message(msg, channel_id)
                            if processed_msg:
                                processed_messages.append(processed_msg)
                        
                        if processed_messages:
                            logger.info(f"Storing batch of {len(processed_messages)} messages from #{channel.name}")
                            self.db.store_messages(processed_messages)
                            message_count += len(processed_messages)
                    except Exception as e:
                        logger.error(f"Failed to store batch: {e}")
                        logger.error(f"Error details: {str(e)}")

            # Get all message dates to check for gaps
            message_dates = self.db.get_message_dates(channel_id)
            if message_dates:
                # Filter dates based on cutoff if set
                if cutoff_date:
                    message_dates = [d for d in message_dates if datetime.fromisoformat(d) >= cutoff_date]
                
                # Sort dates based on order setting
                message_dates.sort(reverse=not self.oldest_first)
                gaps = []
                for i in range(len(message_dates) - 1):
                    current = datetime.fromisoformat(message_dates[i])
                    next_date = datetime.fromisoformat(message_dates[i + 1])
                    # Compare dates based on order
                    date_diff = (next_date - current).days if self.oldest_first else (current - next_date).days
                    if date_diff > 7:
                        gaps.append((current, next_date) if self.oldest_first else (next_date, current))
                
                if gaps:
                    logger.info(f"Found {len(gaps)} gaps (>1 week) in message history for #{channel.name}")
                    for start, end in gaps:
                        gap_message_count = 0
                        current_batch = []
                        logger.info(f"Searching for messages in #{channel.name} between {start} and {end} (gap of {abs((end - start).days)} days)")
                        async for message in channel.history(limit=None, after=start, before=end, oldest_first=self.oldest_first):
                            # Skip messages from the bot
                            if message.author.id == self.bot_user_id:
                                continue
                                
                            current_batch.append(message)
                            gap_message_count += 1
                            
                            # Store batch when it reaches the threshold
                            if len(current_batch) >= 100:
                                try:
                                    processed_messages = []
                                    for msg in current_batch:
                                        processed_msg = await self._process_message(msg, channel_id)
                                        if processed_msg:
                                            processed_messages.append(processed_msg)
                                    
                                    if processed_messages:
                                        logger.info(f"Storing batch of {len(processed_messages)} messages from gap in #{channel.name}")
                                        self.db.store_messages(processed_messages)
                                        message_count += len(processed_messages)
                                        if gap_message_count % 100 == 0:
                                            logger.info(f"Found {gap_message_count} messages in current gap for #{channel.name}")
                                    
                                    current_batch = []
                                    await asyncio.sleep(0.1)
                                except Exception as e:
                                    logger.error(f"Failed to store batch: {e}")
                                    logger.error(f"Error details: {str(e)}")
                        
                        # Process any remaining messages from the gap
                        if current_batch:
                            try:
                                processed_messages = []
                                for msg in current_batch:
                                    processed_msg = await self._process_message(msg, channel_id)
                                    if processed_msg:
                                        processed_messages.append(processed_msg)
                                
                                if processed_messages:
                                    logger.info(f"Storing final gap batch of {len(processed_messages)} messages from #{channel.name}")
                                    self.db.store_messages(processed_messages)
                                    message_count += len(processed_messages)
                            except Exception as e:
                                logger.error(f"Failed to store batch: {e}")
                                logger.error(f"Error details: {str(e)}")
                        
                        logger.info(f"Finished gap search in #{channel.name}, found {gap_message_count} messages")
            
            logger.info(f"Found {message_count} new messages to archive")
            logger.info(f"Archive complete - processed {message_count} new messages")
            
            channel_duration = (datetime.now() - channel_start_time).total_seconds()
            logger.info(f"Finished archive of #{channel.name} in {channel_duration:.2f}s")
            
        except discord.HTTPException as e:
            if e.code == 429:  # Rate limit error
                logger.warning(f"Hit rate limit while processing #{channel.name}: {e}")
                retry_after = e.retry_after if hasattr(e, 'retry_after') else 5
                logger.info(f"Waiting {retry_after}s before continuing")
                await asyncio.sleep(retry_after)
            else:
                logger.error(f"HTTP error in channel {channel.name}: {e}")
        except Exception as e:
            logger.error(f"Error archiving channel {channel.name}: {e}")
        finally:
            # Don't close the connection here as it's reused across channels
            pass

    async def _process_message(self, message, channel_id):
        """Process a single message and store it in the database."""
        try:
            message_start_time = datetime.now()
            
            # Calculate total reaction count and get reactors
            reaction_count = sum(reaction.count for reaction in message.reactions) if message.reactions else 0
            reactors = []  # Always initialize as empty list
            
            if reaction_count > 0 and message.reactions:
                reaction_start_time = datetime.now()
                reactor_ids = set()
                try:
                    # Always process reactions in in-depth mode, or if the message is new
                    if self.in_depth or not self.db.message_exists(message.id):
                        logger.info(f"Processing reactions for message {message.id}: {len(message.reactions)} types, {reaction_count} total reactions")
                        
                        guild = self.get_guild(self.guild_id)
                        
                        for reaction in message.reactions:
                            reaction_process_start = datetime.now()
                            await self._wait_for_rate_limit()
                            try:
                                async for user in reaction.users(limit=50):
                                    reactor_ids.add(user.id)
                                    if hasattr(user, 'id') and not self.db.get_member(user.id):
                                        logger.debug(f"Fetching new member info for reactor {user.id}")
                                        await self._wait_for_rate_limit()
                                        member = guild.get_member(user.id) if guild else None
                                        role_ids = json.dumps([role.id for role in member.roles]) if member and member.roles else None
                                        guild_join_date = member.joined_at.isoformat() if member and member.joined_at else None

                                        self.db.create_or_update_member(
                                            member_id=user.id,
                                            username=user.name,
                                            display_name=getattr(user, 'display_name', None),
                                            global_name=getattr(user, 'global_name', None),
                                            avatar_url=str(user.avatar.url) if user.avatar else None,
                                            discriminator=getattr(user, 'discriminator', None),
                                            bot=getattr(user, 'bot', False),
                                            system=getattr(user, 'system', False),
                                            accent_color=getattr(user, 'accent_color', None),
                                            banner_url=str(user.banner.url) if getattr(user, 'banner', None) else None,
                                            discord_created_at=user.created_at.isoformat() if hasattr(user, 'created_at') else None,
                                            guild_join_date=guild_join_date,
                                            role_ids=role_ids
                                        )
                                    else:
                                        logger.debug(f"Using cached member info for reactor {user.id}")
                                    
                                reaction_process_duration = (datetime.now() - reaction_process_start).total_seconds()
                                logger.info(f"Processed reaction {reaction} in {reaction_process_duration:.2f}s")
                            except discord.Forbidden:
                                logger.warning(f"Missing permissions to fetch reactors for {reaction} on message {message.id}")
                                continue
                            except Exception as e:
                                logger.warning(f"Error fetching users for reaction {reaction} on message {message.id}: {e}")
                                continue
                        
                        reaction_duration = (datetime.now() - reaction_start_time).total_seconds()
                        if reaction_duration > 2.0:  # Log slow reaction processing
                            logger.warning(f"Slow reaction processing for message {message.id}: {reaction_duration:.2f}s")
                        
                        # Convert reactor_ids to list if we found any
                        if reactor_ids:
                            reactors = list(reactor_ids)
                except Exception as e:
                    logger.warning(f"Could not fetch reactors for message {message.id}: {e}")
                
            # First create or update the member
            if hasattr(message.author, 'id'):
                # Get guild member object to access join date and roles
                guild = self.get_guild(self.guild_id)
                member = guild.get_member(message.author.id) if guild else None
                role_ids = json.dumps([role.id for role in member.roles]) if member and member.roles else None
                guild_join_date = member.joined_at.isoformat() if member and member.joined_at else None

                self.db.create_or_update_member(
                    member_id=message.author.id,
                    username=message.author.name,
                    display_name=getattr(message.author, 'display_name', None),
                    global_name=getattr(message.author, 'global_name', None),
                    avatar_url=str(message.author.avatar.url) if message.author.avatar else None,
                    discriminator=getattr(message.author, 'discriminator', None),
                    bot=getattr(message.author, 'bot', False),
                    system=getattr(message.author, 'system', False),
                    accent_color=getattr(message.author, 'accent_color', None),
                    banner_url=str(message.author.banner.url) if getattr(message.author, 'banner', None) else None,
                    discord_created_at=message.author.created_at.isoformat() if hasattr(message.author, 'created_at') else None,
                    guild_join_date=guild_join_date,
                    role_ids=role_ids
                )

            # Get the actual channel ID and name (use parent forum for threads)
            actual_channel = message.channel
            thread_id = None
            
            if hasattr(message.channel, 'parent') and message.channel.parent:
                actual_channel = message.channel.parent
                # Only set thread_id if it's a regular thread, not a forum post
                if isinstance(message.channel, discord.Thread) and not hasattr(message.channel, 'thread_type'):
                    thread_id = message.channel.id
                    logger.debug(f"Message {message.id} is in a thread. Thread ID: {thread_id}, Parent Channel: {actual_channel.name} (ID: {actual_channel.id})")
                elif hasattr(message.channel, 'thread_type'):
                    # For forum posts, use the forum post's channel ID as the channel_id
                    actual_channel = message.channel
                    logger.debug(f"Message {message.id} is in a forum post. Forum Channel: {actual_channel.name} (ID: {actual_channel.id})")

            # Then create or update the channel using the appropriate channel info
            self.db.create_or_update_channel(
                channel_id=actual_channel.id,
                channel_name=actual_channel.name,
                nsfw=getattr(actual_channel, 'nsfw', False)
            )
            
            processed_message = {
                'id': message.id,
                'message_id': message.id,
                'channel_id': actual_channel.id,
                'author_id': message.author.id,
                'author': {
                    'id': message.author.id,
                    'name': message.author.name,
                    'display_name': getattr(message.author, 'display_name', None),
                    'global_name': getattr(message.author, 'global_name', None),
                    'avatar_url': str(message.author.avatar.url) if message.author.avatar else None,
                    'discriminator': getattr(message.author, 'discriminator', None),
                    'bot': getattr(message.author, 'bot', False),
                    'system': getattr(message.author, 'system', False),
                    'accent_color': getattr(message.author, 'accent_color', None),
                    'banner_url': str(message.author.banner.url) if getattr(message.author, 'banner', None) else None,
                    'discord_created_at': message.author.created_at.isoformat() if hasattr(message.author, 'created_at') else None,
                    'guild_join_date': guild_join_date,
                    'role_ids': role_ids
                },
                'channel': {
                    'id': actual_channel.id,
                    'name': actual_channel.name,
                    'nsfw': getattr(actual_channel, 'nsfw', False)
                },
                'content': message.content,
                'created_at': message.created_at.isoformat(),
                'attachments': [
                    {
                        'url': attachment.url,
                        'filename': attachment.filename
                    } for attachment in message.attachments
                ],
                'embeds': [embed.to_dict() for embed in message.embeds],
                'reaction_count': reaction_count,
                'reactors': reactors,  # This will be a list, not None
                'reference_id': message.reference.message_id if message.reference else None,
                'edited_at': message.edited_at.isoformat() if message.edited_at else None,
                'is_pinned': message.pinned,
                'thread_id': thread_id,
                'message_type': str(message.type),
                'flags': message.flags.value,
                'jump_url': message.jump_url
            }
            
            message_duration = (datetime.now() - message_start_time).total_seconds()
            if message_duration > 3.0:
                logger.warning(f"Very slow message processing: {message_duration:.2f}s for message {message.id}")
            
            return processed_message
            
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            return None

    async def close(self):
        """Properly close the bot and database connection."""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
                self.db = None
            await super().close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Archive Discord messages')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    parser.add_argument('--order', choices=['newest', 'oldest'], default='newest',
                      help='Order to process messages (default: newest)')
    parser.add_argument('--days', type=int, help='Number of days of history to fetch (default: all)')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Number of messages to process in each batch (default: 100)')
    parser.add_argument('--in-depth', action='store_true',
                      help='Perform thorough message checks, re-processing all messages in the time range')
    args = parser.parse_args()
    
    if args.dev:
        logger.info("Running in development mode")
    
    bot = None
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        bot = MessageArchiver(dev_mode=args.dev, order=args.order, days=args.days, 
                            batch_size=args.batch_size, in_depth=args.in_depth)
        
        # Start the bot and keep it running until archiving is complete
        async def runner():
            await bot.start(os.getenv('DISCORD_BOT_TOKEN'))
            # Wait for the bot to be ready and complete archiving
            while not bot.is_closed():
                await asyncio.sleep(1)
        
        # Run the bot until it completes
        loop.run_until_complete(runner())
        
    except discord.LoginFailure:
        logger.error("Failed to login. Please check your Discord token.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Ensure everything is cleaned up properly
        if bot:
            if bot.db:
                bot.db.close()
            if not loop.is_closed():
                loop.run_until_complete(bot.close())
            
        # Clean up the event loop
        try:
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
                remaining_tasks = asyncio.all_tasks(loop)
                if remaining_tasks:
                    loop.run_until_complete(asyncio.gather(*remaining_tasks))
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            if loop.is_running():
                loop.stop()
            if not loop.is_closed():
                loop.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot shutdown initiated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}") 