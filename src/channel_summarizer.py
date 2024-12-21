import logging
logger = logging.getLogger('ChannelSummarizer')

# Standard library imports
from typing import List, Dict, Set
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
import sys

# Third-party imports
import discord
import anthropic

# Local imports
from src.bot import ChannelSummarizerBot
from src.services.summary_service import SummaryService
from src.services.attachment_service import AttachmentService
from src.services.reddit_service import RedditService
from src.db_handler import DatabaseHandler
from utils.error_handler import ErrorHandler



class SummaryCoordinator:
    """Coordinates between services to manage the summarization process"""
    def __init__(self, bot_token: str):
        # Initialize bot with proper intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        self.bot = ChannelSummarizerBot(
            token=bot_token,
            command_prefix="!",
            intents=intents
        )
        
        # Initialize services
        self.db = DatabaseHandler()
        self.claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.error_handler = ErrorHandler(
            notification_channel=None,  # Will be set after bot is ready
            admin_user=int(os.getenv('ADMIN_USER_ID', 0))
        )
        self.bot.error_handler = self.error_handler
        
        # Initialize services with error handler
        self.summary_service = SummaryService(self.claude, self.error_handler)
        self.attachment_service = AttachmentService()
        self.reddit_service = RedditService(self.claude)
        
        # Add event listeners
        @self.bot.event
        async def on_ready():
            logger.info(f'Bot is ready! Logged in as {self.bot.user.name} ({self.bot.user.id})')
            # Set notification channel for error handler
            if notification_channel_id := os.getenv('NOTIFICATION_CHANNEL_ID'):
                self.error_handler.notification_channel = self.bot.get_channel(int(notification_channel_id))
            
            # Check if --run-now flag was used
            if '--run-now' in sys.argv:
                logger.info("Running immediate channel processing due to --run-now flag")
                await self.process_all_channels()

    async def process_all_channels(self):
        """Process all configured channels"""
        try:
            # Get categories from environment
            category_ids_str = os.getenv('CATEGORIES_TO_MONITOR', '')
            logger.debug(f"Retrieved CATEGORIES_TO_MONITOR from env: {category_ids_str}")
            
            category_ids = category_ids_str.split(',')
            if not category_ids or not category_ids[0]:
                logger.error("No category IDs configured")
                return

            logger.info(f"Starting to process categories: {category_ids}")
            for category_id in category_ids:
                try:
                    category_id = int(category_id.strip())
                    category = self.bot.get_channel(category_id)
                    if category and isinstance(category, discord.CategoryChannel):
                        logger.info(f"Processing category: {category.name} ({category_id})")
                        # Process all text channels in this category
                        for channel in category.text_channels:
                            logger.info(f"Processing channel: {channel.name} ({channel.id})")
                            await self.process_channel(channel)
                    else:
                        logger.error(f"Could not find category with ID: {category_id}")
                except ValueError as e:
                    logger.error(f"Invalid category ID format: {category_id}")
                except Exception as e:
                    logger.error(f"Error processing category {category_id}: {str(e)}")
            
            logger.info("Finished processing all categories")
            if '--test' in sys.argv:
                logger.info("Test run completed, shutting down")
                await self.bot.close()
                
        except Exception as e:
            logger.error(f"Error in process_all_channels: {str(e)}")
            if '--test' in sys.argv:
                await self.bot.close()

    async def process_channel(self, channel):
        """Process a channel and generate its summary"""
        messages = await self.bot.get_channel_history(channel.id)
        
        # Process attachments
        attachments = await self.attachment_service.process_channel_attachments(messages)
        
        # Generate summary
        summary = await self.summary_service.generate_summary(messages)
        if summary == "[NOTHING OF NOTE]":
            logger.debug(f"No noteworthy activity in channel {channel.name}")
            return

        # Create thread for summary
        thread = await self.create_summary_thread(channel, summary, attachments)
        
        # Process unused attachments
        if attachments:
            used_message_ids = set(msg['id'] for msg in messages if msg['id'] in summary)
            await self.process_unused_attachments(thread, used_message_ids, attachments[0]['id'])

        # Analyze for Reddit
        await self.reddit_service.analyze_for_reddit(summary)

    async def create_summary_thread(self, channel, summary: str, attachments: List[Dict]) -> discord.Thread:
        """Create and populate a thread with the summary"""
        main_message = await self.bot.safe_send_message(
            channel,
            summary,
            file=attachments[0]['file'] if attachments else None
        )
        thread = await main_message.create_thread(name=f"Summary {datetime.now().strftime('%Y-%m-%d')}")
        return thread

    async def process_unused_attachments(self, thread: discord.Thread, used_message_ids: Set[str], main_attachment_id: str):
        """Process and send unused but popular attachments"""
        unused_attachments = self.attachment_service.get_unused_popular_attachments(
            used_message_ids,
            thread.guild.id,
            main_attachment_id
        )

        if unused_attachments:
            await self.bot.safe_send_message(thread, "\n\n---\n### 📎 Other Popular Attachments")
            
            for att in unused_attachments[:10]:
                msg_content = f"By **{att['username']}**: {att['message_link']}"
                await self.bot.safe_send_message(thread, msg_content, file=att['file'])

        # Add jump link to thread start
        async for first_message in thread.history(oldest_first=True, limit=1):
            await self.bot.safe_send_message(
                thread,
                f"---\n\u200B\n***Click here to jump to the beginning of this thread: {first_message.jump_url}***"
            )
            break

    def start(self):
        """Start the bot"""
        self.bot.run(self.bot.token)