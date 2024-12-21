import logging
import discord
from utils.error_handler import handle_errors
from src.services.scheduler_service import SchedulerService
from datetime import datetime, timedelta
import traceback
import io

logger = logging.getLogger(__name__)

class SummaryManager:
    def __init__(self, bot):
        self.bot = bot
        self.db = bot.db
        self.session = bot.session
        self.summary_service = bot.summary_service
        self.attachment_service = bot.attachment_service
        self.reddit_service = bot.reddit_service
        self.is_test_mode = bot.is_test_mode

        self._test_summary_channel_id = bot.test_summary_channel_id
        self._prod_summary_channel_id = bot.summary_channel_id
        self._categories_to_monitor = bot.category_ids

    @handle_errors("run_daily_summary")
    async def run_daily_summary(self):
        """Orchestrates the daily summary for categories, posts results, stores them in DB."""
        logger.info("run_daily_summary started")
        channel_id = (self._test_summary_channel_id 
                      if self.is_test_mode 
                      else self._prod_summary_channel_id)
        summary_channel = self.bot.get_channel(channel_id)

        if not summary_channel:
            logger.error(f"Could not access summary channel {channel_id}")
            return

        active_channels = False
        date_header_posted = False

        for category_id in self._categories_to_monitor:
            await self.summarize_category(category_id, summary_channel, 
                                          date_header_posted, 
                                          lambda: self._mark_date_header(summary_channel))
            date_header_posted = True  # after first channel, the date header has been posted

        if not active_channels:
            logger.info("No channels had significant activity.")
            await summary_channel.send("No channels had significant activity in the past 24 hours.")
        logger.info("run_daily_summary completed")

    async def summarize_category(self, category_id, summary_channel, 
                                 date_header_posted, mark_date_header):
        category = self.bot.get_channel(category_id)
        if not category:
            logger.warning(f"Could not access category {category_id}")
            return

        channels = [ch for ch in category.channels if isinstance(ch, discord.TextChannel)]
        logger.info(f"Found {len(channels)} text channels in category {category.name}")

        for channel in channels:
            self.attachment_service.clear_cache()
            messages = await self.bot.get_channel_history(channel.id)

            if len(messages) < 20:
                logger.debug(f"Skipping channel {channel.name} (only {len(messages)} messages).")
                continue

            summary = await self.summary_service.generate_summary(messages)
            if "[NOTHING OF NOTE]" in summary:
                logger.debug(f"No noteworthy activity in channel {channel.name}.")
                continue

            # Post the date header only once
            if not date_header_posted:
                await mark_date_header()
                date_header_posted = True

            short_summary = await self.summary_service.generate_short_summary(
                summary, len(messages)
            )

            if not self.is_test_mode:
                self.db.store_daily_summary(
                    channel_id=channel.id,
                    channel_name=channel.name,
                    messages=messages,
                    full_summary=summary,
                    short_summary=short_summary
                )

            await self.post_summary(summary_channel, channel.name, summary, len(messages))

            # Optionally handle Reddit suggestions:
            suggestion = await self.reddit_service.analyze_for_reddit(summary)
            if suggestion:
                all_cached_files = self.attachment_service.get_all_files_sorted()
                await self.reddit_service.prepare_suggestion(suggestion, all_cached_files)

    async def _mark_date_header(self, summary_channel):
        current_date = datetime.utcnow()
        header = f"# 📅 Summary for {current_date.strftime('%A, %B %d, %Y')}"
        await summary_channel.send(header)

    @handle_errors("post_summary")
    async def post_summary(self, summary_channel, source_channel_name, summary, message_count):
        """Post the full summary in the summary channel (plus attachments & threads if desired)."""
        try:
            # Generate and post initial summary
            short_summary = await self.summary_service.generate_short_summary(summary, message_count)
            
            # Get top reacted image for initial message
            top_attachment = None
            main_attachment_id = None
            all_files = self.attachment_service.get_all_files_sorted()
            if all_files:
                top_attachment = all_files[0][0]  # First file from the sorted list
                main_attachment_id = all_files[0][2]  # Get the message ID
            
            # Get the actual channel object to get its ID for mention
            source_channel = discord.utils.get(summary_channel.guild.channels, name=source_channel_name)
            if source_channel:
                channel_mention = f"<#{source_channel.id}>"
            else:
                channel_mention = source_channel_name.replace('_', ' ').title()
            
            # Format title with proper channel mention
            formatted_title = f"# {channel_mention}"
            
            # Send initial message with attachment if available
            initial_message = await self.bot.safe_send_message(
                summary_channel,
                f"{formatted_title}\n{short_summary}",
                file=top_attachment
            )
            
            if not initial_message:
                logger.error("Failed to create initial message - cannot create thread")
                return
            
            # Create thread and add debug logging
            logger.info(f"Creating thread for channel {source_channel_name}")
            thread = await self.bot.create_summary_thread(initial_message, source_channel_name)
            logger.info(f"Thread created: {thread.id}")
            
            # Process topics
            topics = summary.split("### ")
            topics = [t.strip().rstrip('#').strip() for t in topics if t.strip()]
            
            used_message_ids = set()

            for i, topic in enumerate(topics):
                logger.debug(f"Processing topic {i+1}/{len(topics)}")
                topic_used_ids = await self.bot.process_topic(thread, topic, is_first=(i == 0))
                used_message_ids.update(topic_used_ids)

            # Process remaining attachments, excluding the main one
            await self.bot.process_unused_attachments(thread, used_message_ids, main_attachment_id)
            
            logger.info(f"Successfully posted summary to thread for {source_channel_name}")

        except Exception as e:
            logger.error(f"Error in post_summary: {e}")
            logger.debug(traceback.format_exc())
            raise