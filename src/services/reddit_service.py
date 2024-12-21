import logging
import asyncio
import os

logger = logging.getLogger('RedditService')


class RedditService:
    """Handles Reddit-related operations"""

    def __init__(self, claude, admin_id: int = None):
        self.claude = claude
        self.admin_id = admin_id or int(os.getenv('ADMIN_USER_ID', 0))
        self.bot = None  # Will be set by the ChannelSummarizer

    def set_bot(self, bot):
        """Set the bot instance for Discord communications"""
        self.bot = bot

    async def analyze_for_reddit(self, summary):
        """
        Analyze summary for potential Reddit content.
        Return dict if suitable, else None.
        """
        logger.info("Analyzing summary for Reddit potential")

        prompt = f"""Analyze this Discord summary and determine if any topic would make for an engaging Reddit post. 
If you find no suitable topic, respond with exactly "NO_REDDIT_CONTENT". 
Otherwise, respond with a proposed Reddit title on the first line, and the relevant "topic text" on the following lines.

Summary to analyze:
{summary}
"""

        loop = asyncio.get_running_loop()

        def create_reddit_analysis():
            return self.claude.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

        try:
            response = await loop.run_in_executor(None, create_reddit_analysis)
            result = response.content[0].text.strip()
            if result == "NO_REDDIT_CONTENT":
                return None

            lines = result.split('\n')
            if len(lines) >= 2:
                return {'title': lines[0].strip(), 'content': '\n'.join(lines[1:]).strip()}

            logger.warning("Invalid RedditService response format.")
            return None
        except Exception as e:
            logger.error(f"Error getting Reddit suggestion: {e}")
            return None

    async def prepare_suggestion(self, suggestion, files):
        """
        Prepare and send a message to the admin about potential Reddit content
        """
        if not suggestion or not self.admin_id or not self.bot:
            logger.info("No Reddit suggestion to prepare or missing admin/bot configuration.")
            return

        try:
            # Try to fetch the admin user
            admin_user = await self.bot.fetch_user(self.admin_id)
            if not admin_user:
                logger.error(f"Could not find admin user with ID {self.admin_id}")
                return

            # Prepare the message
            message = (
                "🎯 **Potential Reddit Post Detected!**\n\n"
                f"**Suggested Title:** {suggestion['title']}\n\n"
                f"**Content:**\n{suggestion['content']}\n\n"
            )

            if files:
                message += f"**Available Attachments:** {len(files)} files"

            # Send the initial message
            dm_channel = await admin_user.create_dm()
            initial_msg = await self.bot.safe_send_message(dm_channel, message)

            # If there are files, send up to 3 of them
            for file in files[:6]:
                try:
                    await self.bot.safe_send_message(
                        dm_channel,
                        "Potential attachment for the post:",
                        file=file
                    )
                except Exception as e:
                    logger.error(f"Failed to send attachment to admin: {e}")

        except Exception as e:
            logger.error(f"Failed to send Reddit suggestion to admin: {e}") 