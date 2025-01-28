import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse
import sqlite3

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import anthropic
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.common.db_handler import DatabaseHandler
import traceback
import discord
from discord.ext import commands
import asyncio
import aiohttp
from discord import Webhook, Embed, Color
import re
from src.common.rate_limiter import RateLimiter
from collections import deque, defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsSummarizer:
    def __init__(self, dev_mode=False, discord_client=None, monitored_channels=None):
        logger.info("Initializing NewsSummarizer...")
        
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Initialize database handler in dev mode
        self.db = DatabaseHandler(dev_mode=dev_mode)
        logger.info(f"Database handler initialized in {'dev' if dev_mode else 'production'} mode using path: {self.db.db_path}")
        
        # Get Discord bot token and channel IDs
        self.discord_token = os.getenv('DISCORD_BOT_TOKEN')
        self.dev_mode = dev_mode
        self.logger = logger  # Add this line to store logger reference
        
        # Get channel IDs based on mode and parameters
        self.channel_ids = []
        
        # If specific channels are provided, they take precedence
        if monitored_channels:
            logger.info(f"Using provided monitored channels: {monitored_channels}")
            self.channel_ids = monitored_channels
        # Otherwise use environment variables based on mode
        elif dev_mode:
            test_channels = os.getenv('TEST_DATA_CHANNEL')
            if test_channels:
                self.channel_ids = [int(cid.strip()) for cid in test_channels.split(',') if cid.strip()]
                logger.info(f"Dev mode: Using test channels: {self.channel_ids}")
            else:
                self.channel_ids = [int(os.getenv('DEV_SUMMARY_CHANNEL_ID'))]
        else:
            self.channel_ids = [int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))]
            
        if not self.discord_token or not all(self.channel_ids):
            raise ValueError("DISCORD_BOT_TOKEN or channel IDs not found in environment")
        
        # Use existing bot instance if provided, otherwise create new one
        if discord_client:
            self.discord_client = discord_client
        else:
            # Initialize Discord client with proper intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.messages = True
            intents.members = True
            intents.presences = True
            
            self.discord_client = commands.Bot(
                command_prefix="!",
                intents=intents,
                heartbeat_timeout=60.0,
                guild_ready_timeout=10.0,
                gateway_queue_size=512
            )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        
        # Initialize Claude client
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.claude = anthropic.Anthropic(api_key=api_key)
        logger.info("Claude client initialized")

    def get_all_channel_ids(self):
        """Get all channel IDs from the database."""
        logger.info("\nGetting all channel IDs from database:")
        
        try:
            # Get all unique channel IDs
            channel_query = """
                SELECT DISTINCT channel_id
                FROM messages
                WHERE channel_id IS NOT NULL
            """
            results = self.db.execute_query(channel_query)
            channel_ids = [row[0] for row in results]
            
            logger.info(f"Found {len(channel_ids)} channels in database")
            
            # Get message counts for each channel
            channel_stats = """
                SELECT 
                    channel_id,
                    COUNT(*) as msg_count,
                    MIN(created_at) as earliest,
                    MAX(created_at) as latest
                FROM messages 
                GROUP BY channel_id
            """
            stats = self.db.execute_query(channel_stats)
            
            logger.info("\nChannel statistics:")
            for row in stats:
                channel_id, count, earliest, latest = row
                logger.info(f"Channel {channel_id}:")
                logger.info(f"  - Total messages: {count}")
                logger.info(f"  - Date range: {earliest} to {latest}")
            
            return channel_ids
                
        except Exception as e:
            logger.error(f"Error getting channel IDs: {e}")
            logger.error(traceback.format_exc())
            return []

    def check_database_content(self, channel_id: int):
        """Check what data we have in the database."""
        logger.info("\nChecking database content:")
        
        try:
            # Get total message count
            total_query = "SELECT COUNT(*) FROM messages WHERE channel_id = ?"
            total_count = self.db.execute_query(total_query, (channel_id,))[0][0]
            logger.info(f"Total messages in database: {total_count}")
            
            # Get message count and date range per channel
            channel_query = """
                SELECT 
                    channel_id,
                    COUNT(*) as msg_count,
                    MIN(created_at) as earliest,
                    MAX(created_at) as latest
                FROM messages 
                GROUP BY channel_id
            """
            results = self.db.execute_query(channel_query)
            
            logger.info("\nMessage distribution by channel:")
            for row in results:
                channel_id, count, earliest, latest = row
                logger.info(f"Channel {channel_id}:")
                logger.info(f"  - Messages: {count}")
                logger.info(f"  - Date range: {earliest} to {latest}")
                
        except Exception as e:
            logger.error(f"Error checking database content: {e}")
            logger.error(traceback.format_exc())

    async def get_channel_messages(self) -> List[Dict[str, Any]]:
        """Get all messages from the past 24 hours for the monitored channels."""
        logger = logging.getLogger(__name__)
        
        logger.info("Getting all messages from database for past 24 hours")
        logger.info(f"[DEBUG] self.channel_ids: {self.channel_ids}")
        
        try:
            # Get messages from the past 24 hours
            query = """
                SELECT 
                    m.message_id, m.author_id, m.channel_id, m.content, m.created_at,
                    m.thread_id, m.reference_id, m.attachments, m.reaction_count,
                    m.reactors, m.jump_url, m.is_deleted,
                    mem.username, mem.server_nick, mem.global_name
                FROM messages m
                LEFT JOIN members mem ON m.author_id = mem.member_id
                WHERE m.channel_id IN ({})
                AND m.created_at > datetime('now', '-24 hours')
                AND (m.is_deleted IS NULL OR m.is_deleted = FALSE)
                ORDER BY m.created_at ASC;
            """.format(','.join('?' * len(self.channel_ids)))
            
            # Execute query in a thread-safe way
            def db_operation():
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute(query, self.channel_ids)
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                return rows
            
            # Run in executor
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, db_operation)
            
            logger.info(f"Retrieved {len(rows)} messages from database")
            
            messages = []
            for row in rows:
                message = {
                    'message_id': row[0],
                    'author_id': row[1], 
                    'channel_id': row[2],
                    'content': row[3],
                    'created_at': row[4],
                    'thread_id': row[5],
                    'reference_id': row[6],
                    'attachments': json.loads(row[7]) if row[7] else [],
                    'reactions': json.loads(row[9]) if row[9] else [], # Using reactors field
                    'reaction_count': row[8],
                    'jump_url': row[10],
                    'author_name': row[13] or row[14] or row[12] or 'Unknown User' # server_nick or global_name or username
                }
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages from database: {e}")
            return []


    
    def _format_single_message(self, msg):
        """Helper method to format a single message with all its details."""
        output = []
        
        # Start with timestamp and author
        output.append(f"=== Message from {msg['author_name']} ===")
        
        # Handle created_at timestamp
        if isinstance(msg['created_at'], str):
            try:
                timestamp = datetime.fromisoformat(msg['created_at'])
            except ValueError:
                try:
                    timestamp = datetime.fromtimestamp(float(msg['created_at']))
                except ValueError:
                    logger.error(f"Could not parse timestamp: {msg['created_at']}")
                    timestamp = None
        elif isinstance(msg['created_at'], datetime):
            timestamp = msg['created_at']
        else:
            logger.error(f"Unexpected created_at type: {type(msg['created_at'])}")
            timestamp = None
        
        if timestamp:
            output.append(f"Time: {timestamp.isoformat()}")
        else:
            output.append(f"Time: Unknown")
        
        # Add content
        output.append(f"Content: {msg['content']}")
        
        # Add reaction information
        if msg['reactions']:
            reaction_info = f"Reactions: {len(msg['reactions'])}"
            output.append(reaction_info)
        
        # Add detailed attachment information
        if msg['attachments']:
            output.append("Attachments:")
            for attachment in msg['attachments']:
                if isinstance(attachment, str):
                    output.append(f"- {attachment}")
                elif isinstance(attachment, dict):
                    url = attachment.get('url', attachment.get('proxy_url', ''))
                    filename = attachment.get('filename', 'unknown')
                    content_type = attachment.get('content_type', 'unknown')
                    output.append(f"- {filename} ({content_type}): {url}")
        
        # Add message link using jump_url if available, otherwise construct it
        if msg.get('jump_url'):
            output.append(f"Message link: {msg['jump_url']}")
        elif msg.get('message_id'):
            # Get guild ID from environment based on mode
            guild_id = os.getenv('DEV_GUILD_ID' if self.dev_mode else 'GUILD_ID')
            if guild_id:
                output.append(f"Message link: https://discord.com/channels/{guild_id}/{msg['channel_id']}/{msg['message_id']}")
        
        return "\n".join(output) + "\n"

    def format_messages_for_claude(self, messages):
        """Format messages for Claude analysis."""
        conversation = """You MUST respond with ONLY a JSON array containing news items. NO introduction text, NO explanation, NO markdown formatting.

If there are no significant news items, respond with exactly "[NO SIGNIFICANT NEWS]".
Otherwise, respond with ONLY a JSON array in this exact format:

[
 {
   "title": "BFL ship new Controlnets for FluxText",
   "mainText": "A new ComfyUI analytics node has been developed to track and analyze data pipeline components, including inputs, outputs, and embeddings. This enhancement aims to provide more controllable prompting capabilities:",
   "mainFile": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342.mp4, https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425343.png",
   "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342",
   "subTopics": [
     {
       "text": "Here's another example of **Kijai** using it in combination with **Redux** - **Kijai** noted that it worked better than the previous version:",
       "file": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342.png",
       "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342"
     }
   ]
 }
]

Focus on these types of content:
1. New features or tools that were announced
2. Demos or images that got a lot of attention (especially messages with many reactions)
3. Focus on the things that people seem most excited about or commented/reacted to on a lot
4. Focus on AI art and AI art-related tools
5. Call out notable achievements or demonstrations
6. Important community announcements

IMPORTANT REQUIREMENTS FOR MEDIA AND LINKS:
1. Each topic MUST have at least one Discord message link (jump_url) and should try to include multiple relevant attachments
2. AGGRESSIVELY search for related media - include ALL images, videos, or links that are part of the same discussion. For each topic, try to find at least 2-3 related images/videos/examples if they exist
3. If you find multiple related pieces of media, include them all in mainFile as a comma-separated list
4. For each subtopic that references media or a demo, you MUST include both the media link and the Discord message link
5. Prioritize messages with reactions or responses when selecting media to include
6. Be careful not to bias towards just the first messages.
7. If a topic has interesting follow-up discussions or examples, include those as subtopics even if they don't have media
8. Always end with a colon if there are attachments or links ":"
9. Don't share the same attachment or link multiple times - even across different subtopics

Requirements for the response:
1. Must be valid JSON in exactly the above format
2. Each news item must have all fields: title, mainText, mainFile (can be multiple comma-separated), messageLink, and subTopics
3. subTopics can include:
   - file (can be multiple comma-separated)
   - link (external links)
   - messageLink (required for all subtopics)
   - Both file and link can be included if relevant
4. Always end with a colon if there are attachments or links ":"
5. All usernames must be in bold with ** (e.g., "**username**") - ALWAYS try to give credit to the creator
6. If there are no significant news items, respond with exactly "[NO SIGNIFICANT NEWS]"
7. Include NOTHING other than the JSON response or "[NO SIGNIFICANT NEWS]"
8. Don't repeat the same item or leave any empty fields
9. When you're referring to groups of community members, refer to them as Banodocians 
10. Don't be hyperbolic or overly enthusiastic
11. If something seems to be a subjective opinion but still noteworthy, mention it as such: "Draken felt...", etc.

Here are the messages to analyze:

"""
        # Format each message using the existing helper method
        for msg in messages:
            conversation += self._format_single_message(msg)

        conversation += "\nRemember: Respond with ONLY the JSON array or '[NO SIGNIFICANT NEWS]'. NO other text."
        return conversation

    async def generate_news_summary(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a news summary from the messages using Claude."""
        logger = logging.getLogger(__name__)
        
        if not messages:
            logger.warning("No messages to analyze")
            return "[NO MESSAGES TO ANALYZE]"
        
        logger.info("Generating news summary with Claude...")
        
        formatted_prompt = self.format_messages_for_claude(messages)
        
        if self.dev_mode:
            logger.info("\n=== SENDING TO CLAUDE API ===")
            logger.info("Model: claude-3-5-sonnet-latest")
            logger.info(f"System prompt: You are a helpful AI assistant that analyzes Discord messages and extracts main news items and announcements. Focus on new features, popular demos, community announcements, and notable achievements. Format your response as a LIST of JSON objects, where each object has fields: title (string), mainText (string), mainFile (string or null), messageLink (string or null). Even if there is only one news item, it must be wrapped in a list. Credit community members as 'Banodocians'.")
            logger.info(f"User prompt:\n{formatted_prompt}")
            logger.info("=== END OF API REQUEST ===\n")
        
        try:
            # Run the blocking Claude API call in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,  # Use default executor
                lambda: self.claude.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=8192,
                    messages=[{
                        "role": "user",
                        "content": formatted_prompt
                    }]
                )
            )
            
            # Get the text content from the first message
            summary = response.content[0].text.strip()
            logger.info("Received response from Claude")
            
            if self.dev_mode:
                logger.info("\n=== CLAUDE API RESPONSE ===")
                logger.info(f"Raw response:\n{summary}")
                logger.info("=== END OF API RESPONSE ===\n")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with Claude: {e}")
            logger.error(f"Full error: {traceback.format_exc()}")
            return None

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

    def format_news_for_discord(self, news_items_json: str) -> List[Dict[str, str]]:
        """Format JSON news items into Discord messages."""
        try:
            # Handle special case responses
            if news_items_json in ["[NO SIGNIFICANT NEWS]", "[NO MESSAGES TO ANALYZE]"]:
                return [{"content": news_items_json}]

            # Try to parse as JSON
            try:
                # Find the first '[' to get the start of JSON if there's any preamble
                json_start = news_items_json.find('[')
                if json_start == -1:
                    # If no JSON found, return the text as is
                    return [{"content": news_items_json}]
                
                # Extract and parse JSON
                json_str = news_items_json[json_start:]
                news_items = json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the text as is
                return [{"content": news_items_json}]

            messages_to_send = []
            
            for item in news_items:
                # Start with main topic header and text
                main_content = [f"## {item['title']}\n"]
                
                # Add mainText and messageLink
                if item['messageLink'] and item['messageLink'] != "unknown":
                    main_content.append(f"{item['mainText']} {item['messageLink']}")
                else:
                    main_content.append(item['mainText'])
                
                messages_to_send.append({"content": "\n".join(main_content)})
                
                # Add main file attachments as separate messages if present
                if item.get('mainFile') and item['mainFile'] not in ["unknown", "null", None]:
                    for file in item['mainFile'].split(','):
                        file = file.strip()
                        if file:
                            messages_to_send.append({"content": f"{file}"})
                    
                # Add subtopics if present
                if 'subTopics' in item and item['subTopics']:
                    for subtopic in item['subTopics']:
                        content = []
                        
                        # Add text and messageLink
                        if subtopic.get('messageLink') and subtopic['messageLink'] != "unknown":
                            content.append(f"• {subtopic['text']} {subtopic['messageLink']}")
                        else:
                            content.append(f"• {subtopic['text']}")
                            
                        messages_to_send.append({"content": "\n".join(content)})
                        
                        # Add subtopic file attachments as separate messages
                        if subtopic.get('file') and subtopic['file'] not in ["unknown", "null", None]:
                            for file in subtopic['file'].split(','):
                                file = file.strip()
                                if file:
                                    messages_to_send.append({"content": f"{file}"})
            
            return messages_to_send
            
        except Exception as e:
            logger.error(f"Error formatting news items: {e}")
            logger.error(traceback.format_exc())
            return [{"content": f"Error formatting news items: {str(e)}"}]

    async def post_to_discord(self, news_items, target_channel=None):
        """Post news items to Discord using bot client."""
        try:
            # If target_channel is provided, use it directly
            channels_to_post = [target_channel] if target_channel else [self.discord_client.get_channel(cid) for cid in self.channel_ids]
            
            for channel in channels_to_post:
                if not channel:
                    logger.error(f"Could not find channel")
                    continue
                    
                if isinstance(news_items, str):
                    # Convert string input into list of messages format
                    if news_items == "[NO SIGNIFICANT NEWS]" or news_items == "[NO MESSAGES TO ANALYZE]":
                        messages = [{"content": news_items}]
                    else:
                        # Split into chunks if needed (Discord has 2000 char limit)
                        messages = [{"content": chunk} for chunk in news_items.split('\n\n') if chunk.strip()]
                elif isinstance(news_items, list):
                    messages = news_items
                else:
                    logger.error(f"Unexpected news_items type: {type(news_items)}")
                    raise ValueError(f"Unexpected news_items type: {type(news_items)}")

                # Send each message one at a time
                for msg in messages:
                    if msg.get("content", "").strip():
                        await self.safe_send_message(channel, msg["content"])
                        await asyncio.sleep(1)
                
                logger.info(f"Successfully sent all news items to channel {channel.id}")
                
        except Exception as e:
            logger.error(f"Error posting to Discord: {e}")
            raise

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

    def is_media_url(self, url):
        """Check if URL is an image or video."""
        media_patterns = r'\.(png|jpg|jpeg|gif|webp|mp4|webm|mov)$'
        return bool(re.search(media_patterns, url.lower()))

    async def run(self):
        """Run the news summary generation process."""
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("Starting news summary generation...")
            
            # Get messages from the past 24 hours
            messages = await self.get_channel_messages()
            if not messages:
                logger.warning("No messages found to analyze")
                return
                
            logger.info(f"Retrieved {len(messages)} messages from the past 24 hours")
            
            # Generate summary
            summary = await self.generate_news_summary(messages)
            if not summary:
                logger.error("Failed to generate summary")
                return
                
            if self.dev_mode:
                # In dev mode, print the formatted summary
                logger.info("=== Generated Summary ===")
                logger.info(json.dumps(summary, indent=2))
                logger.info("=== End of Summary ===")
            else:
                # In production mode, post to Discord
                await self.post_summary_to_discord(summary)
                
        except Exception as e:
            logger.error(f"Error in news summary process: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()

    async def setup_discord(self):
        """Initialize Discord connection with proper error handling"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Create ready event
            ready = asyncio.Event()
            
            # Set up the ready event handler
            @self.discord_client.event
            async def on_ready():
                logger.info(f"Logged in as {self.discord_client.user}")
                ready.set()

            # Handle connection errors
            @self.discord_client.event
            async def on_error(event, *args, **kwargs):
                logger.error(f"Discord error in {event}: {sys.exc_info()}")

            @self.discord_client.event
            async def on_disconnect():
                logger.warning("Discord client disconnected")
                
            # Start the client without blocking
            client_task = asyncio.create_task(self.discord_client.start(self.discord_token))
            
            # Wait for ready with timeout and retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(ready.wait(), timeout=30)
                    logger.info("Discord client ready")
                    break
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timed out waiting for Discord client to be ready (attempt {attempt + 1}/{max_retries})")
                        # Reset the event and try again
                        ready.clear()
                        continue
                    else:
                        logger.error("Failed to initialize Discord client after all retries")
                        raise
            
        except Exception as e:
            logger.error(f"Failed to initialize Discord client: {e}")
            logger.error(traceback.format_exc())
            raise

    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            logger.info("Starting cleanup...")
            
            if hasattr(self, 'session') and self.session and not self.session.closed:
                logger.info("Closing aiohttp session...")
                await self.session.close()
            
            if hasattr(self, 'claude'):
                logger.info("Cleaning up Claude client...")
                self.claude = None
            
            if self.discord_client and not self.discord_client.is_closed():
                logger.info("Closing Discord client...")
                await self.discord_client.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.error(traceback.format_exc())
            # Still try to close Discord client even if we had an error
            if self.discord_client and not self.discord_client.is_closed():
                await self.discord_client.close()

    async def combine_channel_summaries(self, channel_summaries: List[Dict[str, Any]]) -> str:
        """Combine and filter the most interesting summaries from all channels."""
        logger.info("Combining channel summaries...")

        if not channel_summaries:
            return "[NO SIGNIFICANT NEWS]"

        # Format the prompt for Claude
        prompt = """You are analyzing summaries from different Discord channels to pick the most interesting and impactful updates.
Review these summaries and select the most interesting items based on:
1. Technical significance and innovation
2. Community impact and engagement (reactions, discussions)
3. Visual impressiveness and demo quality
4. Broader relevance to AI art/generation
5. Practical utility for users

For each selected item, you MUST preserve ALL of these fields exactly as they appear in the input:
- title
- mainText
- mainFile (array of file links)
- messageLink (Discord message link)
- subTopics (array of subtopics, each with title and text)

The input is a list of JSON summaries from different channels. Return ONLY a JSON array containing the top 3-5 most interesting items.
Each item in your response must maintain the exact same JSON structure as the input items.
DO NOT modify, summarize, or remove any fields from the original items you select.
DO NOT add any explanation or text outside the JSON array.

Here are the channel summaries to analyze:
"""
        # Add each channel's summary
        for summary in channel_summaries:
            if summary and summary not in ["[NO SIGNIFICANT NEWS]", "[NO MESSAGES TO ANALYZE]"]:
                prompt += f"\n{summary}\n"

        prompt += "\nRemember: Return ONLY a JSON array with the 3-5 most interesting items. Each item must preserve ALL fields (title, mainText, mainFile, messageLink, subTopics) exactly as they appear in the input."

        try:
            # Run the blocking Claude API call in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.claude.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=8192,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
            )
            
            # Get the text content from the first message
            filtered_summary = response.content[0].text.strip()
            logger.info("Received filtered summary from Claude")
            
            if self.dev_mode:
                logger.info("\n=== FILTERED SUMMARY ===")
                logger.info(filtered_summary)
                logger.info("=== END OF FILTERED SUMMARY ===\n")
            
            return filtered_summary

        except Exception as e:
            logger.error(f"Error combining summaries with Claude: {e}")
            logger.error(f"Full error: {traceback.format_exc()}")
            return None

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate and post news summaries.')
        parser.add_argument('--dev', action='store_true', help='Run in development mode')
        parser.add_argument('--channel', type=int, help='Specific channel ID to analyze')
        args = parser.parse_args()
        
        logger.info("Starting news summarizer...")
        monitored_channels = [args.channel] if args.channel else None
        summarizer = NewsSummarizer(dev_mode=args.dev, monitored_channels=monitored_channels)
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the summarizer
        try:
            # Initialize Discord first
            loop.run_until_complete(summarizer.setup_discord())
            # Then run the main summarizer
            loop.run_until_complete(summarizer.run())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received - shutting down...")
        finally:
            # Cleanup tasks
            try:
                loop.run_until_complete(asyncio.sleep(1))  # Give tasks a moment to complete
                
                # Cancel all running tasks
                tasks = [t for t in asyncio.all_tasks(loop) 
                        if not t.done() and t != asyncio.current_task(loop)]
                if tasks:
                    logger.info(f"Cancelling {len(tasks)} pending tasks...")
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                logger.error(traceback.format_exc())
            finally:
                if not loop.is_closed():
                    logger.info("Closing event loop...")
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - shutting down...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 