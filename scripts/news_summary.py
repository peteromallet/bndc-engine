import os
import sys
from pathlib import Path
from typing import List, Optional
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
        
        # Get channel IDs based on mode
        self.channel_ids = []
        if dev_mode:
            test_channels = os.getenv('TEST_DATA_CHANNEL')
            if test_channels:
                # Split by commas and convert to integers, handling spaces
                self.channel_ids = [int(cid.strip()) for cid in test_channels.split(',') if cid.strip()]
                logger.info(f"Dev mode: Using test channels: {self.channel_ids}")
            else:
                self.channel_ids = [int(os.getenv('DEV_SUMMARY_CHANNEL_ID'))]
        else:
            # Use provided monitored channels if available, otherwise fall back to summary channel
            if monitored_channels:
                logger.info(f"[DEBUG] Received monitored_channels: {monitored_channels}")
                self.channel_ids = monitored_channels
                logger.info(f"[DEBUG] Set self.channel_ids to: {self.channel_ids}")
            else:
                logger.warning("[DEBUG] No monitored_channels provided!")
                self.channel_ids = [int(os.getenv('PRODUCTION_SUMMARY_CHANNEL_ID'))]
                logger.info(f"[DEBUG] Using fallback summary channel: {self.channel_ids}")
            
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

    def get_channel_messages(self, channel_id: Optional[int] = None) -> List[dict]:
        """Get messages from the past 24 hours from the database."""
        logger.info("Getting all messages from database for past 24 hours")
        
        try:
            # If channel_id is provided, use it, otherwise use all channel_ids
            target_channels = [channel_id] if channel_id else self.channel_ids
            logger.info(f"[DEBUG] self.channel_ids: {self.channel_ids}")
            logger.info(f"[DEBUG] target_channels: {target_channels}")
            
            if not target_channels:
                logger.warning("No target channels specified")
                return []

            # Get messages from all relevant channels
            query = """
                WITH monitored_channels AS (
                    -- Direct channels (not categories)
                    SELECT c.channel_id, c.channel_name
                    FROM channels c 
                    WHERE c.channel_id IN ({})
                    UNION
                    -- Channels within monitored categories
                    SELECT c1.channel_id, c1.channel_name
                    FROM channels c1 
                    JOIN channels c2 ON c1.category_id = c2.channel_id
                    WHERE c2.channel_id IN ({})
                )
                SELECT 
                    m.message_id, m.channel_id, m.author_id, m.content,
                    m.created_at, m.attachments, m.embeds, m.reaction_count,
                    m.reactors, m.reference_id, m.edited_at, m.is_pinned,
                    m.thread_id, m.message_type, m.flags, m.jump_url,
                    m.indexed_at,
                    mem.username, mem.server_nick, mem.global_name,
                    mc.channel_name
                FROM messages m
                LEFT JOIN members mem ON m.author_id = mem.member_id
                JOIN monitored_channels mc ON m.channel_id = mc.channel_id
                WHERE m.created_at > datetime('now', '-1 day')
                ORDER BY m.created_at DESC
                LIMIT 1000
            """
            
            channel_list = ','.join(str(cid) for cid in target_channels)
            formatted_msg_query = query.format(channel_list, channel_list)
            logger.info(f"[DEBUG] Messages Query: {formatted_msg_query}")
            
            # Set row factory to return dictionaries
            self.db.conn.row_factory = sqlite3.Row
            cursor = self.db.conn.cursor()
            cursor.execute(formatted_msg_query)
            results = cursor.fetchall()
            messages = []
            
            for row in results:
                try:
                    # Handle timezone-aware datetime strings
                    created_at = row['created_at']
                    edited_at = row['edited_at']
                    indexed_at = row['indexed_at']
                    
                    # Parse JSON fields
                    attachments = json.loads(row['attachments']) if row['attachments'] else []
                    embeds = json.loads(row['embeds']) if row['embeds'] else []
                    reactors = json.loads(row['reactors']) if row['reactors'] else []
                    
                    # Convert database row to message dict format
                    message = {
                        'message_id': row['message_id'],
                        'channel_id': row['channel_id'],
                        'channel_name': row['channel_name'],
                        'author_id': row['author_id'],
                        'content': row['content'] or '',  # Ensure content is never None
                        'created_at': created_at,
                        'attachments': attachments,
                        'embeds': embeds,
                        'reaction_count': row['reaction_count'] or 0,
                        'reactors': reactors,
                        'reference_id': row['reference_id'],
                        'edited_at': edited_at,
                        'is_pinned': bool(row['is_pinned']),
                        'thread_id': row['thread_id'],
                        'message_type': row['message_type'].replace('MessageType.', '') if row['message_type'] else 'default',
                        'flags': row['flags'] or 0,
                        'jump_url': row['jump_url'],
                        'indexed_at': indexed_at,
                        'author_name': row['server_nick'] or row['global_name'] or row['username'] or 'Unknown User'
                    }
                    messages.append(message)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error processing message {row['message_id']}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message {row['message_id']}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Retrieved {len(messages)} messages from database")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages from database: {e}")
            logger.error(traceback.format_exc())
            return []

    def format_messages_for_claude(self, messages):
        """Format messages for Claude analysis."""
        conversation = """Please analyze the below Discord messages and extract the main news items and announcements from the past 24 hours.

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
6. For each subtopic that references media or a demo, you MUST include both the media link and the Discord message link
7. Prioritize messages with reactions when selecting media to include
6. Be careful not to bias towards just the first messages.
8. If a topic has interesting follow-up discussions or examples, include those as subtopics even if they don't have media
9. Always end with a colon if there are attachments or links ":"
10: Don't share the same attachment or link multiple times

Return the results in EXACTLY this format with one JSON object per news item:

[
 {
   "title": "BFL ship new Controlnets for FluxText",
   "mainText": "A new ComfyUI analytics node has been developed to track and analyze data pipeline components, including inputs, outputs, and embeddings. This enhancement aims to provide more controllable prompting capabilities:",
   "mainFile": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342.mp4, https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425343.png",
   "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342",
   "subTopics": [
     {
       "text": "Here's another example of Kijai using it in combination with Redux - Kijai noted that it worked better than the previous version:",
       "file": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342.png, https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425344.png",
       "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/4532454353425342"
     },
     {
       "text": "While here's an example by Gigantosorouus that got a lot of reactions - Banodocians thought it was impressive:",
       "file": "https://discord.com/channels/1076117621407223829/1138865343314530324/765434534654645.png",
       "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/765434534654645"
     },
     {
       "text": "Here's a link people for using it in ComfyUI that people found interesting too - toyxyz shared that it has native support already:",
       "link": "github.com/froggers/comfydifx",
       "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/765434534654645"
     },
     {
       "text": "Interesting discussion about performance implications - merkwarpper says it runs 10x faster than other tools:",
       "messageLink": "https://discord.com/channels/1076117621407223829/1138865343314530324/765434534654650"
     }
   ]
 }
]

Requirements for the response:
1. Must be valid JSON in exactly this format
2. Each news item must have all fields: title, mainText, mainFile (can be multiple comma-separated), messageLink, and subTopics
3. subTopics can include:
   - file (can be multiple comma-separated)
   - link (external links)
   - messageLink (required for all subtopics)
   - Both file and link can be included if relevant
4. Always end with a colon if there are attachments or links ":"
5. All usernames must be in bold with ** (e.g., "**username**") - ALWAYS try to give credit to the creator
6. If there are no significant news items, respond with exactly "[NO SIGNIFICANT NEWS]"
7. Include NOTHING other than the JSON response
8. Don't repeat the same item or leave any empty fields
9. Always refere to community members as a group as Banodocians 

Here are the messages to analyze:

"""

        # Group messages by time proximity (within 5 minutes)
        messages = sorted(messages, key=lambda x: x['created_at'] if isinstance(x['created_at'], datetime) else datetime.fromisoformat(str(x['created_at'])))
        message_groups = []
        current_group = []
        
        for msg in messages:
            # Ensure created_at is a datetime object
            if isinstance(msg['created_at'], str):
                try:
                    msg_time = datetime.fromisoformat(msg['created_at'])
                except ValueError:
                    # If the string is not in ISO format, try to parse it as a timestamp
                    try:
                        msg_time = datetime.fromtimestamp(float(msg['created_at']))
                    except ValueError:
                        logger.error(f"Could not parse timestamp: {msg['created_at']}")
                        continue
            elif isinstance(msg['created_at'], datetime):
                msg_time = msg['created_at']
            else:
                logger.error(f"Unexpected created_at type: {type(msg['created_at'])}")
                continue
                
            if not current_group:
                current_group = [msg]
            else:
                last_msg = current_group[-1]
                last_time = last_msg['created_at'] if isinstance(last_msg['created_at'], datetime) else datetime.fromisoformat(str(last_msg['created_at']))
                
                # If messages are within 5 minutes, add to current group
                if abs((msg_time - last_time).total_seconds()) <= 300:
                    current_group.append(msg)
                else:
                    message_groups.append(current_group)
                    current_group = [msg]
        
        if current_group:
            message_groups.append(current_group)

        # Process each group
        for group in message_groups:
            # Collect all attachments in the group
            attachments_summary = []
            for msg in group:
                if msg.get('attachments'):
                    try:
                        attachments = msg['attachments']
                        if isinstance(attachments, str):
                            attachments = json.loads(attachments)
                        
                        for attachment in attachments:
                            if isinstance(attachment, str):
                                attachments_summary.append(f"- {attachment}")
                            elif isinstance(attachment, dict):
                                url = attachment.get('url', attachment.get('proxy_url', ''))
                                if url:
                                    attachments_summary.append(f"- {url}")
                    except Exception as e:
                        logger.error(f"Error processing attachments: {e}")
                        continue

            # Add group header with attachments summary
            conversation += "=== Related Messages Group ===\n"
            if attachments_summary:
                conversation += "Group Attachments Summary:\n"
                conversation += "\n".join(attachments_summary) + "\n\n"
            
            for msg in group:
                try:
                    # Start with timestamp, author, and content
                    conversation += f"=== Message from {msg['author_name']} ===\n"
                    
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
                        conversation += f"Time: {timestamp.isoformat()}\n"
                    else:
                        conversation += f"Time: Unknown\n"
                    
                    conversation += f"Content: {msg['content']}\n"
                    
                    # Add reaction information
                    if msg['reaction_count']:
                        conversation += f"Reactions: {msg['reaction_count']}"
                        if msg['reactors']:
                            unique_reactors = len(msg['reactors'])
                            conversation += f" (from {unique_reactors} unique users)"
                        conversation += "\n"
                    
                    # Add detailed attachment information
                    if msg['attachments']:
                        conversation += "Attachments:\n"
                        for attachment in msg['attachments']:
                            if isinstance(attachment, str):
                                # If it's a string, it's probably a URL
                                conversation += f"- {attachment}\n"
                            elif isinstance(attachment, dict):
                                # If it's a dict, extract the URL and any other relevant info
                                url = attachment.get('url', attachment.get('proxy_url', ''))
                                filename = attachment.get('filename', 'unknown')
                                content_type = attachment.get('content_type', 'unknown')
                                conversation += f"- {filename} ({content_type}): {url}\n"
                    
                    # Add message link
                    if msg['jump_url']:
                        conversation += f"Message link: {msg['jump_url']}\n"
                    
                    conversation += "\n"  # Add blank line between messages
                    
                except Exception as e:
                    logger.error(f"Error formatting message: {e}")
                    continue
            conversation += "=== End of Group ===\n\n"

        # Log the exact prompt being sent to Claude (with truncated message content)
        logger.info("\nEXACT PROMPT BEING SENT TO CLAUDE (messages truncated):")
        # Split the conversation into the prompt part and messages part
        prompt_parts = conversation.split("Here are the messages to analyze")
        logger.info(f"=== PROMPT PART ===\n{prompt_parts[0]}")
        logger.info("=== MESSAGES PART (first 500 chars) ===")
        logger.info(prompt_parts[1][:500] + "...")
        logger.info("=== END OF PROMPT ===")

        logger.info(f"Formatted {len(messages)} messages for analysis")
        
        # Add final reminder after all messages
        conversation += "\nRemember to be thorough in finding ALL related media and discussions. Include multiple attachments when available and create subtopics for interesting follow-up discussions. Only return the JSON response in the EXACT format based on the messages above - no introduction or closing text:"

        return conversation

    async def generate_news_summary(self, messages: List[dict]) -> str:
        """Generate a news summary using Claude."""
        if not messages:
            logger.warning("No messages to analyze")
            return "[NO MESSAGES TO ANALYZE]"
            
        logger.info("Generating summary with Claude")
        conversation = self.format_messages_for_claude(messages)
        
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
                        "content": conversation
                    }]
                )
            )
            logger.info("Successfully generated summary with Claude")
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating summary with Claude: {e}")
            raise

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

    def format_news_for_discord(self, news_items_json: str) -> List[dict]:
        """Format JSON news items into Discord markdown, returning a list of messages to send."""
        try:
            if news_items_json == "[NO SIGNIFICANT NEWS]" or news_items_json == "[NO MESSAGES TO ANALYZE]":
                return [{"content": news_items_json}]
                
            news_items = json.loads(news_items_json)
            messages_to_send = []
            
            for item in news_items:
                # Start with main topic header and text
                main_content = [
                    f"## {item['title']}\n",
                    f"{item['mainText']} {item['messageLink']}\n"
                ]
                messages_to_send.append({"content": "\n".join(main_content)})
                
                # Add main files if they exist and aren't placeholders
                if 'mainFile' in item and not item['mainFile'].startswith('⁠'):
                    # Split multiple files and add each as a separate message
                    files = [f.strip() for f in item['mainFile'].split(',')]
                    for file in files:
                        # Skip Discord CDN links that contain "ex=", "is=", and "hm=" parameters
                        if "cdn.discordapp.com" in file and all(param in file for param in ["ex=", "is=", "hm="]):
                            continue
                        messages_to_send.append({"content": f"📎 {file}"})
                
                # Add subtopics
                if 'subTopics' in item and item['subTopics']:
                    for subtopic in item['subTopics']:
                        # Start with bullet point and text
                        content = []
                        
                        # Add text and message link together
                        if 'messageLink' in subtopic:
                            content.append(f"• {subtopic['text']} {subtopic['messageLink']}")
                        else:
                            content.append(f"• {subtopic['text']}")
                            
                        messages_to_send.append({"content": "\n".join(content)})
                        
                        # Add files if they exist and aren't placeholders
                        if 'file' in subtopic and not subtopic['file'].startswith('⁠'):
                            files = [f.strip() for f in subtopic['file'].split(',')]
                            for file in files:
                                # Skip Discord CDN links that contain "ex=", "is=", and "hm=" parameters
                                if "cdn.discordapp.com" in file and all(param in file for param in ["ex=", "is=", "hm="]):
                                    continue
                                messages_to_send.append({"content": f"  📎 {file}"})
                        
                        # Add link if it exists and isn't a placeholder
                        if 'link' in subtopic and not subtopic['link'].startswith('⁠'):
                            messages_to_send.append({"content": f"  🔗 {subtopic['link']}"})
            
            return messages_to_send
            
        except json.JSONDecodeError:
            # If it's not JSON, return as is
            return [{"content": news_items_json}]
        except Exception as e:
            logger.error(f"Error formatting news items: {e}")
            logger.error(traceback.format_exc())
            return [{"content": "Error formatting news items"}]

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

    async def run_news_summary(self):
        """Main function to run the news summary."""
        logger.info("Starting news summary generation...")
        
        try:
            logger.info("Getting all messages from past 24 hours")
            messages = self.get_channel_messages(None)  # None to get all channels
            logger.info(f"Retrieved {len(messages)} messages from database")

            if messages:
                logger.info(f"Generating summary for {len(messages)} messages...")
                try:
                    summary = await self.generate_news_summary(messages)
                    print("\n" + "="*50)
                    print(f"NEWS SUMMARY FOR {datetime.utcnow().strftime('%A, %B %d, %Y')}")
                    print("="*50 + "\n")
                    print(summary)
                    logger.info("Summary generated and printed successfully")
                    
                    # Post to Discord
                    await self.post_to_discord(summary)
                    logger.info("Successfully posted summary to Discord")
                    
                except Exception as e:
                    logger.error(f"Error in summary generation or posting: {e}")
            else:
                logger.warning("No messages found to analyze")
                print("[NO MESSAGES TO ANALYZE]")
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received - shutting down...")
        except Exception as e:
            logger.error(f"Critical error: {e}")
            sys.exit(1)

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
            
            # Start the client without blocking
            asyncio.create_task(self.discord_client.start(self.discord_token))
            
            # Wait for ready with timeout
            try:
                await asyncio.wait_for(ready.wait(), timeout=30)
                logger.info("Discord client ready")
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for Discord client to be ready")
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

    async def run(self):
        """Main async method to run the summarizer."""
        try:
            # Initialize Discord
            await self.setup_discord()
            
            # Run the summary process
            await self.run_news_summary()
        finally:
            # Ensure cleanup happens
            await self.cleanup()

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate and post news summaries.')
        parser.add_argument('--dev', action='store_true', help='Run in development mode')
        args = parser.parse_args()
        
        logger.info("Starting news summarizer...")
        summarizer = NewsSummarizer(dev_mode=args.dev)
        
        # Create event loop
        loop = asyncio.get_event_loop()
        
        # Run the summarizer
        try:
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