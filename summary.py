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

        # Categories to monitor
        self.category_ids = [
            1212472332484870224,
            1138787029170258001,
            1307827932147744868,
            1275200515407609940,
            1147226523187822713,            
            1221869948469776516,
            1076117621407223830
        ]
        self.summary_channel_id = 1316024582041243668

        print("Bot initialized")

    async def get_channel_history(self, channel_id):
        print(f"Attempting to get history for channel {channel_id}")
        channel = self.get_channel(channel_id)
        if not channel:
            print(f"Could not find channel with ID {channel_id}")
            return []
        
        yesterday = datetime.utcnow() - timedelta(days=1)
        messages = []
        
        # Store attachments in a dictionary keyed by message ID for easy lookup
        self.attachment_cache = {}
        
        async for message in channel.history(after=yesterday, limit=None):
            # Calculate total reaction count for the message
            total_reactions = sum(reaction.count for reaction in message.reactions)
            
            # Cache attachments if the message has any and meets reaction threshold
            if message.attachments and (
                total_reactions >= 3  # Messages with 3+ reactions
                or any(attachment.filename.lower().endswith(('.mp4', '.mov', '.webm')) 
                      for attachment in message.attachments)  # Always include videos
            ):
                attachments = []
                for attachment in message.attachments:
                    try:
                        # Increase timeout for larger files
                        async with self.session.get(attachment.url, timeout=300) as response:
                            if response.status == 200:
                                file_data = await response.read()
                                # Only store if file size is under Discord's upload limit (25MB)
                                if len(file_data) <= 25 * 1024 * 1024:  # 25MB in bytes
                                    attachments.append({
                                        'filename': attachment.filename,
                                        'data': file_data,
                                        'content_type': attachment.content_type,
                                        'reaction_count': total_reactions  # Store reaction count
                                    })
                                else:
                                    print(f"Skipping large file {attachment.filename} ({len(file_data)/1024/1024:.2f}MB)")
                            else:
                                print(f"Failed to download attachment: HTTP {response.status}")
                    except Exception as e:
                        print(f"Failed to download attachment {attachment.filename}: {e}")
                
                if attachments:
                    self.attachment_cache[str(message.id)] = {
                        'attachments': attachments,
                        'reaction_count': total_reactions
                    }
            
            # Get message content, but remove any URLs that would create embeds
            content = message.content
            if content:
                # Remove URLs that are on their own line (common for embeds)
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
        
        print(f"Found {len(messages)} messages in channel {channel.name}")
        return messages

    def get_claude_summary(self, messages):
        if not messages:
            print("No messages to summarize")
            return "[NOTHING OF NOTE]"
        
        print(f"Generating summary for {len(messages)} messages")
        conversation = """Please summarize the interesting and noteworthy Discord happenings and idaes in bullet points. You should extract ideas and information that may be useful to others from conversations. Avoid stuff like bug reports that are circumstantial or not useful to others. Break them into topics and sub-topics with relevant links.

If there's nothing significant or noteworthy in the messages, just respond with exactly "[NOTHING OF NOTE]" (and no other text).

Here are bad topics to cover:
- Bug reports that seem unremarkable and not useful to others
- Messages that are not helpful or informative to others
- Discussions that ultimately have no real value to others
- Humourous messages that are not helpful or informative to others

While here's an example of what a good summary and topic should look like:
• 🏋️ **Person Training for Hunyuan Video is Now Possible**
    - Users are experimenting with training LoRAs using images and videos
    - **Kytra** explained that training can be done on relatively modest hardware (24GB VRAM): <https://discord.com/channels/1076117621407223829/1316079815446757396/1316418253102383135>
    - **TDRussell** provided the repository link: https://github.com/tdrussell/diffusion-pipe

And here's another example of a good summary and topic:

• 🤏 **H264/H265 Compression Techniques for Video Generation Improves Img2Vid**
    - **zeevfarbman** recommended h265 compression for frame degradation with less perceptual impact: <https://discord.com/channels/1076117621407223829/1309520535012638740/1316462339318354011>
    - **johndopamine** suggested using h264 node in MTB node pack for better video generation
    - Codec compression can help "trick" current workflows/models
    - melmass confirmed adding h265 support to their tools: <https://discord.com/channels/1076117621407223829/1309520535012638740/1316786801247260672>

Format requirements:
1. Make sure to ALWAYS include relevant message Discord AND external links as references in this format: <message_url>
2. Use Discord's markdown format (not regular markdown)
3. Use • for main topics and properly indented - for sub-points (4 spaces before the -)
4. Use ** for bold text (especially for usernames and main topics)
5. Keep it simple - just bullet points and sub-points for each topic, no headers or complex formatting
6. ALWAYS include the message author's name in bold (**username**) for each point if there's a specific person who did something, said something important, or seemed to be helpful - mention their username, don't tag them. Call them "Banodocians" instead of "users".
7. Always include a funny or relevant emoji in the topic title
8. Highlight messages with significant reactions (3+ reactions) and mention the reaction counts
9. Use 🔥 to indicate highly-reacted messages (5+ total reactions)

IMPORTANT: For each bullet point, use the EXACT message URL provided in the data - do not write <message_url> but instead use the actual URL from the message data.

Please provide the summary now:\n\n"""
        
        for msg in messages:
            conversation += f"{msg['timestamp']} - {msg['author']}: {msg['content']}"
            if msg['reactions']:
                conversation += f"\nReactions: {msg['reactions']}"
            conversation += f"\nMessage link: {msg['jump_url']}\n\n"
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.claude.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=8192,
                    messages=[
                        {
                            "role": "user",
                            "content": conversation
                        }
                    ]
                )
                
                summary_text = response.content[0].text.strip()
                print("Summary generated successfully")
                return summary_text
            except Exception as e:
                retry_count += 1
                print(f"Error attempt {retry_count}/{max_retries} while generating summary: {e}")
                if retry_count < max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("All retry attempts failed")
                    return "An error occurred while generating the summary."

    async def post_summary(self, channel_id, summary):
        print(f"Attempting to post summary to channel {channel_id}")
        channel = self.get_channel(channel_id)
        if channel:
            try:
                # Collect all files from referenced messages
                all_files = []
                for message_id, cache_data in self.attachment_cache.items():
                    if message_id in summary:
                        for attachment in cache_data['attachments']:
                            all_files.append((
                                attachment,
                                cache_data['reaction_count'],
                                message_id
                            ))
                
                # Sort files by reaction count (highest first)
                all_files.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 10 files or all if less than 10
                files = []
                for attachment, reaction_count, message_id in all_files[:10]:
                    try:
                        if len(attachment['data']) <= 25 * 1024 * 1024:  # 25MB limit
                            file = discord.File(
                                io.BytesIO(attachment['data']),
                                filename=attachment['filename'],
                                description=f"From message ID: {message_id} (🔥 {reaction_count} reactions)"
                            )
                            files.append(file)
                        else:
                            print(f"Skipping large file {attachment['filename']}")
                    except Exception as e:
                        print(f"Failed to create Discord File object: {e}")

                # Split summary if it's too long
                max_length = 1900  # Leave some room for the header
                
                if len(summary) > max_length:
                    # Split into chunks at newlines
                    chunks = []
                    current_chunk = ""
                    
                    for line in summary.split('\n'):
                        if len(current_chunk) + len(line) + 1 <= max_length:
                            current_chunk += line + '\n'
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = line + '\n'
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Send first chunk with files
                    if files:
                        await channel.send(chunks[0], files=files)
                    else:
                        await channel.send(chunks[0])
                    
                    # Send remaining chunks
                    for chunk in chunks[1:]:
                        await channel.send(chunk)
                else:
                    # Send everything in one message if it's short enough
                    if files:
                        await channel.send(summary, files=files)
                    else:
                        await channel.send(summary)
                
                print(f"Successfully posted summary to {channel.name}")
            except Exception as e:
                print(f"Failed to post summary: {e}")
        else:
            print(f"Could not find channel to post summary: {channel_id}")

    async def generate_summary(self):
        print("\nStarting summary generation")
        
        try:
            summary_channel = self.get_channel(self.summary_channel_id)
            if not summary_channel:
                print(f"Error: Could not access summary channel {self.summary_channel_id}")
                return
            
            print(f"Found summary channel: {summary_channel.name}")
            
            # Track if we've found any active channels
            active_channels = False
            
            # Get current date in the desired format
            current_date = datetime.utcnow()
            date_string = current_date.strftime("%A, %B %d")  # e.g., "Monday, October 28"
            # Add the ordinal suffix (th, st, nd, rd)
            day = current_date.day
            if day in (1, 21, 31):
                suffix = "st"
            elif day in (2, 22):
                suffix = "nd"
            elif day in (3, 23):
                suffix = "rd"
            else:
                suffix = "th"
            
            try:
                await summary_channel.send(f"# Summary for {date_string}{suffix}")
                print("Posted header message successfully")
            except Exception as e:
                print(f"Failed to post header: {e}")
                return
            
            # Iterate through each category
            for category_id in self.category_ids:
                category = self.get_channel(category_id)
                if not category:
                    print(f"Error: Could not access category {category_id}")
                    continue
                
                print(f"\nProcessing category: {category.name}")
                
                # Get all text channels in this category
                channels = [channel for channel in category.channels 
                          if isinstance(channel, discord.TextChannel)]
                
                if not channels:
                    print(f"No text channels found in category {category.name}")
                    continue
                
                # Process each channel in the category
                for channel in channels:
                    print(f"\nProcessing channel {channel.name}")
                    try:
                        messages = await self.get_channel_history(channel.id)
                        
                        # Only generate summary if there are enough messages
                        if len(messages) >= 20:
                            summary = self.get_claude_summary(messages)
                            
                            # Only post if there's something noteworthy
                            if "[NOTHING OF NOTE]" not in summary:
                                active_channels = True
                                # Send channel-specific header before the summary
                                await summary_channel.send(f"# <#{channel.id}> Summary\n")
                                await self.post_summary(self.summary_channel_id, summary)
                                # Add line break between channel summaries
                                await summary_channel.send("───────────────────────")
                                
                                print(f"Generated summary for {channel.name} with {len(messages)} messages")
                            else:
                                print(f"No noteworthy activity in {channel.name}")
                            
                            print("Waiting before processing next channel...")
                            await asyncio.sleep(1)
                        else:
                            print(f"Skipping {channel.name} - only {len(messages)} messages in the past 24 hours")
                    except discord.Forbidden:
                        print(f"Missing permissions to access channel {channel.name}")
                        continue
                    except Exception as e:
                        print(f"Error processing channel {channel.name}: {e}")
                        continue
            
            # If no channels had noteworthy activity, send a message saying so
            if not active_channels:
                await summary_channel.send("No channels had significant activity in the past 24 hours.")
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            import traceback
            print("Full error:", traceback.format_exc())

    async def setup_hook(self):
        print("Setup hook started")
        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()
        print("Setup hook completed")

    async def close(self):
        print("Closing bot...")
        # Close aiohttp session if it exists
        if self.session:
            await self.session.close()
        # Call parent's close method
        await super().close()

async def schedule_daily_summary(bot):
    while True:
        # Get current time and target time
        now = datetime.utcnow()
        target = now.replace(hour=10, minute=0, second=0, microsecond=0)
        
        # If we've already passed 10am today, schedule for tomorrow
        if now.hour >= 10:
            target += timedelta(days=1)
            
        # Calculate delay until next run
        delay = (target - now).total_seconds()
        print(f"Waiting {delay/3600:.2f} hours until next summary at {target} UTC")
        
        # Wait until target time
        await asyncio.sleep(delay)
        
        try:
            # Generate and post summary
            await bot.generate_summary()
            print(f"Summary generated successfully at {datetime.utcnow()} UTC")
        except Exception as e:
            print(f"Error generating summary: {e}")
            
        # Add small delay to prevent potential double runs
        await asyncio.sleep(60)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Load environment variables
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not bot_token:
        raise ValueError("Discord bot token not found in environment variables")
    if not anthropic_key:
        raise ValueError("Anthropic API key not found in environment variables")
        
    # Create bot instance
    bot = ChannelSummarizer()
    
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Modify the on_ready event to start the scheduler
    @bot.event
    async def on_ready():
        print(f"Logged in as {bot.user.name} ({bot.user.id})")
        print("Connected to servers:", [guild.name for guild in bot.guilds])
        
        # Start the scheduler
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