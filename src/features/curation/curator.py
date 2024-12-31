import discord
from discord.ext import commands
import asyncio
import os
import logging
import traceback
from src.common.log_handler import LogHandler

class ArtCurator(commands.Bot):
    def __init__(self, logger=None):
        # Setup intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.members = True
        intents.guilds = True
        intents.emojis = True

        super().__init__(command_prefix='!', intents=intents)
        
        # Setup logger
        self.logger = logger or logging.getLogger('ArtCurator')
        
        # Initialize variables that will be set by dev_mode setter
        self._dev_mode = None
        self.art_channel_id = None
        self.curator_ids = []
        
        # Set initial dev mode (this will trigger the setter to load correct IDs)
        self.dev_mode = False
        
        # Add a set to track curators currently in rejection flow
        self._active_rejections = set()
        
        # Register event handlers
        self.setup_events()
        
        # Shutdown flag for clean exit
        self._shutdown_flag = False
        
    @property
    def dev_mode(self):
        return self._dev_mode
        
    @dev_mode.setter
    def dev_mode(self, value):
        self._dev_mode = value
        # Update channel ID based on dev mode
        if value:
            self.art_channel_id = int(os.getenv('DEV_ART_CHANNEL_ID', 0))
            self.curator_ids = [int(id) for id in os.getenv('DEV_CURATOR_IDS', '').split(',') if id]
            self.logger.info(f"Using development art channel: {self.art_channel_id}")
            self.logger.info(f"Using development curator IDs: {self.curator_ids}")
        else:
            self.art_channel_id = int(os.getenv('ART_CHANNEL_ID', 0))
            self.curator_ids = [int(id) for id in os.getenv('CURATOR_IDS', '').split(',') if id]
            self.logger.info(f"Using production art channel: {self.art_channel_id}")
            self.logger.info(f"Using production curator IDs: {self.curator_ids}")
        
    def setup_events(self):
        @self.event
        async def on_ready():
            self.logger.info(f'{self.user} has connected to Discord!')
            self.logger.info(f'Bot is in {len(self.guilds)} guilds')
            self.logger.info(f'Intents configured: {self.intents}')
            self.logger.info(f'Art channel ID: {self.art_channel_id}')
            self.logger.info(f'Curator IDs: {self.curator_ids}')

        @self.event
        async def on_message(message):
            # Ignore messages from the bot itself
            if message.author == self.user:
                return

            # Only log message receipt in dev mode
            if self.dev_mode:
                self.logger.debug(f"Received message from {message.author} in channel {message.channel.id}")

            # Check if message is in the art channel
            if message.channel.id == self.art_channel_id:
                self.logger.info(f"Processing message in art channel from {message.author}")
                allowed_domains = [
                    'youtube.com', 'youtu.be',    # YouTube
                    'vimeo.com',                  # Vimeo
                    'tiktok.com',                 # TikTok
                    'streamable.com',             # Streamable
                    'twitch.tv'                   # Twitch
                ]

                # Check if attachments are valid media files
                has_valid_attachment = any(
                    (attachment.content_type and (
                        attachment.content_type.startswith('image/') or 
                        attachment.content_type.startswith('video/')
                    )) or 
                    (attachment.filename.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.svg',
                         '.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.m4v',
                         '.heic', '.heif',
                         '.apng')
                    ))
                    for attachment in message.attachments
                )

                if self.dev_mode:
                    self.logger.debug(f"Message has valid attachment: {has_valid_attachment}")

                # Check for any links in the message
                links = [word for word in message.content.split() 
                        if 'http://' in word or 'https://' in word]
                
                if self.dev_mode:
                    self.logger.debug(f"Found links in message: {links}")
                
                # Check if any of the links are from allowed domains
                has_valid_link = any(
                    domain in link.lower() 
                    for domain in allowed_domains 
                    for link in links
                )

                if self.dev_mode:
                    self.logger.debug(f"Message has valid link: {has_valid_link}")

                # Only delete if there's no valid content at all
                if not (has_valid_attachment or has_valid_link):
                    has_invalid_links = bool(links)
                    
                    try:
                        await message.delete()
                        self.logger.info(f"Deleted message from {message.author} in channel {message.channel.id} due to invalid content.")
                    except discord.Forbidden:
                        self.logger.error(f"Permission denied: Couldn't delete message from {message.author}.")
                    except Exception as e:
                        self.logger.error(f"Error deleting message: {e}", exc_info=True)
                    
                    try:
                        if has_invalid_links:
                            await message.author.send(
                                f"Hi <@{message.author.id}>,\n\n"
                                "Your message in the art channel was deleted because it contained links "
                                "but none were from platforms that Discord can embed. Please make sure to "
                                "include either media files or links from supported platforms.\n\n"
                                "If you'd like to ask a question about or comment on a specific piece, "
                                "please use the thread under that post."
                            )
                        else:
                            await message.author.send(
                                f"Hi <@{message.author.id}>,\n\n"
                                "Your message in the art channel was deleted because it didn't contain "
                                "a valid image/video file or a link from a platform that Discord can embed.\n\n"
                                "If you'd like to ask a question about or comment on a specific piece, "
                                "please use the thread under that post."
                            )
                        self.logger.info(f"Sent deletion reason DM to {message.author}.")
                    except discord.Forbidden:
                        self.logger.warning(f"Couldn't DM user {message.author}")
                    except Exception as e:
                        self.logger.error(f"Error sending DM to {message.author}: {e}", exc_info=True)
                    return

                # If post contains both a file and links, suppress embeds
                if has_valid_attachment and links:
                    try:
                        await message.edit(suppress=True)
                        self.logger.info(f"Suppressed embeds for message from {message.author}.")
                    except discord.Forbidden:
                        self.logger.error("Bot doesn't have permission to edit messages.")
                    except Exception as e:
                        self.logger.error(f"Error editing message: {e}", exc_info=True)

                # Try to create thread
                try:
                    thread = await message.create_thread(
                        name=f"Discussion: {message.author.display_name}'s Art",
                        auto_archive_duration=10080  # Archives after 1 week of inactivity
                    )
                    self.logger.info(f"Created thread for message from {message.author}.")
                    
                    # Add message about tagging the author
                    await thread.send(f"Make sure to tag <@{message.author.id}> in messages to make sure they see your comment!")
                    self.logger.info(f"Added tagging reminder message to thread.")
                except Exception as e:
                    self.logger.error(f"Error creating thread: {e}", exc_info=True)

            # Process commands after handling the message
            await self.process_commands(message)

        @self.event
        async def on_reaction_add(reaction, user):
            # Skip this handler as we'll handle everything in on_raw_reaction_add
            pass

        @self.event
        async def on_raw_reaction_add(payload):
            if self.dev_mode:
                self.logger.debug(f"Raw reaction event received: {payload.emoji}")
            
            # Check if reaction is X and in art channel
            if (str(payload.emoji) in ['❌', '𝕏', 'X', '🇽'] and 
                payload.channel_id == self.art_channel_id):
                
                self.logger.info(f"X reaction received in art channel from user {payload.user_id}")
                
                if self.dev_mode:
                    self.logger.debug(f"Reaction from user ID {payload.user_id}. Curators are: {self.curator_ids}")
                
                # Get the channel and message
                channel = self.get_channel(payload.channel_id)
                if channel is None:
                    self.logger.error(f"Channel with ID {payload.channel_id} not found.")
                    return
                try:
                    message = await channel.fetch_message(payload.message_id)
                    user = self.get_user(payload.user_id)
                    if user is None:
                        self.logger.error(f"User with ID {payload.user_id} not found.")
                        return
                    
                    # Check if reactor is a curator
                    if payload.user_id in self.curator_ids:
                        self.logger.info(f"Valid curator {user.name} ({payload.user_id}) reacted with X")
                        await self._handle_curator_rejection(message, user)
                    else:
                        self.logger.info(f"Non-curator {user.name} ({payload.user_id}) reacted with X - ignoring")
                        
                except Exception as e:
                    self.logger.error(f"Error fetching message or user: {e}", exc_info=True)
                    return

        @self.event
        async def on_error(event, *args, **kwargs):
            self.logger.error(f'Error in {event}:', exc_info=True)
            traceback.print_exc()

    async def _handle_curator_rejection(self, message, user):
        """Handle the rejection process when a curator reacts with X"""
        # Check if curator is already processing a rejection
        if user.id in self._active_rejections:
            try:
                await user.send("Please complete your current rejection process before starting a new one. Reply to the above message with your reason for rejection or reply 'forget' to stop that rejection.")
                await message.remove_reaction('❌', user)
                self.logger.info(f"Curator {user.name} attempted multiple rejections - blocked.")
                return
            except discord.Forbidden:
                self.logger.error(f"Couldn't DM curator {user.name} about multiple rejections.")
                return

        self._active_rejections.add(user.id)
        self.logger.warning(f"Curator {user.name} initiated a rejection.")
        
        try:
            # Get the content URL or first attachment URL
            content_url = ""
            if message.attachments:
                content_url = message.attachments[0].url
            elif message.content:
                # Extract first URL from content if it exists
                words = message.content.split()
                urls = [word for word in words if word.startswith(('http://', 'https://'))]
                if urls:
                    content_url = urls[0]

            # DM curator asking for reason
            prompt = f"You rejected an art post by <@{message.author.id}>"
            if content_url:
                prompt += f": {content_url}"
            prompt += "\n\nPlease reply with the reason for rejection within 5 minutes or reply 'forget' to stop rejection:"
            
            await user.send(prompt)
            if self.dev_mode:
                self.logger.debug(f"Sent DM to curator {user.name} for reason.")

            def check(m):
                return m.author == user and isinstance(m.channel, discord.DMChannel)
            
            # Wait for curator's response
            try:
                reason_msg = await self.wait_for('message', check=check, timeout=300.0)
                reason = reason_msg.content
                
                # Check if curator wants to cancel
                if reason.lower().strip() == 'forget':
                    await user.send("Rejection cancelled.")
                    await message.remove_reaction('❌', user)
                    self.logger.warning(f"Curator {user.name} cancelled the rejection.")
                    return
                
                if self.dev_mode:
                    self.logger.info(f"Received rejection reason from {user.name}: {reason}")

                # Only proceed with deletion if we got a non-empty reason
                if reason.strip():
                    # Store author before deleting message
                    author = message.author
                    
                    # Delete the post first
                    await message.delete()
                    self.logger.warning(f"Deleted message from {author} as per curator {user.name}.")

                    # Format the reason
                    formatted_reason = "" + reason.strip().replace('\n', '\n> ')
                    
                    # Send DM to the original author
                    try:
                        # Format the message parts separately to handle newlines
                        message_parts = [
                            f"Hi <@{author.id}>,\n\n",
                            f"I'm sorry to say that your art post was removed by curator <@{user.id}>.\n\n",
                            f"**Reason for removal:**\n> {formatted_reason}\n\n"
                        ]
                        
                        # Add file reference if there was an attachment
                        if message.attachments:
                            message_parts.extend([
                                "**Your submission:**\n",
                                f"> {message.attachments[0].url}\n\n"
                            ])
                        elif message.content.strip():  # If no attachment but has content (likely a link)
                            content = message.content.strip()
                            urls = [word for word in content.split() 
                                  if word.startswith(('http://', 'https://'))]
                            
                            # Add the first URL if found
                            if urls:
                                message_parts.extend([
                                    "**Your submission:**\n",
                                    f"> {urls[0]}\n\n"
                                ])
                                
                                # Remove the URL from content and check if there's remaining text
                                remaining_content = ' '.join(
                                    word for word in content.split() 
                                    if not word.startswith(('http://', 'https://'))
                                ).strip()
                                
                                if remaining_content:
                                    message_parts.extend([
                                        "**Your comment:**\n",
                                        f"> {remaining_content}\n\n"
                                    ])
                            else:
                                message_parts.extend([
                                    "**Your comment:**\n",
                                    f"> {content}\n\n"
                                ])

                        # Add footer
                        message_parts.append(f"If you would like to discuss this further, please DM <@{user.id}> directly.")

                        # Join all parts and send
                        final_message = ''.join(message_parts)
                        await author.send(final_message)
                        if self.dev_mode:
                            self.logger.info(f"Sent removal reason DM to {author}.")
                        
                        # Get all threads in the channel
                        threads = await message.guild.active_threads()
                        if self.dev_mode:
                            self.logger.debug(f"Found {len(threads)} active threads")
                            for thread in threads:
                                self.logger.debug(f"Thread {thread.id}: parent={thread.parent_id}, starter={thread.starter_message.id if thread.starter_message else 'None'}")
                        
                        # Find the thread that was started from this message and is in the correct channel
                        message_thread = None
                        for thread in threads:
                            try:
                                if (thread.parent_id == message.channel.id and 
                                    thread.name.startswith(f"Discussion: {message.author.display_name}")):
                                    message_thread = thread
                                    break
                            except Exception as e:
                                if self.dev_mode:
                                    self.logger.debug(f"Error checking thread {thread.id}: {e}")
                                continue
                        
                        if message_thread:
                            # Delete all messages in the thread first
                            try:
                                async for msg in message_thread.history(limit=None, oldest_first=False):
                                    await msg.delete()
                                    if self.dev_mode:
                                        self.logger.info(f"Deleted message in thread for {author}'s post")
                            except Exception as e:
                                self.logger.error(f"Error deleting thread messages: {e}", exc_info=True)

                            # Then delete the thread itself
                            try:
                                await message_thread.delete()
                                if self.dev_mode:
                                    self.logger.info(f"Deleted thread for message from {author}.")
                            except Exception as e:
                                self.logger.error(f"Error deleting thread: {e}", exc_info=True)
                        else:
                            if self.dev_mode:
                                self.logger.debug(f"No thread found for message from {author}.")
                    except Exception as e:
                        self.logger.error(f"Error handling thread deletion: {e}", exc_info=True)
                                
                    except discord.Forbidden:
                        self.logger.warning(f"Couldn't DM original poster {author}.")
                else:
                    await user.send("Empty reason provided. Post will not be deleted.")
                    await message.remove_reaction('❌', user)
                    self.logger.warning(f"Empty reason provided by {user.name}. Reaction removed.")
                
            except asyncio.TimeoutError:
                await user.send("No reason provided within 5 minutes. Post will not be deleted.")
                await message.remove_reaction('❌', user)
                self.logger.warning(f"Curator {user.name} did not provide a reason in time.")
                
        except discord.Forbidden:
            self.logger.error(f"Couldn't DM curator {user.name}.")

        finally:
            # Make sure we remove the curator from active rejections even if there's an error
            self._active_rejections.remove(user.id)
