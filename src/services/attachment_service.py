import discord
import logging
import io
import random
import traceback
import aiohttp

from typing import List, Tuple, Dict, Any, Optional, Set

from utils.errors import APIError

logger = logging.getLogger('AttachmentService')


class Attachment:
    def __init__(self, filename: str, data: bytes, content_type: str, reaction_count: int, username: str):
        self.filename = filename
        self.data = data
        self.content_type = content_type
        self.reaction_count = reaction_count
        self.username = username


class AttachmentService:
    """Handles all attachment-related operations"""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = 25 * 1024 * 1024

    def clear_cache(self):
        """Clears the attachment cache at the start of each channel processing."""
        self.cache.clear()

    async def process_attachment(self, attachment: discord.Attachment, message: discord.Message, session) -> Optional[Attachment]:
        """Download and store an attachment, respecting file size limit, etc."""
        try:
            logger.debug(f"Processing attachment {attachment.filename} from message {message.id}")
            cache_key = f"{message.channel.id}:{message.id}"

            async with session.get(attachment.url, timeout=300) as response:
                if response.status != 200:
                    raise APIError(f"Failed to download attachment: HTTP {response.status}")

                file_data = await response.read()
                if len(file_data) > self.max_size:
                    logger.warning(f"Skipping large file {attachment.filename} ({len(file_data)/1024/1024:.2f}MB)")
                    return None

                total_reactions = sum(r.count for r in message.reactions) if message.reactions else 0

                processed_attachment = Attachment(
                    filename=attachment.filename,
                    data=file_data,
                    content_type=attachment.content_type or '',
                    reaction_count=total_reactions,
                    username=message.author.name
                )

                if cache_key not in self.cache:
                    self.cache[cache_key] = {
                        'attachments': [],
                        'reaction_count': total_reactions,
                        'username': message.author.name,
                        'channel_id': str(message.channel.id)
                    }
                self.cache[cache_key]['attachments'].append(processed_attachment)

                return processed_attachment

        except Exception as e:
            logger.error(f"Failed to process attachment {attachment.filename}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def prepare_chunk_files(self, message_ids: set) -> List[Tuple[discord.File, int, str, str]]:
        """Returns up to top 10 attachments from the cache for the given set of message_ids."""
        files = []
        for message_id in message_ids:
            # The cache key includes channel_id, so we need to find them
            for cache_key, data in self.cache.items():
                ch_id, msg_id = cache_key.split(":", 1)
                if msg_id == message_id:
                    for att in data['attachments']:
                        try:
                            file_obj = discord.File(
                                io.BytesIO(att.data),
                                filename=att.filename,
                                description=f"From message ID: {msg_id} (🔥 {att.reaction_count} reactions)"
                            )
                            files.append((file_obj, att.reaction_count, msg_id, att.username))
                        except Exception as e:
                            logger.error(f"Failed to prepare file {att.filename}: {e}")
        # Sort by reaction count descending
        files.sort(key=lambda x: x[1], reverse=True)
        return files[:10]

    def get_unused_popular_attachments(self, used_message_ids: Set[str], guild_id: int, main_attachment_id: str = None) -> List[Dict[str, Any]]:
        """Return a list of up to 10 attachments that haven't been used but have >= 3 reactions.
        Excludes the main attachment that was used in the thread header."""
        results = []
        for cache_key, data in self.cache.items():
            channel_part, message_part = cache_key.split(":", 1)

            # Skip if message was used in summary or is the main attachment
            if message_part in used_message_ids or message_part == main_attachment_id:
                continue

            if data['reaction_count'] >= 3:
                message_link = f"https://discord.com/channels/{guild_id}/{channel_part}/{message_part}"
                for att in data['attachments']:
                    try:
                        if len(att.data) <= self.max_size:
                            file_obj = discord.File(
                                io.BytesIO(att.data),
                                filename=att.filename,
                                description=f"From {att.username} (🔥 {att.reaction_count} reactions)"
                            )
                            results.append({
                                'file': file_obj,
                                'reaction_count': att.reaction_count,
                                'message_link': message_link,
                                'username': att.username
                            })
                    except Exception as e:
                        logger.error(f"Failed to prepare unused attachment: {e}")
                        continue
        results.sort(key=lambda x: x['reaction_count'], reverse=True)
        return results

    def get_top_media(self) -> Optional[Tuple[discord.File, int, str]]:
        """Return the most popular media from the entire cache. (file, reaction_count, msg_id)"""
        best = None
        best_reactions = 2
        best_msg_id = None
        best_file = None

        for cache_key, data in self.cache.items():
            if data['reaction_count'] > best_reactions:
                for att in data['attachments']:
                    if any(att.filename.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.mov', '.webm')):
                        best = att
                        best_reactions = data['reaction_count']
                        # best overall attachments
                        best_msg_id = cache_key.split(":", 1)[1]

        if best:
            try:
                file_obj = discord.File(
                    io.BytesIO(best.data),
                    filename=best.filename,
                    description=f"Most popular media (🔥 {best_reactions} reactions)"
                )
                best_file = file_obj
            except Exception as e:
                logger.error(f"Failed to prepare top media file: {e}")
                return None
            return (best_file, best_reactions, best_msg_id)

        return None

    def get_all_files_sorted(self) -> List[Tuple[discord.File, int, str, str]]:
        """Return all attachments in sorted order by reaction_count desc. 
           (file, reaction_count, message_id, username)
        """
        all_files = []
        for cache_key, data in self.cache.items():
            ch_part, msg_part = cache_key.split(":", 1)
            for att in data['attachments']:
                try:
                    file_obj = discord.File(
                        io.BytesIO(att.data),
                        filename=att.filename,
                        description=f"From {att.username} (🔥 {att.reaction_count} reactions)"
                    )
                    all_files.append((file_obj, att.reaction_count, msg_part, att.username))
                except Exception as e:
                    logger.warning(f"Could not prepare file for {att.filename}: {e}")
                    continue

        all_files.sort(key=lambda x: x[1], reverse=True)
        return all_files

    async def process_channel_attachments(self, messages: List[Dict]) -> List[Dict]:
        """Process attachments from channel messages
        
        Args:
            messages: List of message dictionaries containing message data
            
        Returns:
            List of processed attachment dictionaries with fields:
            - id: Attachment ID
            - file: Discord File object
            - username: Username who posted
            - message_link: Link to original message
        """
        processed_attachments = []
        
        for message in messages:
            if not message.get('attachments'):
                continue
                
            for attachment in message['attachments']:
                try:
                    # Create discord File object from attachment URL
                    file = await self._download_attachment(attachment['url'])
                    
                    processed_attachments.append({
                        'id': attachment['id'],
                        'file': file,
                        'username': message['author']['name'],
                        'message_link': message['jump_url']
                    })
                except Exception as e:
                    logger.error(f"Error processing attachment: {str(e)}")
                    continue
                    
        return processed_attachments
        
    async def _download_attachment(self, url: str) -> discord.File:
        """Download attachment from URL and return as Discord File object"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    filename = url.split('/')[-1]
                    return discord.File(io.BytesIO(data), filename=filename)
                else:
                    raise Exception(f"Failed to download attachment: {response.status}")