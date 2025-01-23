import asyncio
import discord
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.common.db_handler import DatabaseHandler

logger = logging.getLogger('ReactionBackfill')

class ReactionBackfiller(discord.Client):
    def __init__(self, dev_mode=False):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.reactions = True
        super().__init__(intents=intents)
        
        self.dev_mode = dev_mode
        self.db = DatabaseHandler(dev_mode=dev_mode)
        
        # Set up logging
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        
    async def setup_hook(self):
        """Setup hook to initialize any necessary resources."""
        logger.info("Reaction backfiller initialized and ready")
        
    async def backfill_reactions(self):
        """Backfill reaction data for all messages in the database."""
        try:
            # Get all messages without reaction data
            results = self.db.execute_query("""
                SELECT m.message_id, m.channel_id
                FROM messages m
                WHERE (m.reactors IS NULL
                   OR m.reactors = '[]'
                   OR m.reactors = 'null')
                  AND json_valid(m.attachments)
                  AND m.attachments != '[]'
                ORDER BY m.created_at DESC
            """)
            
            if not results:
                logger.info("No messages found needing reaction backfill")
                return
                
            logger.info(f"Found {len(results)} messages needing reaction backfill")
            
            for row in results:
                try:
                    message_id = row['message_id']
                    channel_id = row['channel_id']
                    
                    channel = self.get_channel(channel_id)
                    if not channel:
                        logger.warning(f"Could not find channel {channel_id}")
                        continue
                        
                    message = await channel.fetch_message(message_id)
                    if not message:
                        logger.warning(f"Could not find message {message_id}")
                        continue
                    
                    # Get list of unique reactors
                    reactors = []
                    reaction_count = 0
                    
                    if message.reactions:
                        for reaction in message.reactions:
                            reaction_count += reaction.count
                            async for user in reaction.users():
                                if user.id not in reactors and user.id != self.user.id:
                                    reactors.append(user.id)
                    
                    # Update database
                    self.db.execute_query("""
                        UPDATE messages
                        SET reaction_count = ?,
                            reactors = ?
                        WHERE message_id = ?
                    """, (reaction_count, json.dumps(reactors), message_id))
                    
                    logger.info(f"Updated reactions for message {message_id}: {len(reactors)} reactors")
                    await asyncio.sleep(0.5)  # Rate limit
                    
                except Exception as e:
                    logger.error(f"Error processing message {row['message_id']}: {e}")
                    continue
            
            logger.info("Reaction backfill complete")
            
        except Exception as e:
            logger.error(f"Error during reaction backfill: {e}")
        finally:
            await self.close()
            
    async def on_ready(self):
        """Called when the client is ready."""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        await self.backfill_reactions()

def main():
    """Main entry point for running the reaction backfiller."""
    try:
        load_dotenv()
        dev_mode = os.getenv('DEV_MODE', '').lower() == 'true'
        
        backfiller = ReactionBackfiller(dev_mode=dev_mode)
        backfiller.run(os.getenv('DISCORD_BOT_TOKEN'))
        
    except Exception as e:
        logger.error(f"Failed to run reaction backfiller: {e}")
        raise

if __name__ == "__main__":
    main() 