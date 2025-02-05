import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from .constants import get_database_path
import time

logger = logging.getLogger('DiscordBot')

class DatabaseHandler:
    def __init__(self, db_path: Optional[str] = None, dev_mode: bool = False):
        """Initialize database connection and ensure directory exists."""
        try:
            # Use provided path or get appropriate path based on mode
            self.db_path = db_path if db_path else get_database_path(dev_mode)
            
            # Ensure the directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure SQLite for better concurrency
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)  # Increase timeout
            self.conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
            self.conn.execute("PRAGMA busy_timeout=30000")  # Set busy timeout to 30 seconds
            self.cursor = self.conn.cursor()
            
            # Initialize all tables
            self.init_db()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def init_db(self):
        """Initialize all database tables."""
        try:
            # Channels table with new fields - define this FIRST since other tables reference it
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id BIGINT PRIMARY KEY,
                    channel_name TEXT NOT NULL,
                    description TEXT,
                    suitable_posts TEXT,
                    unsuitable_posts TEXT,
                    rules TEXT,
                    setup_complete BOOLEAN DEFAULT FALSE,
                    nsfw BOOLEAN DEFAULT FALSE,
                    enriched BOOLEAN DEFAULT FALSE
                )
            """)

            # Channel summary threads table with foreign key
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS channel_summary (
                    channel_id BIGINT,
                    summary_thread_id BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id),
                    PRIMARY KEY (channel_id, created_at)
                )
            """)
            
            # Daily summaries table with foreign key
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    daily_summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    channel_id BIGINT NOT NULL REFERENCES channels(channel_id),
                    full_summary TEXT,
                    short_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, channel_id) ON CONFLICT REPLACE
                )
            """)
            
            # Members table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    member_id BIGINT PRIMARY KEY,
                    username TEXT NOT NULL,
                    global_name TEXT,
                    server_nick TEXT,
                    avatar_url TEXT,
                    discriminator TEXT,
                    bot BOOLEAN DEFAULT FALSE,
                    system BOOLEAN DEFAULT FALSE,
                    accent_color INTEGER,
                    banner_url TEXT,
                    discord_created_at TEXT,
                    guild_join_date TEXT,
                    role_ids TEXT,  /* JSON array of role IDs */
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Messages table with FTS support
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id BIGINT PRIMARY KEY,
                    channel_id BIGINT NOT NULL,
                    author_id BIGINT NOT NULL,
                    content TEXT,
                    created_at TEXT,
                    attachments TEXT,
                    embeds TEXT,
                    reaction_count INTEGER,
                    reactors TEXT,
                    reference_id BIGINT,
                    edited_at TEXT,
                    is_pinned BOOLEAN,
                    thread_id BIGINT,
                    message_type TEXT,
                    flags INTEGER,
                    jump_url TEXT,
                    is_deleted BOOLEAN DEFAULT FALSE,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES members(member_id),
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                )
            """)
            
            # Full-text search for messages
            self.cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='message_id'
                )
            """)
            
            # Create indexes
            self._create_indexes()
            
            self.conn.commit()
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _create_indexes(self):
        """Create all necessary indexes."""
        indexes = [
            ("idx_channel_id", "messages(channel_id)"),
            ("idx_created_at", "messages(created_at)"),
            ("idx_author_id", "messages(author_id)"),
            ("idx_reference_id", "messages(reference_id)"),
            ("idx_daily_summaries_date", "daily_summaries(date)"),
            ("idx_daily_summaries_channel", "daily_summaries(channel_id)"),
            ("idx_members_username", "members(username)")
        ]
        
        for index_name, index_def in indexes:
            try:
                self.cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} ON {index_def}
                """)
            except sqlite3.Error as e:
                logger.error(f"Error creating index {index_name}: {e}")

    def _execute_with_retry(self, operation, *args, max_retries=3, initial_delay=0.1):
        """Execute a database operation with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return operation(*args)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Database locked, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise

    def store_messages(self, messages: List[Dict]):
        """Store messages in the database with retry logic."""
        def _store_operation():
            try:
                for message in messages:
                    try:
                        # Get and validate message ID
                        message_id = message.get('message_id')
                        if message_id is None:
                            # Fall back to 'id' if message_id isn't present
                            message_id = message.get('id')
                            if message_id is None:
                                raise ValueError("Message must have either 'message_id' or 'id' field")
                        
                        # Check if message exists and update if it does
                        if self.message_exists(message_id):
                            self.update_message(message)
                            continue

                        # If message doesn't exist, proceed with insert as before
                        # Ensure all fields are properly serialized
                        attachments = message.get('attachments', [])
                        embeds = message.get('embeds', [])
                        reactors = message.get('reactors', [])
                        
                        # Convert to empty lists if None or 'null'
                        if attachments is None or attachments == 'null':
                            attachments = []
                        if embeds is None or embeds == 'null':
                            embeds = []
                        if reactors is None or reactors == 'null':
                            reactors = []
                        
                        attachments_json = json.dumps(attachments if isinstance(attachments, (list, dict)) else [])
                        embeds_json = json.dumps(embeds if isinstance(embeds, (list, dict)) else [])
                        reactors_json = json.dumps(reactors if isinstance(reactors, (list, dict)) else [])
                        
                        created_at = message.get('created_at')
                        if created_at:
                            created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
                        edited_at = message.get('edited_at')
                        if edited_at:
                            edited_at = edited_at.isoformat() if hasattr(edited_at, 'isoformat') else str(edited_at)
                        
                        try:
                            self.cursor.execute("""
                                INSERT INTO messages 
                                (message_id, channel_id, author_id,
                                 content, created_at, attachments, embeds, reaction_count, 
                                 reactors, reference_id, edited_at, is_pinned, thread_id, 
                                 message_type, flags, jump_url)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                message_id,  # Use Discord's message ID as primary key
                                message.get('channel_id'),
                                message.get('author_id') or (message.get('author', {}).get('id')),
                                message.get('content'),
                                created_at,
                                attachments_json,
                                embeds_json,
                                message.get('reaction_count', 0),
                                reactors_json,
                                message.get('reference_id'),
                                edited_at,
                                message.get('is_pinned', False),
                                message.get('thread_id'),
                                message.get('message_type'),
                                message.get('flags', 0),
                                message.get('jump_url')
                            ))
                        except sqlite3.IntegrityError as e:
                            if "UNIQUE constraint failed" in str(e):
                                # Fetch the existing message from the database
                                self.cursor.execute("""
                                    SELECT message_id, channel_id, author_id, content, created_at, 
                                           attachments, embeds, reaction_count, reactors, edited_at
                                    FROM messages 
                                    WHERE message_id = ?
                                """, (message_id,))
                                existing_msg = self.cursor.fetchone()
                                
                                logger.error("UNIQUE constraint failure details:")
                                logger.error(f"Attempted to insert message:")
                                logger.error(f"  Message ID: {message_id}")
                                logger.error(f"  Channel ID: {message.get('channel_id')}")
                                logger.error(f"  Author ID: {message.get('author_id') or (message.get('author', {}).get('id'))}")
                                logger.error(f"  Content: {message.get('content')}")
                                logger.error(f"  Created at: {created_at}")
                                logger.error(f"  Reaction count: {message.get('reaction_count', 0)}")
                                logger.error(f"  Thread ID: {message.get('thread_id')}")
                                
                                if existing_msg:
                                    logger.error("Existing message in database:")
                                    logger.error(f"  Message ID: {existing_msg[0]}")
                                    logger.error(f"  Channel ID: {existing_msg[1]}")
                                    logger.error(f"  Author ID: {existing_msg[2]}")
                                    logger.error(f"  Content: {existing_msg[3]}")
                                    logger.error(f"  Created at: {existing_msg[4]}")
                                    logger.error(f"  Reaction count: {existing_msg[7]}")
                                    
                                    # Check if messages are identical
                                    differences = []
                                    if message.get('channel_id') != existing_msg[1]:
                                        differences.append(f"Channel ID: {message.get('channel_id')} vs {existing_msg[1]}")
                                    if message.get('author_id') != existing_msg[2]:
                                        differences.append(f"Author ID: {message.get('author_id')} vs {existing_msg[2]}")
                                    if message.get('content') != existing_msg[3]:
                                        differences.append("Content differs")
                                    if created_at != existing_msg[4]:
                                        differences.append(f"Created at: {created_at} vs {existing_msg[4]}")
                                    
                                    if differences:
                                        logger.error("Differences found between messages:")
                                        for diff in differences:
                                            logger.error(f"  {diff}")
                                    else:
                                        logger.error("Messages appear to be identical - likely duplicate insert attempt")
                                else:
                                    logger.error("Could not find existing message in database despite UNIQUE constraint failure")
                            raise
                        
                    except Exception as e:
                        logger.error(f"Error processing individual message: {e}")
                        logger.error(f"Conflicting message_id: {message.get('message_id') or message.get('id')}")
                        logger.error(f"Channel ID: {message.get('channel_id')}")
                        logger.error(f"Author ID: {message.get('author_id') or (message.get('author', {}).get('id'))}")
                        logger.error(f"Created at: {message.get('created_at')}")
                        logger.debug(f"Problem message: {json.dumps(message, default=str)}")
                        continue
                
                self.conn.commit()
                logger.debug(f"Stored {len(messages)} messages")
            except Exception as e:
                self.conn.rollback()
                raise

        return self._execute_with_retry(_store_operation)

    def get_last_message_id(self, channel_id: int) -> Optional[int]:
        """Get the ID of the last archived message for a channel."""
        self.cursor.execute("""
            SELECT MAX(message_id) FROM messages WHERE channel_id = ?
        """, (channel_id,))
        result = self.cursor.fetchone()
        return result[0] if result and result[0] else None

    def search_messages(self, query: str, channel_id: Optional[int] = None) -> List[Dict]:
        """Search messages using FTS index."""
        try:
            sql = """
                SELECT m.*
                FROM messages_fts fts
                JOIN messages m ON fts.rowid = m.message_id
                WHERE fts.content MATCH ?
            """
            params = [query]
            
            if channel_id:
                sql += " AND m.channel_id = ?"
                params.append(channel_id)
                
            sql += " ORDER BY m.created_at DESC LIMIT 100"
            
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            self.conn.row_factory = None
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []

    # Summary Methods
    def store_daily_summary(self, 
                          channel_id: int,
                          full_summary: Optional[str],
                          short_summary: Optional[str],
                          date: Optional[datetime] = None) -> bool:
        """Store daily channel summary."""
        if date is None:
            date = datetime.utcnow().date()
            
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO daily_summaries 
                (date, channel_id, full_summary, short_summary)
                VALUES (?, ?, ?, ?)
            """, (
                date.isoformat() if isinstance(date, datetime) else date,  # Ensure date is ISO format
                channel_id,
                full_summary,
                short_summary
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing daily summary: {e}")
            self.conn.rollback()
            return False

    # Thread Management Methods
    def get_summary_thread_id(self, channel_id: int) -> Optional[int]:
        """Get the summary thread ID for a channel."""
        try:
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT summary_thread_id 
                FROM channel_summary 
                WHERE channel_id = ?
            """, (channel_id,))
            result = cursor.fetchone()
            cursor.close()
            self.conn.row_factory = None
            return result['summary_thread_id'] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting summary thread ID: {e}")
            return None

    def update_summary_thread(self, channel_id: int, thread_id: Optional[int]):
        """Update or delete the summary thread ID for a channel."""
        try:
            if thread_id is None:
                self.cursor.execute("""
                    DELETE FROM channel_summary 
                    WHERE channel_id = ?
                """, (channel_id,))
            else:
                self.cursor.execute("""
                    INSERT OR REPLACE INTO channel_summary 
                    (channel_id, summary_thread_id, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (channel_id, thread_id))
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Error updating summary thread: {e}")
            self.conn.rollback()

    def close(self):
        """Close the database connection."""
        try:
            if self.conn:
                self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    def get_all_message_ids(self, channel_id: int) -> List[int]:
        """Get all message IDs that have been archived for a channel."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT message_id FROM messages WHERE channel_id = ?",
            (channel_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_message_date_range(self, channel_id: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the earliest and latest message dates for a channel."""
        try:
            self.cursor.execute("""
                SELECT MIN(created_at), MAX(created_at)
                FROM messages 
                WHERE channel_id = ?
            """, (channel_id,))
            result = self.cursor.fetchone()
            
            if result and result[0] and result[1]:
                return (
                    datetime.fromisoformat(result[0]),
                    datetime.fromisoformat(result[1])
                )
            return None, None
        except Exception as e:
            logger.error(f"Error getting message date range: {e}")
            return None, None

    def get_message_dates(self, channel_id: int) -> List[str]:
        """Get all message dates for a channel to check for gaps."""
        try:
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT created_at
                FROM messages 
                WHERE channel_id = ?
                ORDER BY created_at
            """, (channel_id,))
            results = [dict(row)['created_at'] for row in cursor.fetchall()]
            cursor.close()
            self.conn.row_factory = None
            return results
        except Exception as e:
            logger.error(f"Error getting message dates: {e}")
            return []

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute a SQL query and return the results."""
        try:
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            self.conn.row_factory = None
            return results
        except sqlite3.Error as e:
            logger.error(f"Database error executing query: {query}\nError: {e}")
            self.conn.rollback()
            raise

    def get_member(self, member_id: int) -> Optional[Dict]:
        """Get a member by their ID."""
        try:
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT *
                FROM members
                WHERE member_id = ?
            """, (member_id,))
            result = cursor.fetchone()
            cursor.close()
            self.conn.row_factory = None
            
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"Error getting member {member_id}: {e}")
            return None

    def message_exists(self, message_id: int) -> bool:
        """Check if a message exists in the database."""
        try:
            self.cursor.execute("""
                SELECT 1 FROM messages WHERE message_id = ?
            """, (message_id,))
            return bool(self.cursor.fetchone())
        except Exception as e:
            logger.error(f"Error checking if message {message_id} exists: {e}")
            return False

    def update_message(self, message: Dict) -> bool:
        """Update an existing message with new data, particularly reactions."""
        try:
            # Get and validate message ID
            message_id = message.get('message_id') or message.get('id')
            if message_id is None:
                raise ValueError("Message must have either 'message_id' or 'id' field")

            # Ensure all fields are properly serialized
            attachments = message.get('attachments', [])
            embeds = message.get('embeds', [])
            reactors = message.get('reactors', [])
            
            # Convert to empty lists if None or 'null'
            if attachments is None or attachments == 'null':
                attachments = []
            if embeds is None or embeds == 'null':
                embeds = []
            if reactors is None or reactors == 'null':
                reactors = []
            
            attachments_json = json.dumps(attachments if isinstance(attachments, (list, dict)) else [])
            embeds_json = json.dumps(embeds if isinstance(embeds, (list, dict)) else [])
            reactors_json = json.dumps(reactors if isinstance(reactors, (list, dict)) else [])
            
            edited_at = message.get('edited_at')
            if edited_at:
                edited_at = edited_at.isoformat() if hasattr(edited_at, 'isoformat') else str(edited_at)

            # Update the message with new data
            self.cursor.execute("""
                UPDATE messages 
                SET content = COALESCE(?, content),
                    attachments = COALESCE(?, attachments),
                    embeds = COALESCE(?, embeds),
                    reaction_count = COALESCE(?, reaction_count),
                    reactors = COALESCE(?, reactors),
                    edited_at = COALESCE(?, edited_at),
                    is_pinned = COALESCE(?, is_pinned),
                    flags = COALESCE(?, flags),
                    indexed_at = CURRENT_TIMESTAMP
                WHERE message_id = ?
            """, (
                message.get('content'),
                attachments_json,
                embeds_json,
                message.get('reaction_count', 0),
                reactors_json,
                edited_at,
                message.get('is_pinned'),
                message.get('flags'),
                message_id
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating message {message.get('message_id')}: {e}")
            self.conn.rollback()
            return False

    def create_or_update_member(self, member_id: int, username: str, display_name: Optional[str] = None, 
                              global_name: Optional[str] = None, avatar_url: Optional[str] = None,
                              discriminator: Optional[str] = None, bot: bool = False, 
                              system: bool = False, accent_color: Optional[int] = None,
                              banner_url: Optional[str] = None, discord_created_at: Optional[str] = None,
                              guild_join_date: Optional[str] = None, role_ids: Optional[str] = None) -> bool:
        """Create or update a member in the database."""
        try:
            existing_member = self.get_member(member_id)
            if existing_member:
                # Update existing member if any field changed
                changes = []
                if existing_member['username'] != username:
                    changes.append(f"username: {existing_member['username']} -> {username}")
                if existing_member['server_nick'] != display_name:
                    changes.append(f"server_nick: {existing_member['server_nick']} -> {display_name}")
                if existing_member['global_name'] != global_name:
                    changes.append(f"global_name: {existing_member['global_name']} -> {global_name}")
                if existing_member['avatar_url'] != avatar_url:
                    changes.append(f"avatar_url changed")
                if existing_member['discriminator'] != discriminator:
                    changes.append(f"discriminator: {existing_member['discriminator']} -> {discriminator}")
                if existing_member['bot'] != bot:
                    changes.append(f"bot: {existing_member['bot']} -> {bot}")
                if existing_member['system'] != system:
                    changes.append(f"system: {existing_member['system']} -> {system}")
                if existing_member['accent_color'] != accent_color:
                    changes.append(f"accent_color: {existing_member['accent_color']} -> {accent_color}")
                if existing_member['banner_url'] != banner_url:
                    changes.append(f"banner_url changed")
                if existing_member['discord_created_at'] != discord_created_at:
                    changes.append(f"discord_created_at: {existing_member['discord_created_at']} -> {discord_created_at}")
                if existing_member['guild_join_date'] != guild_join_date:
                    changes.append(f"guild_join_date: {existing_member['guild_join_date']} -> {guild_join_date}")
                if existing_member['role_ids'] != role_ids:
                    changes.append(f"role_ids changed")

                if changes:
                    logger.info(f"Updating member {member_id} ({username}). Changes: {', '.join(changes)}")
                    # Don't update fields that are None in the new data
                    update_fields = []
                    update_values = []
                    if username is not None:
                        update_fields.append("username = ?")
                        update_values.append(username)
                    if display_name is not None:
                        update_fields.append("server_nick = ?")
                        update_values.append(display_name)
                    if global_name is not None:
                        update_fields.append("global_name = ?")
                        update_values.append(global_name)
                    if avatar_url is not None:
                        update_fields.append("avatar_url = ?")
                        update_values.append(avatar_url)
                    if discriminator is not None:
                        update_fields.append("discriminator = ?")
                        update_values.append(discriminator)
                    if bot is not None:
                        update_fields.append("bot = ?")
                        update_values.append(bot)
                    if system is not None:
                        update_fields.append("system = ?")
                        update_values.append(system)
                    if accent_color is not None:
                        update_fields.append("accent_color = ?")
                        update_values.append(accent_color)
                    if banner_url is not None:
                        update_fields.append("banner_url = ?")
                        update_values.append(banner_url)
                    if discord_created_at is not None:
                        update_fields.append("discord_created_at = ?")
                        update_values.append(discord_created_at)
                    if guild_join_date is not None:
                        update_fields.append("guild_join_date = ?")
                        update_values.append(guild_join_date)
                    if role_ids is not None:
                        update_fields.append("role_ids = ?")
                        update_values.append(role_ids)
                    
                    update_fields.append("updated_at = CURRENT_TIMESTAMP")
                    
                    # Add member_id to values
                    update_values.append(member_id)
                    
                    update_sql = f"""
                        UPDATE members
                        SET {', '.join(update_fields)}
                        WHERE member_id = ?
                    """
                    self.cursor.execute(update_sql, tuple(update_values))
            else:
                # Create new member
                logger.info(f"Creating new member {member_id} ({username})")
                self.cursor.execute("""
                    INSERT INTO members 
                    (member_id, username, server_nick, global_name, avatar_url, 
                     discriminator, bot, system, accent_color, banner_url, 
                     discord_created_at, guild_join_date, role_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (member_id, username, display_name, global_name, avatar_url,
                      discriminator, bot, system, accent_color, banner_url,
                      discord_created_at, guild_join_date, role_ids))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating/updating member {member_id}: {e}")
            logger.error(f"Current values - username: {username}, display_name: {display_name}, global_name: {global_name}")
            if existing_member:
                logger.error(f"Existing values - username: {existing_member['username']}, server_nick: {existing_member['server_nick']}, global_name: {existing_member['global_name']}")
            self.conn.rollback()
            return False

    def get_channel(self, channel_id: int) -> Optional[Dict]:
        """Get a channel by its ID."""
        try:
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT channel_id, channel_name, description, suitable_posts, 
                       unsuitable_posts, rules, setup_complete, nsfw, enriched
                FROM channels
                WHERE channel_id = ?
            """, (channel_id,))
            result = cursor.fetchone()
            cursor.close()
            self.conn.row_factory = None
            
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"Error getting channel {channel_id}: {e}")
            return None

    def create_or_update_channel(self, channel_id: int, channel_name: str, 
                               nsfw: bool = False) -> bool:
        """Create or update a channel in the database."""
        try:
            existing_channel = self.get_channel(channel_id)
            if existing_channel:
                # Update if name or nsfw status changed
                if (existing_channel['channel_name'] != channel_name or
                    existing_channel['nsfw'] != nsfw):
                    self.cursor.execute("""
                        UPDATE channels
                        SET channel_name = ?, nsfw = ?
                        WHERE channel_id = ?
                    """, (channel_name, nsfw, channel_id))
            else:
                # Create new channel
                self.cursor.execute("""
                    INSERT INTO channels 
                    (channel_id, channel_name, description, suitable_posts, 
                     unsuitable_posts, rules, setup_complete, nsfw, enriched)
                    VALUES (?, ?, '', '', '', '', FALSE, ?, FALSE)
                """, (channel_id, channel_name, nsfw))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating/updating channel {channel_id}: {e}")
            self.conn.rollback()
            return False

    def get_messages_after(self, date: datetime) -> List[Dict]:
        """Get all messages after a specific date."""
        try:
            self.cursor.execute("""
                SELECT message_id, channel_id, author_id, content, created_at, 
                       attachments, embeds, reaction_count, reactors, reference_id, 
                       edited_at, is_pinned, thread_id, message_type, flags, jump_url
                FROM messages 
                WHERE created_at > ?
                ORDER BY created_at DESC
            """, (date.isoformat(),))
            
            rows = self.cursor.fetchall()
            messages = []
            
            for row in rows:
                message = {
                    'message_id': row[0],
                    'channel_id': row[1],
                    'author_id': row[2],
                    'content': row[3],
                    'created_at': row[4],
                    'attachments': json.loads(row[5]) if row[5] else [],
                    'embeds': json.loads(row[6]) if row[6] else [],
                    'reaction_count': row[7],
                    'reactors': json.loads(row[8]) if row[8] else [],
                    'reference_id': row[9],
                    'edited_at': row[10],
                    'is_pinned': bool(row[11]),
                    'thread_id': row[12],
                    'message_type': row[13],
                    'flags': row[14],
                    'jump_url': row[15]
                }
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages after {date}: {e}")
            return []
