import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from .constants import get_database_path

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
            
            self.conn = sqlite3.connect(self.db_path)
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
                    channel_id BIGINT PRIMARY KEY,
                    summary_thread_id BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                )
            """)
            
            # Daily summaries table with foreign key
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    channel_id BIGINT NOT NULL,
                    message_count INTEGER NOT NULL,
                    full_summary TEXT,
                    short_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, channel_id) ON CONFLICT REPLACE,
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                )
            """)
            
            # Members table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    id BIGINT PRIMARY KEY,
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
                    id BIGINT PRIMARY KEY,
                    message_id BIGINT,
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
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES members(id),
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                )
            """)
            
            # Full-text search for messages
            self.cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='id'
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

    # Message Archive Methods
    def store_messages(self, messages: List[Dict]):
        """Store messages in database, skipping duplicates."""
        try:
            # Debug log the message format
            if messages:
                logger.debug(f"First message format: {json.dumps(messages[0], default=str)}")
            
            # First check which messages already exist
            message_ids = [msg.get('message_id') or msg.get('id') for msg in messages]  # Handle both ID formats
            placeholders = ','.join('?' * len(message_ids))
            
            self.cursor.execute(f"""
                SELECT message_id 
                FROM messages 
                WHERE message_id IN ({placeholders})
            """, message_ids)
            
            existing_ids = {row[0] for row in self.cursor.fetchall()}
            new_messages = [msg for msg in messages if (msg.get('message_id') or msg.get('id')) not in existing_ids]
            
            if not new_messages:
                logger.debug("No new messages to store - all messages already exist")
                return
            
            # Insert only new messages
            for message in new_messages:
                try:
                    # Debug log each message being processed
                    logger.debug(f"Processing message: {json.dumps(message, default=str)}")
                    
                    # First create or update the member
                    if 'author' in message:
                        author = message['author']
                        self.create_or_update_member(
                            member_id=author.get('id'),
                            username=author.get('name') or author.get('username'),
                            display_name=author.get('display_name'),
                            global_name=author.get('global_name'),
                            avatar_url=author.get('avatar_url'),
                            discriminator=author.get('discriminator'),
                            bot=author.get('bot', False),
                            system=author.get('system', False),
                            accent_color=author.get('accent_color'),
                            banner_url=author.get('banner_url'),
                            discord_created_at=author.get('discord_created_at'),
                            guild_join_date=author.get('guild_join_date'),
                            role_ids=author.get('role_ids')
                        )

                    # Then create or update the channel
                    if 'channel' in message:
                        channel = message['channel']
                        self.create_or_update_channel(
                            channel_id=channel.get('id'),
                            channel_name=channel.get('name'),
                            nsfw=channel.get('nsfw', False)
                        )
                    
                    # Ensure all fields are properly serialized
                    attachments_json = json.dumps(message.get('attachments', []) if isinstance(message.get('attachments'), (list, dict)) else json.loads(message.get('attachments', '[]')))
                    embeds_json = json.dumps(message.get('embeds', []) if isinstance(message.get('embeds'), (list, dict)) else json.loads(message.get('embeds', '[]')))
                    reactors_json = json.dumps(message.get('reactors', []) if isinstance(message.get('reactors'), (list, dict)) else json.loads(message.get('reactors', '[]')))
                    created_at = message.get('created_at')
                    if created_at:
                        created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
                    edited_at = message.get('edited_at')
                    if edited_at:
                        edited_at = edited_at.isoformat() if hasattr(edited_at, 'isoformat') else str(edited_at)
                    
                    # Use either message_id or id
                    message_id = message.get('message_id') or message.get('id')
                    
                    self.cursor.execute("""
                        INSERT INTO messages 
                        (id, message_id, channel_id, author_id,
                         content, created_at, attachments, embeds, reaction_count, 
                         reactors, reference_id, edited_at, is_pinned, thread_id, 
                         message_type, flags, jump_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        message_id,  # Use same ID for both fields since we don't have a separate internal ID
                        message_id,
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
                except Exception as e:
                    logger.error(f"Error processing individual message: {e}")
                    logger.debug(f"Problem message: {json.dumps(message, default=str)}")
                    continue
            
            self.conn.commit()
            logger.debug(f"Stored {len(new_messages)} new messages (skipped {len(existing_ids)} duplicates)")
            
        except Exception as e:
            logger.error(f"Error storing messages: {e}")
            self.conn.rollback()
            raise

    def get_last_message_id(self, channel_id: int) -> Optional[int]:
        """Get the ID of the last archived message for a channel."""
        self.cursor.execute("""
            SELECT MAX(id) FROM messages WHERE channel_id = ?
        """, (channel_id,))
        result = self.cursor.fetchone()
        return result[0] if result and result[0] else None

    def search_messages(self, query: str, channel_id: Optional[int] = None) -> List[Dict]:
        """Search messages using FTS index."""
        try:
            sql = """
                SELECT m.*
                FROM messages_fts fts
                JOIN messages m ON fts.rowid = m.id
                WHERE fts.content MATCH ?
            """
            params = [query]
            
            if channel_id:
                sql += " AND m.channel_id = ?"
                params.append(channel_id)
                
            sql += " ORDER BY m.created_at DESC LIMIT 100"
            
            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            
            return [{
                'id': row[0],
                'channel_id': row[1],
                'author_id': row[2],
                'author_name': row[3],
                'content': row[4],
                'created_at': row[5],
                'attachments': json.loads(row[6]),
                'reactions': json.loads(row[8]),
                'reference_id': row[9],
                'thread_id': row[12]
            } for row in rows]
            
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []

    # Summary Methods
    def store_daily_summary(self, 
                          channel_id: int,
                          channel_name: str,
                          messages: List[Any],
                          full_summary: Optional[str],
                          short_summary: Optional[str],
                          date: Optional[datetime] = None) -> bool:
        """Store daily channel summary."""
        if date is None:
            date = datetime.utcnow().date()
            
        try:
            # First create or update the channel record
            self.create_or_update_channel(channel_id, channel_name)
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO daily_summaries 
                (date, channel_id, message_count, 
                 full_summary, short_summary)
                VALUES (?, ?, ?, ?, ?)
            """, (
                date.isoformat() if isinstance(date, datetime) else date,  # Ensure date is ISO format
                channel_id,
                len(messages),
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
            self.cursor.execute("""
                SELECT summary_thread_id 
                FROM channel_summary 
                WHERE channel_id = ?
            """, (channel_id,))
            result = self.cursor.fetchone()
            return result[0] if result else None
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
            self.cursor.execute("""
                SELECT created_at
                FROM messages 
                WHERE channel_id = ?
                ORDER BY created_at
            """, (channel_id,))
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting message dates: {e}")
            return []

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute a SQL query and return the results."""
        try:
            self.cursor.execute(query, params)
            self.conn.commit()  # Commit after each query
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error executing query: {query}\nError: {e}")
            self.conn.rollback()
            raise

    def get_member(self, member_id: int) -> Optional[Dict]:
        """Get a member by their ID."""
        try:
            self.cursor.execute("""
                SELECT id, username, server_nick, global_name, avatar_url, discriminator,
                       bot, system, accent_color, banner_url, discord_created_at,
                       guild_join_date, role_ids, created_at, updated_at
                FROM members
                WHERE id = ?
            """, (member_id,))
            result = self.cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'username': result[1],
                    'server_nick': result[2],
                    'global_name': result[3],
                    'avatar_url': result[4],
                    'discriminator': result[5],
                    'bot': bool(result[6]),
                    'system': bool(result[7]),
                    'accent_color': result[8],
                    'banner_url': result[9],
                    'discord_created_at': result[10],
                    'guild_join_date': result[11],
                    'role_ids': result[12],
                    'created_at': result[13],
                    'updated_at': result[14]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting member {member_id}: {e}")
            return None

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
                if (existing_member['username'] != username or 
                    existing_member['server_nick'] != display_name or
                    existing_member['global_name'] != global_name or
                    existing_member['avatar_url'] != avatar_url or
                    existing_member['discriminator'] != discriminator or
                    existing_member['bot'] != bot or
                    existing_member['system'] != system or
                    existing_member['accent_color'] != accent_color or
                    existing_member['banner_url'] != banner_url or
                    existing_member['discord_created_at'] != discord_created_at or
                    existing_member['guild_join_date'] != guild_join_date or
                    existing_member['role_ids'] != role_ids):
                    self.cursor.execute("""
                        UPDATE members
                        SET username = ?, server_nick = ?, global_name = ?, avatar_url = ?, 
                            discriminator = ?, bot = ?, system = ?, accent_color = ?,
                            banner_url = ?, discord_created_at = ?, guild_join_date = ?, 
                            role_ids = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (username, display_name, global_name, avatar_url, 
                          discriminator, bot, system, accent_color,
                          banner_url, discord_created_at, guild_join_date,
                          role_ids, member_id))
            else:
                # Create new member
                self.cursor.execute("""
                    INSERT INTO members 
                    (id, username, server_nick, global_name, avatar_url, 
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
            self.conn.rollback()
            return False

    def get_channel(self, channel_id: int) -> Optional[Dict]:
        """Get a channel by its ID."""
        try:
            self.cursor.execute("""
                SELECT channel_id, channel_name, description, suitable_posts, 
                       unsuitable_posts, rules, setup_complete, nsfw, enriched
                FROM channels
                WHERE channel_id = ?
            """, (channel_id,))
            result = self.cursor.fetchone()
            if result:
                return {
                    'channel_id': result[0],
                    'channel_name': result[1],
                    'description': result[2],
                    'suitable_posts': result[3],
                    'unsuitable_posts': result[4],
                    'rules': result[5],
                    'setup_complete': bool(result[6]),
                    'nsfw': bool(result[7]),
                    'enriched': bool(result[8])
                }
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
