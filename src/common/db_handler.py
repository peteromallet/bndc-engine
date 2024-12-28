import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from .constants import DATABASE_PATH

logger = logging.getLogger('DiscordBot')

class DatabaseHandler:
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database connection and ensure directory exists."""
        try:
            # Ensure the directory exists
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            
            # Initialize all tables
            self.init_db()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def init_db(self):
        """Initialize all database tables."""
        try:
            # Messages table with FTS support
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGINT PRIMARY KEY,
                    message_id BIGINT,
                    channel_id BIGINT,
                    author_id BIGINT,
                    author_name TEXT,
                    content TEXT,
                    created_at TEXT,
                    attachments TEXT,
                    embeds TEXT,
                    reactions TEXT,
                    reference_id BIGINT,
                    edited_at TEXT,
                    is_pinned BOOLEAN,
                    thread_id BIGINT,
                    message_type TEXT,
                    flags INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            
            # Channel summaries table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    channel_id BIGINT NOT NULL,
                    channel_name TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    raw_messages TEXT NOT NULL,
                    full_summary TEXT,
                    short_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, channel_id) ON CONFLICT REPLACE
                )
            """)
            
            # Channel summary threads table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS channel_summary (
                    channel_id BIGINT PRIMARY KEY,
                    summary_thread_id BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            ("idx_daily_summaries_channel", "daily_summaries(channel_id)")
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
        """Store messages in database."""
        try:
            for message in messages:
                self.cursor.execute("""
                    INSERT OR IGNORE INTO messages 
                    (id, message_id, channel_id, author_id, author_name, content, created_at, 
                     attachments, embeds, reactions, reference_id, edited_at,
                     is_pinned, thread_id, message_type, flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message['id'],
                    message['message_id'],
                    message['channel_id'],
                    message['author_id'],
                    message['author_name'],
                    message['content'],
                    message['created_at'],
                    json.dumps(message['attachments']),
                    json.dumps(message['embeds']),
                    json.dumps(message['reactions']),
                    message['reference_id'],
                    message['edited_at'],
                    message['is_pinned'],
                    message['thread_id'],
                    message['message_type'],
                    message['flags']
                ))
                
                # Update FTS index
                if message['content']:
                    self.cursor.execute("""
                        INSERT OR REPLACE INTO messages_fts(rowid, content)
                        VALUES (?, ?)
                    """, (message['id'], message['content']))
            
            self.conn.commit()
            
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
            # Convert messages to serializable format
            messages_data = []
            for msg in messages:
                message_dict = {
                    'content': msg.content,
                    'author': msg.author.name,
                    'timestamp': msg.created_at.isoformat(),
                    'jump_url': msg.jump_url,
                    'reactions': sum(reaction.count for reaction in msg.reactions) if msg.reactions else 0,
                    'id': str(msg.id),
                    'attachments': [
                        {
                            'filename': attachment.filename,
                            'url': attachment.url
                        } for attachment in msg.attachments
                    ]
                }
                messages_data.append(message_dict)
            
            messages_json = json.dumps(messages_data)
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO daily_summaries 
                (date, channel_id, channel_name, message_count, raw_messages, 
                 full_summary, short_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date.isoformat(),
                channel_id,
                channel_name,
                len(messages),
                messages_json,
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
