import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger('ChannelSummarizer')

class DatabaseHandler:
    def __init__(self, db_path: str = "channel_summaries.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_channel_summary_table()

    def init_db(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create daily_summaries table with a unique constraint
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        channel_id BIGINT NOT NULL,
                        channel_name TEXT NOT NULL,
                        message_count INTEGER NOT NULL,
                        raw_messages TEXT NOT NULL,  -- JSON string of messages
                        full_summary TEXT,
                        short_summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, channel_id) ON CONFLICT REPLACE
                    )
                """)
                
                # Create an index on date and channel_id
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_daily_summaries_date_channel 
                    ON daily_summaries(date, channel_id)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def store_daily_summary(self, 
                          channel_id: int,
                          channel_name: str,
                          messages: List[Dict],
                          full_summary: Optional[str],
                          short_summary: Optional[str],
                          date: Optional[datetime] = None) -> bool:
        """Store daily channel summary and messages."""
        if date is None:
            date = datetime.utcnow().date()
        
        try:
            # Convert messages to JSON string
            messages_json = json.dumps([{
                'content': msg['content'],
                'author': msg['author'],
                'timestamp': msg['timestamp'].isoformat(),
                'jump_url': msg['jump_url'],
                'reactions': msg['reactions'],
                'id': msg['id'],
                'attachments': msg.get('attachments', [])
            } for msg in messages])
            
            query = """
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT,
                channel_id INTEGER,
                channel_name TEXT,
                message_count INTEGER,
                raw_messages TEXT,
                full_summary TEXT,
                short_summary TEXT,
                PRIMARY KEY (date, channel_id)
            )
            """
            self.execute_query(query)
            
            # Using parameterized query to prevent SQL injection
            insert_query = """
            INSERT OR REPLACE INTO daily_summaries 
            (date, channel_id, channel_name, message_count, raw_messages, full_summary, short_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                date.isoformat(),
                channel_id,
                channel_name,
                len(messages),
                messages_json,
                full_summary,
                short_summary
            )
            
            self.execute_query(insert_query, params)
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error storing summary: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing summary: {e}")
            return False

    def get_channel_summary(self, channel_id: int, date: datetime) -> Optional[Dict]:
        """Retrieve a specific channel summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM daily_summaries 
                    WHERE channel_id = ? AND date = ?
                """, (channel_id, date.date().isoformat()))
                
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['raw_messages'] = json.loads(result['raw_messages'])
                    return result
                    
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving summary: {e}")
            return None

    def get_channel_history(self, channel_id: int, 
                          start_date: datetime, 
                          end_date: datetime) -> List[Dict]:
        """Retrieve channel summaries for a date range."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM daily_summaries 
                    WHERE channel_id = ? AND date BETWEEN ? AND ?
                    ORDER BY date DESC
                """, (
                    channel_id, 
                    start_date.date().isoformat(), 
                    end_date.date().isoformat()
                ))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['raw_messages'] = json.loads(result['raw_messages'])
                    results.append(result)
                    
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving history: {e}")
            return []

    def create_channel_summary_table(self):
        """Create the channel_summary table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS channel_summary (
            channel_id BIGINT PRIMARY KEY,
            summary_thread_id BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute_query(query)

    def get_summary_thread_id(self, channel_id: int) -> Optional[int]:
        """Get the summary thread ID for a channel if it exists."""
        try:
            query = """
            SELECT summary_thread_id 
            FROM channel_summary 
            WHERE channel_id = ?
            """
            result = self.execute_query(query, (channel_id,), fetch_one=True)
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving summary thread ID: {e}")
            return None

    def verify_thread_exists(self, thread_id: int) -> bool:
        """Verify if a thread exists in the database and is valid."""
        try:
            query = """
            SELECT COUNT(*) 
            FROM channel_summary 
            WHERE summary_thread_id = ? AND summary_thread_id IS NOT NULL
            """
            result = self.execute_query(query, (thread_id,), fetch_one=True)
            return bool(result[0]) if result else False
        except sqlite3.Error as e:
            logger.error(f"Database error verifying thread: {e}")
            return False

    def update_summary_thread(self, channel_id: int, thread_id: Optional[int]):
        """Insert or update the summary thread ID for a channel."""
        if thread_id is None:
            # If thread_id is None, delete the entry instead of updating
            query = """
            DELETE FROM channel_summary 
            WHERE channel_id = ?
            """
            self.execute_query(query, (channel_id,))
        else:
            query = """
            INSERT INTO channel_summary (channel_id, summary_thread_id, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (channel_id) 
            DO UPDATE SET 
                summary_thread_id = EXCLUDED.summary_thread_id,
                updated_at = CURRENT_TIMESTAMP
            """
            self.execute_query(query, (channel_id, thread_id))

    def execute_query(self, query, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False):
        """Execute a database query with optional fetching."""
        try:
            self.cursor.execute(query, params)
            if fetch_one:
                return self.cursor.fetchone()
            if fetch_all:
                return self.cursor.fetchall()
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
