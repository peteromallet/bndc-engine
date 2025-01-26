import os
import sys
import time
import argparse
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import logging
from typing import List, Set, Dict
from src.common.constants import get_database_path
from src.common.schema import get_schema_tuples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_table_columns(cursor, table_name: str) -> Dict[str, dict]:
    """Get current columns in the table with their full definitions."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {}
    for row in cursor.fetchall():
        # row format: (cid, name, type, notnull, dflt_value, pk)
        columns[row[1]] = {
            'type': row[2],
            'notnull': row[3],
            'default': row[4],
            'primary_key': row[5]
        }
    return columns

def get_desired_schema() -> List[tuple]:
    """Get the desired schema structure for the messages table."""
    return [
        ("message_id", "BIGINT PRIMARY KEY"),
        ("channel_id", "BIGINT NOT NULL REFERENCES channels(channel_id)"),
        ("author_id", "BIGINT NOT NULL REFERENCES members(member_id)"),
        ("content", "TEXT"),
        ("created_at", "TEXT"),
        ("attachments", "TEXT"),
        ("embeds", "TEXT"),
        ("reaction_count", "INTEGER"),
        ("reactors", "TEXT"),
        ("reference_id", "BIGINT"),
        ("edited_at", "TEXT"),
        ("is_pinned", "BOOLEAN"),
        ("thread_id", "BIGINT"),
        ("message_type", "TEXT"),
        ("flags", "INTEGER"),
        ("jump_url", "TEXT"),
        ("is_deleted", "BOOLEAN DEFAULT FALSE"),
        ("indexed_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    ]

def get_desired_members_schema() -> List[tuple]:
    """Get the desired schema structure for the members table."""
    return [
        ("member_id", "BIGINT PRIMARY KEY"),
        ("username", "TEXT NOT NULL"),
        ("global_name", "TEXT"),
        ("server_nick", "TEXT"),
        ("avatar_url", "TEXT"),
        ("discriminator", "TEXT"),
        ("bot", "BOOLEAN DEFAULT FALSE"),
        ("system", "BOOLEAN DEFAULT FALSE"),
        ("accent_color", "INTEGER"),
        ("banner_url", "TEXT"),
        ("discord_created_at", "TEXT"),
        ("guild_join_date", "TEXT"),
        ("role_ids", "TEXT"),  # JSON array of role IDs
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("notifications", "TEXT DEFAULT '[]'")
    ]

def get_desired_daily_summaries_schema() -> List[tuple]:
    """Get the desired schema structure for the daily_summaries table."""
    return [
        ("daily_summary_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("date", "TEXT NOT NULL"),
        ("channel_id", "BIGINT NOT NULL REFERENCES channels(channel_id)"),
        ("full_summary", "TEXT"),
        ("short_summary", "TEXT"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    ]

def create_messages_table(cursor):
    """Create the messages table if it doesn't exist."""
    schema = get_desired_schema()
    columns_def = ", ".join([f"{name} {type_}" for name, type_ in schema])
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS messages (
        {columns_def}
    )
    """
    cursor.execute(create_sql)

def table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None

def backup_table(cursor, table_name: str):
    """Create a backup of the table before migration."""
    backup_name = f"{table_name}_backup_{int(time.time())}"
    cursor.execute(f"""
        CREATE TABLE {backup_name} AS 
        SELECT * FROM {table_name}
    """)
    return backup_name

def validate_migration(cursor, original_count: int):
    """Validate that no data was lost during migration."""
    cursor.execute("SELECT COUNT(*) FROM messages")
    new_count = cursor.fetchone()[0]
    if new_count != original_count:
        raise ValueError(
            f"Data loss detected! Original row count: {original_count}, "
            f"New row count: {new_count}"
        )

def create_temp_table_and_migrate_data(cursor, desired_schema: List[tuple], existing_columns: Dict[str, dict]):
    """Create a temporary table with the desired schema and migrate the data."""
    # Get original row count
    cursor.execute("SELECT COUNT(*) FROM messages")
    original_count = cursor.fetchone()[0]
    
    # Create backup
    backup_name = backup_table(cursor, "messages")
    logger.info(f"Created backup table: {backup_name}")
    
    try:
        # Create column definitions string
        columns_def = ", ".join([f"{name} {type_}" for name, type_ in desired_schema])
        
        # Create new table and copy data
        cursor.execute(f"""
            CREATE TABLE messages_new (
                {columns_def}
            )
        """)
        
        # Copy data, using message_id since we've already migrated from id
        cursor.execute("""
            INSERT INTO messages_new 
            SELECT message_id, channel_id, author_id,
                   content, created_at, attachments, embeds, reaction_count,
                   reactors, reference_id, edited_at, is_pinned, thread_id,
                   message_type, flags, jump_url, FALSE as is_deleted, indexed_at
            FROM messages
        """)
        
        # Validate before dropping old table
        cursor.execute("SELECT COUNT(*) FROM messages_new")
        if cursor.fetchone()[0] != original_count:
            raise ValueError("Row count mismatch before table swap")
        
        # Drop old table and rename new one
        cursor.execute("DROP TABLE messages")
        cursor.execute("ALTER TABLE messages_new RENAME TO messages")
        
        # Final validation
        validate_migration(cursor, original_count)
        
    except Exception as e:
        # If anything goes wrong, we can restore from backup
        logger.error(f"Migration failed: {e}")
        cursor.execute("DROP TABLE IF EXISTS messages")
        cursor.execute(f"ALTER TABLE {backup_name} RENAME TO messages")
        raise

def migrate_messages_table(cursor):
    """Migrate messages table to use message_id as primary key."""
    logger.info("Starting messages table migration")
    
    try:
        # Get current columns with their definitions
        existing_columns = get_table_columns(cursor, "messages")
        desired_schema = get_desired_schema()
        
        # Find missing and extra columns
        desired_column_names = {name for name, _ in desired_schema}
        existing_column_names = set(existing_columns.keys())
        
        missing_columns = [
            (name, type_) for name, type_ in desired_schema 
            if name not in existing_columns
        ]
        extra_columns = existing_column_names - desired_column_names
        
        # Also check if any column types have changed
        type_changes = [
            (name, type_) for name, type_ in desired_schema
            if name in existing_columns and existing_columns[name]['type'] != type_
        ]
        
        # Check if message_id is not the primary key
        primary_key_wrong = any(
            col['primary_key'] == 1 and name != 'message_id'
            for name, col in existing_columns.items()
        )
        
        if not (missing_columns or extra_columns or type_changes or primary_key_wrong):
            logger.info("Messages table schema is up to date")
            return

        if missing_columns:
            logger.info(f"Found missing columns: {[col[0] for col in missing_columns]}")
        if extra_columns:
            logger.info(f"Found columns to remove: {list(extra_columns)}")
        if type_changes:
            logger.info(f"Found columns with changed types: {[col[0] for col in type_changes]}")
        if primary_key_wrong:
            logger.info("Primary key needs to be changed to message_id")

        # Create backup
        backup_name = backup_table(cursor, "messages")
        logger.info(f"Created backup table: {backup_name}")
        
        try:
            # Get original row count
            cursor.execute("SELECT COUNT(*) FROM messages")
            original_count = cursor.fetchone()[0]
            
            # Create new table with correct schema
            columns_def = ", ".join([f"{name} {type_}" for name, type_ in desired_schema])
            cursor.execute(f"""
                CREATE TABLE messages_new (
                    {columns_def}
                )
            """)
            
            # Copy data, using message_id since we've already migrated from id
            cursor.execute("""
                INSERT INTO messages_new 
                SELECT message_id, channel_id, author_id,
                       content, created_at, attachments, embeds, reaction_count,
                       reactors, reference_id, edited_at, is_pinned, thread_id,
                       message_type, flags, jump_url, FALSE as is_deleted, indexed_at
                FROM messages
            """)
            
            # Validate before dropping old table
            cursor.execute("SELECT COUNT(*) FROM messages_new")
            if cursor.fetchone()[0] != original_count:
                raise ValueError("Row count mismatch before table swap")
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE messages")
            cursor.execute("ALTER TABLE messages_new RENAME TO messages")
            
            # Recreate indexes
            cursor.execute("CREATE INDEX idx_channel_id ON messages(channel_id)")
            cursor.execute("CREATE INDEX idx_created_at ON messages(created_at)")
            cursor.execute("CREATE INDEX idx_author_id ON messages(author_id)")
            cursor.execute("CREATE INDEX idx_reference_id ON messages(reference_id)")
            
            logger.info("Successfully migrated messages table")
            
        except Exception as e:
            # If anything goes wrong, we can restore from backup
            logger.error(f"Migration failed: {e}")
            cursor.execute("DROP TABLE IF EXISTS messages")
            cursor.execute(f"ALTER TABLE {backup_name} RENAME TO messages")
            raise
            
    except Exception as e:
        logger.error(f"Error during messages table migration: {e}")
        raise

def migrate_members_table(cursor):
    """Migrate members table to use member_id as primary key."""
    logger.info("Starting members table migration")
    
    try:
        # Get current columns with their definitions
        existing_columns = get_table_columns(cursor, "members")
        desired_schema = get_desired_members_schema()
        
        # Find missing and extra columns
        desired_column_names = {name for name, _ in desired_schema}
        existing_column_names = set(existing_columns.keys())
        
        missing_columns = [
            (name, type_) for name, type_ in desired_schema 
            if name not in existing_columns
        ]
        extra_columns = existing_column_names - desired_column_names
        
        # Also check if any column types have changed
        type_changes = [
            (name, type_) for name, type_ in desired_schema
            if name in existing_columns and existing_columns[name]['type'] != type_
        ]
        
        # Check if member_id is not the primary key
        primary_key_wrong = any(
            col['primary_key'] == 1 and name != 'member_id'
            for name, col in existing_columns.items()
        )
        
        if not (missing_columns or extra_columns or type_changes or primary_key_wrong):
            logger.info("Members table schema is up to date")
            return

        if missing_columns:
            logger.info(f"Found missing columns: {[col[0] for col in missing_columns]}")
        if extra_columns:
            logger.info(f"Found columns to remove: {list(extra_columns)}")
        if type_changes:
            logger.info(f"Found columns with changed types: {[col[0] for col in type_changes]}")
        if primary_key_wrong:
            logger.info("Primary key needs to be changed to member_id")

        # Create backup
        backup_name = backup_table(cursor, "members")
        logger.info(f"Created backup table: {backup_name}")
        
        try:
            # Get original row count
            cursor.execute("SELECT COUNT(*) FROM members")
            original_count = cursor.fetchone()[0]
            
            # Create new table with correct schema
            columns_def = ", ".join([f"{name} {type_}" for name, type_ in desired_schema])
            cursor.execute(f"""
                CREATE TABLE members_new (
                    {columns_def}
                )
            """)
            
            # Copy data, using member_id since we've already migrated from id
            cursor.execute("""
                INSERT INTO members_new 
                SELECT member_id, username, global_name, server_nick,
                       avatar_url, discriminator, bot, system, accent_color,
                       banner_url, discord_created_at, guild_join_date, role_ids,
                       created_at, updated_at, '[]' as notifications
                FROM members
            """)
            
            # Validate before dropping old table
            cursor.execute("SELECT COUNT(*) FROM members_new")
            if cursor.fetchone()[0] != original_count:
                raise ValueError("Row count mismatch before table swap")
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE members")
            cursor.execute("ALTER TABLE members_new RENAME TO members")
            
            logger.info("Successfully migrated members table")
            
        except Exception as e:
            # If anything goes wrong, we can restore from backup
            logger.error(f"Migration failed: {e}")
            cursor.execute("DROP TABLE IF EXISTS members")
            cursor.execute(f"ALTER TABLE {backup_name} RENAME TO members")
            raise
            
    except Exception as e:
        logger.error(f"Error during members table migration: {e}")
        raise

def migrate_daily_summaries(cursor):
    """Migrate daily_summaries table to use daily_summary_id as primary key."""
    logger.info("Starting daily_summaries migration")
    
    try:
        # Get current columns with their definitions
        existing_columns = get_table_columns(cursor, "daily_summaries")
        desired_schema = get_desired_daily_summaries_schema()
        
        # Find missing and extra columns
        desired_column_names = {name for name, _ in desired_schema}
        existing_column_names = set(existing_columns.keys())
        
        missing_columns = [
            (name, type_) for name, type_ in desired_schema 
            if name not in existing_columns
        ]
        extra_columns = existing_column_names - desired_column_names
        
        # Also check if any column types have changed
        type_changes = [
            (name, type_) for name, type_ in desired_schema
            if name in existing_columns and existing_columns[name]['type'] != type_
        ]
        
        # Check if daily_summary_id is not the primary key
        primary_key_wrong = any(
            col['primary_key'] == 1 and name != 'daily_summary_id'
            for name, col in existing_columns.items()
        )
        
        if not (missing_columns or extra_columns or type_changes or primary_key_wrong):
            logger.info("Daily summaries table schema is up to date")
            return

        if missing_columns:
            logger.info(f"Found missing columns: {[col[0] for col in missing_columns]}")
        if extra_columns:
            logger.info(f"Found columns to remove: {list(extra_columns)}")
        if type_changes:
            logger.info(f"Found columns with changed types: {[col[0] for col in type_changes]}")
        if primary_key_wrong:
            logger.info("Primary key needs to be changed to daily_summary_id")

        # Create backup
        backup_name = backup_table(cursor, "daily_summaries")
        logger.info(f"Created backup table: {backup_name}")
        
        try:
            # Get original row count
            cursor.execute("SELECT COUNT(*) FROM daily_summaries")
            original_count = cursor.fetchone()[0]
            
            # Create new table with correct schema
            columns_def = ", ".join([f"{name} {type_}" for name, type_ in desired_schema])
            cursor.execute(f"""
                CREATE TABLE daily_summaries_new (
                    {columns_def},
                    UNIQUE(date, channel_id) ON CONFLICT REPLACE
                )
            """)
            
            # Copy data, using id as daily_summary_id
            cursor.execute("""
                INSERT INTO daily_summaries_new 
                (daily_summary_id, date, channel_id, 
                 full_summary, short_summary, created_at)
                SELECT 
                    daily_summary_id, date, channel_id,
                    full_summary, short_summary, created_at
                FROM daily_summaries
            """)
            
            # Validate before dropping old table
            cursor.execute("SELECT COUNT(*) FROM daily_summaries_new")
            if cursor.fetchone()[0] != original_count:
                raise ValueError("Row count mismatch before table swap")
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE daily_summaries")
            cursor.execute("ALTER TABLE daily_summaries_new RENAME TO daily_summaries")
            
            logger.info("Successfully migrated daily_summaries table")
            
        except Exception as e:
            # If anything goes wrong, we can restore from backup
            logger.error(f"Migration failed: {e}")
            cursor.execute("DROP TABLE IF EXISTS daily_summaries")
            cursor.execute(f"ALTER TABLE {backup_name} RENAME TO daily_summaries")
            raise
            
    except Exception as e:
        logger.error(f"Error during daily_summaries migration: {e}")
        raise

def migrate_remove_raw_messages(cursor):
    """Remove raw_messages column from daily_summaries table."""
    logger.info("Starting raw_messages removal migration")
    
    try:
        # First check if raw_messages column exists
        cursor.execute("PRAGMA table_info(daily_summaries)")
        columns = {col[1]: col for col in cursor.fetchall()}
        
        if 'raw_messages' not in columns:
            logger.info("raw_messages column already removed")
            return
            
        # Get original row count
        cursor.execute("SELECT COUNT(*) FROM daily_summaries")
        original_count = cursor.fetchone()[0]
        
        # Create backup
        backup_name = backup_table(cursor, "daily_summaries")
        logger.info(f"Created backup table: {backup_name}")
        
        try:
            # Create new table without raw_messages
            cursor.execute("""
                CREATE TABLE daily_summaries_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    channel_id BIGINT NOT NULL,
                    full_summary TEXT,
                    short_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, channel_id) ON CONFLICT REPLACE,
                    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
                )
            """)
            
            # Copy data to new table
            cursor.execute("""
                INSERT INTO daily_summaries_new 
                (id, date, channel_id, 
                 full_summary, short_summary, created_at)
                SELECT 
                    id, date, channel_id,
                    full_summary, short_summary, created_at
                FROM daily_summaries
            """)
            
            # Validate before dropping old table
            cursor.execute("SELECT COUNT(*) FROM daily_summaries_new")
            if cursor.fetchone()[0] != original_count:
                raise ValueError("Row count mismatch before table swap")
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE daily_summaries")
            cursor.execute("ALTER TABLE daily_summaries_new RENAME TO daily_summaries")
            
            logger.info("Successfully removed raw_messages column")
            
        except Exception as e:
            # If anything goes wrong, we can restore from backup
            logger.error(f"Migration failed: {e}")
            cursor.execute("DROP TABLE IF EXISTS daily_summaries")
            cursor.execute(f"ALTER TABLE {backup_name} RENAME TO daily_summaries")
            raise
            
    except Exception as e:
        logger.error(f"Error during raw_messages removal migration: {e}")
        raise

def cleanup_backup_tables(cursor):
    """Clean up any backup tables from previous migrations."""
    try:
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_backup_%'")
        backup_tables = cursor.fetchall()
        
        for (table_name,) in backup_tables:
            logger.info(f"Dropping backup table: {table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
    except Exception as e:
        logger.error(f"Error cleaning up backup tables: {e}")
        raise

def migrate_database(dev_mode: bool = False):
    conn = None
    try:
        # Get appropriate database path
        db_path = get_database_path(dev_mode)
        logger.info(f"Using database at: {db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.execute("BEGIN TRANSACTION")  # Explicit transaction
        cursor = conn.cursor()

        # Run migrations in order
        migrate_members_table(cursor)  # Run members migration first
        migrate_messages_table(cursor)  # Then messages since it depends on members
        migrate_daily_summaries(cursor)
        migrate_remove_raw_messages(cursor)
        
        # Clean up backup tables
        cleanup_backup_tables(cursor)
        
        # Commit all changes
        conn.commit()
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Migrate database schema')
    parser.add_argument('--dev', action='store_true', help='Use development database')
    args = parser.parse_args()

    try:
        # Process development database
        if args.dev:
            logger.info("Starting database migration")
            logger.info(f"Processing development database at: {get_database_path(True)}")
            migrate_database(dev_mode=True)
            logger.info("Migration completed successfully for development database")
        else:
            # Process both databases
            logger.info("Starting database migration")
            
            # Process development database
            logger.info(f"Processing development database at: {get_database_path(True)}")
            migrate_database(dev_mode=True)
            logger.info("Migration completed successfully for development database")
            
            # Process production database
            logger.info(f"Processing production database at: {get_database_path(False)}")
            migrate_database(dev_mode=False)
            logger.info("Migration completed successfully for production database")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main() 