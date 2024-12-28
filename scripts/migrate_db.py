import sqlite3
import logging
from typing import List, Set, Dict
from src.common.constants import DATABASE_PATH

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
    """Define the desired schema structure."""
    return [
        ("id", "BIGINT PRIMARY KEY"),
        ("message_id", "BIGINT"),
        ("channel_id", "BIGINT"),
        ("author_id", "BIGINT"),
        ("author_name", "TEXT"),
        ("content", "TEXT"),
        ("created_at", "TEXT"),
        ("attachments", "TEXT"),
        ("embeds", "TEXT"),
        ("reactions", "TEXT"),
        ("reference_id", "BIGINT"),
        ("edited_at", "TEXT"),
        ("is_pinned", "BOOLEAN"),
        ("thread_id", "BIGINT"),
        ("message_type", "TEXT"),
        ("flags", "INTEGER"),
        ("indexed_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
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

def migrate_database():
    try:
        # Connect to the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check if messages table exists
        if not table_exists(cursor, "messages"):
            logger.info("Messages table not found. Creating...")
            create_messages_table(cursor)
            conn.commit()
            logger.info("Messages table created successfully")
            return

        # Get current columns with their definitions
        existing_columns = get_table_columns(cursor, "messages")
        desired_schema = get_desired_schema()
        
        # Find missing columns
        missing_columns = [
            (name, type_) for name, type_ in desired_schema 
            if name not in existing_columns
        ]
        
        if not missing_columns:
            logger.info("Database schema is up to date")
            return

        logger.info(f"Found missing columns: {[col[0] for col in missing_columns]}")

        # Add missing columns
        for col_name, col_type in missing_columns:
            try:
                alter_sql = f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}"
                cursor.execute(alter_sql)
                logger.info(f"Added column: {col_name}")

                # Handle special cases for data migration
                if col_name == 'message_id':
                    cursor.execute("UPDATE messages SET message_id = id WHERE message_id IS NULL")
                    logger.info("Updated message_id values from id column")
                
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    logger.error(f"Error adding column {col_name}: {e}")
                continue

        # Commit the changes
        conn.commit()
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database() 