import os
import sys
import time
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import logging
from typing import List, Set, Dict
from src.common.constants import DATABASE_PATH
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
    """Get the desired schema structure."""
    return get_schema_tuples()

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
        
        # Get list of columns that exist in both old and new schema
        common_columns = [name for name, _ in desired_schema if name in existing_columns]
        columns_to_copy = ", ".join(common_columns)
        
        # Create new table and copy data
        cursor.execute(f"""
            CREATE TABLE messages_new (
                {columns_def}
            )
        """)
        
        cursor.execute(f"""
            INSERT INTO messages_new ({columns_to_copy})
            SELECT {columns_to_copy}
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

def migrate_database():
    conn = None
    try:
        # Connect to the database
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("BEGIN TRANSACTION")  # Explicit transaction
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
        
        # Find missing and extra columns
        desired_column_names = {name for name, _ in desired_schema}
        existing_column_names = set(existing_columns.keys())
        
        missing_columns = [
            (name, type_) for name, type_ in desired_schema 
            if name not in existing_columns
        ]
        extra_columns = existing_column_names - desired_column_names
        
        if not missing_columns and not extra_columns:
            logger.info("Database schema is up to date")
            return

        if missing_columns:
            logger.info(f"Found missing columns: {[col[0] for col in missing_columns]}")
        if extra_columns:
            logger.info(f"Found columns to remove: {list(extra_columns)}")

        # If there are any schema changes, recreate the table
        logger.info("Recreating table with updated schema...")
        create_temp_table_and_migrate_data(cursor, desired_schema, existing_columns)
        
        # Handle special cases for data migration
        if 'message_id' in [col[0] for col in missing_columns]:
            cursor.execute("UPDATE messages SET message_id = id WHERE message_id IS NULL")
            logger.info("Updated message_id values from id column")

        # Commit the changes
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

if __name__ == "__main__":
    migrate_database() 