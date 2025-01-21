import sqlite3
import os
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_db(db_path):
    """Connect to a database and return the connection."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        raise

def print_table_schema(conn, table_name):
    """Print the schema of a table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        schema = cursor.fetchone()
        if schema:
            logger.info(f"Schema for {table_name}:")
            logger.info(schema[0])
        
        # Print a sample row
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        row = cursor.fetchone()
        if row:
            logger.info(f"Sample row from {table_name}:")
            logger.info(dict(row))
    except Exception as e:
        logger.error(f"Error getting schema for {table_name}: {e}")

def get_unique_channel_ids(source_conn):
    """Get all unique channel IDs from both summary tables."""
    try:
        cursor = source_conn.cursor()
        
        # Get channel IDs from both tables
        cursor.execute("""
            SELECT DISTINCT channel_id FROM channel_summary
            UNION
            SELECT DISTINCT channel_id FROM daily_summaries
        """)
        
        return [row['channel_id'] for row in cursor.fetchall()]
        
    except Exception as e:
        logger.error(f"Error getting unique channel IDs: {e}")
        raise

def ensure_channels_exist(source_conn, dest_conn, channel_ids):
    """Ensure all needed channels exist in the channels table."""
    try:
        dest_cursor = dest_conn.cursor()
        
        # For each channel ID
        for channel_id in channel_ids:
            try:
                # Check if channel exists in destination
                dest_cursor.execute("SELECT 1 FROM channels WHERE id = ?", (channel_id,))
                if not dest_cursor.fetchone():
                    # If it doesn't exist, insert it
                    # We'll use a placeholder name since we don't have the actual name
                    dest_cursor.execute("""
                        INSERT INTO channels (id, name, type)
                        VALUES (?, ?, ?)
                    """, (channel_id, f"channel_{channel_id}", "text"))
                    logger.info(f"Added missing channel {channel_id} to channels table")
            except Exception as e:
                logger.error(f"Error ensuring channel {channel_id} exists: {e}")
                continue
                
        dest_conn.commit()
        logger.info("Channel existence check completed")
        
    except Exception as e:
        logger.error(f"Error in channel existence check: {e}")
        raise

def migrate_channel_summaries(source_conn, dest_conn):
    """Migrate data from channel_summary table."""
    try:
        # Get data from source
        cursor = source_conn.cursor()
        cursor.execute("SELECT * FROM channel_summary")
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} rows in source channel_summary table")

        # Insert into destination
        dest_cursor = dest_conn.cursor()
        for row in rows:
            try:
                # Convert row to dict for debugging
                row_dict = dict(row)
                logger.debug(f"Processing row: {row_dict}")
                
                dest_cursor.execute("""
                    INSERT OR REPLACE INTO channel_summary 
                    (channel_id, summary_thread_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    row_dict['channel_id'],
                    row_dict['summary_thread_id'],
                    row_dict.get('created_at'),  # Use get() to handle missing timestamps
                    row_dict.get('updated_at')
                ))
                logger.debug(f"Migrated channel summary for channel {row_dict['channel_id']}")
            except Exception as e:
                logger.error(f"Error migrating channel summary row: {e}")
                logger.error(f"Row data: {row_dict}")
                continue

        dest_conn.commit()
        logger.info("Channel summaries migration completed")

    except Exception as e:
        logger.error(f"Error in channel summaries migration: {e}")
        raise

def migrate_daily_summaries(source_conn, dest_conn):
    """Migrate data from daily_summaries table."""
    try:
        # Get data from source
        cursor = source_conn.cursor()
        cursor.execute("SELECT * FROM daily_summaries")
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} rows in source daily_summaries table")

        # Get destination table schema
        dest_cursor = dest_conn.cursor()
        dest_cursor.execute("PRAGMA table_info(daily_summaries)")
        dest_columns = [col[1] for col in dest_cursor.fetchall()]
        logger.info(f"Destination columns: {dest_columns}")

        # Insert into destination
        dest_cursor = dest_conn.cursor()
        for row in rows:
            try:
                # Convert row to dict for debugging
                row_dict = dict(row)
                logger.debug(f"Processing row: {row_dict}")
                
                # Build insert statement based on destination schema
                columns = []
                values = []
                
                # Required fields
                columns.extend(['date', 'channel_id', 'message_count', 'raw_messages'])
                values.extend([
                    row_dict['date'],
                    row_dict['channel_id'],
                    row_dict['message_count'],
                    row_dict['raw_messages']  # Use raw_messages directly from source
                ])
                
                # Optional fields
                if 'full_summary' in dest_columns and 'full_summary' in row_dict:
                    columns.append('full_summary')
                    values.append(row_dict['full_summary'])
                if 'short_summary' in dest_columns and 'short_summary' in row_dict:
                    columns.append('short_summary')
                    values.append(row_dict['short_summary'])
                if 'created_at' in dest_columns and 'created_at' in row_dict:
                    columns.append('created_at')
                    values.append(row_dict['created_at'])

                # Construct and execute insert statement
                placeholders = ','.join(['?' for _ in values])
                columns_str = ','.join(columns)
                dest_cursor.execute(f"""
                    INSERT OR REPLACE INTO daily_summaries 
                    ({columns_str})
                    VALUES ({placeholders})
                """, values)
                
                logger.debug(f"Migrated daily summary for channel {row_dict['channel_id']} on {row_dict['date']}")
            except Exception as e:
                logger.error(f"Error migrating daily summary row: {e}")
                logger.error(f"Row data: {row_dict}")
                continue

        dest_conn.commit()
        logger.info("Daily summaries migration completed")

    except Exception as e:
        logger.error(f"Error in daily summaries migration: {e}")
        raise

def main():
    try:
        # Get database paths
        source_db = 'data.db'
        dest_db = os.path.join('data', 'production.db')

        # Ensure source database exists
        if not os.path.exists(source_db):
            logger.error(f"Source database {source_db} not found")
            return

        # Ensure destination directory exists
        os.makedirs('data', exist_ok=True)

        logger.info(f"Starting migration from {source_db} to {dest_db}")

        # Connect to databases
        source_conn = connect_to_db(source_db)
        dest_conn = connect_to_db(dest_db)

        try:
            # Print schemas for comparison
            logger.info("=== Source Database Schemas ===")
            print_table_schema(source_conn, "channel_summary")
            print_table_schema(source_conn, "daily_summaries")
            
            logger.info("\n=== Destination Database Schemas ===")
            print_table_schema(dest_conn, "channel_summary")
            print_table_schema(dest_conn, "daily_summaries")
            
            # Get all unique channel IDs that need to exist
            channel_ids = get_unique_channel_ids(source_conn)
            logger.info(f"Found {len(channel_ids)} unique channels to migrate")
            
            # Ensure all needed channels exist in destination
            ensure_channels_exist(source_conn, dest_conn, channel_ids)
            
            # Now migrate the summaries
            migrate_channel_summaries(source_conn, dest_conn)
            migrate_daily_summaries(source_conn, dest_conn)
            
            logger.info("Migration completed successfully")

        finally:
            # Close connections
            source_conn.close()
            dest_conn.close()

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main() 