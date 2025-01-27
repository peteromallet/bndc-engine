#!/usr/bin/env python3

import sqlite3
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_channel_summary_table(db_path):
    """Migrate the channel_summary table to the new schema."""
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Backup existing data
        logger.info("Backing up existing data...")
        cursor.execute("SELECT * FROM channel_summary")
        existing_data = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Found {len(existing_data)} existing records")

        # Drop the existing table
        logger.info("Dropping existing table...")
        cursor.execute("DROP TABLE channel_summary")

        # Create the table with new schema
        logger.info("Creating new table with updated schema...")
        cursor.execute("""
            CREATE TABLE channel_summary (
                channel_id BIGINT,
                summary_thread_id BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id),
                PRIMARY KEY (channel_id, created_at)
            )
        """)

        # Restore the data
        if existing_data:
            logger.info("Restoring existing data...")
            for row in existing_data:
                cursor.execute("""
                    INSERT INTO channel_summary 
                    (channel_id, summary_thread_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    row['channel_id'],
                    row['summary_thread_id'],
                    row['created_at'],
                    row['updated_at']
                ))

        # Commit the changes
        conn.commit()
        logger.info("Migration completed successfully")

    except Exception as e:
        logger.error(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    # Get the database path
    db_path = os.path.join('data', 'production.db')
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        sys.exit(1)

    # Run the migration
    try:
        migrate_channel_summary_table(db_path)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1) 