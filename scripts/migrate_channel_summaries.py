"""Script to migrate data from channel_summaries.db to data.db"""
import sqlite3
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from src.common.constants import DATABASE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='migration_channel_summaries.log'
)

def get_table_info(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def migrate_data():
    try:
        if not Path('channel_summaries.db').exists():
            print("Error: channel_summaries.db not found!")
            return
        
        old_conn = sqlite3.connect('channel_summaries.db')
        new_conn = sqlite3.connect(DATABASE_PATH)
        
        old_cur = old_conn.cursor()
        new_cur = new_conn.cursor()

        # Print existing table structures
        print("\nExisting table structure in data.db:")
        for table in ['daily_summaries', 'channel_summary']:
            print(f"\n{table} table columns:")
            columns = get_table_info(new_cur, table)
            for col in columns:
                print(f"  {col}")
        
        # Migrate daily_summaries table
        print("\nChecking daily_summaries table...")
        old_cur.execute("SELECT COUNT(*) FROM daily_summaries")
        count = old_cur.fetchone()[0]
        print(f"Found {count} records in daily_summaries table")
        
        if count > 0:
            old_cur.execute("""
                SELECT date, channel_id, channel_name, message_count, 
                       raw_messages, full_summary, short_summary, 
                       created_at
                FROM daily_summaries
            """)
            daily_summaries = old_cur.fetchall()
            
            for summary in daily_summaries:
                try:
                    new_cur.execute("""
                        INSERT INTO daily_summaries 
                        (date, channel_id, channel_name, message_count,
                         raw_messages, full_summary, short_summary,
                         created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, summary)
                    print(f"Migrated daily summary for channel {summary[1]} date {summary[0]}")
                except sqlite3.Error as e:
                    print(f"Error migrating daily summary for channel {summary[1]} date {summary[0]}: {e}")

        # Migrate channel_summary table
        print("\nChecking channel_summary table...")
        old_cur.execute("SELECT COUNT(*) FROM channel_summary")
        count = old_cur.fetchone()[0]
        print(f"Found {count} records in channel_summary table")
        
        if count > 0:
            old_cur.execute("""
                SELECT channel_id, summary_thread_id, created_at, 
                       updated_at
                FROM channel_summary
            """)
            channel_summaries = old_cur.fetchall()
            
            for summary in channel_summaries:
                try:
                    new_cur.execute("""
                        INSERT INTO channel_summary 
                        (channel_id, summary_thread_id, created_at,
                         updated_at)
                        VALUES (?, ?, ?, ?)
                    """, summary)
                    print(f"Migrated channel summary for channel {summary[0]}")
                except sqlite3.Error as e:
                    print(f"Error migrating channel summary for channel {summary[0]}: {e}")
        
        # Commit changes
        new_conn.commit()
        print("\nMigration completed successfully")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        logging.error(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
        raise
    finally:
        if 'old_conn' in locals():
            old_conn.close()
        if 'new_conn' in locals():
            new_conn.close()

if __name__ == "__main__":
    print("Starting migration from channel_summaries.db to data.db...")
    migrate_data() 