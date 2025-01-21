"""Common constants used throughout the application."""
import os

# Database
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
PROD_DATABASE_PATH = os.path.join(DATA_DIR, 'production.db')
DEV_DATABASE_PATH = os.path.join(DATA_DIR, 'dev.db')

def get_database_path(dev_mode: bool = False) -> str:
    """Get the appropriate database path based on mode."""
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    return DEV_DATABASE_PATH if dev_mode else PROD_DATABASE_PATH 