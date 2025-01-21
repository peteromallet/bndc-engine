# BNDC Engine

A Discord bot that generates comprehensive daily summaries of channel activities using Claude 3.5 Sonnet. The bot monitors specified channels and categories, processes messages, and creates organized summaries with media handling, thread management, and historical archiving capabilities.

## Features

- 📊 Daily automated summaries at 10:00 UTC
- 🧵 Creates and maintains summary threads for each channel
- 📱 Handles images, videos, and other attachments with smart caching
- ⭐ Tracks reaction counts and highlights popular content
- 🤖 AI-powered summaries using Claude 3.5 Sonnet
- 📝 Organizes content into topics with emoji headers
- 🔗 Preserves Discord message links for reference
- ⚡ Rate limiting and exponential backoff for API calls
- 🗄️ Database storage for historical summaries
- 📊 Advanced logging system with detailed error tracking
- 📚 Historical message archiving and backup
- 📈 Enhanced daily summaries with:
  - Topic clustering and categorization
  - Sentiment analysis
  - User engagement metrics
  - Media content analysis
- 🔍 Comprehensive data curation and cleanup tools
- 🛠️ Development mode for testing with sample data
- 📂 Flexible monitoring of both categories and individual channels
- 🎯 Smart media handling with collages and video combinations

## Setup

### Installation

1. Clone the repository:
```bash
git clone https://github.com/peteromallet/bndc-engine.git
cd bndc-engine

```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```
# Required Bot Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token
ANTHROPIC_API_KEY=your_anthropic_api_key

# Main Configuration
GUILD_ID=your_server_id
PRODUCTION_SUMMARY_CHANNEL_ID=channel_id_for_summaries
CATEGORIES_TO_MONITOR=category_id1,category_id2
ADMIN_USER_ID=your_admin_user_id

# Development Configuration (optional)
DEV_GUILD_ID=dev_server_id
DEV_SUMMARY_CHANNEL_ID=dev_channel_id
DEV_CATEGORIES_TO_MONITOR=dev_category_ids
TEST_DATA_CHANNEL=test_channel_id

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=discord_bot.log
MAX_LOG_SIZE=10485760  # 10MB
LOG_BACKUP_COUNT=5
```

### Running the Bot

Basic operation:
```bash
python main.py
```

Development mode:
```bash
python main.py --dev
```

Run summary immediately:
```bash
python main.py --run-now
```

### Bot Permissions

The bot requires the following Discord permissions:
- Read Messages/View Channels
- Send Messages
- Create Public Threads
- Send Messages in Threads
- Manage Messages (for pinning)
- Read Message History
- Attach Files
- Add Reactions
- View Channel
- Manage Threads

### Development Mode

Run the bot in development mode to:
- Use test data instead of live channels
- Test in a development server
- Avoid affecting production data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Logging System

The bot includes a robust logging system that:
- Tracks all bot operations and interactions
- Provides detailed error reporting and stack traces
- Maintains separate log files for different components
- Implements log rotation and cleanup
- Supports different logging levels (DEBUG, INFO, WARNING, ERROR)

## Historical Archiving

The bot automatically archives Discord messages and content:
- Stores complete message history with metadata
- Archives attachments and media content
- Maintains reaction and engagement data
- Supports data migration and backup
- Includes cleanup tools for managing archived data

Archive management commands:
```bash
# Archive specific channels or date ranges
python scripts/archive_discord.py --channel-id <channel_id> --start-date YYYY-MM-DD

# Clean up test or temporary data
python scripts/cleanup_test_data.py

# Migrate database schema
python scripts/migrate_db.py
```