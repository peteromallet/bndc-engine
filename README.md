# Discord Summary Bot

A Discord bot that generates daily summaries of channel activities using Claude 3.5 Sonnet. The bot monitors specified categories, processes messages, and creates comprehensive summaries including relevant attachments and reactions.

## Features

- 📊 Daily automated summaries at 10:00 UTC
- 🔍 Monitors specified Discord categories and channels
- 📎 Tracks and reposts popular attachments (images, videos)
- ⭐ Highlights messages with significant reactions
- 🤖 AI-powered summaries using Claude 3.5 Sonnet
- 📝 Organizes content into topics and sub-topics
- 🔗 Preserves message links for reference
## Setup

### Prerequisites

- Python 3.8 or higher
- Discord Bot Token
- Anthropic API Key (for Claude)
- Discord Server with appropriate permissions

### Installation

1. Clone the repository:
```bash
git clone https://github.com/peteromallet/discord_summary_bot.git
cd discord-summary-bot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
DISCORD_TOKEN=your_discord_bot_token
ANTHROPIC_API_KEY=your_anthropic_api_key
GUILD_ID=your_server_id
SUMMARY_CHANNEL_ID=channel_id_for_summaries
CATEGORIES_TO_MONITOR=category_id1,category_id2
```

### Configuration

1. Create a Discord application and bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable necessary bot permissions:
   - Read Messages/View Channels
   - Send Messages
   - Read Message History
   - Attach Files
3. Invite the bot to your server using the OAuth2 URL generator

### Running the Bot

1. Start the bot:
```bash
python summary.py
```

2. The bot will automatically:
   - Monitor specified categories
   - Generate daily summaries at 10:00 UTC
   - Post summaries in the designated channel

```

This setup guide provides clear instructions for installing, configuring, and running the bot, making it easy for users to get started with the project.