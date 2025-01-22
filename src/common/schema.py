from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Column:
    name: str
    type: str
    description: str
    nullable: bool = False
    default: Optional[str] = None

def get_messages_schema() -> List[Column]:
    """Define the schema structure for the messages table."""
    return [
        Column("message_id", "BIGINT PRIMARY KEY", "Discord's message ID"),
        Column("channel_id", "BIGINT", "Discord channel ID where message was sent"),
        Column("author_id", "BIGINT", "Discord user ID of message author"),
        Column("content", "TEXT", "Text content of the message"),
        Column("created_at", "TEXT", "Timestamp when message was created"),
        Column("attachments", "TEXT", "JSON string of message attachments"),
        Column("embeds", "TEXT", "JSON string of message embeds"),
        Column("reaction_count", "INTEGER", "Total number of reactions on the message", default="0"),
        Column("reactors", "TEXT", "JSON string of user IDs who reacted", nullable=True),
        Column("reference_id", "BIGINT", "ID of message being replied to", nullable=True),
        Column("edited_at", "TEXT", "Timestamp of last edit", nullable=True),
        Column("is_pinned", "BOOLEAN", "Whether message is pinned"),
        Column("thread_id", "BIGINT", "ID of thread if message started one", nullable=True),
        Column("message_type", "TEXT", "Type of message (default, reply, etc)"),
        Column("flags", "INTEGER", "Discord message flags"),
        Column("jump_url", "TEXT", "URL to jump to message in Discord"),
        Column("indexed_at", "TIMESTAMP", "When message was indexed in database", default="CURRENT_TIMESTAMP")
    ]

def get_members_schema() -> List[Column]:
    """Define the schema structure for the members table."""
    return [
        Column("id", "BIGINT PRIMARY KEY", "Discord user ID"),
        Column("username", "TEXT NOT NULL", "Discord username"),
        Column("display_name", "TEXT", "Display name in the server", nullable=True),
        Column("created_at", "TIMESTAMP", "When member was first added to database", 
               default="CURRENT_TIMESTAMP"),
        Column("updated_at", "TIMESTAMP", "When member was last updated", 
               default="CURRENT_TIMESTAMP")
    ]

def get_schema_tuples() -> List[Tuple[str, str]]:
    """Get all schema definitions as tuples of (name, type)."""
    messages_schema = [(col.name, col.type) for col in get_messages_schema()]
    members_schema = [(col.name, col.type) for col in get_members_schema()]
    return messages_schema + members_schema 