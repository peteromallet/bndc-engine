from dataclasses import dataclass
from typing import List, Optional

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
        Column("id", "BIGINT PRIMARY KEY", "Unique identifier for the message"),
        Column("message_id", "BIGINT", "Discord's message ID"),
        Column("channel_id", "BIGINT", "Discord channel ID where message was sent"),
        Column("channel_name", "TEXT", "Name of the channel"),
        Column("author_id", "BIGINT", "Discord user ID of message author"),
        Column("author_name", "TEXT", "Username of message author"),
        Column("author_avatar_url", "TEXT", "URL to author's avatar image"),
        Column("content", "TEXT", "Text content of the message"),
        Column("created_at", "TEXT", "Timestamp when message was created"),
        Column("attachments", "TEXT", "JSON string of message attachments"),
        Column("embeds", "TEXT", "JSON string of message embeds"),
        Column("reactions", "TEXT", "JSON string of message reactions"),
        Column("reference_id", "BIGINT", "ID of message being replied to"),
        Column("edited_at", "TEXT", "Timestamp of last edit", nullable=True),
        Column("is_pinned", "BOOLEAN", "Whether message is pinned"),
        Column("thread_id", "BIGINT", "ID of thread if message started one", nullable=True),
        Column("message_type", "TEXT", "Type of message (default, reply, etc)"),
        Column("flags", "INTEGER", "Discord message flags"),
        Column("jump_url", "TEXT", "URL to jump to message in Discord"),
        Column("indexed_at", "TIMESTAMP", "When message was indexed in database", 
               default="CURRENT_TIMESTAMP")
    ]

def get_schema_tuples() -> List[tuple]:
    """Convert schema to simple (name, type) tuples for migration."""
    def build_type(col: Column) -> str:
        type_def = col.type
        if not col.nullable:
            type_def += " NOT NULL"
        if col.default is not None:
            type_def += f" DEFAULT {col.default}"
        return type_def

    return [(col.name, build_type(col)) for col in get_messages_schema()] 