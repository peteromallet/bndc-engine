import logging
import asyncio
import re
import anthropic
from utils.errors import SummaryError

from typing import List, Dict, Any
from utils.error_handler import handle_errors

logger = logging.getLogger('SummaryService')


class MessageFormatter:
    @staticmethod
    def chunk_content(content: str, max_length: int = 1900):
        """Split content into chunks while preserving message links."""
        chunks = []
        current_chunk = ""
        current_chunk_links = set()

        lines = content.split('\n')
        for line in lines:
            message_links = set(re.findall(r'https://discord\.com/channels/\d+/\d+/(\d+)', line))

            # If the line triggers a new chunk...
            if (any(line.startswith(emoji) for emoji in ['🎥', '💻', '🎬', '🤖', '📱', '💡', '🔧', '🎨', '📊'])
                and current_chunk):
                chunks.append((current_chunk, current_chunk_links))
                current_chunk = ""
                current_chunk_links = set()
                current_chunk += '\n---\n\n'

            # Check if adding this line would exceed max_length
            if len(current_chunk) + len(line) + 2 <= max_length:
                current_chunk += line + '\n'
                current_chunk_links.update(message_links)
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_chunk_links))
                current_chunk = line + '\n'
                current_chunk_links = set(message_links)

        if current_chunk:
            chunks.append((current_chunk, current_chunk_links))

        return chunks


class SummaryService:
    """Handles all summary generation and Claude API interactions"""

    def __init__(self, claude, error_handler=None):
        self.claude = claude
        self.error_handler = error_handler
        self.max_tokens = 200000

    @handle_errors("generate_summary")
    async def generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a summary of the messages"""
        try:
            # Calculate approximate token count (rough estimate)
            total_text = "\n".join(msg.get('content', '') for msg in messages)
            estimated_tokens = len(total_text.split()) * 1.3  # Rough token estimation
            
            if estimated_tokens > self.max_tokens:
                logger.info(f"Message history too long ({estimated_tokens:.0f} tokens), chunking into smaller pieces")
                return await self._generate_chunked_summary(messages)
            
            return await self._generate_single_summary(messages)
            
        except Exception as e:
            logger.error(f"Error in generate_summary of {self.__class__.__name__}: {str(e)}")
            if self.error_handler:
                await self.error_handler.handle_error(e)
            return "[NOTHING OF NOTE]"
            
    async def _generate_chunked_summary(self, messages: List[Dict]) -> str:
        """Generate summary by breaking messages into chunks"""
        # Estimate average tokens per message
        sample_text = "\n".join(msg.get('content', '') for msg in messages[:100])
        avg_tokens_per_msg = len(sample_text.split()) * 1.3 / min(100, len(messages))
        
        # Calculate chunk size to stay under token limit with safety margin
        target_tokens = self.max_tokens * 0.4  # 40% of max to account for prompt and overhead
        chunk_size = max(100, min(5000, int(target_tokens / avg_tokens_per_msg)))
        
        logger.info(f"Using chunk size of {chunk_size} messages (avg {avg_tokens_per_msg:.1f} tokens/msg)")
        
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
        logger.info(f"Split into {len(chunks)} chunks")
        
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            try:
                logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} messages)")
                chunk_summary = await self._generate_single_summary(chunk)
                if chunk_summary != "[NOTHING OF NOTE]":
                    summaries.append(chunk_summary)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
                
        if not summaries:
            return "[NOTHING OF NOTE]"
            
        # Generate final summary of summaries
        try:
            final_summary = await self._generate_single_summary(
                [{'content': summary} for summary in summaries],
                is_meta_summary=True
            )
            return final_summary
        except Exception as e:
            logger.error(f"Error generating final summary: {str(e)}")
            return "\n\n".join(summaries[:3])  # Return first 3 chunk summaries if meta-summary fails
            
    async def _generate_single_summary(self, messages: List[Dict], is_meta_summary: bool = False) -> str:
        """Generate summary for a single chunk of messages"""
        if not messages:
            logger.info("No messages to summarize")
            return "[NOTHING OF NOTE]"
            
        logger.info(f"Generating summary for {len(messages)} messages")
        
        # Build different prompt based on whether this is a meta-summary
        if is_meta_summary:
            conversation = """Please combine these summaries into a single coherent summary. Keep the same format with topics and bullet points. Merge similar topics and maintain all Discord links and external links.

Summaries to combine:
"""
            for summary in messages:
                conversation += f"\n{summary['content']}\n"
        else:
            conversation = self.build_full_prompt(messages)

        loop = asyncio.get_running_loop()
        
        # Synchronous call in executor
        def create_summary():
            return self.claude.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=8192,
                messages=[{"role": "user", "content": conversation}],
                timeout=120
            )

        try:
            response = await loop.run_in_executor(None, create_summary)
            if not hasattr(response, 'content') or not response.content:
                raise ValueError("Invalid response from Claude API.")

            summary_text = response.content[0].text.strip()
            return summary_text
            
        except Exception as e:
            logger.error(f"Error in _generate_single_summary: {str(e)}")
            if self.error_handler:
                await self.error_handler.handle_error(e)
            return "[NOTHING OF NOTE]"

    @handle_errors("generate_short_summary")
    async def generate_short_summary(self, full_summary: str, message_count: int) -> str:
        """
        Create exactly 3 bullet points summarizing key developments
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                conversation = f"""Create exactly 3 bullet points summarizing key developments. STRICT format requirements:
1. The FIRST LINE MUST BE EXACTLY: __📨 {message_count} messages sent__
2. Then three bullet points that:
   - Start with -
   - Give a short summary of one of the main topics from the full summary
   - Bold the most important finding/result/insight using **
   - Keep each to a single line
3. Then EXACTLY one blank line with only a zero-width space
4. DO NOT MODIFY THE MESSAGE COUNT OR FORMAT IN ANY WAY

Required format:
"__📨 {message_count} messages sent__

- [Main topic 1]
- [Main topic 2]
- [Main topic 3]
\u200B"

Full summary to work from:
{full_summary}
"""

                loop = asyncio.get_running_loop()

                def create_short_summary():
                    return self.claude.messages.create(
                        model="claude-3-5-haiku-latest",
                        max_tokens=8192,
                        messages=[{"role": "user", "content": conversation}],
                    )

                response = await loop.run_in_executor(None, create_short_summary)
                return response.content[0].text.strip()

            except anthropic.exceptions.AnthropicException as e:
                logger.error(f"Anthropic error attempt {retry_count+1}/{max_retries}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(5)
                else:
                    raise SummaryError("Reached max retries due to Anthropic error.", e)

            except TimeoutError as e:
                logger.error(f"Timeout attempt {retry_count+1}/{max_retries}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(5)
                else:
                    raise SummaryError("Timed out too many times while generating short summary.", e)

            except Exception as e:
                logger.error(f"Unexpected error in generate_short_summary: {e}")
                # If it’s truly unknown, raise your own custom error to avoid swallowing.
                raise SummaryError("Unexpected error in short summary generation", e)

    def build_full_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Builds the conversation text for Claude with instructions and message data.
        """
        prompt = """Please summarize the interesting and noteworthy Discord happenings and ideas in bullet points. ALWAYS include Discord links and external links. You should extract ideas and information that may be useful to others from conversations. Avoid stuff like bug reports that are not generally helpful. Break them into topics and sub-topics.

If there's no significant content, respond with exactly "[NOTHING OF NOTE]".

Requirements:
1. ALWAYS include Discord links and external links 
2. Use Discord's markdown format
3. Use bullet points for each main topic
4. Start each main topic with "### " followed by an emoji
5. Use ** for bold text
6. Summaries should be helpful to others, ignoring ephemeral or trivial conversations
7. Keep it simple - bullet points only
8. Mention the message author's name in bold if relevant
9. Do not include extra commentary or disclaimers

Example:

### 🤏 **Compression Techniques for Video** 
- **zeevfarbman** recommended h265 compression: https://discord.com/channels/xxxx/xxxx/xxxx

Now here is the conversation:
"""

        for msg in messages:
            prompt += f"{msg['timestamp']} - {msg['author']}: {msg['content']}"
            if msg['reactions']:
                prompt += f"\nReactions: {msg['reactions']}"
            prompt += f"\nDiscord link: {msg['jump_url']}\n\n"

        return prompt