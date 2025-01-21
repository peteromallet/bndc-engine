import os
import sys
import praw
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RedditScraper:
    def __init__(self, subreddits: List[str], output_dir: str = "data/reddit_posts", comment_limit: int = 100, comment_depth: int = 3):
        """Initialize the Reddit scraper.
        
        Args:
            subreddits: List of subreddit names to scrape
            output_dir: Directory to save the scraped data
            comment_limit: Maximum number of top-level comments to fetch per post
            comment_depth: Maximum depth of comment replies to fetch
        """
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        self.subreddits = subreddits
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comment_limit = comment_limit
        self.comment_depth = comment_depth
    
    def process_comment(self, comment, depth: int = 0) -> Optional[Dict]:
        """Process a comment and its replies recursively.
        
        Args:
            comment: PRAW comment object
            depth: Current depth in comment tree
            
        Returns:
            Dictionary containing comment data and replies
        """
        if depth > self.comment_depth:
            return None
            
        try:
            comment_data = {
                'id': comment.id,
                'author': str(comment.author) if comment.author else '[deleted]',
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'edited': comment.edited if hasattr(comment, 'edited') else False,
                'is_submitter': comment.is_submitter if hasattr(comment, 'is_submitter') else False,
                'stickied': comment.stickied if hasattr(comment, 'stickied') else False,
                'permalink': f"https://reddit.com{comment.permalink}",
                'depth': depth,
                'replies': []
            }
            
            # Process replies if they exist and we haven't reached max depth
            if depth < self.comment_depth and hasattr(comment, 'replies'):
                # Ensure replies are loaded
                comment.replies.replace_more(limit=0)
                for reply in comment.replies:
                    reply_data = self.process_comment(reply, depth + 1)
                    if reply_data:
                        comment_data['replies'].append(reply_data)
            
            return comment_data
            
        except Exception as e:
            logger.error(f"Error processing comment {comment.id}: {str(e)}")
            return None
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 50) -> List[Dict]:
        """Scrape top posts and their comments from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit to scrape
            limit: Number of top posts to fetch
            
        Returns:
            List of post data dictionaries
        """
        logger.info(f"Scraping top {limit} posts from r/{subreddit_name}")
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for post in subreddit.top(time_filter='day', limit=limit):
                # Fetch the full comment tree
                post.comments.replace_more(limit=0)  # Remove MoreComments objects
                
                comments = []
                for comment in post.comments[:self.comment_limit]:  # Limit top-level comments
                    comment_data = self.process_comment(comment)
                    if comment_data:
                        comments.append(comment_data)
                
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'author': str(post.author) if post.author else '[deleted]',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'permalink': f"https://reddit.com{post.permalink}",
                    'is_self': post.is_self,
                    'selftext': post.selftext if post.is_self else None,
                    'link_flair_text': post.link_flair_text,
                    'over_18': post.over_18,
                    'stickied': post.stickied,
                    'scraped_at': datetime.utcnow().isoformat(),
                    'comments': comments
                }
                posts.append(post_data)
                logger.info(f"Scraped post {post.id} with {len(comments)} top-level comments")
                
                # Sleep briefly between posts to avoid rate limiting
                time.sleep(0.5)
                
            logger.info(f"Successfully scraped {len(posts)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
        
        return posts
    
    def save_posts(self, subreddit_name: str, posts: List[Dict]):
        """Save scraped posts to a JSON file.
        
        Args:
            subreddit_name: Name of the subreddit
            posts: List of post data to save
        """
        if not posts:
            logger.warning(f"No posts to save for r/{subreddit_name}")
            return
        
        # Create date-based directory structure
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        output_path = self.output_dir / date_str
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        filename = f"{subreddit_name}_{date_str}.json"
        file_path = output_path / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(posts)} posts to {file_path}")
        except Exception as e:
            logger.error(f"Error saving posts for r/{subreddit_name}: {str(e)}")
    
    def run(self):
        """Run the scraper for all configured subreddits."""
        for subreddit in self.subreddits:
            try:
                posts = self.scrape_subreddit(subreddit)
                self.save_posts(subreddit, posts)
                # Sleep briefly between subreddits to avoid rate limiting
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error processing subreddit {subreddit}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Scrape top posts from specified subreddits')
    parser.add_argument('--subreddits', nargs='+', required=True,
                      help='List of subreddit names to scrape')
    parser.add_argument('--output-dir', default='data/reddit_posts',
                      help='Directory to save scraped data (default: data/reddit_posts)')
    parser.add_argument('--comment-limit', type=int, default=100,
                      help='Maximum number of top-level comments to fetch per post (default: 100)')
    parser.add_argument('--comment-depth', type=int, default=3,
                      help='Maximum depth of comment replies to fetch (default: 3)')
    args = parser.parse_args()
    
    try:
        scraper = RedditScraper(
            args.subreddits,
            args.output_dir,
            comment_limit=args.comment_limit,
            comment_depth=args.comment_depth
        )
        scraper.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 