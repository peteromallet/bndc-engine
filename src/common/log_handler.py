import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

class LineCountRotatingFileHandler(RotatingFileHandler):
    """A handler that rotates based on both size and line count"""
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False, max_lines=None):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.max_lines = max_lines
        
        # Initialize line count from existing file
        if os.path.exists(filename):
            with open(filename, 'r', encoding=encoding or 'utf-8') as f:
                self.line_count = sum(1 for _ in f)
        else:
            self.line_count = 0
    
    def doRollover(self):
        """Override doRollover to keep last N lines"""
        if self.stream:
            self.stream.close()
            self.stream = None
            
        if self.max_lines:
            # Read all lines from the current file
            with open(self.baseFilename, 'r', encoding=self.encoding) as f:
                lines = f.readlines()
            
            # Keep only the last max_lines
            lines = lines[-self.max_lines:]
            
            # Write the last max_lines back to the file
            with open(self.baseFilename, 'w', encoding=self.encoding) as f:
                f.writelines(lines)
            
            self.line_count = len(lines)
        
        if not self.delay:
            self.stream = self._open()
    
    def emit(self, record):
        """Emit a record and check line count"""
        if self.max_lines and self.line_count >= self.max_lines:
            self.doRollover()
            
        super().emit(record)
        self.line_count += 1

class LogHandler:
    """Handles logging configuration with separate files for dev/prod and rotation"""
    
    def __init__(self, 
                 logger_name: str = 'Application',
                 prod_log_file: str = 'app.log',
                 dev_log_file: Optional[str] = 'app_dev.log'):
        """
        Initialize the log handler.
        
        Args:
            logger_name: Name of the logger
            prod_log_file: Path to production log file
            dev_log_file: Path to development log file (optional)
        """
        self.logger_name = logger_name
        self.prod_log_file = prod_log_file
        self.dev_log_file = dev_log_file
        self.logger = None

    def setup_logging(self, dev_mode: bool = False) -> logging.Logger:
        """
        Configure logging with separate files for dev/prod and rotation.
        
        Args:
            dev_mode: Whether to run in development mode
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG if dev_mode else logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if dev_mode else logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Production file handler (always active)
        prod_handler = RotatingFileHandler(
            self.prod_log_file,
            maxBytes=1024 * 1024,  # 1MB per file
            backupCount=5,
            encoding='utf-8'
        )
        prod_handler.setLevel(logging.INFO)
        prod_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        prod_handler.setFormatter(prod_formatter)
        logger.addHandler(prod_handler)
        
        # Development file handler (only when in dev mode)
        if dev_mode and self.dev_log_file:
            dev_handler = LineCountRotatingFileHandler(
                self.dev_log_file,
                maxBytes=512 * 1024,  # 512KB per file
                backupCount=200,
                encoding='utf-8',
                max_lines=200
            )
            dev_handler.setLevel(logging.DEBUG)
            dev_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            dev_handler.setFormatter(dev_formatter)
            
            # Only do rollover if file exists AND exceeds limits
            if os.path.exists(self.dev_log_file):
                if os.path.getsize(self.dev_log_file) > 512 * 1024:
                    dev_handler.doRollover()
                else:
                    # Count lines in existing file
                    with open(self.dev_log_file, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    if line_count > 200:
                        dev_handler.doRollover()
            
            logger.addHandler(dev_handler)
        
        # Log the logging configuration
        logger.info(f"Logging configured in {'development' if dev_mode else 'production'} mode")
        logger.info(f"Production log file: {self.prod_log_file}")
        if dev_mode and self.dev_log_file:
            logger.info(f"Development log file: {self.dev_log_file}")
        
        self.logger = logger
        return logger

    def get_logger(self) -> Optional[logging.Logger]:
        """Get the configured logger instance."""
        return self.logger 