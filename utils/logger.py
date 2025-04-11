"""
Logging configuration for the application.
Sets up logging for various components.
"""
import logging
import sys
from pathlib import Path
from config.config import LOGGING_CONFIG

def setup_logger(name='rndv_ghandi'):
    """
    Set up and configure the application logger.
    
    Args:
        name (str, optional): Name of the logger. Defaults to 'rndv_ghandi'.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    # If the logger already has handlers, don't add more
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if LOGGING_CONFIG.get('file'):
        log_path = Path(LOGGING_CONFIG['file'])
        
        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info("Logger initialized with level %s", LOGGING_CONFIG['level'])
    return logger

# Create a singleton logger instance
logger = setup_logger()
