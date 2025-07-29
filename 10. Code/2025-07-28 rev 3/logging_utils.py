# logging_utils.py
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from config import LOG_LEVEL

class EDTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("America/New_York"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

def setup_logging(filename=None, level=LOG_LEVEL):
    """
    Set up logging configuration with console and optional file output in EDT.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = EDTFormatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if filename provided)
    if filename:
        try:
            # Only create directory if filename has a path component
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(log_level)
            file_formatter = EDTFormatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Failed to set up file logging for {filename}: {e}")
            file_handler = logging.NullHandler()
    else:
        file_handler = logging.NullHandler()

    # Log initialization
    logger.info(f"Logging initialized with level {level}, file: {filename if filename else 'None'} at {datetime.now(ZoneInfo('America/New_York'))}")
    return logger, file_handler