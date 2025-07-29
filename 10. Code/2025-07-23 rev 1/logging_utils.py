# logging_utils.py
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

# Global flag to prevent multiple logger configurations
_LOGGER_CONFIGURED = False

class EDTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("America/New_York"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]

def setup_logging(level=logging.DEBUG):
    """
    Set up logging configuration for the trading bot.

    :param level: Logging level for trading logic (e.g., logging.DEBUG).
    :return: Tuple of (logger, file_handler).
    """
    global _LOGGER_CONFIGURED
    logger = logging.getLogger()
    if _LOGGER_CONFIGURED:
        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
        logging.debug(f"Logger already configured, handler count: {len(logger.handlers)}")
        return logger, file_handler

    logger.setLevel(level)

    # Create file handler with overwrite mode
    file_handler = logging.FileHandler('trade_alpaca_grok4.log', mode='w')
    file_handler.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter with EDT timezone
    formatter = EDTFormatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log handler count for debugging
    logging.debug(f"Logger configured, handler count: {len(logger.handlers)}")

    # Suppress matplotlib font logs
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    _LOGGER_CONFIGURED = True
    return logger, file_handler