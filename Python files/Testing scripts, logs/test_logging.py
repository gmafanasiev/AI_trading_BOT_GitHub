# test_logging.py
from logging_utils import setup_logging
logger, file_handler = setup_logging('/home/gmafanasiev/ai_trading_bot/test_log.log')
logger.info("Test log message")
file_handler.flush()
print("Logging setup completed")