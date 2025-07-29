# test_modules.py
import logging
from logging_utils import setup_logging
from data_utils import prepare_grok4_input
import pandas as pd
import numpy as np

# Setup logging
logger, file_handler = setup_logging('/home/gmafanasiev/ai_trading_bot/test_log.log')

# Mock data for prepare_grok4_input
mock_data = pd.DataFrame({'close': np.random.rand(60) * 100})
stats = prepare_grok4_input(mock_data, 60)
logger.info(f"Mock stats: {stats}")
file_handler.flush()
print("Module interaction test passed")