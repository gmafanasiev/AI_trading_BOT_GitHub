# test_imports.py (create temporarily in /home/gmafanasiev/ai_trading_bot/)
import sys
sys.path.insert(0, '.')

import config
import logging_utils
import clients
import data_utils
import prediction
import trade_analysis
import order_execution
import trading_loop
print("All modules imported successfully")