# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# xAI API credentials
XAI_API_KEY = os.getenv("XAI_API_KEY")
URL = "https://api.x.ai/v1/chat/completions"

# Trading parameters
SYMBOLS = ["AAPL"]
TRADE_MODE = 'simulation'  # Options: 'simulation', 'paper', 'live'
SEQUENCE_LENGTH = 5  # Updated for quicker predictions
UPPER_THRESHOLD = 0.65
LOWER_THRESHOLD = 0.35
ADJUSTMENT_INTERVAL = 30  # Changed from 300 to 30 seconds
MAX_POSITION_PCT = 0.10
MAX_EQUITY = 100000.0
STOP_LOSS_PCT = 0.10
TAKE_PROFIT_PCT = 0.06
TRAILING_PCT = 0.02
RISK_PER_TRADE = 0.01
TIMEFRAME = "Minute"
COOLDOWN_SECONDS = 60  # Cooldown period after a trade in seconds
SESSION_DURATION = "5 minutes"  # Options: "end_of_day", "<number> seconds", "<number> minutes"

# Position tracking
open_positions = {symbol: {"type": None, "qty": 0, "entry_price": 0.0} for symbol in SYMBOLS}
sl_prices = {symbol: 0.0 for symbol in SYMBOLS}
tp_prices = {symbol: 0.0 for symbol in SYMBOLS}