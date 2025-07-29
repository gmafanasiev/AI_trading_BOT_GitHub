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
SYMBOLS = ["AAPL", "NVDA", "AMD"]
TRADE_MODE = 'paper'  # Options: 'simulation', 'paper', 'live'
BACKTEST_MODE = True  # Set to True for backtest mode
SEQUENCE_LENGTH = 15  # Reduced to 15 for free tier historical data limit (last 15 minutes)
NUM_DAYS_HISTORY = 5  # Increased for more context
UPPER_THRESHOLD = 0.65  # my original value
LOWER_THRESHOLD = 0.35  # my original value
ADJUSTMENT_INTERVAL = 30  # Seconds
MAX_POSITION_PCT = 0.10
MAX_EQUITY = 100000.0
STOP_LOSS_PCT = 0.10
TAKE_PROFIT_PCT = 0.06
TRAILING_PCT = 0.02
RISK_PER_TRADE = 0.10  # Increased to 10% for 10x higher trade value
TIMEFRAME = "Minute"
COOLDOWN_SECONDS = 60  # Cooldown period after a trade in seconds
SESSION_DURATION = "60 minutes"  # Options: "end_of_day", "<number> seconds", "<number> minutes"
LONG_EXIT_THRESHOLD = 0.4  # Exit long if prediction < this
SHORT_EXIT_THRESHOLD = 0.6  # Exit short if prediction > this
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
RR_RATIO = 1.5  # Reward-to-risk ratio for take-profit
MAX_DRAWDOWN = 0.05  # 5% drawdown limit to pause trading

# Additional parameters moved from modules
VOLATILITY = 0.005  # For simulated data in data_utils.py
TEMPERATURE = 0.3  # For Grok API in prediction.py
MARKET_CORR = 0.80  # Placeholder for prompt in prediction.py
SUPPORTS = {"AAPL": [211.00, 212.00, 213.00], "NVDA": [148.00, 171.00, 172.00], "AMD": [160.00, 163.00, 168.00]}  # Per-symbol, fetched/updated
RESISTANCES = {"AAPL": [214.00, 215.00, 217.00], "NVDA": [173.00, 174.00, 178.00], "AMD": [172.00, 173.00, 177.00]}  # Per-symbol, fetched/updated
POLL_TIMEOUT = 10  # Timeout for poll_order_fill in order_execution.py
LOG_LEVEL = "INFO"  # Logging level for setup_logging in logging_utils.py

# Position tracking
open_positions = {symbol: {"type": None, "qty": 0, "entry_price": 0.0} for symbol in SYMBOLS}
sl_prices = {symbol: 0.0 for symbol in SYMBOLS}
tp_prices = {symbol: 0.0 for symbol in SYMBOLS}