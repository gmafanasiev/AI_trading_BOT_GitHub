# data_utils.py
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from config import SEQUENCE_LENGTH, NUM_DAYS_HISTORY, VOLATILITY, TRADE_MODE, ALPACA_API_KEY, ALPACA_API_SECRET, ATR_PERIOD

def fetch_bar_data(data_client: StockHistoricalDataClient, symbol: str, timeframe: TimeFrame, limit: int, file_handler) -> pd.DataFrame:
    """
    Fetch minute bar data for a given symbol.
    Returns a DataFrame with OHLCV data.
    """
    ny_tz = ZoneInfo("America/New_York")
    try:
        end_time = datetime.now(ny_tz)
        start_time = end_time - timedelta(days=2)  # Fetch 2 days to ensure enough data
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_time,
            end=end_time,
            limit=limit,
            feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(request_params).df
        if len(bars) == 0:
            logging.warning(f"Fetched only 0 bars for {symbol} (less than requested {limit})")
            return fetch_mock_bar_data(symbol, limit, file_handler)
        if len(bars) < limit:
            logging.warning(f"Fetched only {len(bars)} bars for {symbol} (less than requested {limit})")
        logging.info(f"Fetched {len(bars)} bars for {symbol}, latest price: ${bars['close'].iloc[-1]:.2f}")
        file_handler.flush()
        return bars
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        file_handler.flush()
        return fetch_mock_bar_data(symbol, limit, file_handler)

def fetch_daily_data(data_client: StockHistoricalDataClient, symbol: str, limit: int, file_handler) -> pd.DataFrame:
    """
    Fetch daily bar data for a given symbol.
    Returns a DataFrame with daily OHLCV data.
    """
    ny_tz = ZoneInfo("America/New_York")
    try:
        end_time = datetime.now(ny_tz)
        start_time = end_time - timedelta(days=limit)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_time,
            end=end_time,
            limit=limit,
            feed=DataFeed.IEX
        )
        daily_bars = data_client.get_stock_bars(request_params).df
        if len(daily_bars) == 0:
            logging.warning(f"Fetched only 0 daily bars for {symbol} (less than requested {limit})")
            return fetch_mock_daily_data(symbol, limit, file_handler)
        if len(daily_bars) < limit:
            logging.warning(f"Fetched only {len(daily_bars)} daily bars for {symbol} (less than requested {limit})")
        logging.info(f"Fetched {len(daily_bars)} daily bars for {symbol}")
        file_handler.flush()
        return daily_bars
    except Exception as e:
        logging.error(f"Error fetching daily data for {symbol}: {e}")
        file_handler.flush()
        return fetch_mock_daily_data(symbol, limit, file_handler)

def fetch_mock_bar_data(symbol: str, limit: int, file_handler) -> pd.DataFrame:
    """Generate mock minute bar data with realistic trend when live data is unavailable."""
    ny_tz = ZoneInfo("America/New_York")
    timestamps = [datetime.now(ny_tz) - timedelta(minutes=i) for i in range(limit)]
    base_price = 190.0  # Starting price for AAPL
    # Simulate realistic trend with 1-2% daily volatility
    returns = np.random.normal(0, VOLATILITY / np.sqrt(390), limit)  # Minute-level returns
    prices = base_price * np.exp(np.cumsum(returns))  # Geometric Brownian motion
    mock_data = {
        'open': prices,
        'high': [p + np.random.normal(0.5, VOLATILITY) for p in prices],
        'low': [p - np.random.normal(0.5, VOLATILITY) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in range(limit)],
        'timestamp': timestamps
    }
    bars = pd.DataFrame(mock_data)
    bars['high'] = bars[['open', 'high', 'close']].max(axis=1)
    bars['low'] = bars[['open', 'low', 'close']].min(axis=1)
    bars.set_index('timestamp', inplace=True)
    logging.info(f"Generated {len(bars)} mock bars for {symbol}, latest price: ${bars['close'].iloc[-1]:.2f}")
    file_handler.flush()
    return bars

def fetch_mock_daily_data(symbol: str, limit: int, file_handler) -> pd.DataFrame:
    """Generate mock daily bar data with realistic trend when live data is unavailable."""
    ny_tz = ZoneInfo("America/New_York")
    timestamps = [datetime.now(ny_tz) - timedelta(days=i) for i in range(limit)]
    base_price = 190.0  # Starting price for AAPL
    # Simulate daily volatility (~1-2%)
    returns = np.random.normal(0, VOLATILITY * 2, limit)  # Daily returns
    prices = base_price * np.exp(np.cumsum(returns))
    mock_data = {
        'open': prices,
        'high': [p + np.random.normal(1.0, VOLATILITY * 2) for p in prices],
        'low': [p - np.random.normal(1.0, VOLATILITY * 2) for p in prices],
        'close': prices,
        'volume': [np.random.randint(10000, 100000) for _ in range(limit)],
        'timestamp': timestamps
    }
    daily_bars = pd.DataFrame(mock_data)
    daily_bars['high'] = daily_bars[['open', 'high', 'close']].max(axis=1)
    daily_bars['low'] = daily_bars[['open', 'low', 'close']].min(axis=1)
    daily_bars.set_index('timestamp', inplace=True)
    logging.info(f"Generated {len(daily_bars)} mock daily bars for {symbol}, latest price: ${daily_bars['close'].iloc[-1]:.2f}")
    file_handler.flush()
    return daily_bars

def calculate_atr(bars: pd.DataFrame, period: int) -> float:
    """Calculate the Average True Range (ATR)."""
    if len(bars) < period:
        return 1.0  # Default fallback
    high_low = bars['high'] - bars['low']
    high_close = np.abs(bars['high'] - bars['close'].shift())
    low_close = np.abs(bars['low'] - bars['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    return atr

def prepare_grok4_input(bars: pd.DataFrame, daily_bars: pd.DataFrame, sequence_length: int, symbol: str, file_handler) -> dict:
    """
    Prepare statistical inputs for Grok-4 prediction model from bar data.
    """
    if len(bars) < sequence_length:
        logging.warning(f"No minute bars available for {sequence_length} sequence length")
        file_handler.flush()
        return None
    stats = {}
    stats['close_prices'] = bars['close'].values
    stats['volume'] = bars['volume'].values
    stats['returns'] = bars['close'].pct_change().dropna().values
    stats['volatility'] = np.std(stats['returns']) * np.sqrt(252 * 390) if len(stats['returns']) > 0 else 0.0
    stats['mean_return'] = np.mean(stats['returns']) if len(stats['returns']) > 0 else 0.0
    stats['last_close'] = bars['close'].iloc[-1] if not bars.empty else 190.0
    stats['atr'] = calculate_atr(bars, ATR_PERIOD)
    if not daily_bars.empty:
        stats['daily_closes'] = daily_bars['close'].values
        stats['daily_volatility'] = daily_bars['close'].pct_change().std() * np.sqrt(252) if len(daily_bars) > 1 else 0.0
        stats['daily_mean_return'] = daily_bars['close'].pct_change().mean() if len(daily_bars) > 1 else 0.0
    else:
        stats['daily_closes'] = []
        stats['daily_volatility'] = 0.0
        stats['daily_mean_return'] = 0.0
    logging.debug(f"Prepared stats for {symbol}: {stats}")
    file_handler.flush()
    return stats