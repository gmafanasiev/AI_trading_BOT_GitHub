# data_utils.py
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import SEQUENCE_LENGTH, NUM_DAYS_HISTORY, VOLATILITY, TRADE_MODE, ALPACA_API_KEY, ALPACA_API_SECRET

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
            limit=limit
        )
        bars = data_client.get_stock_bars(request_params).df
        if len(bars) == 0:
            logging.warning(f"Fetched only 0 bars for {symbol} (less than requested {limit})")
            return fetch_mock_bar_data(symbol, limit, file_handler)
        if len(bars) < limit:
            logging.warning(f"Fetched only {len(bars)} bars for {symbol} (less than requested {limit})")
        logging.info(f"Fetched {len(bars)} bars for {symbol}, latest price: ${bars['close'][-1]:.2f}")
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
            limit=limit
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
    """Generate mock minute bar data when live data is unavailable."""
    ny_tz = ZoneInfo("America/New_York")
    timestamps = [datetime.now(ny_tz) - timedelta(minutes=i) for i in range(limit)]
    last_price = 190.0  # Default price
    mock_data = {
        'open': [last_price + np.random.normal(0, VOLATILITY) for _ in range(limit)],
        'high': [last_price + np.random.normal(0.5, VOLATILITY) for _ in range(limit)],
        'low': [last_price + np.random.normal(-0.5, VOLATILITY) for _ in range(limit)],
        'close': [last_price + np.random.normal(0, VOLATILITY) for _ in range(limit)],
        'volume': [np.random.randint(1000, 10000) for _ in range(limit)],
        'timestamp': timestamps
    }
    bars = pd.DataFrame(mock_data)
    bars.set_index('timestamp', inplace=True)
    logging.info(f"Generated {len(bars)} mock bars for {symbol}")
    file_handler.flush()
    return bars

def fetch_mock_daily_data(symbol: str, limit: int, file_handler) -> pd.DataFrame:
    """Generate mock daily bar data when live data is unavailable."""
    ny_tz = ZoneInfo("America/New_York")
    timestamps = [datetime.now(ny_tz) - timedelta(days=i) for i in range(limit)]
    mock_data = {
        'open': [190.0 + np.random.normal(0, VOLATILITY * 2) for _ in range(limit)],
        'high': [190.0 + np.random.normal(1.0, VOLATILITY * 2) for _ in range(limit)],
        'low': [190.0 + np.random.normal(-1.0, VOLATILITY * 2) for _ in range(limit)],
        'close': [190.0 + np.random.normal(0, VOLATILITY * 2) for _ in range(limit)],
        'volume': [np.random.randint(10000, 100000) for _ in range(limit)],
        'timestamp': timestamps
    }
    daily_bars = pd.DataFrame(mock_data)
    daily_bars.set_index('timestamp', inplace=True)
    logging.info(f"Generated {len(daily_bars)} mock daily bars for {symbol}")
    file_handler.flush()
    return daily_bars

def prepare_grok4_input(bars: pd.DataFrame, daily_bars: pd.DataFrame, sequence_length: int) -> dict:
    """
    Prepare statistical inputs for Grok-4 prediction model from bar data.
    """
    if len(bars) < sequence_length:
        logging.warning(f"No minute bars available for {sequence_length} sequence length")
        file_handler.flush()
        return None
    stats = {}
    stats['returns'] = bars['close'].pct_change().dropna().values
    stats['volatility'] = np.std(stats['returns']) * np.sqrt(252 * 390) if len(stats['returns']) > 0 else 0.0
    stats['mean_return'] = np.mean(stats['returns']) if len(stats['returns']) > 0 else 0.0
    stats['last_close'] = bars['close'][-1] if not bars.empty else 190.0
    if not daily_bars.empty:
        stats['daily_volatility'] = daily_bars['close'].pct_change().std() * np.sqrt(252) if len(daily_bars) > 1 else 0.0
        stats['daily_mean_return'] = daily_bars['close'].pct_change().mean() if len(daily_bars) > 1 else 0.0
    else:
        stats['daily_volatility'] = 0.0
        stats['daily_mean_return'] = 0.0
    file_handler.flush()
    return stats