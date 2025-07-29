# data_utils.py
import pandas as pd
import numpy as np
import random
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import TRADE_MODE, ALPACA_API_KEY, ALPACA_API_SECRET, VOLATILITY, NUM_DAYS_HISTORY

"""
Data utilities module for fetching and preparing bar data.
"""

async def fetch_bar_data(data_client: StockHistoricalDataClient, symbol: str, timeframe, limit: int, file_handler) -> pd.DataFrame:
    """
    Fetch bar data for a symbol, either from Alpaca or simulated data.

    :param data_client: Alpaca data client.
    :param symbol: Stock symbol.
    :param timeframe: TimeFrame object.
    :param limit: Number of bars to fetch.
    :param file_handler: Logging file handler.
    :return: DataFrame with bar data.
    """
    try:
        if TRADE_MODE == 'simulation':
            # Simulate price data
            last_price = 200.0  # Starting price, adjust based on historical_analysis_AAPL.json
            bars = pd.DataFrame({
                'close': [last_price * (1 + random.uniform(-VOLATILITY, VOLATILITY)) for _ in range(limit)],
                'open': [last_price * (1 + random.uniform(-VOLATILITY, VOLATILITY)) for _ in range(limit)],
                'high': [last_price * (1 + random.uniform(0, VOLATILITY)) for _ in range(limit)],
                'low': [last_price * (1 - random.uniform(0, VOLATILITY)) for _ in range(limit)],
                'volume': [random.randint(1000, 10000) for _ in range(limit)],
                'timestamp': [pd.Timestamp.now() - pd.Timedelta(minutes=i) for i in range(limit-1, -1, -1)]
            })
            # Ensure prices are within realistic range ($190-$210)
            bars['close'] = bars['close'].clip(190, 210)
            bars['open'] = bars['open'].clip(190, 210)
            bars['high'] = bars['high'].clip(190, 210)
            bars['low'] = bars['low'].clip(190, 210)
            logging.info(f"Simulated {limit} bars for {symbol}, price: ${bars.iloc[-1]['close']:.2f}")
            file_handler.flush()
            return bars
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                limit=limit
            )
            bars = data_client.get_stock_bars(request)  # Remove 'await' as it's sync
            df = bars.df if TRADE_MODE == 'paper' else bars[symbol].df  # Handle structure differences
            if len(df) < limit:
                logging.warning(f"Fetched only {len(df)} bars for {symbol} (less than requested {limit})")
            logging.info(f"Fetched {len(df)} bars for {symbol}, latest price: ${df.iloc[-1]['close']:.2f}")
            file_handler.flush()
            return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        file_handler.flush()
        return pd.DataFrame()

async def fetch_daily_data(data_client: StockHistoricalDataClient, symbol: str, num_days: int, file_handler) -> pd.DataFrame:
    """
    Fetch daily bar data for longer-term context.

    :param data_client: Alpaca data client.
    :param symbol: Stock symbol.
    :param num_days: Number of daily bars to fetch.
    :param file_handler: Logging file handler.
    :return: DataFrame with daily bar data.
    """
    try:
        if TRADE_MODE == 'simulation':
            # Simulate daily data
            last_price = 200.0
            daily_bars = pd.DataFrame({
                'close': [last_price * (1 + random.uniform(-VOLATILITY * 10, VOLATILITY * 10)) for _ in range(num_days)],
                'volume': [random.randint(100000, 1000000) for _ in range(num_days)],
                'timestamp': [pd.Timestamp.now() - pd.Timedelta(days=i) for i in range(num_days-1, -1, -1)]
            })
            daily_bars['close'] = daily_bars['close'].clip(180, 220)
            logging.info(f"Simulated {num_days} daily bars for {symbol}")
            file_handler.flush()
            return daily_bars
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                limit=num_days
            )
            bars = data_client.get_stock_bars(request)
            df = bars.df if TRADE_MODE == 'paper' else bars[symbol].df
            if len(df) < num_days:
                logging.warning(f"Fetched only {len(df)} daily bars for {symbol} (less than requested {num_days})")
            logging.info(f"Fetched {len(df)} daily bars for {symbol}")
            file_handler.flush()
            return df
    except Exception as e:
        logging.error(f"Error fetching daily data for {symbol}: {e}")
        file_handler.flush()
        return pd.DataFrame()

def prepare_grok4_input(bars: pd.DataFrame, daily_bars: pd.DataFrame, sequence_length: int) -> dict:
    """
    Prepare input data for Grok-4 prediction, including daily context.

    :param bars: DataFrame with minute bar data.
    :param daily_bars: DataFrame with daily bar data.
    :param sequence_length: Number of bars for input sequence.
    :return: Dict with processed stats or None if invalid.
    """
    try:
        if len(bars) == 0:
            logging.warning(f"No minute bars available for {sequence_length} sequence length")
            return None
        # Use available bars even if fewer than sequence_length
        actual_length = min(len(bars), sequence_length)
        if actual_length < sequence_length:
            logging.warning(f"Using {actual_length} bars for {sequence_length} sequence length")
        stats = {
            'close_prices': bars['close'].tail(actual_length).tolist(),
            'volume': bars['volume'].tail(actual_length).tolist(),
            'returns': bars['close'].pct_change().tail(actual_length).tolist(),
            'volatility': bars['close'].pct_change().tail(actual_length).std() if len(bars) > 1 else 0.05,
            'market_corr': 0.80,  # Placeholder, adjust as needed
            'supports': [210.03, 207.22, 199.26],  # From historical_analysis_AAPL.json
            'resistances': [213.48, 216.23, 214.65, 213.34]
        }
        if not daily_bars.empty:
            stats['daily_closes'] = daily_bars['close'].tolist()
            stats['daily_returns'] = daily_bars['close'].pct_change().tolist()
            stats['daily_volatility'] = daily_bars['close'].pct_change().std() if len(daily_bars) > 1 else 0.05
        return stats
    except Exception as e:
        logging.error(f"Error preparing Grok-4 input: {e}")
        return None