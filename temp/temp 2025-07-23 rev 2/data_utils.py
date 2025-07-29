# data_utils.py
import pandas as pd
import numpy as np
import random
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from config import TRADE_MODE, ALPACA_API_KEY, ALPACA_API_SECRET

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
            volatility = 0.10  # Changed from 0.15
            last_price = 200.0  # Starting price, adjust based on historical_analysis_AAPL.json
            bars = pd.DataFrame({
                'close': [last_price * (1 + random.uniform(-volatility, volatility)) for _ in range(limit)],
                'open': [last_price * (1 + random.uniform(-volatility, volatility)) for _ in range(limit)],
                'high': [last_price * (1 + random.uniform(0, volatility)) for _ in range(limit)],
                'low': [last_price * (1 - random.uniform(0, volatility)) for _ in range(limit)],
                'volume': [random.randint(1000, 10000) for _ in range(limit)],
                'timestamp': [pd.Timestamp.now() - pd.Timedelta(minutes=i) for i in range(limit-1, -1, -1)]
            })
            # Ensure prices are within realistic range ($190-$210)
            bars['close'] = bars['close'].clip(190, 210)
            bars['open'] = bars['open'].clip(190, 210)
            bars['high'] = bars['high'].clip(190, 210)
            bars['low'] = bars['low'].clip(190, 210)
            logging.info(f"Simulated {limit} bars for {symbol}, price: ${bars.iloc[-1]['close']:.2f}")  # Shortened as requested
            file_handler.flush()
            return bars
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                limit=limit
            )
            bars = await data_client.get_stock_bars(request)
            df = bars[symbol].df
            logging.info(f"Fetched {len(df)} bars for {symbol}, latest price: ${df.iloc[-1]['close']:.2f}")
            file_handler.flush()
            return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        file_handler.flush()
        return pd.DataFrame()

def prepare_grok4_input(bars: pd.DataFrame, sequence_length: int) -> dict:
    """
    Prepare input data for Grok-4 prediction.

    :param bars: DataFrame with bar data.
    :param sequence_length: Number of bars for input sequence.
    :return: Dict with processed stats or None if invalid.
    """
    try:
        if len(bars) < sequence_length:
            logging.warning(f"Insufficient bars for sequence length {sequence_length}")
            return None
        stats = {
            'close_prices': bars['close'].tail(sequence_length).tolist(),
            'volume': bars['volume'].tail(sequence_length).tolist(),
            'returns': bars['close'].pct_change().tail(sequence_length).tolist(),
            'volatility': bars['close'].pct_change().tail(sequence_length).std(),
            'market_corr': 0.80,  # Placeholder, adjust as needed
            'supports': [210.03, 207.22, 199.26],  # From historical_analysis_AAPL.json
            'resistances': [213.48, 216.23, 214.65, 213.34]
        }
        return stats
    except Exception as e:
        logging.error(f"Error preparing Grok-4 input: {e}")
        return None