# data_utils.py
import logging
import numpy
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
import time
from config import (
    SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    NUM_DAYS_HISTORY,
    VOLATILITY,
    TRADE_MODE,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ATR_PERIOD,
    RSI_PERIOD,
    SMA_PERIOD,
    SUPPORTS,
    RESISTANCES,
    BASE_PRICES
)
from indicators import calculate_atr, calculate_rsi, calculate_sma

# Set up logging
logger = logging.getLogger(__name__)

# Cache for bar data
_bar_cache = {}

def fetch_bar_data(data_client: StockHistoricalDataClient, symbol: str, timeframe: TimeFrame, limit: int, file_handler, retries: int = 5, retry_delay: float = 1.0) -> tuple[pd.DataFrame, float]:
    """
    Fetch minute bar data for a given symbol with retry mechanism.
    Returns a tuple of (DataFrame with OHLCV data, latest price), using cache for recent data.
    """
    ny_tz = ZoneInfo("America/New_York")
    cache_key = f"{symbol}_{timeframe.value}_{limit}"
    current_time = datetime.now(ny_tz)

    if cache_key in _bar_cache:
        cached_data, cache_time = _bar_cache[cache_key]
        if isinstance(cache_time, datetime) and (current_time - cache_time).total_seconds() < 60:
            latest_price = cached_data['close'].iloc[-1]
            logger.debug(f"Using cached bar data for {symbol}, latest price: ${latest_price:.2f}")
            file_handler.flush()
            return cached_data, latest_price
        else:
            logger.debug(f"Cache expired or invalid for {symbol}, fetching new data")

    for attempt in range(retries):
        try:
            end_time = current_time
            # Dynamic window: extend further near market close or if fewer bars
            window_minutes = limit + 60 if attempt < retries - 1 else limit * 4
            start_time = end_time - timedelta(minutes=window_minutes)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time,
                limit=limit,
                feed=DataFeed.IEX
            )
            bars = data_client.get_stock_bars(request_params).df
            logger.debug(f"Attempt {attempt + 1}: Fetched {len(bars)} bars for {symbol}, timestamp range: {bars.index[0] if not bars.empty else 'empty'} to {bars.index[-1] if not bars.empty else 'empty'}")
            if bars.empty:
                logger.warning(f"Fetched 0 bars for {symbol} (requested {limit})")
                if attempt == retries - 1:
                    bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                    latest_price = bars['close'].iloc[-1]
                    logger.info(f"Using mock data for {symbol} after {retries} failed attempts, mock price: ${latest_price:.2f}")
                    file_handler.flush()
                    return bars, latest_price
                time.sleep(retry_delay)
                continue
            elif len(bars) < MIN_SEQUENCE_LENGTH:
                logger.warning(f"Fetched only {len(bars)} bars for {symbol} (minimum {MIN_SEQUENCE_LENGTH}), attempt {attempt + 1}")
                if attempt == retries - 1:
                    bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                    latest_price = bars['close'].iloc[-1]
                    logger.info(f"Using mock data for {symbol} after {retries} failed attempts, mock price: ${latest_price:.2f}")
                    file_handler.flush()
                    return bars, latest_price
                time.sleep(retry_delay)
                continue

            latest_price = bars['close'].iloc[-1]
            logger.info(f"Fetched {len(bars)} minute bars for {symbol}, latest price: ${latest_price:.2f} (source: historical bars)")

            if isinstance(bars.index, pd.DatetimeIndex):
                logger.debug(f"Index is already DatetimeIndex for {symbol}, no reset needed")
                bars.index.name = 'timestamp'
            else:
                if isinstance(bars.index, pd.MultiIndex):
                    logger.debug(f"Multi-index detected for {symbol}, resetting to single timestamp index")
                    bars = bars.reset_index()
                    if 'timestamp' not in bars.columns:
                        logger.error(f"No 'timestamp' column after resetting multi-index for {symbol}: {bars.columns}, sample: {bars.head(1).to_dict()}")
                        if attempt == retries - 1:
                            bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                            latest_price = bars['close'].iloc[-1]
                            logger.info(f"Using mock data for {symbol} after {retries} failed attempts, mock price: ${latest_price:.2f}")
                            file_handler.flush()
                            return bars, latest_price
                        time.sleep(retry_delay)
                        continue
                    bars['timestamp'] = pd.to_datetime(bars['timestamp'], errors='coerce', utc=True).dt.tz_convert(ny_tz)
                    bars = bars.drop(columns=['symbol'] if 'symbol' in bars.columns else [])
                    bars.set_index('timestamp', inplace=True)
                    if bars.index.hasnans:
                        logger.error(f"Invalid timestamps in bars for {symbol}, falling back to mock data")
                        if attempt == retries - 1:
                            bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                            latest_price = bars['close'].iloc[-1]
                            logger.info(f"Using mock data for {symbol} after {retries} failed attempts, mock price: ${latest_price:.2f}")
                            file_handler.flush()
                            return bars, latest_price
                        time.sleep(retry_delay)
                        continue
                else:
                    logger.error(f"Unexpected index type for {symbol}: {type(bars.index)}, columns: {bars.columns}, sample: {bars.head(1).to_dict()}")
                    if attempt == retries - 1:
                        bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                        latest_price = bars['close'].iloc[-1]
                        logger.info(f"Using mock data for {symbol} after {retries} failed attempts, mock price: ${latest_price:.2f}")
                        file_handler.flush()
                        return bars, latest_price
                    time.sleep(retry_delay)
                    continue

            latest_timestamp = pd.Timestamp(bars.index[-1]).tz_convert(ny_tz).to_pydatetime()
            if (current_time - latest_timestamp).total_seconds() > 3600:  # Relaxed to 1 hour
                logger.warning(f"Stale data for {symbol}: latest timestamp {latest_timestamp} too old")
                if attempt == retries - 1:
                    bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                    latest_price = bars['close'].iloc[-1]
                    logger.info(f"Using mock data for {symbol} after {retries} failed attempts, mock price: ${latest_price:.2f}")
                    file_handler.flush()
                    return bars, latest_price
                time.sleep(retry_delay)
                continue

            _bar_cache[cache_key] = (bars, current_time)
            file_handler.flush()
            return bars, latest_price
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                logger.error(f"Max retries reached for {symbol}, using mock data")
                bars = fetch_mock_bar_data(symbol, limit, file_handler, pd.DataFrame())
                latest_price = bars['close'].iloc[-1]
                logger.info(f"Using mock data for {symbol}, mock price: ${latest_price:.2f}")
                file_handler.flush()
                return bars, latest_price
            time.sleep(retry_delay)

def fetch_mock_bar_data(symbol: str, limit: int, file_handler, daily_bars: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """Generate mock minute bar data with realistic trend when live data is unavailable."""
    ny_tz = ZoneInfo("America/New_York")
    timestamps = [datetime.now(ny_tz) - timedelta(minutes=i) for i in range(limit)]
    base_price = BASE_PRICES.get(symbol, 100.0)
    if not daily_bars.empty and 'close' in daily_bars.columns:
        base_price = daily_bars['close'].iloc[-1]
        volatility = daily_bars['close'].pct_change().std() * numpy.sqrt(252) if len(daily_bars) > 1 else VOLATILITY
    else:
        volatility = VOLATILITY
    returns = numpy.random.normal(0, volatility / numpy.sqrt(390), limit)
    prices = base_price * numpy.exp(numpy.cumsum(returns))
    mock_data = {
        'open': prices,
        'high': [p + numpy.random.normal(0.5, volatility) for p in prices],
        'low': [p - numpy.random.normal(0.5, volatility) for p in prices],
        'close': prices,
        'volume': [numpy.random.randint(1000, 10000) for _ in range(limit)],
        'timestamp': timestamps
    }
    bars = pd.DataFrame(mock_data)
    bars['high'] = bars[['open', 'high', 'close']].max(axis=1)
    bars['low'] = bars[['open', 'low', 'close']].min(axis=1)
    bars.set_index('timestamp', inplace=True)
    logger.info(f"Generated {len(bars)} mock bars for {symbol}, latest price: ${bars['close'].iloc[-1]:.2f}")
    file_handler.flush()
    return bars

def fetch_mock_daily_data(symbol: str, limit: int, file_handler) -> pd.DataFrame:
    """Generate mock daily bar data with realistic trend when live data is unavailable."""
    ny_tz = ZoneInfo("America/New_York")
    timestamps = [datetime.now(ny_tz) - timedelta(days=i) for i in range(limit)]
    base_price = BASE_PRICES.get(symbol, 100.0)
    returns = numpy.random.normal(0, VOLATILITY * 2, limit)
    prices = base_price * numpy.exp(numpy.cumsum(returns))
    mock_data = {
        'open': prices,
        'high': [p + numpy.random.normal(1.0, VOLATILITY * 2) for p in prices],
        'low': [p - numpy.random.normal(1.0, VOLATILITY * 2) for p in prices],
        'close': prices,
        'volume': [numpy.random.randint(10000, 100000) for _ in range(limit)],
        'timestamp': timestamps
    }
    daily_bars = pd.DataFrame(mock_data)
    daily_bars['high'] = daily_bars[['open', 'high', 'close']].max(axis=1)
    daily_bars['low'] = daily_bars[['open', 'low', 'close']].min(axis=1)
    daily_bars.set_index('timestamp', inplace=True)
    logger.info(f"Generated {len(daily_bars)} mock daily bars for {symbol}, latest price: ${daily_bars['close'].iloc[-1]:.2f}")
    file_handler.flush()
    return daily_bars

def fetch_daily_data(data_client: StockHistoricalDataClient, symbol: str, limit: int, file_handler) -> pd.DataFrame:
    """
    Fetch daily bar data for a given symbol.
    Returns a DataFrame with daily OHLCV data.
    """
    ny_tz = ZoneInfo("America/New_York")
    try:
        end_time = datetime.now(ny_tz)
        start_time = end_time - timedelta(days=limit + 10)  # Extended window for daily bars
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_time,
            end=end_time,
            limit=limit,
            feed=DataFeed.IEX
        )
        daily_bars = data_client.get_stock_bars(request_params).df
        logger.debug(f"Fetched {len(daily_bars)} daily bars for {symbol}")
        if daily_bars.empty:
            logger.warning(f"Fetched 0 daily bars for {symbol} (requested {limit})")
            daily_bars = fetch_mock_daily_data(symbol, limit, file_handler)
        elif len(daily_bars) < limit:
            logger.warning(f"Fetched only {len(daily_bars)} daily bars for {symbol} (requested {limit})")
        if isinstance(daily_bars.index, pd.MultiIndex):
            daily_bars = daily_bars.reset_index()
            if 'timestamp' in daily_bars.columns:
                daily_bars['timestamp'] = pd.to_datetime(daily_bars['timestamp'], errors='coerce', utc=True).dt.tz_convert(ny_tz)
                daily_bars = daily_bars.drop(columns=['symbol'] if 'symbol' in daily_bars.columns else [])
                daily_bars.set_index('timestamp', inplace=True)
        logger.info(f"Fetched {len(daily_bars)} daily bars for {symbol}")
        file_handler.flush()
        return daily_bars
    except Exception as e:
        logger.error(f"Error fetching daily data for {symbol}: {e}")
        daily_bars = fetch_mock_daily_data(symbol, limit, file_handler)
        file_handler.flush()
        return daily_bars

def prepare_grok4_input(bars: pd.DataFrame, daily_bars: pd.DataFrame, sequence_length: int, symbol: str, file_handler) -> dict:
    """Prepare statistical inputs for Grok-4 prediction model from bar data."""
    logger.debug(f"Input bars for {symbol}: length={len(bars)}, columns={bars.columns}, sample={bars.head(1).to_dict()}")
    logger.debug(f"Input daily_bars for {symbol}: length={len(daily_bars)}, columns={daily_bars.columns}, sample={daily_bars.head(1).to_dict()}")

    if len(bars) < MIN_SEQUENCE_LENGTH:
        logger.warning(f"Insufficient bars ({len(bars)} < {MIN_SEQUENCE_LENGTH}) for {symbol}; using mock data")
        bars = fetch_mock_bar_data(symbol, sequence_length, file_handler, daily_bars)
        latest_price = bars['close'].iloc[-1]
    elif len(bars) < sequence_length:
        logger.warning(f"Using {len(bars)} bars for {symbol} (less than ideal {sequence_length}, but sufficient >= {MIN_SEQUENCE_LENGTH})")
        latest_price = bars['close'].iloc[-1]
    else:
        latest_price = bars['close'].iloc[-1]

    stats = {}
    closes = bars['close'].values
    stats['close_prices'] = closes
    stats['volume'] = bars['volume'].values
    stats['returns'] = bars['close'].pct_change().dropna().values
    stats['volatility'] = numpy.std(stats['returns']) * numpy.sqrt(252 * 390) if len(stats['returns']) > 0 else 0.0
    stats['mean_return'] = numpy.mean(stats['returns']) if len(stats['returns']) > 0 else 0.0
    stats['last_close'] = latest_price
    stats['atr'] = calculate_atr(bars, min(ATR_PERIOD, len(bars) - 1), symbol) if len(bars) > 1 else 0.0
    stats['rsi'] = calculate_rsi(closes, min(RSI_PERIOD, len(closes) - 1)) if len(closes) > 1 else 50.0
    stats['sma'] = calculate_sma(closes, min(SMA_PERIOD, len(closes) - 1)) if len(closes) > 1 else closes[-1]
    stats['supports'] = SUPPORTS.get(symbol, [])
    stats['resistances'] = RESISTANCES.get(symbol, [])
    if not daily_bars.empty:
        stats['daily_closes'] = daily_bars['close'].values
        stats['daily_volatility'] = daily_bars['close'].pct_change().std() * numpy.sqrt(252) if len(daily_bars) > 1 else 0.0
        stats['daily_mean_return'] = daily_bars['close'].pct_change().mean() if len(daily_bars) > 1 else 0.0
    else:
        stats['daily_closes'] = []
        stats['daily_volatility'] = 0.0
        stats['daily_mean_return'] = 0.0
    logger.debug(f"Prepared stats for {symbol}: {stats}")
    file_handler.flush()
    return stats