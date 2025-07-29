# backtest.py
import asyncio
import logging
import random
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from trading_loop import trading_logic
from trade_analysis import analyze_trades
from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    SYMBOLS,
    SEQUENCE_LENGTH,
    NUM_DAYS_HISTORY,
    TRADE_MODE,
    BACKTEST_PERIOD_DAYS,
    GRID_SEARCH,
    UPPER_THRESHOLDS,
    LOWER_THRESHOLDS,
    UPPER_THRESHOLD,
    LOWER_THRESHOLD,
    LOG_LEVEL
)
from logging_utils import setup_logging
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from order_execution import poll_order_fill, update_position

class MockOrder:
    def __init__(self, id, symbol, qty, side, filled_qty, filled_avg_price):
        self.id = id
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.status = "FILLED" if filled_qty > 0 else "PENDING"

class MockTradingClient:
    def __init__(self):
        self.positions = {symbol: {'qty': 0, 'avg_entry_price': 0.0} for symbol in SYMBOLS}
        self.equity = 100000.0
        self.clock = MockClock(is_open=True)
        self.orders = []

    def get_open_position(self, symbol):
        pos = self.positions.get(symbol, {'qty': 0, 'avg_entry_price': 0.0})
        return MockPosition(pos['qty'], pos['avg_entry_price'])

    def get_account(self):
        return MockAccount(self.equity)

    def get_clock(self):
        return self.clock

    async def submit_order(self, market_order: MarketOrderRequest):
        order_id = len(self.orders) + 1
        filled_qty = market_order.qty
        filled_price = market_order.current_price if hasattr(market_order, 'current_price') else random.uniform(190, 220)
        side = 'buy' if market_order.side == OrderSide.BUY else 'sell'
        order = MockOrder(order_id, market_order.symbol, filled_qty, side, filled_qty, filled_price)
        self.orders.append(order)
        await update_position(self, market_order.symbol, side, 'long' if side == 'buy' else 'close long', filled_qty, filled_price, logging.getLogger(), 0.0, 0.0)
        return order

    def get_order_by_id(self, order_id):
        for order in self.orders:
            if order.id == order_id:
                return order
        return None

class MockPosition:
    def __init__(self, qty, avg_entry_price=0.0):
        self.qty = qty
        self.avg_entry_price = avg_entry_price

class MockAccount:
    def __init__(self, equity):
        self.equity = equity

class MockClock:
    def __init__(self, is_open):
        self.is_open = is_open

async def process_symbol(data_client, trading_client, symbol, start_date, end_date, log_file, file_handler, last_trade_time, upper_threshold, lower_threshold):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date,
        feed=DataFeed.IEX
    )
    bars_df = data_client.get_stock_bars(request_params).df
    if bars_df.empty:
        logging.warning(f"No historical data for {symbol}")
        return
    bars = bars_df.loc[symbol] if isinstance(bars_df.index, pd.MultiIndex) else bars_df

    daily_request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        feed=DataFeed.IEX
    )
    daily_bars_df = data_client.get_stock_bars(daily_request_params).df
    if daily_bars_df.empty:
        logging.warning(f"No historical daily data for {symbol}")
        return
    daily_bars = daily_bars_df.loc[symbol] if isinstance(daily_bars_df.index, pd.MultiIndex) else daily_bars_df

    for i in range(SEQUENCE_LENGTH, len(bars)):
        timestamp = bars.index[i]
        past_bars = bars.iloc[i-SEQUENCE_LENGTH:i]
        current_price = bars.iloc[i]['close']
        past_daily_bars = daily_bars[daily_bars.index <= timestamp].tail(NUM_DAYS_HISTORY)

        await trading_logic(
            trading_client,
            data_client,
            symbol,
            current_price,
            log_file,
            file_handler,
            last_trade_time,
            current_time=timestamp,
            historical_bars=past_bars,
            historical_daily_bars=past_daily_bars,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )
        await asyncio.sleep(0.1)  # Improved Ctrl+C handling

async def single_backtest(symbols, start_date, end_date, upper_threshold, lower_threshold, run_id):
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    trading_client = MockTradingClient()
    log_file = f'backtest_log_{run_id}.log' if GRID_SEARCH else 'backtest_log.log'
    logger, file_handler = setup_logging(filename=log_file, level=LOG_LEVEL)

    # Ensure log file exists
    try:
        with open(log_file, 'w') as f:
            f.write(f"Backtest {run_id} started at {datetime.now(ZoneInfo('America/New_York'))}\n")
        file_handler.flush()
    except Exception as e:
        logging.error(f"Failed to create log file {log_file}: {e}")
        return {'pl': 0.0, 'win_rate': 0.0, 'completed_trades': 0, 'total_trades': 0, 'equity': 100000.0, 'drawdown': 0.0}, log_file

    logging.info(f"Starting backtest {run_id} for symbols {symbols} from {start_date} to {end_date} with upper_threshold={upper_threshold}, lower_threshold={lower_threshold}")

    last_trade_time = {symbol: None for symbol in symbols}
    tasks = [process_symbol(data_client, trading_client, symbol, start_date, end_date, log_file, file_handler, last_trade_time, upper_threshold, lower_threshold) for symbol in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)
    file_handler.flush()  # Ensure all logs are written

    try:
        summary = analyze_trades(log_file, trading_client)
    except FileNotFoundError:
        logging.error(f"Log file {log_file} not found during analysis, returning default summary")
        summary = {'pl': 0.0, 'win_rate': 0.0, 'completed_trades': 0, 'total_trades': 0, 'equity': 100000.0, 'drawdown': 0.0}
    logging.info(f"Backtest {run_id} summary: Total P/L: ${summary['pl']:.2f}, Win Rate: {summary['win_rate']:.2%}, Trades: {summary['completed_trades']}")
    return summary, log_file

async def grid_search_backtest(symbols=SYMBOLS, start_date=None, end_date=None):
    if not start_date:
        start_date = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=BACKTEST_PERIOD_DAYS)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')

    results = []
    run_id = 1
    for upper in UPPER_THRESHOLDS:
        for lower in LOWER_THRESHOLDS:
            if upper > lower:  # Valid threshold pair
                summary, log_file = await single_backtest(symbols, start_date, end_date, upper, lower, run_id)
                results.append({
                    'run_id': run_id,
                    'upper_threshold': upper,
                    'lower_threshold': lower,
                    'pl': summary['pl'],
                    'win_rate': summary['win_rate'],
                    'trades': summary['completed_trades'],
                    'log_file': log_file
                })
                run_id += 1

    results_df = pd.DataFrame(results)
    results_df.to_csv('grid_search_results.csv', index=False)
    if not results_df.empty:
        best_result = results_df.loc[results_df['win_rate'].idxmax()]
        logging.info(f"Best grid search result: Run {best_result['run_id']}, Upper: {best_result['upper_threshold']}, Lower: {best_result['lower_threshold']}, P/L: ${best_result['pl']:.2f}, Win Rate: {best_result['win_rate']:.2%}, Trades: {best_result['trades']}")
    else:
        logging.warning("No valid grid search results")
    return results_df

async def backtest(symbols=SYMBOLS, start_date=None, end_date=None):
    """
    Backtest the trading bot on historical data for given symbols.
    """
    if GRID_SEARCH:
        return await grid_search_backtest(symbols, start_date, end_date)
    else:
        summary, _ = await single_backtest(symbols, start_date, end_date, UPPER_THRESHOLD, LOWER_THRESHOLD, 1)
        return summary