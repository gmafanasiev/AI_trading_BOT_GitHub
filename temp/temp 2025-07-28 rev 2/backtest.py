# backtest.py
import asyncio
import logging
import random  # Added for mock filled_price if needed; but using current_price
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from trading_loop import trading_logic
from trade_analysis import analyze_trades
from config import ALPACA_API_KEY, ALPACA_API_SECRET, SYMBOLS, SEQUENCE_LENGTH, NUM_DAYS_HISTORY, TRADE_MODE
from logging_utils import setup_logging  # Import to setup logging
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from order_execution import poll_order_fill, update_position  # Import to use in simulation

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
        self.orders = []  # To track submitted orders

    def get_open_position(self, symbol):
        pos = self.positions.get(symbol, {'qty': 0, 'avg_entry_price': 0.0})
        return MockPosition(pos['qty'], pos['avg_entry_price'])

    def get_account(self):
        return MockAccount(self.equity)

    def get_clock(self):
        return self.clock

    def submit_order(self, market_order: MarketOrderRequest):
        # Simulate immediate fill for backtest
        order_id = len(self.orders) + 1
        filled_qty = market_order.qty
        filled_price = random.uniform(190, 220)  # Mock price; in real backtest, use historical
        side = 'buy' if market_order.side == OrderSide.BUY else 'sell'
        order = MockOrder(order_id, market_order.symbol, filled_qty, side, filled_qty, filled_price)
        self.orders.append(order)
        # Update position immediately
        asyncio.run(update_position(self, market_order.symbol, side, 'long' if side == 'buy' else 'close long', filled_qty, filled_price, logging.getLogger(), 0.0, 0.0))
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

async def backtest(symbols=SYMBOLS, start_date=None, end_date=None):
    """
    Backtest the trading bot on historical data for given symbols.
    """
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    trading_client = MockTradingClient()  # Mock for simulation
    log_file = 'backtest_log.log'
    logger, file_handler = setup_logging(level=logging.INFO)  # Setup logging here, no 'file' arg

    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year back
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logging.info(f"Starting backtest for symbols {symbols} from {start_date} to {end_date}")

    last_trade_time = {symbol: None for symbol in symbols}
    for symbol in symbols:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX  # Free tier
        )
        bars = data_client.get_stock_bars(request_params).df
        if bars.empty:
            logging.warning(f"No historical data for {symbol}")
            continue

        for timestamp, bar in bars.iterrows():
            current_price = bar['close']
            await trading_logic(trading_client, data_client, symbol, current_price, log_file, file_handler, last_trade_time)

    summary = analyze_trades(log_file, trading_client)
    logging.info(f"Backtest summary: Total P/L: ${summary['pl']:.2f}, Win Rate: {summary['win_rate']:.2%}, Trades: {summary['completed_trades']}")
    return summary