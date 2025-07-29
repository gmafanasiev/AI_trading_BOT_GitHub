# trading_loop.py
import asyncio
import logging
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.live import StockDataStream
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from data_utils import fetch_bar_data, prepare_grok4_input
from prediction import get_grok4_prediction_and_adjustments
from order_execution import get_position, execute_order
from trade_analysis import analyze_trades
from config import (
    SYMBOLS,
    TRADE_MODE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    open_positions,
    sl_prices,
    tp_prices,
    SEQUENCE_LENGTH,
    UPPER_THRESHOLD,
    LOWER_THRESHOLD,
    ADJUSTMENT_INTERVAL,
    MAX_POSITION_PCT,
    MAX_EQUITY,
    TRAILING_PCT,
    TIMEFRAME,
    URL,
    XAI_API_KEY,
    COOLDOWN_SECONDS,
    SESSION_DURATION,
    ALPACA_API_KEY,
    ALPACA_API_SECRET
)

"""
Main trading loop for AI-powered day trading with Grok-3 and Alpaca.
"""

async def is_cooldown_active(symbol: str, last_trade_time: dict) -> bool:
    """
    Check if a cooldown is active for the given symbol.

    :param symbol: Stock symbol.
    :param last_trade_time: Dictionary tracking last trade time for each symbol.
    :return: True if cooldown is active, False otherwise.
    """
    if symbol not in last_trade_time or last_trade_time[symbol] is None:
        return False
    time_since_last_trade = (datetime.now(ZoneInfo("America/New_York")) - last_trade_time[symbol]).total_seconds()
    is_active = time_since_last_trade < COOLDOWN_SECONDS
    if is_active:
        remaining_seconds = COOLDOWN_SECONDS - time_since_last_trade
        logging.info(f"Cooldown active for {symbol}, skipping trade (remaining: {remaining_seconds:.1f} seconds)")
    return is_active

async def trading_logic(trading_client: TradingClient, data_client: StockHistoricalDataClient, symbol: str, current_price: float, log_file_path: str, file_handler, last_trade_time: dict):
    """
    Trading logic for a single symbol.

    :param trading_client: Alpaca trading client.
    :param data_client: Alpaca data client.
    :param symbol: Stock symbol.
    :param current_price: Current price of the symbol.
    :param log_file_path: Path to log file.
    :param file_handler: Logging file handler.
    :param last_trade_time: Dictionary tracking last trade time for each symbol.
    """
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Fetched'} 60 bars for {symbol}, price: ${current_price:.2f}")
    file_handler.flush()

    position = await get_position(trading_client, symbol)
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Current'} position for {symbol}: {position} shares")
    file_handler.flush()

    trade_summary = analyze_trades(log_file_path)
    
    max_position_value = MAX_EQUITY * MAX_POSITION_PCT
    qty = min(int((MAX_EQUITY * 0.01) / current_price), int(max_position_value / current_price))  # Use default risk; adjust if needed later

    if position != 0:
        if position > 0:  # Long position
            if current_price <= sl_prices[symbol]:
                logging.info(f"Trailing stop-loss hit for {symbol} long: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, position, "sell", "close long", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
            elif current_price >= tp_prices[symbol]:
                logging.info(f"Take-profit hit for {symbol} long: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, position, "sell", "close long", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
            else:
                sl_prices[symbol] = max(sl_prices[symbol], current_price * (1 - TRAILING_PCT))
                tp_prices[symbol] = min(tp_prices[symbol], current_price * (1 + TRAILING_PCT))
                logging.info(f"Trailing SL updated for {symbol} long to ${sl_prices[symbol]:.2f}")
                logging.info(f"Trailing TP updated for {symbol} long to ${tp_prices[symbol]:.2f}")
        elif position < 0:  # Short position
            if current_price >= sl_prices[symbol]:
                logging.info(f"Trailing stop-loss hit for {symbol} short: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), "buy", "close short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
            elif current_price <= tp_prices[symbol]:
                logging.info(f"Take-profit hit for {symbol} short: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), "buy", "close short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
            else:
                sl_prices[symbol] = min(sl_prices[symbol], current_price * (1 + TRAILING_PCT))
                tp_prices[symbol] = max(tp_prices[symbol], current_price * (1 - TRAILING_PCT))
                logging.info(f"Trailing SL updated for {symbol} short to ${sl_prices[symbol]:.2f}")
                logging.info(f"Trailing TP updated for {symbol} short to ${tp_prices[symbol]:.2f}")
        file_handler.flush()
    
    if position == 0:
        if await is_cooldown_active(symbol, last_trade_time):
            file_handler.flush()
            return
        
        timeframe = TimeFrame.Minute  # Assuming TIMEFRAME == "Minute"; adjust if needed for other timeframes
        bars = await fetch_bar_data(data_client, symbol, timeframe, SEQUENCE_LENGTH, file_handler)
        stats = prepare_grok4_input(bars, SEQUENCE_LENGTH)
        if stats is None:
            file_handler.flush()
            return
        
        prediction, threshold, risk_per_trade = await get_grok4_prediction_and_adjustments(
            URL, XAI_API_KEY, stats, symbol, trade_summary
        )
        logging.info(f"{symbol} - Current Price: ${current_price:.2f}, Grok 3 Prediction: {prediction:.2f}, Position: {position}")
        
        qty = min(int((MAX_EQUITY * risk_per_trade) / current_price), int(max_position_value / current_price))  # Update qty with adjusted risk
        
        if prediction >= threshold:
            if qty > 0:
                await execute_order(trading_client, symbol, qty, "buy", "long", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
        elif prediction <= (1 - threshold):
            if qty > 0:
                await execute_order(trading_client, symbol, qty, "sell", "short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))

    trade_summary = analyze_trades(log_file_path)
    logging.info(f"Parsed {trade_summary['total_trades']} trades, {trade_summary['completed_trades']} completed, calculated total P/L: ${trade_summary['pl']:.2f}, win rate: {trade_summary['win_rate']:.2f}")
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Account'} Equity: ${trade_summary['equity']:.2f}")
    logging.info(f"Saved equity plot to equity_plot.png")
    file_handler.flush()

async def trading_loop(trading_client: TradingClient, data_client: StockHistoricalDataClient, log_file_path: str, file_handler):
    """
    Main trading loop for AI-powered day trading with Grok-3 and Alpaca.

    :param trading_client: Alpaca trading client.
    :param data_client: Alpaca data client.
    :param log_file_path: Path to log file.
    :param file_handler: Logging file handler.
    """
    stream = StockDataStream(ALPACA_API_KEY, ALPACA_API_SECRET)
    last_trade_time = {symbol: None for symbol in SYMBOLS}
    start_time = datetime.now(ZoneInfo("America/New_York"))
    ny_tz = ZoneInfo("America/New_York")
    logging.info(f"Starting trading session at {start_time.strftime('%Y-%m-%d %H:%M:%S')} with duration: {SESSION_DURATION}")

    async def handle_bar(bar):
        await trading_logic(trading_client, data_client, bar.symbol, bar.close, log_file_path, file_handler, last_trade_time)

    async def close_all_positions():
        """Close all open positions at market close."""
        for symbol in SYMBOLS:
            position = await get_position(trading_client, symbol)
            if position != 0:
                order_type = "sell" if position > 0 else "buy"
                trade_type = "close long" if position > 0 else "close short"
                if TRADE_MODE == 'simulation':
                    current_price = 190.0 + random.uniform(-20, 20)
                else:
                    latest_bar = data_client.get_stock_latest_bar(symbol)
                    current_price = latest_bar[symbol].close
                logging.info(f"Closing {symbol} position at market close: {position} shares at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), order_type, trade_type, current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))

    async def process_symbols():
        tasks = []
        for symbol in SYMBOLS:
            if TRADE_MODE == 'simulation':
                tasks.append(trading_logic(trading_client, data_client, symbol, 190.0 + random.uniform(-20, 20), log_file_path, file_handler, last_trade_time))
            else:
                await stream.subscribe_bars(handle_bar, symbol)
        if TRADE_MODE == 'simulation':
            await asyncio.gather(*tasks)

    async def run_trading_loop():
        for symbol in SYMBOLS:
            open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
            sl_prices[symbol] = 0.0
            tp_prices[symbol] = 0.0
        while True:
            # Parse SESSION_DURATION
            session_seconds = None
            if SESSION_DURATION == "end_of_day":
                now = datetime.now(ny_tz)
                market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                if now >= market_close:
                    market_close += timedelta(days=1)  # Next trading day
                session_seconds = (market_close - now).total_seconds()
            else:
                try:
                    parts = SESSION_DURATION.split()
                    if len(parts) != 2:
                        raise ValueError("SESSION_DURATION must be 'end_of_day' or '<number> <unit>'")
                    value, unit = parts
                    value = float(value)
                    if value <= 0:
                        raise ValueError("SESSION_DURATION value must be positive")
                    if unit == "minutes":
                        session_seconds = value * 60
                    elif unit == "seconds":
                        session_seconds = value
                    else:
                        raise ValueError("SESSION_DURATION unit must be 'seconds' or 'minutes'")
                except (ValueError, AttributeError) as e:
                    logging.error(f"Invalid SESSION_DURATION format: {SESSION_DURATION}, error: {str(e)}, defaulting to 3600 seconds")
                    session_seconds = 3600

            # Check if session duration exceeded
            if (datetime.now(ZoneInfo("America/New_York")) - start_time).total_seconds() >= session_seconds:
                logging.info("Session duration reached, closing positions and stopping trading loop")
                await close_all_positions()
                file_handler.flush()
                logging.info("Trading loop terminated")
                break

            # Check for market close
            now = datetime.now(ny_tz)
            if SESSION_DURATION == "end_of_day" and now.hour >= 16:
                logging.info("Market close reached, closing positions and stopping trading loop")
                await close_all_positions()
                file_handler.flush()
                logging.info("Trading loop terminated")
                break

            await process_symbols()
            logging.info("Starting sleep")
            for i in range(ADJUSTMENT_INTERVAL, 0, -10):
                logging.info(f"Sleep countdown: {i} seconds remaining")
                file_handler.flush()
                await asyncio.sleep(10)
            logging.info("Completed sleep")

    if TRADE_MODE != 'simulation':
        await stream.run()
    await run_trading_loop()