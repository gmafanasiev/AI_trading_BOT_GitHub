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
from alpaca.data.requests import StockLatestBarRequest
from data_utils import fetch_bar_data, fetch_daily_data, fetch_mock_bar_data, fetch_mock_daily_data, prepare_grok4_input
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
    MAX_DRAWDOWN,
    MAX_EQUITY,
    TRAILING_PCT,
    TIMEFRAME,
    URL,
    XAI_API_KEY,
    COOLDOWN_SECONDS,
    SESSION_DURATION,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    LONG_EXIT_THRESHOLD,
    SHORT_EXIT_THRESHOLD,
    RISK_PER_TRADE,
    NUM_DAYS_HISTORY,
    ATR_MULTIPLIER,
    RR_RATIO,
)

"""
Main trading loop for AI-powered day trading with Grok-3 and Alpaca.
"""

async def is_cooldown_active(symbol: str, last_trade_time: dict) -> bool:
    if symbol not in last_trade_time or last_trade_time[symbol] is None:
        return False
    time_since_last_trade = (datetime.now(ZoneInfo("America/New_York")) - last_trade_time[symbol]).total_seconds()
    is_active = time_since_last_trade < COOLDOWN_SECONDS
    if is_active:
        remaining_seconds = COOLDOWN_SECONDS - time_since_last_trade
        logging.info(f"Cooldown active for {symbol}, skipping trade (remaining: {remaining_seconds:.1f} seconds)")
    return is_active

async def trading_logic(trading_client: TradingClient, data_client: StockHistoricalDataClient, symbol: str, current_price: float, log_file_path: str, file_handler, last_trade_time: dict):
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Fetched'} {SEQUENCE_LENGTH} bars for {symbol}, price: ${current_price:.2f}")
    file_handler.flush()

    position = await get_position(trading_client, symbol)
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Current'} position for {symbol}: {position} shares")
    file_handler.flush()

    trade_summary = analyze_trades(log_file_path, trading_client)

    if trade_summary['drawdown'] < -MAX_DRAWDOWN:
        logging.info(f"Max drawdown exceeded ({trade_summary['drawdown']:.2%}), skipping trade for {symbol}")
        file_handler.flush()
        return

    # Check market hours
    use_mock_data = TRADE_MODE == 'simulation'
    if not use_mock_data:
        try:
            clock = trading_client.get_clock()
            use_mock_data = not clock.is_open
            if use_mock_data:
                logging.info(f"Market closed for {symbol}, using mock data")
        except Exception as e:
            logging.error(f"Error checking market hours for {symbol}: {e}, using mock data")
            use_mock_data = True

    # Fetch data for Grok prediction
    timeframe = TimeFrame.Minute
    if use_mock_data:
        bars = fetch_mock_bar_data(symbol, SEQUENCE_LENGTH, file_handler)  # Sync function, no await
        daily_bars = fetch_mock_daily_data(symbol, NUM_DAYS_HISTORY, file_handler)  # Sync function
        current_price = bars['close'].iloc[-1]  # Update current_price with mock data
    else:
        bars = await fetch_bar_data(data_client, symbol, timeframe, SEQUENCE_LENGTH, file_handler)  # Async function
        daily_bars = await fetch_daily_data(data_client, symbol, NUM_DAYS_HISTORY, file_handler)  # Async function
    stats = prepare_grok4_input(bars, daily_bars, SEQUENCE_LENGTH)
    logging.debug(f"Stats for {symbol}: {stats}")
    if stats is None:
        logging.warning(f"Using default Grok prediction for {symbol} due to insufficient data")
        file_handler.flush()
        prediction = 0.65  # Default to trigger trade for testing
        threshold = UPPER_THRESHOLD
        risk_per_trade = 0.01
    else:
        try:
            prediction, threshold, risk_per_trade = await get_grok4_prediction_and_adjustments(
                URL, XAI_API_KEY, stats, symbol, trade_summary
            )
        except Exception as e:
            logging.error(f"Failed to fetch Grok prediction for {symbol}: {e}, using default prediction")
            prediction = 0.65  # Adjusted for off-hours testing
            threshold = UPPER_THRESHOLD
            risk_per_trade = 0.01
    logging.info(f"{symbol} - Current Price: ${current_price:.2f}, Grok 3 Prediction: {prediction:.2f}, Position: {position}")

    # Dynamic stop loss and position sizing based on ATR
    atr = stats.get('atr', current_price * STOP_LOSS_PCT / ATR_MULTIPLIER) if stats else current_price * STOP_LOSS_PCT / ATR_MULTIPLIER
    sl_distance = atr * ATR_MULTIPLIER
    tp_distance = sl_distance * RR_RATIO
    max_position_value = trade_summary['equity'] * MAX_POSITION_PCT
    qty = min(int((trade_summary['equity'] * risk_per_trade) / sl_distance), int(max_position_value / current_price))
    logging.info(f"{symbol} - Calculated Quantity: {qty}, SL Distance: ${sl_distance:.2f}, TP Distance: ${tp_distance:.2f}")

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
            elif prediction < LONG_EXIT_THRESHOLD:  # Early exit if bearish
                logging.info(f"Bearish prediction ({prediction:.2f}) for long: Closing at ${current_price:.2f}")
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
            elif prediction > SHORT_EXIT_THRESHOLD:  # Early exit if bullish
                logging.info(f"Bullish prediction ({prediction:.2f}) for short: Closing at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), "buy", "close short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
            else:
                sl_prices[symbol] = min(sl_prices[symbol], current_price * (1 + TRAILING_PCT))
                tp_prices[symbol] = max(tp_prices[symbol], current_price * (1 - TRAILING_PCT))
                logging.info(f"Trailing SL updated for {symbol} short to ${sl_prices[symbol]:.2f}")
                logging.info(f"Trailing TP updated for {symbol} long to ${tp_prices[symbol]:.2f}")
        file_handler.flush()

    if position == 0:
        if await is_cooldown_active(symbol, last_trade_time):
            file_handler.flush()
            return
        if prediction >= threshold:
            if qty > 0:
                await execute_order(trading_client, symbol, qty, "buy", "long", current_price, log_file_path, file_handler, sl_distance, tp_distance)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
        elif prediction <= (1 - threshold):
            if qty > 0:
                await execute_order(trading_client, symbol, qty, "sell", "short", current_price, log_file_path, file_handler, sl_distance, tp_distance)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))

    trade_summary = analyze_trades(log_file_path, trading_client)
    logging.info(f"Parsed {trade_summary['total_trades']} trades, {trade_summary['completed_trades']} completed, calculated total P/L: ${trade_summary['pl']:.2f}, win rate: {trade_summary['win_rate']:.2f}")
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Account'} Equity: ${trade_summary['equity']:.2f}")
    logging.info(f"Saved equity plot to equity_plot.png")
    file_handler.flush()

async def trading_loop(trading_client: TradingClient, data_client: StockHistoricalDataClient, log_file_path: str, file_handler, stream: StockDataStream):
    """
    Main trading loop for AI-powered day trading with Grok-3 and Alpaca.
    """
    last_trade_time = {symbol: None for symbol in SYMBOLS}
    start_time = datetime.now(ZoneInfo("America/New_York")).timestamp()  # Use timestamp for precision
    ny_tz = ZoneInfo("America/New_York")
    logging.info(f"Starting trading session at {datetime.fromtimestamp(start_time, ny_tz).strftime('%Y-%m-%d %H:%M:%S')} with duration: {SESSION_DURATION}")
    subscribed = False  # Track subscription status
    websocket_task = None

    async def handle_bar(bar):
        await trading_logic(trading_client, data_client, bar.symbol, bar.close, log_file_path, file_handler, last_trade_time)

    async def close_all_positions():
        """Close all open positions at market close or session end."""
        any_positions_closed = False
        for symbol in SYMBOLS:
            position = await get_position(trading_client, symbol)
            if position != 0:
                any_positions_closed = True
                order_type = "sell" if position > 0 else "buy"
                trade_type = "close long" if position > 0 else "close short"
                if TRADE_MODE == 'simulation':
                    current_price = 190.0 + random.uniform(-20, 20)
                else:
                    request_params = StockLatestBarRequest(symbol_or_symbols=symbol)
                    try:
                        latest_bar = await data_client.get_stock_latest_bar(request_params)
                        current_price = latest_bar[symbol].close
                    except Exception as e:
                        logging.error(f"Error fetching latest bar for {symbol}: {e}")
                        current_price = 190.0  # Fallback price
                logging.info(f"Closing {symbol} position at session end: {position} shares at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), order_type, trade_type, current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
        if not any_positions_closed:
            logging.info("No open positions to close at session end")

    async def process_symbols():
        tasks = []
        for symbol in SYMBOLS:
            use_mock_data = TRADE_MODE == 'simulation'
            current_price = 190.0  # Fallback price
            if not use_mock_data:
                try:
                    clock = trading_client.get_clock()
                    use_mock_data = not clock.is_open
                    if use_mock_data:
                        logging.info(f"Market closed for {symbol}, using mock data")
                except Exception as e:
                    logging.error(f"Error checking market hours for {symbol}: {e}, using mock data")
                    use_mock_data = True
            if use_mock_data:
                bars = fetch_mock_bar_data(symbol, SEQUENCE_LENGTH, file_handler)  # Sync function, no await
                current_price = bars['close'].iloc[-1]  # Use latest mock price
            else:
                try:
                    request_params = StockLatestBarRequest(symbol_or_symbols=symbol)
                    latest_bar = await data_client.get_stock_latest_bar(request_params)
                    current_price = latest_bar[symbol].close
                except Exception as e:
                    logging.warning(f"Error fetching latest bar for {symbol}: {e}, using fallback price ${current_price:.2f}")
            tasks.append(trading_logic(trading_client, data_client, symbol, current_price, log_file_path, file_handler, last_trade_time))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_trading_loop():
        for symbol in SYMBOLS:
            open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
            sl_prices[symbol] = 0.0
            tp_prices[symbol] = 0.0
        try:
            if TRADE_MODE != 'simulation':
                # Subscribe to bars only once
                nonlocal subscribed, websocket_task
                if not subscribed:
                    for symbol in SYMBOLS:
                        stream.subscribe_bars(handle_bar, symbol)
                    subscribed = True
                    logging.info(f"Subscribed to bars for {SYMBOLS}")
                # Run WebSocket in a cancellable task
                async def run_websocket():
                    loop = asyncio.get_running_loop()
                    try:
                        await loop.run_in_executor(None, stream.run)
                    except asyncio.CancelledError:
                        await stream.stop_ws()
                        logging.info("WebSocket task cancelled and stopped")
                        raise
                    except Exception as e:
                        logging.error(f"WebSocket error: {e}")
                        await stream.stop_ws()
                        raise
                websocket_task = asyncio.create_task(run_websocket())
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
                elapsed = datetime.now(ZoneInfo("America/New_York")).timestamp() - start_time
                if elapsed >= session_seconds:
                    logging.info(f"Session duration reached (elapsed: {elapsed:.1f}s, target: {session_seconds}s), closing positions and stopping trading loop")
                    await close_all_positions()
                    file_handler.flush()
                    logging.info("Trading loop terminated")
                    if TRADE_MODE != 'simulation':
                        stream.unsubscribe_bars(*SYMBOLS)
                        subscribed = False
                    break

                # Check for market close
                now = datetime.now(ny_tz)
                if SESSION_DURATION == "end_of_day" and now.hour >= 16:
                    logging.info("Market close reached, closing positions and stopping trading loop")
                    await close_all_positions()
                    file_handler.flush()
                    logging.info("Trading loop terminated")
                    if TRADE_MODE != 'simulation':
                        stream.unsubscribe_bars(*SYMBOLS)
                        subscribed = False
                    break

                await process_symbols()
                logging.info("Starting sleep")
                for i in range(ADJUSTMENT_INTERVAL, 0, -10):
                    logging.info(f"Sleep countdown: {i} seconds remaining")
                    file_handler.flush()
                    await asyncio.sleep(10)
                logging.info("Completed sleep")
        except asyncio.CancelledError:
            logging.info("Trading loop cancelled, closing positions and WebSocket")
            await close_all_positions()
            if TRADE_MODE != 'simulation':
                stream.unsubscribe_bars(*SYMBOLS)
                subscribed = False
                await stream.stop_ws()
            file_handler.flush()
            logging.info("Trading loop terminated")
            if TRADE_MODE != 'simulation' and websocket_task:
                websocket_task.cancel()
                try:
                    await websocket_task
                except asyncio.CancelledError:
                    pass
            raise
        except Exception as e:
            logging.error(f"Unexpected error in trading loop: {e}")
            raise
        finally:
            if TRADE_MODE != 'simulation' and websocket_task:
                websocket_task.cancel()
                try:
                    await websocket_task
                except asyncio.CancelledError:
                    pass
                await stream.stop_ws()
                subscribed = False
            file_handler.flush()
            logging.info("Trading loop cleanup completed")

    await run_trading_loop()