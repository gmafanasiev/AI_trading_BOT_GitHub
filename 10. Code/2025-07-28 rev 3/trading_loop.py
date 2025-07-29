# trading_loop.py
import asyncio
import logging
import random
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.live import StockDataStream
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.enums import DataFeed
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
    BACKTEST_MODE
)

async def is_cooldown_active(symbol: str, last_trade_time: dict, current_time: datetime = None) -> bool:
    if symbol not in last_trade_time or last_trade_time[symbol] is None:
        return False
    if current_time is None:
        current_time = datetime.now(ZoneInfo("America/New_York"))
    time_since = (current_time - last_trade_time[symbol]).total_seconds()
    if time_since < COOLDOWN_SECONDS:
        logging.info(f"Cooldown active for {symbol}, skipping trade (remaining: {COOLDOWN_SECONDS - time_since:.1f} seconds)")
        return True
    return False

async def trading_logic(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    symbol: str,
    current_price: float,
    log_file_path: str,
    file_handler,
    last_trade_time: dict,
    current_time: datetime = None,
    historical_bars=None,
    historical_daily_bars=None,
    upper_threshold=UPPER_THRESHOLD,
    lower_threshold=LOWER_THRESHOLD
):
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Fetched'} {SEQUENCE_LENGTH} bars for {symbol}, price: ${current_price:.2f}")
    position = await get_position(trading_client, symbol)
    logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Current'} position for {symbol}: {position} shares")
    trade_summary = analyze_trades(log_file_path, trading_client)
    if trade_summary['drawdown'] < -MAX_DRAWDOWN:
        logging.info(f"Max drawdown exceeded ({trade_summary['drawdown']:.2%}), skipping trade for {symbol}")
        file_handler.flush()
        return

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

    timeframe = TimeFrame.Minute
    if historical_bars is None:
        if use_mock_data:
            bars = fetch_mock_bar_data(symbol, SEQUENCE_LENGTH, file_handler)
            daily_bars = fetch_mock_daily_data(symbol, NUM_DAYS_HISTORY, file_handler)
            current_price = bars['close'].iloc[-1]
        else:
            bars = fetch_bar_data(data_client, symbol, timeframe, SEQUENCE_LENGTH, file_handler)
            daily_bars = fetch_daily_data(data_client, symbol, NUM_DAYS_HISTORY, file_handler)
    else:
        bars = historical_bars
        daily_bars = historical_daily_bars if historical_daily_bars is not None else pd.DataFrame()

    stats = prepare_grok4_input(bars, daily_bars, SEQUENCE_LENGTH, symbol, file_handler)
    if stats is None:
        logging.warning(f"Using default Grok prediction for {symbol} due to insufficient data")
        prediction = 0.65
        threshold = upper_threshold
        risk_per_trade = 0.01
    else:
        try:
            prediction, threshold, risk_per_trade = await get_grok4_prediction_and_adjustments(URL, XAI_API_KEY, stats, symbol, trade_summary)
        except Exception as e:
            logging.error(f"Failed to fetch Grok prediction for {symbol}: {e}, using default prediction")
            prediction = 0.65
            threshold = upper_threshold
            risk_per_trade = 0.01
    logging.info(f"{symbol} - Current Price: ${current_price:.2f}, Grok 3 Prediction: {prediction:.2f}, Position: {position}")

    atr = stats.get('atr', current_price * STOP_LOSS_PCT / ATR_MULTIPLIER) if stats else current_price * STOP_LOSS_PCT / ATR_MULTIPLIER
    sl_distance = atr * ATR_MULTIPLIER
    tp_distance = sl_distance * RR_RATIO
    max_position_value = trade_summary['equity'] * MAX_POSITION_PCT
    qty = min(int((trade_summary['equity'] * risk_per_trade) / sl_distance), int(max_position_value / current_price))
    logging.info(f"{symbol} - Calculated Quantity: {qty}, SL Distance: ${sl_distance:.2f}, TP Distance: ${tp_distance:.2f}")

    if position != 0:
        if position > 0:  # Long
            if current_price <= sl_prices[symbol]:
                logging.info(f"Trailing stop-loss hit for {symbol} long: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, position, "sell", "close long", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
            elif current_price >= tp_prices[symbol]:
                logging.info(f"Take-profit hit for {symbol} long: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, position, "sell", "close long", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
            elif prediction < LONG_EXIT_THRESHOLD:
                logging.info(f"Bearish prediction ({prediction:.2f}) for long: Closing at ${current_price:.2f}")
                await execute_order(trading_client, symbol, position, "sell", "close long", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
            else:
                sl_prices[symbol] = max(sl_prices[symbol], current_price * (1 - TRAILING_PCT))
                tp_prices[symbol] = min(tp_prices[symbol], current_price * (1 + TRAILING_PCT))
                logging.info(f"Trailing SL updated for {symbol} long to ${sl_prices[symbol]:.2f}")
                logging.info(f"Trailing TP updated for {symbol} long to ${tp_prices[symbol]:.2f}")
        else:  # Short
            if current_price >= sl_prices[symbol]:
                logging.info(f"Trailing stop-loss hit for {symbol} short: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), "buy", "close short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
            elif current_price <= tp_prices[symbol]:
                logging.info(f"Take-profit hit for {symbol} short: Closed at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), "buy", "close short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
            elif prediction > SHORT_EXIT_THRESHOLD:
                logging.info(f"Bullish prediction ({prediction:.2f}) for short: Closing at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), "buy", "close short", current_price, log_file_path, file_handler)
                last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
            else:
                sl_prices[symbol] = min(sl_prices[symbol], current_price * (1 + TRAILING_PCT))
                tp_prices[symbol] = max(tp_prices[symbol], current_price * (1 - TRAILING_PCT))
                logging.info(f"Trailing SL updated for {symbol} short to ${sl_prices[symbol]:.2f}")
                logging.info(f"Trailing TP updated for {symbol} short to ${tp_prices[symbol]:.2f}")
        file_handler.flush()
    else:
        if await is_cooldown_active(symbol, last_trade_time, current_time):
            file_handler.flush()
            return
        if prediction >= upper_threshold and qty > 0:
            await execute_order(trading_client, symbol, qty, "buy", "long", current_price, log_file_path, file_handler, sl_distance, tp_distance)
            last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))
        elif prediction <= (1 - lower_threshold) and qty > 0:
            await execute_order(trading_client, symbol, qty, "sell", "short", current_price, log_file_path, file_handler, sl_distance, tp_distance)
            last_trade_time[symbol] = current_time or datetime.now(ZoneInfo("America/New_York"))

    if not hasattr(trading_logic, 'plot_counter'):
        trading_logic.plot_counter = {sym: 0 for sym in SYMBOLS}
    trading_logic.plot_counter[symbol] = trading_logic.plot_counter.get(symbol, 0) + 1
    if trading_logic.plot_counter[symbol] % 100 == 0:
        trade_summary = analyze_trades(log_file_path, trading_client)
        logging.info(f"Parsed {trade_summary['total_trades']} trades, {trade_summary['completed_trades']} completed, total P/L: ${trade_summary['pl']:.2f}, win rate: {trade_summary['win_rate']:.2f}")
        logging.info(f"{'Simulated' if TRADE_MODE == 'simulation' else 'Account'} Equity: ${trade_summary['equity']:.2f}")
        logging.info("Saved equity plot to equity_plot.png")
    file_handler.flush()

async def trading_loop(trading_client: TradingClient, data_client: StockHistoricalDataClient, log_file_path: str, file_handler, stream: StockDataStream):
    last_trade_time = {symbol: None for symbol in SYMBOLS}
    start_time = datetime.now(ZoneInfo("America/New_York")).timestamp()
    ny_tz = ZoneInfo("America/New_York")
    logging.info(f"Starting trading session at {datetime.fromtimestamp(start_time, ny_tz).strftime('%Y-%m-%d %H:%M:%S')} with duration: {SESSION_DURATION}")
    subscribed = False
    websocket_task = None

    async def handle_bar(bar):
        await trading_logic(trading_client, data_client, bar.symbol, bar.close, log_file_path, file_handler, last_trade_time)

    async def close_all_positions():
        any_closed = False
        for symbol in SYMBOLS:
            position = await get_position(trading_client, symbol)
            if position != 0:
                any_closed = True
                order_type = "sell" if position > 0 else "buy"
                trade_type = "close long" if position > 0 else "close short"
                current_price = 190.0 + random.uniform(-20, 20) if TRADE_MODE == 'simulation' else (data_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))[symbol].close or 190.0)
                logging.info(f"Closing {symbol} position at session end: {position} shares at ${current_price:.2f}")
                await execute_order(trading_client, symbol, abs(position), order_type, trade_type, current_price, log_file_path, file_handler)
                last_trade_time[symbol] = datetime.now(ZoneInfo("America/New_York"))
        if not any_closed:
            logging.info("No open positions to close at session end")

    async def process_symbols():
        tasks = []
        for symbol in SYMBOLS:
            use_mock_data = TRADE_MODE == 'simulation'
            if not use_mock_data:
                try:
                    use_mock_data = not trading_client.get_clock().is_open
                    if use_mock_data:
                        logging.info(f"Market closed for {symbol}, using mock data")
                except Exception as e:
                    logging.error(f"Error checking market hours for {symbol}: {e}, using mock data")
                    use_mock_data = True
            current_price = 190.0
            if use_mock_data:
                bars = fetch_mock_bar_data(symbol, SEQUENCE_LENGTH, file_handler)
                current_price = bars['close'].iloc[-1]
            else:
                try:
                    current_price = data_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))[symbol].close
                except Exception as e:
                    logging.warning(f"Error fetching latest bar for {symbol}: {e}, using fallback price ${current_price:.2f}")
            tasks.append(trading_logic(trading_client, data_client, symbol, current_price, log_file_path, file_handler, last_trade_time))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def parse_session_seconds() -> float:
        if SESSION_DURATION == "end_of_day":
            now = datetime.now(ny_tz)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if now >= market_close:
                market_close += timedelta(days=1)
            return (market_close - now).total_seconds()
        try:
            value, unit = SESSION_DURATION.split()
            value = float(value)
            if value <= 0:
                raise ValueError("SESSION_DURATION value must be positive")
            if unit == "minutes":
                return value * 60
            elif unit == "seconds":
                return value
            else:
                raise ValueError("SESSION_DURATION unit must be 'seconds' or 'minutes'")
        except Exception as e:
            logging.error(f"Invalid SESSION_DURATION: {SESSION_DURATION}, {str(e)}, defaulting to 3600s")
            return 3600.0

    for symbol in SYMBOLS:
        open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
        sl_prices[symbol] = 0.0
        tp_prices[symbol] = 0.0

    try:
        if TRADE_MODE != 'simulation':
            subscribed = False
            websocket_task = None
            if not subscribed:
                for symbol in SYMBOLS:
                    stream.subscribe_bars(handle_bar, symbol)
                subscribed = True
                logging.info(f"Subscribed to bars for {SYMBOLS}")
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
            session_seconds = parse_session_seconds()
            elapsed = datetime.now(ZoneInfo("America/New_York")).timestamp() - start_time
            now = datetime.now(ny_tz)
            if elapsed >= session_seconds or (SESSION_DURATION == "end_of_day" and now.hour >= 16):
                logging.info("Session/market end reached, closing positions and stopping")
                await close_all_positions()
                logging.info("Trading loop terminated")
                if TRADE_MODE != 'simulation':
                    stream.unsubscribe_bars(*SYMBOLS)
                    subscribed = False
                file_handler.flush()
                break
            await process_symbols()
            logging.info("Starting sleep")
            for i in range(ADJUSTMENT_INTERVAL, 0, -10):
                logging.info(f"Sleep countdown: {i} seconds remaining")
                await asyncio.sleep(10)
            logging.info("Completed sleep")
            file_handler.flush()
    except asyncio.CancelledError:
        logging.info("Trading loop cancelled, closing positions and WebSocket")
        await close_all_positions()
        if TRADE_MODE != 'simulation':
            stream.unsubscribe_bars(*SYMBOLS)
            subscribed = False
            await stream.stop_ws()
        logging.info("Trading loop terminated")
        file_handler.flush()
        if TRADE_MODE != 'simulation' and websocket_task:
            websocket_task.cancel()
            await websocket_task
        raise
    except Exception as e:
        logging.error(f"Unexpected error in trading loop: {e}")
        raise
    finally:
        if TRADE_MODE != 'simulation' and websocket_task:
            websocket_task.cancel()
            await websocket_task
            await stream.stop_ws()
            subscribed = False
        logging.info("Trading loop cleanup completed")
        file_handler.flush()