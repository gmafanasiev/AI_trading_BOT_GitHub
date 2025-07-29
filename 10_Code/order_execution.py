# order_execution.py
import logging
import asyncio
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.common.exceptions import APIError
from config import open_positions, sl_prices, tp_prices, STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRADE_MODE, POLL_TIMEOUT, BACKTEST_MODE

"""
Order execution module for managing positions and orders for the AI trading bot.
"""

async def get_position(trading_client: TradingClient, symbol: str) -> float:
    """Get current position quantity for a symbol."""
    if TRADE_MODE == 'simulation' or BACKTEST_MODE:
        return open_positions[symbol]['qty']  # Use in-memory state for simulation
    try:
        position = trading_client.get_open_position(symbol)
        qty = float(position.qty)
        open_positions[symbol] = {'type': 'long' if qty > 0 else 'short', 'qty': qty, 'entry_price': float(position.avg_entry_price or 0.0)}
        return qty
    except (ValueError, APIError) as e:
        if "position does not exist" in str(e).lower():
            open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
            return 0
        logging.error(f"Error fetching position for {symbol}: {e}")
        return open_positions[symbol]['qty']

async def poll_order_fill(trading_client: TradingClient, order_id: str, timeout: int = POLL_TIMEOUT) -> tuple[float, float]:
    """Poll for order fill status and return filled qty and avg price."""
    filled_qty = 0.0
    filled_price = 0.0
    for _ in range(timeout):
        await asyncio.sleep(1)
        try:
            order = trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                filled_qty = float(order.filled_qty)
                filled_price = float(order.filled_avg_price or 0.0)
                break
        except Exception as e:
            logging.error(f"Error polling order {order_id}: {e}")
    return filled_qty, filled_price

async def update_position(trading_client: TradingClient, symbol: str, side: str, trade_type: str, filled_qty: float, filled_price: float, file_handler, sl_distance: float = 0.0, tp_distance: float = 0.0):
    """Update position state based on trade execution."""
    position = open_positions[symbol]
    pl = 0.0
    if side == 'buy':
        if trade_type == 'long':
            if position['type'] == 'long' and position['qty'] > 0:
                total_qty = position['qty'] + filled_qty
                position['entry_price'] = (position['entry_price'] * position['qty'] + filled_price * filled_qty) / total_qty
                position['qty'] = total_qty
            else:
                position.update({'type': 'long', 'qty': filled_qty, 'entry_price': filled_price})
            sl_prices[symbol] = filled_price - sl_distance
            tp_prices[symbol] = filled_price + tp_distance
            logging.info(f"Buy long: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
        elif trade_type == 'close short':
            if position['type'] == 'short' and position['qty'] <= -filled_qty:
                pl = (position['entry_price'] - filled_price) * abs(filled_qty)
                position['qty'] += filled_qty
                if abs(position['qty']) < 0.001:  # Handle floating-point precision
                    position.update({'type': None, 'qty': 0, 'entry_price': 0.0})
                    logging.info(f"Closed short position for {symbol}: 0 shares remaining")
                    logging.info(f"Calculated P/L for {symbol} short: ${pl:.2f}")
                else:
                    logging.info(f"Partially closed short position for {symbol}: {position['qty']} shares remaining")
            else:
                logging.warning(f"Invalid close short order for {symbol}: Insufficient short position (qty: {position['qty']})")
                return False
    elif side == 'sell':
        if trade_type == 'short':
            if position['type'] == 'short' and position['qty'] < 0:
                total_qty = position['qty'] - filled_qty
                position['entry_price'] = (position['entry_price'] * abs(position['qty']) + filled_price * filled_qty) / abs(total_qty)
                position['qty'] = total_qty
            else:
                position.update({'type': 'short', 'qty': -filled_qty, 'entry_price': filled_price})
            sl_prices[symbol] = filled_price + sl_distance
            tp_prices[symbol] = filled_price - tp_distance
            logging.info(f"Sell short: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
        elif trade_type == 'close long':
            if position['type'] == 'long' and position['qty'] >= filled_qty:
                pl = (filled_price - position['entry_price']) * filled_qty
                position['qty'] -= filled_qty
                if position['qty'] < 0.001:  # Handle floating-point precision
                    position.update({'type': None, 'qty': 0, 'entry_price': 0.0})
                    logging.info(f"Closed long position for {symbol}: 0 shares remaining")
                    logging.info(f"Calculated P/L for {symbol} long: ${pl:.2f}")
                else:
                    logging.info(f"Partially closed long position for {symbol}: {position['qty']} shares remaining")
            else:
                logging.warning(f"Invalid close long order for {symbol}: Insufficient long position (qty: {position['qty']})")
                return False
    file_handler.flush()
    return True

async def execute_order(trading_client: TradingClient, symbol: str, qty: float, side: str, trade_type: str, current_price: float, log_file_path: str, file_handler, sl_distance: float = 0.0, tp_distance: float = 0.0) -> tuple[bool, str]:
    """
    Execute a market order and update position state.
    """
    if qty <= 0:
        logging.warning(f"Invalid quantity {qty} for {side} order on {symbol}")
        file_handler.flush()
        return False, f"Invalid quantity {qty}"

    if TRADE_MODE == 'simulation' or BACKTEST_MODE:
        logging.info(f"Simulated {side.upper()} order (Alpaca, {trade_type}): {qty} shares of {symbol} at ${current_price:.2f}")
        success = await update_position(trading_client, symbol, side, trade_type, qty, current_price, file_handler, sl_distance, tp_distance)
        return success, ""
    else:  # 'paper' or 'live'
        try:
            # Check market hours
            clock = trading_client.get_clock()
            if not clock.is_open:
                logging.warning(f"Market closed, cannot submit {side.upper()} order for {symbol} in {TRADE_MODE} mode")
                return False, "Market closed"
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            market_order = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY)
            order = trading_client.submit_order(market_order)
            logging.info(f"Placed {side.upper()} order (Alpaca, {trade_type}): {qty} shares of {symbol} at ${current_price:.2f}, Order ID: {order.id}")
            filled_qty, filled_price = await poll_order_fill(trading_client, order.id)
            if filled_qty == 0:
                logging.warning(f"Order {order.id} not filled after polling, using estimated qty/price")
                filled_qty, filled_price = qty, current_price
            success = await update_position(trading_client, symbol, side, trade_type, filled_qty, filled_price, file_handler, sl_distance, tp_distance)
            return success, ""
        except ValueError as e:
            logging.error(f"Error placing {side} order for {symbol}: {e}")
            file_handler.flush()
            return False, str(e)