# order_execution.py
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from config import open_positions, sl_prices, tp_prices, STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRADE_MODE

"""
Order execution module for managing positions and orders for the AI trading bot.
"""

def update_position_and_levels(symbol, qty, side, trade_type, current_price):
    """Helper to update position, calculate P/L, and set SL/TP levels."""
    position = open_positions[symbol]
    pl = 0.0
    if side == 'buy':
        if trade_type == 'long':
            if position['type'] == 'long' and position['qty'] > 0:
                total_qty = position['qty'] + qty
                position['entry_price'] = (position['entry_price'] * position['qty'] + current_price * qty) / total_qty
                position['qty'] = total_qty
            else:
                position['type'] = 'long'
                position['qty'] = qty
                position['entry_price'] = current_price
            sl_prices[symbol] = current_price * (1 - STOP_LOSS_PCT)
            tp_prices[symbol] = current_price * (1 + TAKE_PROFIT_PCT)
            logging.info(f"Buy long: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
        elif trade_type == 'close short':
            if position['type'] == 'short' and position['qty'] <= -qty:
                pl = (position['entry_price'] - current_price) * abs(position['qty'])
                position['qty'] += qty
                if position['qty'] >= 0:
                    position.update({'type': None, 'qty': 0, 'entry_price': 0.0})
                    logging.info(f"Closed short position for {symbol}: 0 shares remaining")
                    logging.info(f"Calculated P/L for {symbol} short: ${pl:.2f}")
                else:
                    logging.info(f"Partially closed short position for {symbol}: {position['qty']} shares remaining")
            else:
                logging.warning(f"Invalid close short order for {symbol}: Insufficient short position (qty: {position['qty']})")
                return 0.0, False
    elif side == 'sell':
        if trade_type == 'short':
            if position['type'] == 'short' and position['qty'] < 0:
                total_qty = position['qty'] - qty
                position['entry_price'] = (position['entry_price'] * abs(position['qty']) + current_price * qty) / abs(total_qty)
                position['qty'] = total_qty
            else:
                position['type'] = 'short'
                position['qty'] = -qty
                position['entry_price'] = current_price
            sl_prices[symbol] = current_price * (1 + STOP_LOSS_PCT)
            tp_prices[symbol] = current_price * (1 - TAKE_PROFIT_PCT)
            logging.info(f"Sell short: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
        elif trade_type == 'close long':
            if position['type'] == 'long' and position['qty'] >= qty:
                pl = (current_price - position['entry_price']) * position['qty']
                position['qty'] -= qty
                if position['qty'] <= 0:
                    position.update({'type': None, 'qty': 0, 'entry_price': 0.0})
                    logging.info(f"Closed long position for {symbol}: 0 shares remaining")
                    logging.info(f"Calculated P/L for {symbol} long: ${pl:.2f}")
                else:
                    logging.info(f"Partially closed long position for {symbol}: {position['qty']} shares remaining")
            else:
                logging.warning(f"Invalid close long order for {symbol}: Insufficient long position (qty: {position['qty']})")
                return 0.0, False
    return pl, True

async def get_position(trading_client: TradingClient, symbol: str) -> float:
    """
    Get current position quantity for a symbol.
    """
    if TRADE_MODE == 'simulation':
        return open_positions[symbol]['qty']
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except (ValueError, APIError) as e:
        if "position does not exist" in str(e).lower():
            return 0
        logging.error(f"Error fetching position for {symbol}: {e}")
        return 0

async def execute_order(trading_client: TradingClient, symbol: str, qty: float, side: str, trade_type: str, current_price: float, log_file_path: str, file_handler) -> tuple[bool, str]:
    """
    Execute a market order and update position state.
    """
    if qty <= 0:
        logging.warning(f"Invalid quantity {qty} for {side} order on {symbol}")
        file_handler.flush()
        return False, f"Invalid quantity {qty}"

    if TRADE_MODE == 'simulation':
        logging.info(f"Simulated {side.upper()} order (Alpaca, {trade_type}): {qty} shares of {symbol} at ${current_price:.2f}")
    else:
        try:
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            market_order = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY)
            order = trading_client.submit_order(market_order)
            logging.info(f"Placed {side.upper()} order (Alpaca, {trade_type}): {qty} shares of {symbol} at ${current_price:.2f}, Order ID: {order.id}")
        except ValueError as e:
            logging.error(f"Error placing {side} order for {symbol}: {e}")
            file_handler.flush()
            return False, str(e)

    pl, success = update_position_and_levels(symbol, qty, side, trade_type, current_price)
    if not success:
        file_handler.flush()
        return False, "Position update failed"

    current_position = await get_position(trading_client, symbol)
    if current_position == 0:
        sl_prices[symbol] = 0.0
        tp_prices[symbol] = 0.0
        open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
        logging.info(f"Position for {symbol} fully closed, reset SL and TP prices")
    elif current_position > 0 and trade_type == 'short':
        logging.warning(f"Position for {symbol} is positive ({current_position}) after {trade_type} order")
    elif current_position < 0 and trade_type == 'long':
        logging.warning(f"Position for {symbol} is negative ({current_position}) after {trade_type} order")

    file_handler.flush()
    return True, ""