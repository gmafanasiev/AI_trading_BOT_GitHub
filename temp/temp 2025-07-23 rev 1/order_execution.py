# order_execution.py
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from config import open_positions, sl_prices, tp_prices, STOP_LOSS_PCT, TAKE_PROFIT_PCT, SIMULATION_MODE

"""
Order execution module for managing positions and orders for the AI trading bot.
"""

async def get_position(trading_client: TradingClient, symbol: str) -> float:
    """
    Get current position quantity for a symbol.

    :param trading_client: Alpaca trading client.
    :param symbol: Stock symbol.
    :return: Position quantity (positive for long, negative for short, 0 if none).
    """
    if SIMULATION_MODE:
        qty = open_positions[symbol]['qty']
        return qty

    try:
        position = trading_client.get_open_position(symbol)
        qty = float(position.qty)
        return qty
    except (ValueError, APIError) as e:
        if "position does not exist" in str(e).lower():
            return 0
        logging.error(f"Error fetching position for {symbol}: {e}")
        return 0

async def execute_order(trading_client: TradingClient, symbol: str, qty: float, side: str, trade_type: str, current_price: float, log_file_path: str, file_handler) -> tuple[bool, str]:
    """
    Execute a market order and update position state.

    :param trading_client: Alpaca trading client.
    :param symbol: Stock symbol.
    :param qty: Quantity to trade.
    :param side: 'buy' or 'sell'.
    :param trade_type: 'long', 'short', 'close long', or 'close short'.
    :param current_price: Current price.
    :param log_file_path: Path to log file.
    :param file_handler: Logging file handler.
    :return: Tuple of (success: bool, message: str).
    """
    if qty <= 0:
        logging.warning(f"Invalid quantity {qty} for {side} order on {symbol}")
        file_handler.flush()
        return False, f"Invalid quantity {qty}"

    if SIMULATION_MODE:
        logging.info(f"Simulated {side.upper()} order (Alpaca, {trade_type}): {qty} shares of {symbol} at ${current_price:.2f}")
        file_handler.flush()

        pl = 0.0
        if side == 'buy':
            if trade_type == 'long':
                if open_positions[symbol]['type'] == 'long' and open_positions[symbol]['qty'] > 0:
                    total_qty = open_positions[symbol]['qty'] + qty
                    open_positions[symbol]['entry_price'] = (
                        open_positions[symbol]['entry_price'] * open_positions[symbol]['qty'] + current_price * qty
                    ) / total_qty
                    open_positions[symbol]['qty'] = total_qty
                else:
                    open_positions[symbol]['type'] = 'long'
                    open_positions[symbol]['qty'] = qty
                    open_positions[symbol]['entry_price'] = current_price
                sl_prices[symbol] = current_price * (1 - STOP_LOSS_PCT)
                tp_prices[symbol] = current_price * (1 + TAKE_PROFIT_PCT)
                logging.info(f"Buy long: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
                file_handler.flush()
            elif trade_type == 'close short':
                if open_positions[symbol]['type'] == 'short' and open_positions[symbol]['qty'] <= -qty:
                    pl = (open_positions[symbol]['entry_price'] - current_price) * abs(open_positions[symbol]['qty'])
                    open_positions[symbol]['qty'] += qty
                    if open_positions[symbol]['qty'] >= 0:
                        open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
                        logging.info(f"Closed short position for {symbol}: 0 shares remaining")
                        logging.info(f"Calculated P/L for {symbol} short: ${pl:.2f}")
                        file_handler.flush()
                    else:
                        logging.info(f"Partially closed short position for {symbol}: {open_positions[symbol]['qty']} shares remaining")
                        file_handler.flush()
                else:
                    logging.warning(f"Invalid close short order for {symbol}: Insufficient short position (qty: {open_positions[symbol]['qty']})")
                    file_handler.flush()
                    return False, f"Insufficient short position to close (qty: {open_positions[symbol]['qty']})"
        elif side == 'sell':
            if trade_type == 'short':
                if open_positions[symbol]['type'] == 'short' and open_positions[symbol]['qty'] < 0:
                    total_qty = open_positions[symbol]['qty'] - qty
                    open_positions[symbol]['entry_price'] = (
                        open_positions[symbol]['entry_price'] * abs(open_positions[symbol]['qty']) + current_price * qty
                    ) / abs(total_qty)
                    open_positions[symbol]['qty'] = total_qty
                else:
                    open_positions[symbol]['type'] = 'short'
                    open_positions[symbol]['qty'] = -qty  # Negative qty for short
                    open_positions[symbol]['entry_price'] = current_price
                sl_prices[symbol] = current_price * (1 + STOP_LOSS_PCT)
                tp_prices[symbol] = current_price * (1 - TAKE_PROFIT_PCT)
                logging.info(f"Sell short: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
                file_handler.flush()
            elif trade_type == 'close long':
                if open_positions[symbol]['type'] == 'long' and open_positions[symbol]['qty'] >= qty:
                    pl = (current_price - open_positions[symbol]['entry_price']) * open_positions[symbol]['qty']
                    open_positions[symbol]['qty'] -= qty
                    if open_positions[symbol]['qty'] <= 0:
                        open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
                        logging.info(f"Closed long position for {symbol}: 0 shares remaining")
                        logging.info(f"Calculated P/L for {symbol} long: ${pl:.2f}")
                        file_handler.flush()
                    else:
                        logging.info(f"Partially closed long position for {symbol}: {open_positions[symbol]['qty']} shares remaining")
                        file_handler.flush()
                else:
                    logging.warning(f"Invalid close long order for {symbol}: Insufficient long position (qty: {open_positions[symbol]['qty']})")
                    file_handler.flush()
                    return False, f"Insufficient long position to close (qty: {open_positions[symbol]['qty']})"

        current_position = await get_position(trading_client, symbol)
        if current_position == 0:
            sl_prices[symbol] = 0.0
            tp_prices[symbol] = 0.0
            open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
            logging.info(f"Position for {symbol} fully closed, reset SL and TP prices")
            file_handler.flush()
        elif current_position > 0 and trade_type == 'short':
            logging.warning(f"Position for {symbol} is positive ({current_position}) after {trade_type} order")
            file_handler.flush()
        elif current_position < 0 and trade_type == 'long':
            logging.warning(f"Position for {symbol} is negative ({current_position}) after {trade_type} order")
            file_handler.flush()

        return True, ""
    else:
        try:
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            market_order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            order = trading_client.submit_order(market_order)
            pl = 0.0
            logging.info(f"Placed {side.upper()} order (Alpaca, {trade_type}): {qty} shares of {symbol} at ${current_price:.2f}, Order ID: {order.id}")
            file_handler.flush()

            if side == 'buy':
                if trade_type == 'long':
                    if open_positions[symbol]['type'] == 'long' and open_positions[symbol]['qty'] > 0:
                        total_qty = open_positions[symbol]['qty'] + qty
                        open_positions[symbol]['entry_price'] = (
                            open_positions[symbol]['entry_price'] * open_positions[symbol]['qty'] + current_price * qty
                        ) / total_qty
                        open_positions[symbol]['qty'] = total_qty
                    else:
                        open_positions[symbol]['type'] = 'long'
                        open_positions[symbol]['qty'] = qty
                        open_positions[symbol]['entry_price'] = current_price
                    sl_prices[symbol] = current_price * (1 - STOP_LOSS_PCT)
                    tp_prices[symbol] = current_price * (1 + TAKE_PROFIT_PCT)
                    logging.info(f"Buy long: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
                    file_handler.flush()
                elif trade_type == 'close short':
                    if open_positions[symbol]['type'] == 'short' and open_positions[symbol]['qty'] <= -qty:
                        pl = (open_positions[symbol]['entry_price'] - current_price) * abs(open_positions[symbol]['qty'])
                        open_positions[symbol]['qty'] += qty
                        if open_positions[symbol]['qty'] >= 0:
                            open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
                            logging.info(f"Closed short position for {symbol}: 0 shares remaining")
                            logging.info(f"Calculated P/L for {symbol} short: ${pl:.2f}")
                            file_handler.flush()
                        else:
                            logging.info(f"Partially closed short position for {symbol}: {open_positions[symbol]['qty']} shares remaining")
                            file_handler.flush()
                    else:
                        logging.warning(f"Invalid close short order for {symbol}: Insufficient short position (qty: {open_positions[symbol]['qty']})")
                        file_handler.flush()
                        return False, f"Insufficient short position to close (qty: {open_positions[symbol]['qty']})"
            elif side == 'sell':
                if trade_type == 'short':
                    if open_positions[symbol]['type'] == 'short' and open_positions[symbol]['qty'] < 0:
                        total_qty = open_positions[symbol]['qty'] - qty
                        open_positions[symbol]['entry_price'] = (
                            open_positions[symbol]['entry_price'] * abs(open_positions[symbol]['qty']) + current_price * qty
                        ) / abs(total_qty)
                        open_positions[symbol]['qty'] = total_qty
                    else:
                        open_positions[symbol]['type'] = 'short'
                        open_positions[symbol]['qty'] = -qty  # Negative qty for short
                        open_positions[symbol]['entry_price'] = current_price
                    sl_prices[symbol] = current_price * (1 + STOP_LOSS_PCT)
                    tp_prices[symbol] = current_price * (1 - TAKE_PROFIT_PCT)
                    logging.info(f"Sell short: SL ${sl_prices[symbol]:.2f}, TP ${tp_prices[symbol]:.2f}")
                    file_handler.flush()
                elif trade_type == 'close long':
                    if open_positions[symbol]['type'] == 'long' and open_positions[symbol]['qty'] >= qty:
                        pl = (current_price - open_positions[symbol]['entry_price']) * open_positions[symbol]['qty']
                        open_positions[symbol]['qty'] -= qty
                        if open_positions[symbol]['qty'] <= 0:
                            open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
                            logging.info(f"Closed long position for {symbol}: 0 shares remaining")
                            logging.info(f"Calculated P/L for {symbol} long: ${pl:.2f}")
                            file_handler.flush()
                        else:
                            logging.info(f"Partially closed long position for {symbol}: {open_positions[symbol]['qty']} shares remaining")
                            file_handler.flush()
                    else:
                        logging.warning(f"Invalid close long order for {symbol}: Insufficient long position (qty: {open_positions[symbol]['qty']})")
                        file_handler.flush()
                        return False, f"Insufficient long position to close (qty: {open_positions[symbol]['qty']})"

            current_position = await get_position(trading_client, symbol)
            if current_position == 0:
                sl_prices[symbol] = 0.0
                tp_prices[symbol] = 0.0
                open_positions[symbol] = {'type': None, 'qty': 0, 'entry_price': 0.0}
                logging.info(f"Position for {symbol} fully closed, reset SL and TP prices")
                file_handler.flush()
            elif current_position > 0 and trade_type == 'short':
                logging.warning(f"Position for {symbol} is positive ({current_position}) after {trade_type} order")
                file_handler.flush()
            elif current_position < 0 and trade_type == 'long':
                logging.warning(f"Position for {symbol} is negative ({current_position}) after {trade_type} order")
                file_handler.flush()

            return True, ""
        except ValueError as e:
            logging.error(f"Error placing {side} order for {symbol}: {e}")
            file_handler.flush()
            return False, str(e)