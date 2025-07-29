# trade_analysis.py
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from config import TRADE_MODE

def analyze_trades(log_file: str, trading_client=None) -> dict:
    """
    Analyze trades from the log file and generate equity plot.

    :param log_file: Path to the trade log file.
    :param trading_client: Optional TradingClient for real equity fetch.
    :return: Dictionary with trade summary (total trades, completed trades, P/L, win rate, equity).
    """
    trades = []
    total_pl = 0.0
    completed_trades = 0
    wins = 0
    equity = 100000.0  # Starting equity
    if TRADE_MODE != 'simulation' and trading_client:
        try:
            equity = float(trading_client.get_account().equity)
        except Exception as e:
            logging.warning(f"Failed to fetch real equity: {e}, using default {equity}")
    equity_curve = [equity]  # Start with initial equity
    trade_times = [datetime.now()]  # Initial timestamp
    trade_types = ['Start']  # Initial type
    time_threshold = datetime.now() - timedelta(hours=24)  # Parse trades within last 24 hours

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if not timestamp_match:
                continue
            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            if timestamp < time_threshold:
                continue

            # Parse buy/sell orders
            order_match = re.search(r'(Simulated|Placed) (BUY|SELL) order \(Alpaca, (\w+)\): ([\d.]+) shares of (\w+) at \$([\d.]+)', line)
            if order_match and timestamp >= time_threshold:
                side = order_match.group(2).lower()
                trade_type = 'long' if side == 'buy' and order_match.group(3) == 'long' else 'close short' if side == 'buy' else 'short' if side == 'sell' and order_match.group(3) == 'short' else 'close long'
                trades.append({
                    "symbol": order_match.group(5),
                    "type": side,
                    "price": float(order_match.group(6)),
                    "id": f"{side}_{len(trades)}",
                    "qty": float(order_match.group(4)),
                    "trade_type": trade_type,
                    "timestamp": timestamp
                })
                trade_times.append(timestamp)
                trade_types.append('L' if trade_type in ['long', 'close long'] else 'S')
                equity_curve.append(equity)
                logging.debug(f"Parsed {side.upper()} order for {order_match.group(5)}: {trade_type}, qty: {order_match.group(4)}")
            # Parse P/L
            pl_match = re.search(r'Calculated P/L for (\w+) (long|short): \$([-]?[\d.]+)', line)
            if pl_match and timestamp >= time_threshold:
                pl = float(pl_match.group(3))
                total_pl += pl
                completed_trades += 1
                if pl > 0:
                    wins += 1
                equity += pl
                trade_times.append(timestamp)
                trade_types.append('C')
                equity_curve.append(equity)
                logging.debug(f"Parsed P/L for {pl_match.group(1)}: ${pl:.2f}")

        win_rate = wins / completed_trades if completed_trades > 0 else 0.0
        logging.debug(f"Parsed {len(trades)} trades, {completed_trades} completed, total P/L: ${total_pl:.2f}")

        # Generate and save equity plot
        if len(trade_times) >= 1 and len(trade_times) == len(equity_curve) == len(trade_types):
            plt.figure(figsize=(10, 6))
            plt.plot(trade_times, equity_curve, label='Equity Curve', color='#1f77b4', marker='o')
            for i, (time, equity, trade_type) in enumerate(zip(trade_times, equity_curve, trade_types)):
                plt.annotate(f"{trade_type}{i}", (time, equity), textcoords="offset points", xytext=(0,10), ha='center')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
            plt.xticks(rotation=45)
            plt.xlabel('Time (S/L/C, Index)')
            plt.title('Equity Curve')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            try:
                plt.savefig('equity_plot.png', bbox_inches='tight', format='png')
                plt.close()
                logging.info("Equity plot saved to equity_plot.png")
            except Exception as e:
                logging.error(f"Error saving equity plot: {e}")
        else:
            logging.warning(f"Skipping equity plot: trade_times ({len(trade_times)}), equity_curve ({len(equity_curve)}), trade_types ({len(trade_types)}) mismatched")

        return {
            "total_trades": len(trades),
            "completed_trades": completed_trades,
            "pl": total_pl,
            "win_rate": win_rate,
            "equity": equity
        }
    except Exception as e:
        logging.error(f"Error analyzing trades: {e}")
        return {"total_trades": 0, "completed_trades": 0, "pl": 0.0, "win_rate": 0.0, "equity": equity}