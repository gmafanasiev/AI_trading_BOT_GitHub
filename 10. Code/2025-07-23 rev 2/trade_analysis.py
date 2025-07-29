# trade_analysis.py
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def analyze_trades(log_file: str) -> dict:
    """
    Analyze trades from the log file and generate equity plot.

    :param log_file: Path to the trade log file.
    :return: Dictionary with trade summary (total trades, completed trades, P/L, win rate, equity).
    """
    trades = []
    total_pl = 0.0
    completed_trades = 0
    wins = 0
    equity = 100000.0  # Starting equity
    equity_curve = []  # Track equity for plotting
    trade_times = []  # Track trade timestamps
    trade_types = []  # Track trade types (short/long)
    time_threshold = datetime.now() - timedelta(hours=24)  # Parse trades within last 24 hours

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                if timestamp < time_threshold and ("BUY order" in line or "SELL order" in line or "Calculated P/L" in line):
                    logging.debug(f"Skipping old trade at {timestamp}")
                    continue

            # Parse both simulated and placed buy/sell orders
            if "BUY order" in line:
                match = re.search(r'(Simulated|Placed) BUY order \(Alpaca, (\w+)\): (\d+) shares of (\w+) at \$([\d.]+)', line)
                if match and timestamp >= time_threshold:
                    trade_type = 'long' if match.group(2) == 'long' else 'close short'
                    trades.append({
                        "symbol": match.group(4),
                        "type": "buy",
                        "price": float(match.group(5)),
                        "id": f"buy_{len(trades)}",
                        "qty": int(match.group(3)),
                        "trade_type": trade_type,
                        "timestamp": timestamp
                    })
                    trade_times.append(timestamp)
                    trade_types.append('L' if trade_type == 'long' else 'S')
                    equity_curve.append(equity)  # Append current equity for trade entry
            elif "SELL order" in line:
                match = re.search(r'(Simulated|Placed) SELL order \(Alpaca, (\w+)\): (\d+) shares of (\w+) at \$([\d.]+)', line)
                if match and timestamp >= time_threshold:
                    trade_type = 'short' if match.group(2) == 'short' else 'close long'
                    trades.append({
                        "symbol": match.group(4),
                        "type": "sell",
                        "price": float(match.group(5)),
                        "id": f"sell_{len(trades)}",
                        "qty": int(match.group(3)),
                        "trade_type": trade_type,
                        "timestamp": timestamp
                    })
                    trade_times.append(timestamp)
                    trade_types.append('S' if trade_type == 'short' else 'L')
                    equity_curve.append(equity)  # Append current equity for trade entry
            elif "Calculated P/L for AAPL long" in line or "Calculated P/L for AAPL short" in line:
                match = re.search(r'Calculated P/L for AAPL (long|short): \$([-]?[\d.]+)', line)
                if match and timestamp >= time_threshold:
                    pl = float(match.group(2))
                    total_pl += pl
                    completed_trades += 1
                    if pl > 0:
                        wins += 1
                    equity += pl
                    equity_curve.append(equity)  # Append updated equity for completed trade

        win_rate = wins / completed_trades if completed_trades > 0 else 0.0
        logging.debug(f"Parsed {len(trades)} recent trades, {completed_trades} completed, total P/L: ${total_pl:.2f}")

        # Generate and save equity plot
        plt.figure(figsize=(10, 6))
        # Use trade_times for x-axis, fallback to trade numbers if no trades
        if trade_times and len(trade_times) == len(equity_curve):
            plt.plot(trade_times, equity_curve, label='Equity Curve', color='#1f77b4', marker='o')
            for i, (time, equity, trade_type) in enumerate(zip(trade_times, equity_curve, trade_types)):
                plt.annotate(f"{trade_type}{i+1}", (time, equity), textcoords="offset points", xytext=(0,10), ha='center')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
        else:
            plt.plot(range(len(equity_curve)), equity_curve, label='Equity Curve', color='#1f77b4', marker='o')
            plt.xlabel('Trade Number')
        plt.title('Equity Curve with Trade Types')
        plt.xlabel('Date/Time (Short/Long, Trade Number)' if trade_times and len(trade_times) == len(equity_curve) else 'Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('equity_plot.png', bbox_inches='tight', format='png')
        plt.close()

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