# trade_analysis.py
import re
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_trades(log_file_path, trading_client):
    """
    Analyzes trade logs to compute total P/L, win rate, number of trades, equity, and drawdown.
    Saves equity curve plot to 'equity_plot.png'.
    """
    total_pl = 0.0
    wins = 0
    total_trades = 0
    completed_trades = 0
    equity = trading_client.get_account().equity
    max_equity = equity
    drawdown = 0.0
    equity_curve = []
    timestamps = []

    try:
        if not os.path.exists(log_file_path):
            logging.error(f"Log file {log_file_path} not found, returning default summary")
            return {
                'pl': total_pl,
                'win_rate': 0.0,
                'total_trades': total_trades,
                'completed_trades': completed_trades,
                'equity': equity,
                'drawdown': drawdown
            }

        with open(log_file_path, 'r') as file:
            for line in file:
                # Parse P/L
                pl_match = re.search(r'Calculated P/L for .*?: \$([-]?\d+\.\d+)', line)
                if pl_match:
                    pl = float(pl_match.group(1))
                    total_pl += pl
                    completed_trades += 1
                    if pl > 0:
                        wins += 1

                # Parse trade initiations
                if 'Simulated BUY order' in line or 'Simulated SELL order' in line:
                    total_trades += 1

                # Parse equity and timestamp
                equity_match = re.search(r'Account Equity: \$(\d+\.\d+)', line)
                timestamp_match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if equity_match and timestamp_match:
                    current_equity = float(equity_match.group(1))
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    max_equity = max(max_equity, current_equity)
                    drawdown = min(drawdown, (current_equity - max_equity) / max_equity)
                    equity_curve.append(current_equity)
                    timestamps.append(timestamp)

        win_rate = wins / completed_trades if completed_trades > 0 else 0.0

        # Plot equity curve
        if equity_curve and timestamps:
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, equity_curve, label='Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity ($)')
            plt.title('Equity Curve')
            plt.legend()
            plt.grid(True)
            try:
                plt.savefig('equity_plot.png')
                plt.close()
            except Exception as e:
                logging.error(f"Failed to save equity plot: {e}")

    except Exception as e:
        logging.error(f"Error analyzing trades: {e}")
        win_rate = 0.0

    return {
        'pl': total_pl,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'completed_trades': completed_trades,
        'equity': equity,
        'drawdown': drawdown
    }