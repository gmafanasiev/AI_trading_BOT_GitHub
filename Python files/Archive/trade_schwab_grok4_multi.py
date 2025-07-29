import asyncio
import pandas as pd
import numpy as np
import requests
import logging
import re
from datetime import datetime, time, timedelta, timezone
from schwab.auth import easy_client
from schwab.orders.equities import equity_buy_market, equity_sell_market
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(filename='/home/gmafanasiev/trading.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Schwab and xAI API credentials
SCHWAB_APP_KEY = "5AqBefcOK5NJbuv0twAo6s41Ulc2cRlr"
SCHWAB_APP_SECRET = "o76s0GYzvACjHO1e"
SCHWAB_CALLBACK_URL = "https://127.0.0.1:8182"
XAI_API_KEY = "xai-BU2X5OnnITYxBZN5Q3dc8mNrbyw4RrxdrM7caEZvaDZJpASF0e8v6S6IX6HLlQ98WLCvlsK5Ik2HLGhr"
TOKEN_PATH = "/home/gmafanasiev/token.json"

# Trading parameters
SYMBOLS = ["AMD", "MSFT", "NVDA"]
SEQUENCE_LENGTH = 60
RISK_PER_TRADE = 0.01  # Initial 1% risk, divided across stocks
MAX_EQUITY = 1000  # Maximum equity assumption per trade
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
TIMEFRAME = "DAY"
UPPER_THRESHOLD = 0.65  # Buy if > this (no-trade buffer upper)
LOWER_THRESHOLD = 0.35  # Sell if < this (no-trade buffer lower)
ADJUSTMENT_INTERVAL = 30 * 60  # Adjust every 30 minutes

# Initialize Schwab client
client = easy_client(
    api_key=SCHWAB_APP_KEY,
    app_secret=SCHWAB_APP_SECRET,
    callback_url=SCHWAB_CALLBACK_URL,
    token_path=TOKEN_PATH
)

async def fetch_bar_data(symbol, timeframe="DAY", limit=60):
    resp = client.get_price_history_every_day(symbol, limit=limit)
    if resp.status_code == 200:
        data = resp.json()
        prices = [candle["close"] for candle in data["candles"]]
        df = pd.DataFrame(prices, columns=["close"])
        return df
    else:
        logging.info(f"Failed to fetch data for {symbol}: {resp.status_code}")
        return pd.DataFrame()

def prepare_grok4_input(data, sequence_length):
    prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    sequences = []
    for i in range(len(scaled_prices) - sequence_length):
        sequences.append(scaled_prices[i:i + sequence_length])
    return np.array(sequences)

def get_grok4_prediction(data, symbol):
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    data_str = ",".join(map(str, data.flatten()))
    payload = {
        "prompt": f"Predict {symbol} price direction (0 for down, 1 for up) based on 60 days of scaled closing prices: {data_str}",
        "model": "grok-4"
    }
    response = requests.post("https://api.x.ai/v1/predict", json=payload, headers=headers)
    return float(response.json()["prediction"])

def analyze_trades(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()

        trades = []
        buy_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Placed BUY order \(dry run, (long|close short)\): (\d+) shares of (\w+) at \$([\d.]+)'
        sell_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Placed SELL order \(dry run, (short|close long)\): (\d+) shares of (\w+) at \$([\d.]+)'

        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        for line in log_content.split('\n'):
            buy_match = re.search(buy_pattern, line)
            if buy_match and buy_match.group(1).startswith(today):
                trades.append({
                    'timestamp': buy_match.group(1),
                    'type': 'BUY',
                    'trade_type': buy_match.group(2),
                    'symbol': buy_match.group(4),
                    'qty': int(buy_match.group(3)),
                    'price': float(buy_match.group(5))
                })
            sell_match = re.search(sell_pattern, line)
            if sell_match and sell_match.group(1).startswith(today):
                trades.append({
                    'timestamp': sell_match.group(1),
                    'type': 'SELL',
                    'trade_type': sell_match.group(2),
                    'symbol': sell_match.group(4),
                    'qty': int(sell_match.group(3)),
                    'price': float(sell_match.group(5))
                })

        pl_by_stock = {symbol: {'long': 0, 'short': 0} for symbol in SYMBOLS}
        for i in range(1, len(trades)):
            prev_trade = trades[i-1]
            curr_trade = trades[i]
            if prev_trade['symbol'] == curr_trade['symbol']:
                qty = min(prev_trade['qty'], curr_trade['qty'])
                if prev_trade['type'] == 'BUY' and prev_trade['trade_type'] == 'long' and curr_trade['type'] == 'SELL' and curr_trade['trade_type'] == 'close long':
                    pl = (curr_trade['price'] - prev_trade['price']) * qty
                    pl_by_stock[prev_trade['symbol']]['long'] += pl
                elif prev_trade['type'] == 'SELL' and prev_trade['trade_type'] == 'short' and curr_trade['type'] == 'BUY' and curr_trade['trade_type'] == 'close short':
                    pl = (prev_trade['price'] - curr_trade['price']) * qty
                    pl_by_stock[prev_trade['symbol']]['short'] += pl

        total_pl = sum(sum(pl.values()) for pl in pl_by_stock.values())
        total_trades = len(trades)
        win_rate = sum(1 for pl in pl_by_stock.values() for v in pl.values() if v > 0) / max(total_trades, 1)

        return {'pl': total_pl, 'trades': total_trades, 'win_rate': win_rate, 'pl_by_stock': pl_by_stock}
    except Exception as e:
        logging.info(f"Error analyzing trades: {str(e)}")
        return {'pl': 0, 'trades': 0, 'win_rate': 0, 'pl_by_stock': {symbol: {'long': 0, 'short': 0} for symbol in SYMBOLS}}

def get_grok4_adjustments(trade_summary):
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    prompt = f"Analyze today's trading: Total P/L: ${trade_summary['pl']:.2f}, Trades: {trade_summary['trades']}, Win Rate: {trade_summary['win_rate']:.2f}. Suggest a prediction threshold (0.45–0.7) and risk per trade (0.005–0.015). Increase risk and lower threshold if win rate is 1.0, otherwise be conservative."
    payload = {
        "prompt": prompt,
        "model": "grok-4"
    }
    response = requests.post("https://api.x.ai/v1/predict", json=payload, headers=headers)
    try:
        result = response.json()
        threshold = min(max(float(result.get("threshold", 0.5)), 0.45), 0.7)
        risk = min(max(float(result.get("risk", 0.01)), 0.005), 0.015)
        return threshold, risk
    except:
        return 0.5, 0.01

def get_position(account_hash, symbol):
    try:
        resp = client.get_account(account_hash)
        if resp.status_code == 200:
            positions = resp.json()["positions"]
            for pos in positions:
                if pos["instrument"]["symbol"] == symbol:
                    return float(pos["quantity"])
        return 0
    except:
        return 0

async def trading_logic():
    # Get account hash
    resp = client.get_account_numbers()
    if resp.status_code != 200:
        logging.info(f"Failed to get account hash: {resp.status_code}")
        return
    account_hash = resp.json()[0]["hashValue"]
    logging.info(f"Account Hash: {account_hash}")

    last_adjustment_time = 0
    while True:
        try:
            # Fetch account info
            resp = client.get_account(account_hash)
            if resp.status_code != 200:
                logging.info(f"Failed to get account info: {resp.status_code}")
                await asyncio.sleep(60)
                continue
            account = resp.json()
            equity = float(account["accountValue"])
            logging.info(f"Account Equity: ${equity:.2f}")

            # Close positions at 3:58 PM ET
            current_time = datetime.now(timezone.utc) - timedelta(hours=4)  # ET time
            if current_time.time() >= time(15, 58):
                for symbol in SYMBOLS:
                    position_qty = get_position(account_hash, symbol)
                    if position_qty > 0:
                        order = equity_sell_market(symbol, position_qty)
                        client.place_order(account_hash, order, dry_run=True)
                        logging.info(f"Closed long position for {symbol} at market close")
                    elif position_qty < 0:
                        order = equity_buy_market(symbol, abs(position_qty))
                        client.place_order(account_hash, order, dry_run=True)
                        logging.info(f"Closed short position for {symbol} at market close")
                await asyncio.sleep(60)
                continue

            # Analyze trades every 30 minutes
            current_time_ts = current_time.timestamp()
            if current_time_ts - last_adjustment_time >= ADJUSTMENT_INTERVAL:
                trade_summary = analyze_trades('/home/gmafanasiev/trading.log')
                threshold, risk_per_trade = get_grok4_adjustments(trade_summary)
                logging.info(f"Grok 4 Adjustments - Threshold: {threshold:.2f}, Risk per Trade: {risk_per_trade:.4f}")
                last_adjustment_time = current_time_ts
            else:
                threshold, risk_per_trade = PREDICTION_THRESHOLD, RISK_PER_TRADE
            risk_per_stock = risk_per_trade / len(SYMBOLS)

            # Process each stock
            for symbol in SYMBOLS:
                bars = await fetch_bar_data(symbol, TIMEFRAME, SEQUENCE_LENGTH + 1)
                if bars.empty:
                    logging.info(f" No data retrieved for {symbol}. Retrying...")
                    continue

                sequences = prepare_grok4_input(bars, SEQUENCE_LENGTH)
                if len(sequences) == 0:
                    logging.info(f"Insufficient data for Grok 4 for {symbol}. Retrying...")
                    continue

                latest_sequence = sequences[-1]
                prediction = get_grok4_prediction(latest_sequence, symbol)
                current_price = bars.iloc[-1]['close']
                position_qty = get_position(account_hash, symbol)
                logging.info(f"{symbol} - Current Price: ${current_price:.2f}, Grok 4 Prediction: {prediction:.2f}, Position: {position_qty}")

                if prediction > UPPER_THRESHOLD and position_qty == 0:
                    risk_amount = min(equity * risk_per_stock, MAX_EQUITY)
                    stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
                    qty = int(risk_amount / (current_price - stop_loss_price))
                    if qty > 0:
                        order = equity_buy_market(symbol, qty)
                        messages, success = client.place_order(account_hash, order, dry_run=True)
                        if success:
                            logging.info(f"Placed BUY order (dry run, long): {qty} shares of {symbol} at ${current_price:.2f}")
                        else:
                            logging.info(f"Buy order failed for {symbol}: {messages}")

                elif prediction < LOWER_THRESHOLD and position_qty == 0:
                    risk_amount = min(equity * risk_per_stock, MAX_EQUITY)
                    stop_loss_price = current_price * (1 + STOP_LOSS_PCT)
                    qty = int(risk_amount / (stop_loss_price - current_price))
                    if qty > 0:
                        order = equity_sell_market(symbol, qty)
                        messages, success = client.place_order(account_hash, order, dry_run=True)
                        if success:
                            logging.info(f"Placed SELL order (dry run, short): {qty} shares of {symbol} at ${current_price:.2f}")
                        else:
                            logging.info(f"Sell order failed for {symbol}: {messages}")

                elif prediction < LOWER_THRESHOLD and position_qty > 0:
                    order = equity_sell_market(symbol, position_qty)
                    messages, success = client.place_order(account_hash, order, dry_run=True)
                    if success:
                        logging.info(f"Placed SELL order (dry run, close long): {position_qty} shares of {symbol} at ${current_price:.2f}")
                    else:
                        logging.info(f"Sell order failed for {symbol}: {messages}")

                elif prediction > UPPER_THRESHOLD and position_qty < 0:
                    order = equity_buy_market(symbol, abs(position_qty))
                    messages, success = client.place_order(account_hash, order, dry_run=True)
                    if success:
                        logging.info(f"Placed BUY order (dry run, close short): {abs(position_qty)} shares of {symbol} at ${current_price:.2f}")
                    else:
                        logging.info(f"Buy order failed for {symbol}: {messages}")

        except Exception as e:
            logging.info(f"Error: {str(e)}")

        await asyncio.sleep(60)

async def main():
    logging.info(f"Starting AI-powered day trader for {SYMBOLS} with Grok 4 and Schwab")
    await trading_logic()

if __name__ == "__main__":
    asyncio.run(main())