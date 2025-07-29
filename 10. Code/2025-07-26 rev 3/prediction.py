# prediction.py
import requests
import re
import json
import random
import logging
import numpy as np
from config import URL, XAI_API_KEY, SEQUENCE_LENGTH, UPPER_THRESHOLD, LOWER_THRESHOLD, TRADE_MODE, TEMPERATURE, MARKET_CORR, SUPPORTS, RESISTANCES

"""
Prediction module for fetching Grok-3 predictions.
"""

def load_historical_data(file_path: str) -> dict:
    """
    Load historical analysis data for a symbol.

    :param file_path: Path to JSON file.
    :return: Dict with historical data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading historical data from {file_path}: {e}")
        return {}

async def get_x_sentiment(url: str, api_key: str, symbol: str) -> str:
    """
    Fetch summarized sentiment from recent X posts and news via xAI API.

    :param url: xAI API URL.
    :param api_key: xAI API key.
    :param symbol: Stock symbol.
    :return: Sentiment summary string.
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = (
            f"Summarize recent sentiment on X (Twitter) and news for {symbol} stock in the last 24 hours: "
            f"bullish, bearish, or neutral? Provide 2-3 key points. Output only the summary, no extra text."
        )
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": TEMPERATURE
        }
        logging.debug(f"Sending sentiment request for {symbol}: {payload}")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        sentiment_str = result.get('choices', [{}])[0].get('message', {}).get('content', 'Neutral sentiment, no key updates.')
        logging.info(f"Fetched X/news sentiment for {symbol}: {sentiment_str}")
        return sentiment_str
    except requests.exceptions.HTTPError as e:
        error_response = e.response.json() if e.response and e.response.text else str(e)
        logging.error(f"HTTP error fetching sentiment for {symbol}: {e}, Response: {error_response}")
        return "Neutral sentiment, no key updates."
    except Exception as e:
        logging.error(f"Error fetching sentiment for {symbol}: {e}")
        return "Neutral sentiment, no key updates."

async def get_grok4_prediction_and_adjustments(url: str, api_key: str, stats: dict, symbol: str, trade_summary: dict) -> tuple[float, float, float]:
    """
    Fetch prediction and adjustments from Grok-3, including sentiment and daily context.

    :param url: xAI API URL.
    :param api_key: xAI API key.
    :param stats: Input stats for prediction.
    :param symbol: Stock symbol.
    :param trade_summary: Trade performance summary.
    :return: Tuple of (prediction, threshold, risk_per_trade).
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        sentiment = await get_x_sentiment(url, api_key, symbol)
        prices = stats.get('close_prices', [200.0] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
        prices_str = ",".join(f"{x:.2f}" for x in prices)
        recent_prices = prices[-5:] if len(prices) >= 5 else prices
        trend_pct = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100) if len(recent_prices) >= 2 and recent_prices[0] != 0 else 0.0
        daily_closes_str = ",".join(f"{x:.2f}" for x in stats.get('daily_closes', [])) if 'daily_closes' in stats else "N/A"
        prompt = (
            f"Predict {symbol} 1-minute price direction as a confidence score (0.0=strong down, 1.0=strong up). "
            f"Recent prices (last {SEQUENCE_LENGTH}): [{prices_str}], "
            f"Recent 5-price trend: {trend_pct:.2f}% (positive=up), "
            f"Volume trend: {np.mean(stats.get('volume', [0])[-5:]):.0f} (avg last 5), "
            f"Returns volatility: {stats.get('volatility', 0.05):.2f}, Market corr: {MARKET_CORR}, "
            f"Supports: {SUPPORTS}, Resistances: {RESISTANCES}. "
            f"ATR: {stats.get('atr', 0.05):.2f}. "
            f"Daily closes (last {len(stats.get('daily_closes', []))} days): [{daily_closes_str}], "
            f"Daily volatility: {stats.get('daily_volatility', 0.05):.2f}. "
            f"Recent X/news sentiment: {sentiment}. "
            f"Output ONLY a float: 0.0,0.1,0.2,0.3,0.7,0.8,0.9,1.0. No text."
        )
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": TEMPERATURE
        }
        logging.debug(f"Sending prediction request for {symbol}: {payload}")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logging.info(f"Grok-3 processed payload for {symbol}")
        result = response.json()

        # Extract prediction
        prediction_str = result.get('choices', [{}])[0].get('message', {}).get('content', '0.5')
        match = re.search(r'(\d+\.\d+)', prediction_str)
        if match:
            prediction = float(match.group(1))
            if prediction not in [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0]:
                logging.warning(f"Invalid prediction {prediction}, defaulting to 0.65 for testing")
                prediction = 0.65  # Adjusted for off-hours testing
            if 0.0 <= prediction <= 1.0:
                if TRADE_MODE == 'simulation' and not (0.0 <= prediction <= 0.3 or 0.7 <= prediction <= 1.0):
                    prediction = random.choice([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
                    logging.warning(f"Simulated prediction for {symbol}: {prediction}")
                logging.info(f"Grok-3 returned prediction for {symbol}: {prediction} (0 to 1)")
                threshold = UPPER_THRESHOLD if trade_summary['win_rate'] < 0.3 else (UPPER_THRESHOLD + LOWER_THRESHOLD) / 2
                risk_per_trade = 0.01 if trade_summary['pl'] > -1000 else 0.005
                return prediction, threshold, risk_per_trade
        logging.warning(f"Invalid prediction format: {prediction_str}, defaulting to 0.65")
        return 0.65, UPPER_THRESHOLD, 0.01
    except requests.exceptions.HTTPError as e:
        error_response = e.response.json() if e.response and e.response.text else str(e)
        logging.error(f"HTTP error fetching prediction for {symbol}: {e}, Response: {error_response}")
        return 0.65, UPPER_THRESHOLD, 0.01
    except Exception as e:
        logging.error(f"Error fetching prediction for {symbol}: {e}")
        return 0.65, UPPER_THRESHOLD, 0.01