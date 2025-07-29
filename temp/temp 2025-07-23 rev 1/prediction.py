# prediction.py
import requests
import re
import json
import random
import logging
from config import URL, XAI_API_KEY, SEQUENCE_LENGTH, UPPER_THRESHOLD, LOWER_THRESHOLD, SIMULATION_MODE

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

async def get_grok4_prediction_and_adjustments(url: str, api_key: str, stats: dict, symbol: str, trade_summary: dict) -> tuple[float, float, float]:
    """
    Fetch prediction and adjustments from Grok-3.

    :param url: xAI API URL.
    :param api_key: xAI API key.
    :param stats: Input stats for prediction.
    :param symbol: Stock symbol.
    :param trade_summary: Trade performance summary.
    :return: Tuple of (prediction, threshold, risk_per_trade).
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prices = stats.get('close_prices', [200.0] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
        prices_str = ",".join(f"{x:.2f}" for x in prices)
        # Calculate recent price trend (percentage change over last 5 prices)
        recent_prices = prices[-5:] if len(prices) >= 5 else prices
        trend_pct = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100) if len(recent_prices) >= 2 and recent_prices[0] != 0 else 0.0
        prompt = (
            f"Predict {symbol} price direction as a float for the next minute. "
            f"Recent prices (last {SEQUENCE_LENGTH}): [{prices_str}], "
            f"Recent 5-price trend: {trend_pct:.2f}% (positive = up, negative = down), "
            f"Volatility: 0.10, Trend: 0.02, Market correlation: 0.80, "
            f"Supports: [210.03, 207.22, 199.26], Resistances: [213.48, 216.23, 214.65, 213.34]. "
            f"Output only 0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, or 1.0 for high confidence."
        )
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.5
        }
        logging.info(f"Grok-3 received payload for {symbol}")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logging.info(f"Grok-3 processed payload for {symbol}")
        result = response.json()

        # Extract prediction
        prediction_str = result.get('choices', [{}])[0].get('message', {}).get('content', '0.5')

        # Try to extract a float from the response (handles verbose responses with encoding issues)
        match = re.search(r'(\d+\.\d+)', prediction_str)
        if match:
            prediction = float(match.group(1))
            if 0.0 <= prediction <= 1.0:
                # Fallback to simulated prediction in simulation mode if mid-range
                if SIMULATION_MODE and not (0.0 <= prediction <= 0.3 or 0.7 <= prediction <= 1.0):
                    prediction = random.choice([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
                    logging.warning(f"Simulated prediction for {symbol}: {prediction}")
                logging.info(f"Grok-3 returned prediction for {symbol}: {prediction} (0 to 1)")
                threshold = UPPER_THRESHOLD if trade_summary['win_rate'] < 0.3 else (UPPER_THRESHOLD + LOWER_THRESHOLD) / 2
                risk_per_trade = 0.01 if trade_summary['pl'] > -1000 else 0.005
                return prediction, threshold, risk_per_trade
        logging.warning(f"Invalid prediction format: {prediction_str}, defaulting to 0.5")
        return 0.5, UPPER_THRESHOLD, 0.01
    except requests.exceptions.HTTPError as e:
        error_response = e.response.json() if e.response and e.response.text else str(e)
        logging.error(f"Error fetching prediction: {e}, Response: {error_response}")
        if "404" in str(e) or "400" in str(e):
            logging.warning(f"Endpoint/model issue. Try model 'grok-4' or check xAI API docs.")
        return 0.5, UPPER_THRESHOLD, 0.01
    except Exception as e:
        logging.error(f"Error fetching prediction: {e}")
        return 0.5, UPPER_THRESHOLD, 0.01