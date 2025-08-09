import requests
import re
import json
import random
import logging
import numpy as np
import pandas as pd
import time
from config import URL, XAI_API_KEY, SEQUENCE_LENGTH, UPPER_THRESHOLD, LOWER_THRESHOLD, TRADE_MODE, TEMPERATURE, MARKET_CORR, SUPPORTS, RESISTANCES, ENSEMBLE_WEIGHTS, SPY_CORRELATIONS, open_positions, AI_TA_RATIO, BASE_PRICES, RISK_PROFILES, RISK_LEVEL

"""
Prediction module for fetching Grok-3 predictions.
"""

SENTIMENT_TTL = 600  # 10 minutes
sentiment_cache = {}  # symbol: (sentiment_str, timestamp)

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
    current_time = time.time()
    if symbol in sentiment_cache:
        sentiment_str, timestamp = sentiment_cache[symbol]
        if current_time - timestamp < SENTIMENT_TTL:
            logging.info(f"Using cached sentiment for {symbol}")
            return sentiment_str
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        sentiment_str = result.get('choices', [{}])[0].get('message', {}).get('content', 'Neutral sentiment, no key updates.')
        sentiment_summary = re.search(r'\b(bullish|bearish|neutral)\b', sentiment_str, re.I).group(0).lower() if re.search(r'\b(bullish|bearish|neutral)\b', sentiment_str, re.I) else 'Neutral'
        logging.info(f"Fetched X/news sentiment for {symbol}: {sentiment_summary}")
        logging.debug(f"Full X/news sentiment for {symbol}: {sentiment_str}")
        sentiment_cache[symbol] = (sentiment_str, current_time)
        return sentiment_str
    except requests.exceptions.HTTPError as e:
        error_response = e.response.json() if e.response and e.response.text else str(e)
        logging.error(f"HTTP error fetching sentiment for {symbol}: {e}, Response: {error_response}")
        sentiment_str = "Neutral sentiment, no key updates."
        sentiment_cache[symbol] = (sentiment_str, current_time)
        return sentiment_str
    except Exception as e:
        logging.error(f"Error fetching sentiment for {symbol}: {e}")
        sentiment_str = "Neutral sentiment, no key updates."
        sentiment_cache[symbol] = (sentiment_str, current_time)
        return sentiment_str

def compute_dynamic_weights(trade_summary: dict) -> dict:
    """
    Adjust ensemble weights based on trade performance and AI_TA_RATIO.

    :param trade_summary: Trade summary with win_rate and pl.
    :return: Adjusted weights dict.
    """
    win_rate = trade_summary['win_rate']
    pl = trade_summary['pl']
    grok_weight = AI_TA_RATIO + (win_rate - 0.5) * 0.1  # Increase Grok weight if win rate > 50%
    grok_weight = max(0.5, min(0.7, grok_weight))  # Clamp between 0.5 and 0.7
    ta_weight = 1 - grok_weight
    rsi_weight = ta_weight * 0.375
    sma_weight = ta_weight * 0.25
    short_sma_weight = ta_weight * 0.125
    macd_weight = ta_weight * 0.25
    if pl < -1000:
        rsi_weight += 0.1  # Increase RSI weight if losses are high
        macd_weight += 0.1
        grok_weight -= 0.2
    return {
        'grok': grok_weight,
        'rsi': rsi_weight,
        'sma': sma_weight,
        'short_sma': short_sma_weight,
        'macd': macd_weight
    }

def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> float:
    """
    Calculate MACD signal.

    :param prices: Array of closing prices.
    :param fast_period: Fast EMA period.
    :param slow_period: Slow EMA period.
    :param signal_period: Signal line period.
    :return: MACD signal value.
    """
    if len(prices) < slow_period:
        return 0.0
    df = pd.DataFrame({'close': prices})
    exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return histogram.iloc[-1]

def ensemble_prediction(grok_pred: float, rsi: float, sma_signal: float, short_sma_signal: float, macd_signal: float, trade_summary: dict) -> float:
    """
    Compute ensemble prediction with dynamic weights.

    :param grok_pred: Grok prediction (0-1).
    :param rsi: RSI value.
    :param sma_signal: SMA signal (-1/0/1).
    :param short_sma_signal: Short SMA signal (-1/0/1).
    :param macd_signal: MACD signal value.
    :param trade_summary: Trade metrics for weights.
    :return: Ensemble prediction (0-1).
    """
    weights = compute_dynamic_weights(trade_summary)
    rsi_signal = 1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.0
    macd_normalized = np.tanh(macd_signal)  # Normalize to -1 to 1
    prediction = (
        weights['grok'] * grok_pred +
        weights['rsi'] * (rsi_signal + 1) / 2 +  # Map -1/0/1 to 0-1
        weights['sma'] * (sma_signal + 1) / 2 +
        weights['short_sma'] * (short_sma_signal + 1) / 2 +
        weights['macd'] * (macd_normalized + 1) / 2
    )
    return prediction

async def get_grok4_prediction_and_adjustments(stats: dict, trade_summary: dict, symbol: str) -> tuple[float, float, float]:
    """
    Fetch Grok-3 prediction and adjustments.

    :param stats: Statistical inputs from data_utils.
    :param trade_summary: Trade analysis summary.
    :param symbol: Stock symbol.
    :return: Tuple of (prediction, threshold, risk_per_trade).
    """
    sentiment = await get_x_sentiment(URL, XAI_API_KEY, symbol)
    historical_data = load_historical_data(f"{symbol}_historical.json")
    prices = stats.get('close_prices', [BASE_PRICES.get(symbol, 100.0)] * SEQUENCE_LENGTH)
    daily_closes = stats.get('daily_closes', [])
    try:
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        prompt = (
            f"Predict the next 5-minute price movement for {symbol} as a confidence score from 0 (strong sell) to 1 (strong buy). "
            f"Current price: ${stats.get('last_close', BASE_PRICES.get(symbol, 100.0)):.2f}. "
            f"Recent closes: {prices[-5:]}. Volatility: {stats.get('volatility', 0.0):.4f}. "
            f"ATR: {stats.get('atr', 0.0):.4f}. RSI: {stats.get('rsi', 50.0):.2f}. "
            f"SMA: {stats.get('sma', 0.0):.2f}. Supports: {SUPPORTS.get(symbol, [])}. "
            f"Resistances: {RESISTANCES.get(symbol, [])}. "
            f"Daily volatility: {stats.get('daily_volatility', 0.0):.4f}. "
            f"Daily closes: {daily_closes[-3:] if daily_closes else []}. "
            f"Market correlation (SPY): {SPY_CORRELATIONS.get(symbol, MARKET_CORR):.2f}. "
            f"Sentiment: {sentiment}. Historical: {historical_data.get('recent_trends', 'No data')}. "
            f"Output in JSON: {{\"prediction\": score, \"rationale\": \"brief explanation\"}}."
        )
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": TEMPERATURE
        }
        response = requests.post(URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract structured output
        response_str = result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        try:
            grok_output = json.loads(response_str)
            grok_pred = float(grok_output.get('prediction', 0.5))
            rationale = grok_output.get('rationale', 'No rationale provided.')
            rationale_summary = rationale[:50] + "..." if len(rationale) > 50 else rationale
            logging.info(f"Grok-3 rationale for {symbol}: {rationale_summary}")
            logging.debug(f"Full Grok-3 rationale for {symbol}: {rationale}")
        except (json.JSONDecodeError, ValueError):
            logging.warning(f"Invalid JSON format: {response_str}, defaulting to 0.65")
            grok_pred = 0.65

        # Ensemble blending with MACD
        rsi = stats.get('rsi', 50.0)
        last_close = stats.get('last_close', BASE_PRICES.get(symbol, 100.0))
        sma = stats.get('sma', last_close)
        short_sma = stats.get('short_sma', last_close)
        sma_signal = 1.0 if last_close > sma else -1.0 if last_close < sma else 0.0
        short_sma_signal = 1.0 if last_close > short_sma else -1.0 if last_close < short_sma else 0.0
        macd_signal = calculate_macd(prices)
        prediction = ensemble_prediction(grok_pred, rsi, sma_signal, short_sma_signal, macd_signal, trade_summary)

        if TRADE_MODE == 'simulation' and not (0.0 <= prediction <= 0.3 or 0.7 <= prediction <= 1.0):
            prediction = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            logging.warning(f"Simulated prediction for {symbol}: {prediction}")

        # Validate predictions for shorts and longs
        if prediction <= RISK_PROFILES[RISK_LEVEL]['LOWER_THRESHOLD']:
            bearish_conditions = (
                stats.get('rsi', 50.0) > RISK_PROFILES[RISK_LEVEL]['SHORT_RSI_THRESHOLD'] or
                any(abs(stats.get('last_close', BASE_PRICES.get(symbol, 100.0)) - r) / r < 0.005 for r in RESISTANCES.get(symbol, [])) or
                'bearish' in sentiment.lower()
            )
            if not bearish_conditions:
                prediction = 0.5
                logging.warning(f"No bearish signals for {symbol}, neutralizing short prediction to 0.5")
        elif prediction >= RISK_PROFILES[RISK_LEVEL]['UPPER_THRESHOLD']:
            bullish_conditions = (
                stats.get('rsi', 50.0) < 30 or
                any(abs(stats.get('last_close', BASE_PRICES.get(symbol, 100.0)) - s) / s < 0.005 for s in SUPPORTS.get(symbol, [])) or
                'bullish' in sentiment.lower()
            )
            if not bullish_conditions:
                prediction = 0.5
                logging.warning(f"No bullish signals for {symbol}, neutralizing long prediction to 0.5")

        logging.info(f"Ensemble prediction for {symbol}: {prediction:.2f} (0 to 1)")

        # Adjust threshold based on win rate to prevent no-trades issue
        threshold = (RISK_PROFILES[RISK_LEVEL]['UPPER_THRESHOLD'] + RISK_PROFILES[RISK_LEVEL]['LOWER_THRESHOLD']) / 2 if trade_summary['win_rate'] < 0.3 else RISK_PROFILES[RISK_LEVEL]['UPPER_THRESHOLD']
        if trade_summary['win_rate'] < 0.3 and stats.get('rsi', 50.0) > RISK_PROFILES[RISK_LEVEL]['SHORT_RSI_THRESHOLD']:
            threshold = RISK_PROFILES[RISK_LEVEL]['LOWER_THRESHOLD']  # Favor shorts for low win rate and high RSI
        risk_per_trade = 0.01 if trade_summary['pl'] > -1000 else 0.005

        # Log prediction for validation to reduce hallucinations
        actual_price_change = (stats.get('last_close', BASE_PRICES.get(symbol, 100.0)) - stats.get('close_prices', [BASE_PRICES.get(symbol, 100.0)] * SEQUENCE_LENGTH)[-2]) / stats.get('close_prices', [BASE_PRICES.get(symbol, 100.0)] * SEQUENCE_LENGTH)[-2] if len(stats.get('close_prices', [])) >= 2 else 0.0
        predicted_direction = 1.0 if prediction >= 0.5 else -1.0
        actual_direction = 1.0 if actual_price_change > 0 else -1.0 if actual_price_change < 0 else 0.0
        if predicted_direction != actual_direction and abs(actual_price_change) > 0.01:  # Significant price movement
            logging.warning(f"Potential hallucination for {symbol}: Predicted direction {predicted_direction:.2f}, Actual direction {actual_direction:.2f}, Price change {actual_price_change:.2%}")

        return prediction, threshold, risk_per_trade
    except requests.exceptions.HTTPError as e:
        error_response = e.response.json() if e.response and e.response.text else str(e)
        logging.error(f"HTTP error fetching prediction for {symbol}: {e}, Response: {error_response}")
        return 0.5, UPPER_THRESHOLD, 0.01
    except Exception as e:
        logging.error(f"Error fetching prediction for {symbol}: {e}")
        return 0.5, UPPER_THRESHOLD, 0.01