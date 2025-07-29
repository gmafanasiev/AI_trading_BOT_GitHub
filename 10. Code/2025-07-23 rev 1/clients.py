# clients.py
import logging

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

"""
Alpaca client initialization module for the AI trading bot.
"""

def init_alpaca_clients(api_key, api_secret):
    """
    Initialize Alpaca trading and data clients.
    
    :param api_key: Alpaca API key.
    :param api_secret: Alpaca API secret.
    :return: Tuple of (trading_client, data_client).
    """
    try:
        trading_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
        data_client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        logging.info("Connected to Alpaca Paper Trading API")
        return trading_client, data_client
    except ValueError as e:
        logging.error("Failed to connect to Alpaca API: %s", e)
        raise