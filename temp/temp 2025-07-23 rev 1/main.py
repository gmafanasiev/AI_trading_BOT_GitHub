# main.py
import asyncio
import logging
from alpaca.trading.client import TradingClient
from trading_loop import trading_loop
from logging_utils import setup_logging
from config import ALPACA_API_KEY, ALPACA_API_SECRET, SIMULATION_MODE

async def main():
    """
    Main entry point for the AI-powered trading bot.
    """
    # Set up logging
    logger, file_handler = setup_logging(level=logging.INFO)
    logger.info("Connected to Alpaca Paper Trading API")
    logger.info("Starting AI-powered day trader for ['AAPL'] with Grok 3 and Alpaca")
    file_handler.flush()

    # Initialize Alpaca trading client
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=SIMULATION_MODE)

    # Run the trading loop
    log_file = 'trade_alpaca_grok4.log'
    await trading_loop(trading_client, log_file, file_handler)
    file_handler.flush()  # Ensure all logs are written to file before exit
    logger.info("Trading bot execution completed")  # Final log to confirm exit

if __name__ == "__main__":
    asyncio.run(main())