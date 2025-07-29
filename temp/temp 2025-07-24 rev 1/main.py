# main.py
import asyncio
import logging
import signal
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from trading_loop import trading_loop
from logging_utils import setup_logging
from config import ALPACA_API_KEY, ALPACA_API_SECRET, TRADE_MODE

async def main():
    """
    Main entry point for the AI-powered trading bot.
    """
    # Set up logging
    logger, file_handler = setup_logging(level=logging.INFO)
    logger.info("Connected to Alpaca Paper Trading API" if TRADE_MODE != 'live' else "Connected to Alpaca Live Trading API")
    logger.info("Starting AI-powered day trader for ['AAPL'] with Grok 3 and Alpaca")
    file_handler.flush()

    # Initialize Alpaca trading and data clients
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=(TRADE_MODE != 'live'))
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # Run the trading loop
    log_file = 'trade_alpaca_grok4.log'
    await trading_loop(trading_client, data_client, log_file, file_handler)
    file_handler.flush()  # Ensure all logs are written to file before exit
    logger.info("Trading bot execution completed")  # Final log to confirm exit

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def shutdown():
        logging.info("Shutdown signal received, stopping bot")
        loop.stop()
    signal.signal(signal.SIGINT, lambda s, f: shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown())
    loop.run_until_complete(main())
    loop.close()