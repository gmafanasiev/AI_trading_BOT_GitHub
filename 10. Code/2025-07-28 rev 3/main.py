# main.py
import asyncio
import logging
import signal
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from trading_loop import trading_loop
from trade_analysis import analyze_trades
from logging_utils import setup_logging
from config import ALPACA_API_KEY, ALPACA_API_SECRET, TRADE_MODE, SYMBOLS, BACKTEST_MODE, LOG_LEVEL
from backtest import backtest

# Set up logging globally with conditional log_file based on mode
log_file = 'backtest_log.log' if BACKTEST_MODE else 'trade_alpaca_grok4.log'
logger, file_handler = setup_logging(filename=log_file, level=LOG_LEVEL)

async def main():
    """
    Main entry point for the AI-powered trading bot.
    """
    if BACKTEST_MODE:
        logger.info("Running backtest mode")
        await backtest()
        return

    logger.info("Connected to Alpaca Paper Trading API" if TRADE_MODE != 'live' else "Connected to Alpaca Live Trading API")
    logger.info(f"Starting AI-powered day trader for {SYMBOLS} with Grok 3 and Alpaca")
    file_handler.flush()

    # Initialize Alpaca trading and data clients
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=(TRADE_MODE != 'live'))
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    stream = StockDataStream(ALPACA_API_KEY, ALPACA_API_SECRET)

    # Run the trading loop
    try:
        await trading_loop(trading_client, data_client, log_file, file_handler, stream)
    except asyncio.CancelledError:
        logger.info("Trading loop cancelled, cleaning up...")
        await stream.stop_ws()  # Ensure WebSocket is closed
        file_handler.flush()
        raise
    finally:
        logger.info("Trading loop completed, generating session plot")
        summary = analyze_trades(log_file, trading_client)
        logger.info(f"Session summary: Total P/L: ${summary['pl']:.2f}, Final Equity: ${summary['equity']:.2f}")
        file_handler.flush()
        logger.info("Trading bot execution completed")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stream = StockDataStream(ALPACA_API_KEY, ALPACA_API_SECRET)  # Initialize stream globally for shutdown

    def shutdown(*args):
        logger.info("Shutdown signal received, stopping bot")
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        # Schedule WebSocket cleanup as a task
        if not loop.is_closed():
            asyncio.ensure_future(stream.stop_ws(), loop=loop)
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    loop.run_until_complete(main())
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()