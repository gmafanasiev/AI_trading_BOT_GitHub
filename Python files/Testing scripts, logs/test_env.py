import config
print("ALPACA_API_KEY:", "Set" if config.ALPACA_API_KEY else "Not set")
print("ALPACA_API_SECRET:", "Set" if config.ALPACA_API_SECRET else "Not set")
print("XAI_API_KEY:", "Set" if config.XAI_API_KEY else "Not set")