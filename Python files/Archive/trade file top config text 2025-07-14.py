# Configure logging
logging.basicConfig(filename='/home/gmafanasiev/trading.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Schwab and xAI API credentials
SCHWAB_APP_KEY = "5AqBefcOK5NJbuv0twAo6s41Ulc2cRlr"
SCHWAB_APP_SECRET = "o76s0GYzvACjHO1e"
SCHWAB_CALLBACK_URL = "https://127.0.0.1:8182"
XAI_API_KEY = "xai-BU2X5OnnITYxBZN5Q3dc8mNrbyw4RrxdrM7caEZvaDZJpASF0e8v6S6IX6HLlQ98WLCvlsK5Ik2HLGhr"
TOKEN_PATH = "/home/gmafanasiev/token.json"