# test_json_access.py
from prediction import load_historical_analysis
import logging
logging.basicConfig(level=logging.INFO)
result = load_historical_analysis('/home/gmafanasiev/historical_analysis_APPL.json', 'AAPL')
print(f"JSON loaded: {result is not None}")