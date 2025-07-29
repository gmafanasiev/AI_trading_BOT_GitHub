"""
Test script for Grok-3 prediction in prediction.py.
"""
from config import URL, XAI_API_KEY
from prediction import get_grok4_prediction_and_adjustments

# Mock real-time stats (from prepare_grok4_input)
mock_data = {
    "mean": 0.5,
    "std": 0.1,
    "min": 0.4,
    "max": 0.6,
    "last_5": [0.45, 0.5, 0.55, 0.6, 0.65]
}

# Mock trade summary
mock_trade_summary = {'pl': 0.0, 'trades': 0, 'win_rate': 0.0}

# Run prediction for AAPL
prediction, threshold, risk = get_grok4_prediction_and_adjustments(
    URL, XAI_API_KEY, mock_data, "AAPL", mock_trade_summary
)

print(f"Prediction: {prediction:.2f} (0=down, 1=up)")
print(f"Threshold: {threshold:.2f}")
print(f"Risk: {risk:.4f}")