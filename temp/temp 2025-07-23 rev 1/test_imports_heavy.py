# test_imports_heavy.py
import importlib
import inspect
import os
import logging
import warnings
import ast
import requests
from pathlib import Path
import sys

# Suppress NumPy reload warning
warnings.filterwarnings('ignore', category=UserWarning, module='importlib',
                        message='The NumPy module was reloaded')

# Preload dependencies to minimize reloads
for dep in ['numpy', 'pandas', 'sklearn', 'matplotlib']:
    if dep not in sys.modules:
        module = importlib.import_module(dep)
        sys.modules[dep] = module

# Set up logging with aligned format
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Directories
PROJECT_DIR = '/home/gmafanasiev/ai_trading_bot'
HOME_DIR = '/home/gmafanasiev'

# List of modules and their expected items
MODULES = {
    'config': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD',
               'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY',
               'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME',
               'URL', 'XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET',
               'SIMULATION_MODE', 'open_positions', 'sl_prices', 'tp_prices'],
    'data_utils': ['fetch_bar_data', 'prepare_grok4_input'],
    'trade_analysis': ['analyze_trades'],
    'prediction': ['get_grok4_prediction_and_adjustments'],
    'trading_loop': ['trading_logic'],
    'order_execution': ['get_position', 'execute_order'],
    'GenerateAAPLJSON': ['AAPLDataProcessor']
}

# Required dependencies
DEPENDENCIES = ['pandas', 'numpy', 'sklearn', 'alpaca', 'aiohttp', 'dotenv', 'matplotlib']

# Variables that should be imported from config.py
CONFIG_IMPORTS = {
    'prediction': ['XAI_API_KEY', 'URL', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD'],
    'trading_loop': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD',
                    'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY',
                    'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME',
                    'URL', 'XAI_API_KEY', 'SIMULATION_MODE', 'open_positions',
                    'sl_prices', 'tp_prices'],
    'order_execution': ['open_positions', 'sl_prices', 'tp_prices', 'STOP_LOSS_PCT',
                       'TAKE_PROFIT_PCT', 'SIMULATION_MODE']
}

def check_dependencies():
    """Check if required libraries are installed."""
    logger.info("Checking dependencies...")
    errors = []
    for dep in DEPENDENCIES:
        try:
            importlib.import_module(dep)
            logger.info(f"Dependency {dep:.<50}: OK")
        except ImportError as e:
            logger.error(f"Dependency {dep:.<50}: Missing. Install with: pip install --user {dep}")
            errors.append(str(e))
    return errors

def check_env_file():
    """Verify .env file exists and contains required keys."""
    logger.info("Checking .env file...")
    env_path = os.path.join(PROJECT_DIR, '.env')
    errors = []
    if not os.path.exists(env_path):
        logger.error(f".env file {env_path:.<50}: Missing")
        errors.append(f"Missing .env file at {env_path}")
        return errors

    from dotenv import load_dotenv
    load_dotenv(env_path)
    required_keys = ['XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET']
    for key in required_keys:
        if not os.getenv(key):
            logger.error(f".env key {key:.<50}: Missing")
            errors.append(f"Missing {key} in .env")
    if not errors:
        logger.info(f".env file {env_path:.<50}: All keys OK")
    return errors

def check_xai_endpoint(config_module):
    """Check if config.py uses the correct xAI API endpoint."""
    expected_url = "https://api.x.ai/v1/chat/completions"
    if hasattr(config_module, 'URL') and config_module.URL == expected_url:
        logger.info(f"xAI API endpoint {config_module.URL:.<50}: OK")
        return True
    else:
        logger.error(f"xAI API endpoint {getattr(config_module, 'URL', 'None'):.<50}: Expected {expected_url}")
        return False

def check_model_prediction(prediction_module):
    """Check if prediction.py uses model: 'grok-3'."""
    with open(prediction_module.__file__, 'r') as f:
        content = f.read()
    if '"model": "grok-3"' in content:
        logger.info(f"Model in prediction.py {'grok-3':.<50}: OK")
        return True
    else:
        logger.error(f"Model in prediction.py {'grok-3':.<50}: Missing or incorrect")
        return False

def check_logging_config(logging_utils_module):
    """Check if logging_utils.py has FileHandler and StreamHandler."""
    try:
        logger, file_handler = logging_utils_module.setup_logging()
        handlers = logger.handlers
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in handlers)
        has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in handlers)
        if has_file_handler and has_stream_handler:
            logger.info(f"Logging handlers {'File and Stream':.<50}: OK")
            return True
        else:
            logger.error(f"Logging handlers {'File and Stream':.<50}: Missing FileHandler or StreamHandler")
            return False
    except Exception as e:
        logger.error(f"Logging configuration check {'File and Stream':.<50}: Error: {str(e)}")
        return False

def check_cooldown_logic(trading_loop_module):
    """Check if trading_loop.py has cooldown logic."""
    with open(trading_loop_module.__file__, 'r') as f:
        content = f.read()
    if "COOLDOWN_SECONDS" in content and "last_trade_time" in content:
        logger.info(f"Cooldown logic {'COOLDOWN_SECONDS':.<50}: OK")
        return True
    else:
        logger.error(f"Cooldown logic {'COOLDOWN_SECONDS':.<50}: Missing")
        return False

def check_trade_filtering(trade_analysis_module):
    """Check if trade_analysis.py has time-based filtering."""
    with open(trade_analysis_module.__file__, 'r') as f:
        content = f.read()
    if "time_threshold" in content and "datetime" in content:
        logger.info(f"Trade filtering {'time_threshold':.<50}: OK")
        return True
    else:
        logger.error(f"Trade filtering {'time_threshold':.<50}: Missing")
        return False

def check_fetch_bar_data_signature(data_utils_module):
    """Check if fetch_bar_data in data_utils.py has the correct signature."""
    try:
        fetch_bar_data = getattr(data_utils_module, 'fetch_bar_data')
        sig = inspect.signature(fetch_bar_data)
        expected_params = ['data_client', 'symbol', 'timeframe', 'limit', 'file_handler']
        actual_params = list(sig.parameters.keys())
        if actual_params == expected_params:
            logger.info(f"fetch_bar_data signature {'data_client, symbol, timeframe, limit, file_handler':.<50}: OK")
            return True
        else:
            logger.error(f"fetch_bar_data signature {'data_client, symbol, timeframe, limit, file_handler':.<50}: Expected {expected_params}, Found {actual_params}")
            return False
    except Exception as e:
        logger.error(f"fetch_bar_data signature check {'data_client, symbol, timeframe, limit, file_handler':.<50}: Error: {str(e)}")
        return False

def check_api_connectivity(config_module):
    """Test xAI API connectivity with a sample request."""
    try:
        headers = {"Authorization": f"Bearer {config_module.XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": "Test connectivity"}],
            "max_tokens": 10
        }
        response = requests.post(config_module.URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        logger.info(f"xAI API connectivity {'Status = 200':.<50}: OK")
        return True
    except Exception as e:
        logger.error(f"xAI API connectivity {'Status = 200':.<50}: Failed: {str(e)}")
        return False

def check_config_imports(module_name, module_path, expected_imports):
    """Check if module imports required variables from config.py."""
    errors = []
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=module_path)

        imported_vars = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'config':
                imported_vars.extend(name.name for name in node.names)

        missing_imports = [var for var in expected_imports if var not in imported_vars]
        if missing_imports:
            logger.error(f"Module {module_name:.<50}: Missing config imports: {', '.join(missing_imports)}")
            errors.append(f"Missing config imports in {module_name}: {', '.join(missing_imports)}")
        else:
            logger.info(f"Module {module_name:.<50}: Config imports OK")
    except Exception as e:
        logger.error(f"Module {module_name:.<50}: Config import check error: {str(e)}")
        errors.append(f"Config import check error in {module_name}: {str(e)}")
    return errors

def check_parameters(config_module):
    """Check consistency of key parameters."""
    expected_params = {
        'UPPER_THRESHOLD': 0.65,
        'LOWER_THRESHOLD': 0.35,
        'STOP_LOSS_PCT': 0.10,
        'SEQUENCE_LENGTH': 60
    }
    errors = []
    for param, expected in expected_params.items():
        if hasattr(config_module, param) and getattr(config_module, param) == expected:
            logger.info(f"Parameter {param:.<50}: OK")
        else:
            logger.error(f"Parameter {param:.<50}: Expected {expected}, Found {getattr(config_module, param, 'None')}")
            errors.append(f"Incorrect parameter {param}: Expected {expected}, Found {getattr(config_module, param, 'None')}")
    # Check ADJUSTMENT_INTERVAL exists and is a positive integer
    if hasattr(config_module, 'ADJUSTMENT_INTERVAL') and isinstance(getattr(config_module, 'ADJUSTMENT_INTERVAL'), int) and getattr(config_module, 'ADJUSTMENT_INTERVAL') > 0:
        logger.info(f"Parameter ADJUSTMENT_INTERVAL:.<50: OK (Value: {getattr(config_module, 'ADJUSTMENT_INTERVAL')})")
    else:
        logger.error(f"Parameter ADJUSTMENT_INTERVAL:.<50: Invalid or missing, Found {getattr(config_module, 'ADJUSTMENT_INTERVAL', 'None')}")
        errors.append(f"Invalid or missing ADJUSTMENT_INTERVAL: Found {getattr(config_module, 'ADJUSTMENT_INTERVAL', 'None')}")
    return errors

def check_module(module_name, expected_items):
    """Check if module exists and contains expected items."""
    if module_name == 'GenerateAAPLJSON':
        module_path = os.path.join(HOME_DIR, f"{module_name}.py")
    else:
        module_path = os.path.join(PROJECT_DIR, f"{module_name}.py")

    errors = []
    if not os.path.exists(module_path):
        if module_name == 'GenerateAAPLJSON':
            logger.warning(f"Module {module_name:.<50}: Optional, not found, continuing")
            return errors
        logger.error(f"Module {module_name:.<50}: Not found at {module_path}")
        errors.append(f"Module {module_name} not found at {module_path}")
        return errors

    # Check config imports for specific modules
    if module_name in CONFIG_IMPORTS:
        errors.extend(check_config_imports(module_name, module_path, CONFIG_IMPORTS[module_name]))

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        logger.info(f"Module {module_name:.<50}: Imported OK")

        missing_items = []
        for item in expected_items:
            if not hasattr(module, item):
                missing_items.append(item)
            else:
                obj = getattr(module, item)
                if inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
                    is_async = inspect.iscoroutinefunction(obj)
                    logger.info(f"  {item:.<48}: {'Async' if is_async else 'Sync'} function OK")
                else:
                    logger.info(f"  {item:.<48}: {type(obj).__name__} OK")

        if missing_items:
            logger.error(f"Module {module_name:.<50}: Missing items: {', '.join(missing_items)}")
            errors.append(f"Missing items in {module_name}: {', '.join(missing_items)}")
        else:
            logger.info(f"Module {module_name:.<50}: All items OK")
    except Exception as e:
        logger.error(f"Module {module_name:.<50}: Import error: {str(e)}")
        errors.append(f"Import error in {module_name}: {str(e)}")
    return errors

def main():
    """Run consistency checks for all modules."""
    logger.info("Starting import consistency check for AI trading bot")
    all_errors = []

    # Check dependencies
    all_errors.extend(check_dependencies())

    # Check .env file
    all_errors.extend(check_env_file())

    # Check modules
    for module_name, items in MODULES.items():
        all_errors.extend(check_module(module_name, items))

    # Check specific configurations
    config_module = sys.modules.get('config')
    prediction_module = sys.modules.get('prediction')
    logging_utils_module = sys.modules.get('logging_utils')
    trading_loop_module = sys.modules.get('trading_loop')
    trade_analysis_module = sys.modules.get('trade_analysis')
    data_utils_module = sys.modules.get('data_utils')

    if config_module:
        all_errors.extend(check_parameters(config_module))
        if not check_xai_endpoint(config_module):
            all_errors.append("Incorrect xAI API endpoint")
        if not check_api_connectivity(config_module):
            all_errors.append("xAI API connectivity failed")
    if prediction_module:
        if not check_model_prediction(prediction_module):
            all_errors.append("Incorrect model in prediction.py")
    if logging_utils_module:
        if not check_logging_config(logging_utils_module):
            all_errors.append("Incorrect logging configuration")
    if trading_loop_module:
        if not check_cooldown_logic(trading_loop_module):
            all_errors.append("Missing cooldown logic in trading_loop.py")
    if trade_analysis_module:
        if not check_trade_filtering(trade_analysis_module):
            all_errors.append("Missing trade filtering in trade_analysis.py")
    if data_utils_module:
        if not check_fetch_bar_data_signature(data_utils_module):
            all_errors.append("Incorrect fetch_bar_data signature in data_utils.py")

    # Check JSON file
    logger.info("Checking JSON file...")
    json_path = os.path.join(HOME_DIR, 'historical_analysis_AAPL.json')
    if not os.path.exists(json_path):
        logger.error(f"JSON file {json_path:.<50}: Missing")
        all_errors.append(f"JSON file missing at {json_path}")
    else:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                import json
                json.load(f)
            logger.info(f"JSON file {json_path:.<50}: Valid")
        except json.JSONDecodeError as e:
            logger.error(f"JSON file {json_path:.<50}: Invalid JSON: {str(e)}")
            all_errors.append(f"Invalid JSON in {json_path}: {str(e)}")

    if all_errors:
        logger.error(f"Consistency checks failed with {len(all_errors)} errors:")
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  Error {i}: {error}")
        sys.exit(1)
    else:
        logger.info("All checks passed! Ready to run main.py")

if __name__ == "__main__":
    # Prevent duplicate runs
    lock_file = os.path.join(PROJECT_DIR, 'test_imports_heavy.lock')
    if os.path.exists(lock_file):
        logger.error(f"Script already running or did not exit cleanly.....: Remove {lock_file} and try again")
        sys.exit(1)

    try:
        with open(lock_file, 'w') as f:
            f.write('running')
        main()
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)