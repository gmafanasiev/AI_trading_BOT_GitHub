# test_imports_heavy.py
import importlib
import inspect
import os
import logging
import warnings
import ast
import requests
import sys
from dotenv import load_dotenv
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='importlib', message='The NumPy module was reloaded')

for dep in ['numpy', 'pandas', 'sklearn', 'matplotlib']:
    if dep not in sys.modules:
        sys.modules[dep] = importlib.import_module(dep)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = '/home/gmafanasiev/ai_trading_bot'
HOME_DIR = '/home/gmafanasiev'

MODULES = {
    'config': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME', 'URL', 'XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'TRADE_MODE', 'open_positions', 'sl_prices', 'tp_prices'],
    'data_utils': ['fetch_bar_data', 'prepare_grok4_input'],
    'trade_analysis': ['analyze_trades'],
    'prediction': ['get_grok4_prediction_and_adjustments'],
    'trading_loop': ['trading_logic'],
    'order_execution': ['get_position', 'execute_order', 'poll_order_fill', 'update_position'],
    'GenerateAAPLJSON': ['AAPLDataProcessor']
}

DEPENDENCIES = ['pandas', 'numpy', 'sklearn', 'alpaca', 'aiohttp', 'dotenv', 'matplotlib']

CONFIG_IMPORTS = {
    'prediction': ['XAI_API_KEY', 'URL', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD'],
    'trading_loop': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME', 'URL', 'XAI_API_KEY', 'TRADE_MODE', 'open_positions', 'sl_prices', 'tp_prices'],
    'order_execution': ['open_positions', 'sl_prices', 'tp_prices', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRADE_MODE']
}

def check_dependencies():
    errors = []
    for dep in DEPENDENCIES:
        try:
            importlib.import_module(dep)
            logger.info(f"Dependency {dep:<20}: {'OK':>10}")
        except ImportError as e:
            logger.error(f"Dependency {dep:<20}: {'Missing':>10}. Install with: pip install --user {dep}")
            errors.append(str(e))
    return errors

def check_env_file():
    env_path = os.path.join(PROJECT_DIR, '.env')
    errors = []
    if not os.path.exists(env_path):
        logger.error(f".env file {os.path.basename(env_path):<20}: {'Missing':>10}")
        errors.append(f"Missing .env file at {env_path}")
        return errors
    load_dotenv(env_path)
    required_keys = ['XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET']
    for key in required_keys:
        if not os.getenv(key):
            logger.error(f".env key {key:<20}: {'Missing':>10}")
            errors.append(f"Missing {key} in .env")
    if not errors:
        logger.info(f".env file {os.path.basename(env_path):<20}: {'All keys OK':>10}")
    return errors

def check_xai_endpoint(config_module):
    expected_url = "https://api.x.ai/v1/chat/completions"
    actual = getattr(config_module, 'URL', 'None')
    if actual == expected_url:
        logger.info(f"xAI API endpoint {actual:<20}: {'OK':>10}")
        return True
    logger.error(f"xAI API endpoint {actual:<20}: {'Expected ' + expected_url:>10}")
    return False

def check_model_prediction(prediction_module):
    with open(prediction_module.__file__, 'r') as f:
        content = f.read()
    if '"model": "grok-3"' in content:
        logger.info(f"Model in prediction.py {'grok-3':<20}: {'OK':>10}")
        return True
    logger.error(f"Model in prediction.py {'grok-3':<20}: {'Missing or incorrect':>10}")
    return False

def check_logging_config(logging_utils_module):
    try:
        logger, file_handler = logging_utils_module.setup_logging()
        handlers = logger.handlers
        has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in handlers)
        if has_file and has_stream:
            logger.info(f"Logging handlers {'File and Stream':<20}: {'OK':>10}")
            return True
        logger.error(f"Logging handlers {'File and Stream':<20}: {'Missing FileHandler or StreamHandler':>10}")
        return False
    except Exception as e:
        logger.error(f"Logging configuration check {'File and Stream':<20}: {'Error: ' + str(e):>10}")
        return False

def check_cooldown_logic(trading_loop_module):
    with open(trading_loop_module.__file__, 'r') as f:
        content = f.read()
    if "COOLDOWN_SECONDS" in content and "last_trade_time" in content:
        logger.info(f"Cooldown logic {'COOLDOWN_SECONDS':<20}: {'OK':>10}")
        return True
    logger.error(f"Cooldown logic {'COOLDOWN_SECONDS':<20}: {'Missing':>10}")
    return False

def check_trade_filtering(trade_analysis_module):
    with open(trade_analysis_module.__file__, 'r') as f:
        content = f.read()
    if "time_threshold" in content and "datetime" in content:
        logger.info(f"Trade filtering {'time_threshold':<20}: {'OK':>10}")
        return True
    logger.error(f"Trade filtering {'time_threshold':<20}: {'Missing':>10}")
    return False

def check_fetch_bar_data_signature(data_utils_module):
    try:
        fetch_bar_data = getattr(data_utils_module, 'fetch_bar_data')
        sig = inspect.signature(fetch_bar_data)
        expected_params = ['data_client', 'symbol', 'timeframe', 'limit', 'file_handler']
        if list(sig.parameters.keys()) == expected_params:
            logger.info(f"fetch_bar_data signature {'data_client, symbol, timeframe, limit, file_handler':<20}: {'OK':>10}")
            return True
        logger.error(f"fetch_bar_data signature {'data_client, symbol, timeframe, limit, file_handler':<20}: {'Expected ' + str(expected_params) + ', Found ' + str(list(sig.parameters.keys())):>10}")
        return False
    except Exception as e:
        logger.error(f"fetch_bar_data signature check {'data_client, symbol, timeframe, limit, file_handler':<20}: {'Error: ' + str(e):>10}")
        return False

def check_api_connectivity(config_module):
    try:
        headers = {"Authorization": f"Bearer {config_module.XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "grok-3", "messages": [{"role": "user", "content": "Test connectivity"}], "max_tokens": 10}
        response = requests.post(config_module.URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        logger.info(f"xAI API connectivity {'Status = 200':<20}: {'OK':>10}")
        return True
    except Exception as e:
        logger.error(f"xAI API connectivity {'Status = 200':<20}: {'Failed: ' + str(e):>10}")
        return False

def check_config_imports(module_name, module_path, expected_imports):
    errors = []
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=module_path)
        imported_vars = {name.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == 'config' for name in node.names}
        missing_imports = [var for var in expected_imports if var not in imported_vars]
        if missing_imports:
            logger.error(f"Module {module_name:<20}: {'Missing config imports: ' + ', '.join(missing_imports):>10}")
            errors.append(f"Missing config imports in {module_name}: {', '.join(missing_imports)}")
        else:
            logger.info(f"Module {module_name:<20}: {'Config imports OK':>10}")
    except Exception as e:
        logger.error(f"Module {module_name:<20}: {'Config import check error: ' + str(e):>10}")
        errors.append(f"Config import check error in {module_name}: {str(e)}")
    return errors

def check_parameters(config_module):
    expected_params = {'UPPER_THRESHOLD': 0.65, 'LOWER_THRESHOLD': 0.35, 'STOP_LOSS_PCT': 0.10, 'SEQUENCE_LENGTH': 5}  # Updated to 5
    errors = []
    for param, expected in expected_params.items():
        actual = getattr(config_module, param, None)
        if actual == expected:
            logger.info(f"Parameter {param:<20}: {'OK':>10}")
        else:
            logger.error(f"Parameter {param:<20}: {'Expected ' + str(expected) + ', Found ' + str(actual):>10}")
            errors.append(f"Incorrect parameter {param}: Expected {expected}, Found {actual}")
    adj_interval = getattr(config_module, 'ADJUSTMENT_INTERVAL', None)
    if isinstance(adj_interval, int) and adj_interval > 0:
        logger.info(f"Parameter {'ADJUSTMENT_INTERVAL':<20}: {'OK (Value: ' + str(adj_interval) + ')':>10}")
    else:
        logger.error(f"Parameter {'ADJUSTMENT_INTERVAL':<20}: {'Invalid or missing, Found ' + str(adj_interval):>10}")
        errors.append(f"Invalid or missing ADJUSTMENT_INTERVAL: Found {adj_interval}")
    return errors

def check_module(module_name, expected_items):
    module_path = os.path.join(HOME_DIR if module_name == 'GenerateAAPLJSON' else PROJECT_DIR, f"{module_name}.py")
    errors = []
    if not os.path.exists(module_path):
        if module_name == 'GenerateAAPLJSON':
            logger.warning(f"Module {module_name:<20}: {'Optional, not found, continuing':>10}")
            return errors
        logger.error(f"Module {module_name:<20}: {'Not found at ' + module_path:>10}")
        errors.append(f"Module {module_name} not found at {module_path}")
        return errors
    try:
        with open(module_path, 'r') as f:
            ast.parse(f.read())
        logger.info(f"Syntax check for {module_name:<20}: {'OK':>10}")
    except SyntaxError as e:
        logger.error(f"Syntax error in {module_name:<20}: {str(e):>10}")
        errors.append(f"Syntax error in {module_name}: {str(e)}")
        return errors
    if module_name in CONFIG_IMPORTS:
        errors.extend(check_config_imports(module_name, module_path, CONFIG_IMPORTS[module_name]))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        logger.info(f"Module {module_name:<20}: {'Imported OK':>10}")
        missing_items = [item for item in expected_items if not hasattr(module, item)]
        for item in expected_items:
            if item in missing_items: continue
            obj = getattr(module, item)
            if inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
                logger.info(f"  {item:<48}: {'Async' if inspect.iscoroutinefunction(obj) else 'Sync'} function OK")
            else:
                logger.info(f"  {item:<48}: {type(obj).__name__} OK")
        if missing_items:
            logger.error(f"Module {module_name:<20}: {'Missing items: ' + ', '.join(missing_items):>10}")
            errors.append(f"Missing items in {module_name}: {', '.join(missing_items)}")
        else:
            logger.info(f"Module {module_name:<20}: {'All items OK':>10}")
    except Exception as e:
        logger.error(f"Module {module_name:<20}: {'Import error: ' + str(e):>10}")
        errors.append(f"Import error in {module_name}: {str(e)}")
    return errors

def main():
    logger.info("Starting import consistency check for AI trading bot")
    all_errors = check_dependencies() + check_env_file()
    for module_name, items in MODULES.items():
        all_errors.extend(check_module(module_name, items))
    config_module = sys.modules.get('config')
    prediction_module = sys.modules.get('prediction')
    logging_utils_module = sys.modules.get('logging_utils')
    trading_loop_module = sys.modules.get('trading_loop')
    trade_analysis_module = sys.modules.get('trade_analysis')
    data_utils_module = sys.modules.get('data_utils')
    if config_module:
        all_errors.extend(check_parameters(config_module))
        if not check_xai_endpoint(config_module): all_errors.append("Incorrect xAI API endpoint")
        if not check_api_connectivity(config_module): all_errors.append("xAI API connectivity failed")
    if prediction_module and not check_model_prediction(prediction_module): all_errors.append("Incorrect model in prediction.py")
    if logging_utils_module and not check_logging_config(logging_utils_module): all_errors.append("Incorrect logging configuration")
    if trading_loop_module and not check_cooldown_logic(trading_loop_module): all_errors.append("Missing cooldown logic in trading_loop.py")
    if trade_analysis_module and not check_trade_filtering(trade_analysis_module): all_errors.append("Missing trade filtering in trade_analysis.py")
    if data_utils_module and not check_fetch_bar_data_signature(data_utils_module): all_errors.append("Incorrect fetch_bar_data signature in data_utils.py")
    json_path = os.path.join(HOME_DIR, 'historical_analysis_AAPL.json')
    if not os.path.exists(json_path):
        logger.error(f"JSON file {os.path.basename(json_path):<20}: {'Missing':>10}")
        all_errors.append(f"JSON file missing at {json_path}")
    else:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                import json
                json.load(f)
            logger.info(f"JSON file {os.path.basename(json_path):<20}: {'Valid':>10}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON file {os.path.basename(json_path):<20}: {'Invalid JSON: ' + str(e):>10}")
            all_errors.append(f"Invalid JSON in {json_path}: {str(e)}")
    if all_errors:
        logger.error(f"Consistency checks failed with {len(all_errors)} errors:")
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  Error {i}: {error}")
        sys.exit(1)
    else:
        logger.info("All checks passed! Ready to run main.py")

if __name__ == "__main__":
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